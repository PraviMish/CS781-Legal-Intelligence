"""Phase-1 guarded generation and the later GQA decoding sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from phase1.models import build_model


@dataclass(frozen=True)
class DecodingConfig:
    name: str = 'legacy_word_guard_v1'
    strategy: str = 'sample'
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.15
    no_repeat_ngram: int = 4
    min_generation_tokens: int = 560
    max_tokens_without_word: int = 3
    seed_offset: int = 0


LEGACY = DecodingConfig()
GREEDY_FULL_PROMPT = DecodingConfig(
    name='greedy_full_prompt', strategy='greedy', temperature=1.0, top_k=None,
    top_p=None, repetition_penalty=1.05, no_repeat_ngram=4,
    min_generation_tokens=280, max_tokens_without_word=5)
LOW_TEMPERATURE = DecodingConfig(
    name='full_prompt_low_temperature', temperature=0.65, top_k=30,
    top_p=0.90, repetition_penalty=1.05, no_repeat_ngram=4,
    min_generation_tokens=280, max_tokens_without_word=6)
CONSERVATIVE = DecodingConfig(
    name='full_prompt_conservative', temperature=0.70, top_k=40,
    top_p=0.90, repetition_penalty=1.08, no_repeat_ngram=4,
    min_generation_tokens=280, max_tokens_without_word=5)
BALANCED = DecodingConfig(
    name='full_prompt_balanced', temperature=0.80, top_k=50,
    top_p=0.95, repetition_penalty=1.10, no_repeat_ngram=4,
    min_generation_tokens=280, max_tokens_without_word=5)
SWEEP_LEGACY = replace(LEGACY, name='legacy_guarded')
CONFIGS = {x.name: x for x in [
    LEGACY, GREEDY_FULL_PROMPT, LOW_TEMPERATURE, CONSERVATIVE, BALANCED, SWEEP_LEGACY,
]}


def exactly_128_words(text: str) -> str | None:
    words = text.strip().split()
    return ' '.join(words[:128]) if len(words) >= 128 else None


def banned_ngram_tokens(generated: list[int], n: int) -> set[int]:
    """Match notebook's generated-continuation-only no-repeat n-gram rule."""
    if n <= 1 or len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):])
    return {generated[i + n - 1] for i in range(len(generated) - n + 1)
            if tuple(generated[i:i+n-1]) == prefix}


class LegalDecoder:
    """Contains the tokenizer-dependent constants instead of global notebook cells."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = [tokenizer.token_to_id(token) for token in ('<pad>', '<bos>', '<eos>')]
        self.bos = tokenizer.token_to_id('<bos>')
        unk = tokenizer.token_to_id('<unk>')
        if any(x is None for x in self.special_ids) or self.bos is None:
            raise ValueError('The legal tokenizer is missing required special tokens')
        self.word_start_ids = []
        for token_id in range(tokenizer.get_vocab_size()):
            if token_id in self.special_ids or token_id == unk:
                continue
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            if piece and piece[0].isspace() and piece.strip() and any(ch.isalpha() for ch in piece.strip()):
                self.word_start_ids.append(token_id)
        if not self.word_start_ids:
            raise ValueError('No natural BPE word-start tokens found')
        self.word_starts = set(self.word_start_ids)

    def _logits(self, logits: torch.Tensor, generated: list[int], cfg: DecodingConfig,
                tokens_without_word: int) -> torch.Tensor:
        logits = logits.clone()
        for sid in self.special_ids:
            logits[:, sid] = -float('inf')
        # Historical sign-aware repetition penalty.
        for token_id in set(generated):
            value = logits[0, token_id]
            logits[0, token_id] = value * cfg.repetition_penalty if value < 0 else value / cfg.repetition_penalty
        banned = banned_ngram_tokens(generated, cfg.no_repeat_ngram)
        if banned:
            logits[:, list(banned)] = -float('inf')
        if tokens_without_word >= cfg.max_tokens_without_word:
            forced = torch.full_like(logits, -float('inf'))
            forced[:, self.word_start_ids] = logits[:, self.word_start_ids]
            if torch.isfinite(forced).any():
                logits = forced
        return logits

    @staticmethod
    def _sample(logits: torch.Tensor, cfg: DecodingConfig, generator: torch.Generator) -> torch.Tensor:
        if cfg.strategy == 'greedy':
            return logits.argmax(dim=-1)
        logits = logits / cfg.temperature
        if cfg.top_k is not None and cfg.top_k > 0:
            k = min(cfg.top_k, logits.size(-1))
            cutoff = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -float('inf')), logits)
        if cfg.top_p is not None and cfg.top_p < 1:
            ordered, indices = logits.sort(dim=-1, descending=True)
            remove = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1) > cfg.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            ordered[remove] = -float('inf')
            filtered = torch.full_like(logits, -float('inf'))
            logits = filtered.scatter(-1, indices, ordered)
        return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)

    @staticmethod
    def _seed(prompt: str, cfg: DecodingConfig, device: torch.device) -> torch.Generator:
        # Original 100M generation seed depends on prompt only; later tuned
        # generation is identified separately by a named decoding configuration.
        text = prompt if cfg.name == 'legacy_word_guard_v1' else f'{cfg.name}||{prompt}'
        seed = int(hashlib.sha1(text.encode('utf-8')).hexdigest()[:8], 16) + cfg.seed_offset
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @torch.no_grad()
    def generate(self, model, architecture: str, prompt: str,
                 config: DecodingConfig = LEGACY, emergency_retry: bool = True) -> str:
        """Generate exactly 128 words; second attempt mirrors original emergency guard."""
        try:
            return self._generate_once(model, architecture, prompt, config)
        except RuntimeError:
            if not emergency_retry:
                raise
            # The sweep switched to guarded sampling on retry, including after
            # greedy decoding failed. The initial decoder used a seed offset.
            if config.name == LEGACY.name:
                backup = replace(LEGACY, max_tokens_without_word=1, seed_offset=7919)
            else:
                backup = replace(LEGACY, name=config.name + '_emergency', max_tokens_without_word=1)
            return self._generate_once(model, architecture, prompt, backup)

    @torch.no_grad()
    def _generate_once(self, model, architecture: str, prompt: str, cfg: DecodingConfig) -> str:
        device = next(model.parameters()).device
        generator = self._seed(prompt, cfg, device)
        generated: list[int] = []
        without_word = 0
        if architecture == 'encoder_decoder':
            src_ids = self.tokenizer.encode(prompt).ids[-model.encoder_max_len:]
            source = torch.tensor([src_ids], dtype=torch.long, device=device)
            memory = model.encode(source)
            current = torch.tensor([[self.bos]], dtype=torch.long, device=device)
            self_caches = None
            memory_caches = None
            maximum = model.decoder_max_len - 1
        else:
            ids = self.tokenizer.encode(prompt).ids
            prompt_limit = model.max_seq_len - cfg.min_generation_tokens
            if prompt_limit <= 0:
                raise ValueError('Generation reserve exceeds context length')
            ids = ids[-prompt_limit:]
            maximum = model.max_seq_len - len(ids)
            current = torch.tensor([ids], dtype=torch.long, device=device)
            if architecture == 'prefix_lm':
                logits, cache = model(current, prefix_length=len(ids), use_cache=True, last_only=True)
            else:
                logits, cache = model(current, use_cache=True, last_only=True)
        for step in range(maximum):
            if architecture == 'encoder_decoder':
                logits, self_caches, memory_caches = model.decode(
                    current, memory, self_caches=self_caches,
                    memory_caches=memory_caches, use_cache=True, last_only=True)
            selected_logits = self._logits(logits[:, -1, :], generated, cfg, without_word)
            next_id = self._sample(selected_logits, cfg, generator)
            token = int(next_id.item())
            generated.append(token)
            without_word = 0 if token in self.word_starts else without_word + 1
            # Exactly match original check frequency / threshold.
            if len(generated) >= 100 and step % 4 == 0:
                completed = exactly_128_words(self.tokenizer.decode(generated))
                if completed is not None:
                    return completed
            current = next_id[:, None]
            if architecture != 'encoder_decoder':
                logits, cache = model(current, past_cache=cache, use_cache=True, last_only=True)
        completed = exactly_128_words(self.tokenizer.decode(generated))
        if completed is None:
            raise RuntimeError(f'Decoder produced fewer than 128 words ({len(generated)} tokens)')
        return completed


def load_trained_model(architecture: str, checkpoint_path: Path, device: str | None = None,
                       vocab_size: int = 32000, context_length: int = 1280):
    """Load a trusted notebook-derived checkpoint with unchanged state keys."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(architecture, vocab_size, context_length).to(device)
    # Historical Phase-1 model files include Python RNG / optimizer objects.
    # Load ONLY checkpoints generated by this project and personally trusted.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state'], strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description='Generate one legal 128-word continuation')
    ap.add_argument('--architecture', choices=list(build_model.__globals__['MODEL_NAMES']), required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--decoding', choices=list(CONFIGS), default=LEGACY.name)
    args = ap.parse_args()
    model = load_trained_model(args.architecture, args.checkpoint)
    generator = LegalDecoder(Tokenizer.from_file(str(args.tokenizer)))
    print(generator.generate(model, args.architecture, args.prompt, CONFIGS[args.decoding]))


if __name__ == '__main__':
    main()
