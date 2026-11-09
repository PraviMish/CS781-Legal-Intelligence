"""Cached greedy decoding and revised document-level aggregation."""

import re
from contextlib import nullcontext

import torch

from .data import BOS, EOS, PAD, UNK, sentences, words


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if torch.device(device).type != "cuda":
        raise ValueError("Use fp32 for CPU inference and training")
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def next_token(logits, generated, step, minimum, penalty):
    scores = logits[0, -1].float().clone()
    scores[[PAD, BOS, UNK]] = -torch.inf
    if step < minimum:
        scores[EOS] = -torch.inf
    if generated:
        used = list(set(generated))
        old = scores[used]
        scores[used] = torch.where(old < 0, old * penalty, old / penalty)
    if len(generated) >= 2:
        banned = {generated[i + 2] for i in range(len(generated) - 2)
                  if generated[i:i + 2] == generated[-2:]}
        if banned:
            scores[list(banned)] = -torch.inf
    if not torch.isfinite(scores).any():
        raise RuntimeError("No finite token satisfies the decoding constraints")
    return int(scores.argmax().item())


@torch.inference_mode()
def generate_passage(model, architecture, source, tokenizer, device, precision="fp32",
                     original=False, max_new_tokens=None):
    model.eval()
    cap = max_new_tokens if max_new_tokens is not None else (192 if original else 260)
    minimum, penalty = (40, 1.12) if original else (32, 1.10)
    tensor = torch.tensor([source], dtype=torch.long, device=device)
    with amp_context(device, precision):
        if architecture == "encoder_decoder":
            memory = model.encode(tensor)
            bos = torch.tensor([[BOS]], device=device)
            logits, cache, memory_cache = model.decode(bos, memory, use_cache=True, last_only=True)
            steps = cap if original else min(cap, model.decoder_max_len - 1)
        else:
            prefix = {"prefix_length": len(source)} if architecture == "prefix_lm" else {}
            logits, cache = model(tensor, use_cache=True, last_only=True, **prefix)
            steps = min(cap, model.max_seq_len - len(source))
    if steps < 1:
        raise ValueError("Generation requires at least one available token")
    generated = []
    for step in range(steps):
        token = next_token(logits, generated, step, minimum, penalty)
        if token == EOS:
            break
        generated.append(token)
        if step + 1 == steps:
            break
        tensor = torch.tensor([[token]], device=device)
        with amp_context(device, precision):
            if architecture == "encoder_decoder":
                logits, cache, memory_cache = model.decode(
                    tensor, memory, self_caches=cache, memory_caches=memory_cache,
                    use_cache=True, last_only=True,
                )
            else:
                logits, cache = model(tensor, past_cache=cache, use_cache=True, last_only=True)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not original:
        ends = list(re.finditer(r"[.!?](?=\s|$)", text))
        if ends and ends[-1].end() >= .65 * len(text):
            text = text[:ends[-1].end()].strip()
    return text


def dedupe(texts):
    accepted = []
    seen = []
    for text in texts:
        for sentence in sentences(text):
            bag = words(sentence)
            if bag and not any(len(bag & old) / len(bag | old) > .73 for old in seen):
                accepted.append(sentence)
                seen.append(bag)
    return " ".join(accepted)


def evidence_filter(text, document):
    """V3 optional lexical filter. Fall back when fewer than 30 words survive."""
    original = document.lower()
    evidence = words(document)
    kept = []
    for sentence in sentences(text):
        bag = words(sentence)
        numbers = set(re.findall(r"(?<!\w)\d{2,}(?:[.,]\d+)*(?!\w)", sentence))
        if bag and len(bag & evidence) / len(bag) >= .52 and all(n in original for n in numbers):
            kept.append(sentence)
    filtered = dedupe(kept)
    return filtered if len(filtered.split()) >= 30 else text


def summarize(model, architecture, inputs, document, device, precision="fp32",
              strategy="hierarchical", policy="raw"):
    original = strategy == "section" or not hasattr(inputs, "passages")
    generated = [
        generate_passage(model, architecture, source, inputs.tokenizer, device, precision, original)
        for source in inputs.prompts(document, strategy)
    ]
    joined = " ".join(text for text in generated if text)
    text = joined if original else dedupe(generated) or joined
    if policy == "grounded":
        return evidence_filter(text, document)
    if policy != "raw":
        raise ValueError(f"Unknown decoding policy {policy}")
    return text
