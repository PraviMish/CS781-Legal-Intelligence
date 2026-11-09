"""Reproducible Phase-1 training and continuation for all four architectures.

Preserves the notebook's next-token/encoder-decoder objectives, batch policy,
AdamW parameter groups, validation sampler, and checkpoint schema. Unlike the
Colab notebook, the same command handles cold start and interruption recovery.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from common.artifacts import append_jsonl, restore_checkpoint, save_checkpoint
from phase1.data import MixedTokenStore, mix_weights, SOURCES
from phase1.models import build_model, parameter_count

MODEL_MICROBATCH = {'decoder_only': 8, 'encoder_decoder': 4, 'prefix_lm': 8, 'efficient_gqa': 8}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model: torch.nn.Module, learning_rate: float = 3e-4) -> torch.optim.Optimizer:
    """Exact original AdamW parameter grouping (matrix weights decay; 1D do not)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{'params': decay, 'weight_decay': 0.1},
              {'params': no_decay, 'weight_decay': 0.0}]
    kwargs = dict(lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    try:
        return torch.optim.AdamW(groups, fused=torch.cuda.is_available(), **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, **kwargs)


def initial_learning_rate(tokens_seen: int, *, warmup: int = 5_000_000,
                          schedule_end: int = 500_000_000, peak: float = 3e-4,
                          floor: float = 3e-5) -> float:
    """Original 5M-token warmup followed by cosine decay over 500M tokens."""
    if tokens_seen < warmup:
        return peak * tokens_seen / max(warmup, 1)
    progress = min(1.0, max(0.0, (tokens_seen - warmup) / (schedule_end - warmup)))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def continuation_learning_rate(tokens_seen: int, *, base_tokens: int,
                               base_lr: float, end_tokens: int = 700_000_000,
                               floor: float = 3e-5) -> float:
    """Continue from saved 100M checkpoint with new cosine schedule to 700M."""
    if tokens_seen <= base_tokens:
        return base_lr
    fraction = min(1.0, max(0.0, (tokens_seen - base_tokens) / (end_tokens - base_tokens)))
    return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * fraction))


def read_source_weights(path: Path) -> dict[str, float]:
    """Load exact reproducible weights rather than guessing from binary sizes."""
    content = json.loads(path.read_text(encoding='utf-8'))
    weights = content.get('source_weights', content)
    if not set(weights).issubset(set(SOURCES)) or any(float(x) <= 0 for x in weights.values()):
        raise ValueError('Invalid source weights')
    return {name: float(value) for name, value in weights.items()}


@torch.no_grad()
def validation_metrics(model, store: MixedTokenStore, *, batch_size: int,
                       context_length: int, device: torch.device, batches: int = 25,
                       seed: int = 42) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed + 999)
    losses = []
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    for _ in range(batches):
        block = store.sample_batch(batch_size, context_length, rng=rng).to(device)
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=device.type == 'cuda'):
            loss = model.loss_on_block(block)
        losses.append(float(loss.item()))
    model.train(was_training)
    average = float(np.mean(losses))
    return average, math.exp(min(average, 20.0))


def train(*, architecture: str, train_dir: Path, val_dir: Path, weights: dict[str, float],
          output_dir: Path, target_tokens: int, resume_checkpoint: Path | None = None,
          val_weights: dict[str, float] | None = None,
          context_length: int = 1280, vocab_size: int = 32000, seed: int = 42,
          microbatch: int | None = None, effective_token_budget: int = 65536,
          checkpoint_every: int = 10_000_000, validate_every: int = 10_000_000,
          validation_batches: int = 25, log_every: int = 20,
          continued_base_tokens: int | None = None,
          continued_base_lr: float | None = None,
          continuation_end: int = 700_000_000) -> Path:
    """Run from scratch or resume a trusted checkpoint until `target_tokens`.

    `resume_checkpoint` is optional. If omitted and `output_dir/latest.pt`
    already exists, that checkpoint is resumed automatically. Original model
    architecture and PyTorch state keys are not changed.
    """
    if architecture not in MODEL_MICROBATCH:
        raise ValueError(f'Unknown architecture: {architecture}')
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / 'latest.pt'
    checkpoint = resume_checkpoint or (latest if latest.exists() else None)
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(architecture, vocab_size, context_length).to(device)
    optimizer = make_optimizer(model)
    tokens_seen, step, best_val_loss = 0, 0, float('inf')
    train_store = MixedTokenStore(train_dir, 'train', weights, seed=seed, min_length=context_length)
    val_store = MixedTokenStore(val_dir, 'val', val_weights or weights, seed=seed+1, min_length=context_length)
    if checkpoint is not None:
        state = restore_checkpoint(checkpoint, model, optimizer, train_store.rng)
        tokens_seen = int(state['tokens_seen'])
        step = int(state['step'])
        best_val_loss = float(state['best_val_loss'])
        print(f'Resumed {architecture} from {checkpoint}, {tokens_seen:,} tokens')
        if continued_base_tokens is not None and tokens_seen <= continued_base_tokens + 1_000_000:
            # The initial ~100M checkpoint's sampler used the 500M cache.
            # A fresh deterministic stream must be started for the complete corpus.
            train_store.rng = np.random.default_rng(seed + 2026)
    if tokens_seen >= target_tokens:
        print(f'Already complete: {tokens_seen:,} >= {target_tokens:,}')
        return checkpoint or latest
    if (continued_base_tokens is None) != (continued_base_lr is None):
        raise ValueError('Both continued_base_tokens and continued_base_lr are required')
    micro = microbatch or MODEL_MICROBATCH[architecture]
    accumulation = max(1, effective_token_budget // (micro * context_length))
    tokens_per_step = micro * context_length * accumulation
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and dtype == torch.float16))
    next_check = (tokens_seen // checkpoint_every + 1) * checkpoint_every
    next_val = (tokens_seen // validate_every + 1) * validate_every
    recent = []
    started = time.monotonic()
    last_log = started
    prev_token = tokens_seen
    print(f'{architecture}: {parameter_count(model):,} parameters, {micro} microbatch, '
          f'{accum} accumulations, {tokens_per_step:,} tokens/step, {device}')
    while tokens_seen < target_tokens:
        if continued_base_tokens is None:
            lr = initial_learning_rate(tokens_seen)
        else:
            lr = continuation_learning_rate(tokens_seen, base_tokens=continued_base_tokens,
                                            base_lr=continued_base_lr, end_tokens=continuation_end)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(accumulation):
            batch = train_store.sample_batch(micro, context_length).to(device)
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=device.type == 'cuda'):
                loss = model.loss_on_block(batch)
            scaler.scale(loss / accumulation).backward()
            accumulated_loss += float(loss.detach().item()) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        tokens_seen += tokens_per_step
        recent.append(accumulated_loss)
        if step % log_every == 0:
            now = time.monotonic()
            speed = (tokens_seen - prev_token) / max(now - last_log, 1e-9)
            append_jsonl(output_dir / 'history.jsonl', {
                'type': 'train', 'step': step, 'tokens_seen': tokens_seen,
                'train_loss': float(np.mean(recent[-log_every:])), 'learning_rate': lr,
                'tokens_per_second': speed, 'elapsed_seconds': now - started})
            print(f'{architecture} step {step}, tokens {tokens_seen:,}, '
                  f'loss={accumulated_loss:.4f}, lr={lr:.2e}, tok/s={speed:,.0f}')
            last_log, prev_token = now, tokens_seen
        if tokens_seen >= next_val or tokens_seen >= target_tokens:
            val_loss, perplexity = validation_metrics(model, val_store, batch_size=micro,
                context_length=context_length, device=device, batches=validation_batches, seed=seed)
            append_jsonl(output_dir / 'history.jsonl', {
                'type': 'validation', 'step': step, 'tokens_seen': tokens_seen,
                'val_loss': val_loss, 'perplexity': perplexity})
            print(f'Validation @ {tokens_seen:,}: loss={val_loss:.4f}, PPL={perplexity:.3f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = output_dir / 'best.pt'
                temp = output_dir / 'best.tmp.pt'
                torch.save({'model_name': architecture, 'model_state': model.state_dict(),
                            'tokens_seen': tokens_seen, 'val_loss': val_loss}, temp)
                temp.replace(path)
            while next_val <= tokens_seen:
                next_val += validate_every
        if tokens_seen >= next_check or tokens_seen >= target_tokens:
            save_checkpoint(latest, model, optimizer, step, tokens_seen, best_val_loss, train_store.rng)
            while next_check <= tokens_seen:
                next_check += checkpoint_every
    return latest


def main() -> None:
    p = argparse.ArgumentParser(description='Run Phase-1 legal language-model training')
    p.add_argument('--architecture', choices=MODEL_MICROBATCH, required=True)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--val-dir', type=Path, required=True)
    p.add_argument('--weights-json', type=Path, required=True)
    p.add_argument('--val-weights-json', type=Path, help='Use original initial-corpus validation mixture during continuation')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--target-tokens', type=int, required=True)
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--context-length', type=int, default=1280)
    p.add_argument('--vocab-size', type=int, default=32000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--microbatch', type=int)
    p.add_argument('--continued-base-tokens', type=int)
    p.add_argument('--continued-base-lr', type=float)
    p.add_argument('--continuation-end', type=int, default=700_000_000)
    args = p.parse_args()
    checkpoint = train(
        architecture=args.architecture, train_dir=args.train_dir, val_dir=args.val_dir,
        weights=read_source_weights(args.weights_json), output_dir=args.output_dir,
        val_weights=(read_source_weights(args.val_weights_json) if args.val_weights_json else None),
        target_tokens=args.target_tokens, resume_checkpoint=args.resume_checkpoint,
        context_length=args.context_length, vocab_size=args.vocab_size, seed=args.seed,
        microbatch=args.microbatch, continued_base_tokens=args.continued_base_tokens,
        continued_base_lr=args.continued_base_lr, continuation_end=args.continuation_end)
    print('Training checkpoint:', checkpoint)


if __name__ == '__main__':
    main()
