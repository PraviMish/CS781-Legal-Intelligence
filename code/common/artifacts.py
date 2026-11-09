"""Small, reusable, filesystem-independent artifact utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as out:
        json.dump(obj, out, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def save_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, step: int,
                    tokens_seen: int, best_val_loss: float,
                    sampler_rng: np.random.Generator) -> None:
    """Save the historical Phase-1 checkpoint schema, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    data = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'tokens_seen': tokens_seen,
        'best_val_loss': best_val_loss,
        'sampler_state': sampler_rng.bit_generator.state,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    torch.save(data, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path: str | Path, model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer | None = None,
                       sampler_rng: np.random.Generator | None = None,
                       restore_rng: bool = True) -> dict:
    """Read a trusted local checkpoint from either the original notebook or this package.

    `torch.load(weights_only=False)` is needed for historical optimizer/RNG states;
    only load trusted checkpoints, never untrusted files downloaded at random.
    """
    data = torch.load(Path(path), map_location='cpu', weights_only=False)
    model.load_state_dict(data['model_state'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer_state'])
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    if sampler_rng is not None and 'sampler_state' in data:
        sampler_rng.bit_generator.state = data['sampler_state']
    if restore_rng:
        state = data.get('torch_rng_state')
        if state is not None:
            torch.set_rng_state(torch.as_tensor(state, dtype=torch.uint8, device='cpu'))
        if torch.cuda.is_available() and data.get('cuda_rng_state'):
            cuda_states = [torch.as_tensor(s, dtype=torch.uint8, device='cpu') for s in data['cuda_rng_state']]
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)
        if 'numpy_rng_state' in data:
            np.random.set_state(data['numpy_rng_state'])
        if 'python_rng_state' in data:
            random.setstate(data['python_rng_state'])
    return data


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
