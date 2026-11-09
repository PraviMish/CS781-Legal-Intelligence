"""Trusted checkpoint loading and batched classifier inference."""

from contextlib import nullcontext

import numpy as np
import torch
from phase1.models import build_model

from .gqa import CJPEGQAClassifier


def amp(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 requires a supported CUDA device. Use --precision fp32 otherwise.")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def choose_precision(device, precision):
    if precision == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    return precision


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def backbone(name, foundation=None, expected_tokens=None):
    model = build_model(name)
    if foundation is not None:
        state = load_checkpoint(foundation)
        if state["model_name"] != name or state["tokens_seen"] != expected_tokens:
            raise ValueError(f"Expected {name} foundation at {expected_tokens} consumed tokens")
        model.load_state_dict(state["model_state"], strict=True)
    return model


def e01_model(checkpoint=None, foundation=None):
    model = CJPEGQAClassifier(backbone("efficient_gqa", foundation, 700047360))
    if checkpoint is not None:
        model.load_state_dict(load_checkpoint(checkpoint)["model"], strict=True)
    return model


@torch.inference_mode()
def predict(model, split, device, precision, batch_size, *, return_logits=False):
    model.eval()
    result = []
    for start in range(0, len(split.ids), batch_size):
        indices = np.arange(start, min(start + batch_size, len(split.ids)))
        with amp(device, precision):
            logits = model(*split.batch(indices, device)).float()
        result.append((logits if return_logits else logits.softmax(-1)[:, 1]).cpu().numpy())
    combined = np.concatenate(result)
    if not np.isfinite(combined).all():
        raise FloatingPointError("Non-finite model outputs")
    return combined
