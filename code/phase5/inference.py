"""Batched inference with the original precision and sigmoid policy."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CaseDataset, build_arrays, load_tokenizer, read_cases
from .metrics import probabilities
from .submission import decode, load_thresholds, write_predictions


def precision_dtype(device, precision):
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if (
            device.type == "cuda") else "fp32"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("Use CUDA for FP16 or select FP32 on CPU")
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


@torch.inference_mode()
def predict_logits(model, dataset, batch_size, device, dtype):
    model.eval()
    parts = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        with autocast(device, dtype):
            logits = model(batch["inputs"].to(device), batch["lengths"].to(device))
        parts.append(logits.float().cpu().numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Model produced nonfinite logits")
    return result


def submit(model, tokenizer, test, thresholds, output, batch_size, device, dtype):
    cases = read_cases(test, labeled=False)
    windows, lengths, _ = build_arrays(cases, load_tokenizer(tokenizer))
    logits = predict_logits(model, CaseDataset(windows, lengths), batch_size, device, dtype)
    ids = [case["id"] for case in cases]
    records = decode(ids, probabilities(logits), load_thresholds(thresholds))
    write_predictions(output, records, ids)
    return len(records)
