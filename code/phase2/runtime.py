"""Portable batching and document-level inference."""
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from .data import IGNORE, build_windows, decode_bio


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def autocast(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def pad_batch(windows, pad_id, supervised=False):
    shape = (len(windows), max(len(window["input_ids"]) for window in windows))
    inputs = torch.full(shape, pad_id, dtype=torch.long)
    targets = torch.full(shape, IGNORE, dtype=torch.long) if supervised else None
    for index, window in enumerate(windows):
        length = len(window["input_ids"])
        inputs[index, :length] = torch.tensor(window["input_ids"])
        if supervised:
            targets[index, :length] = torch.tensor(window["labels"])
    return inputs, targets


@torch.inference_mode()
def predict_documents(model, docs, keys, batch_size, device):
    model.eval()
    windows = build_windows(docs, keys)
    tags = {key: [-1] * len(docs[key]["ids"]) for key in keys}
    for batch in DataLoader(windows, batch_size=batch_size, collate_fn=list):
        inputs, _ = pad_batch(batch, model.pad_id)
        with autocast(device):
            logits = model(inputs.to(device))
        predictions = logits.argmax(dim=-1).tolist()
        for window, row in zip(batch, predictions):
            start, end = window["owned_start"], window["owned_end"]
            offset = window["global_start"]
            tags[window["doc_key"]][offset + start:offset + end] = row[start:end]

    if any(-1 in row for row in tags.values()):
        raise RuntimeError("Some tokens were not covered by a prediction window.")
    return {key: decode_bio(row, docs[key]["offsets"]) for key, row in tags.items()}
