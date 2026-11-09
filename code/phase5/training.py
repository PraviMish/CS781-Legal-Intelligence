"""Document-wise training with the archived objectives and checkpoint schema."""

import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from common.artifacts import write_json
from .inference import autocast, predict_logits
from .metrics import metrics, probabilities

ACCUMULATE = 16


def positive_weights(y):
    frequency = y.sum(axis=0)
    return np.sqrt((len(y) - frequency) / np.maximum(frequency, 1)).clip(1, 8)


def objective(kind, logits, labels, weights=None, teacher=None):
    logits = logits.float()
    if kind == "asl":
        p = logits.sigmoid()
        positive = labels * p.clamp_min(1e-8).log()
        negative = (1 - labels) * (1 - p + .05).clamp(1e-8, 1).log() * p.pow(4)
        return -(positive + negative).mean()
    if kind != "weighted_bce":
        raise ValueError(f"Unknown objective: {kind}")
    hard = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights)
    if teacher is None:
        return hard
    temperature = 2.0
    soft = F.binary_cross_entropy_with_logits(
        logits / temperature, (teacher.float() / temperature).sigmoid())
    return .5 * hard + .5 * temperature ** 2 * soft


def epoch_order(name, epoch, length, seed):
    run_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    generator = torch.Generator().manual_seed((seed + run_seed + epoch * 7919) % (2**63 - 1))
    return torch.randperm(length, generator=generator).tolist()


def atomic_checkpoint(data, path):
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state(device):
    name, values, pos, cached_flag, cached = np.random.get_state()
    state = {"cpu_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "numpy_rng": (name, values.tolist(), pos, cached_flag, cached)}
    if device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    if "cpu_rng" in state:
        torch.set_rng_state(state["cpu_rng"])
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        name, keys, pos, flag, cached = state["numpy_rng"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, flag, cached))
    if "cuda_rng" in state and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def fit(model, run, train, validation, output, device, dtype, seed, save_every, fingerprint):
    """Restore an original or port last.pt at a completed optimizer step."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last, best, history_path = (output / name for name in ("last.pt", "best.pt", "history.json"))
    config = {"arch": run.arch, "method": run.method, "loss": run.loss,
              "lr": run.lr, "epochs": run.epochs, "rank": run.rank}
    if run.student:
        config.update(student=True, teacher=run.teacher)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=run.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    if run.student != (train.teacher is not None):
        raise ValueError("Only E06 requires aligned teacher logits")
    weights = torch.as_tensor(positive_weights(train.y), device=device, dtype=torch.float32)
    first_epoch = start = seen = 0
    loss_sum, best_score = 0.0, -1.0
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True, mmap=True)
        if checkpoint.get("fingerprint", fingerprint) != fingerprint:
            raise ValueError("Training cache, run, seed or teacher differs from last.pt")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        first_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
        start = checkpoint.get("next_batch", 0)
        best_score = checkpoint["best_score"]
        if best_score >= 0 and not best.exists():
            raise ValueError("Restore the corresponding historical best.pt alongside last.pt")
        if start:
            loss_sum = checkpoint.get("loss_sum", 0.0)
            seen = checkpoint.get("seen_batches", 0)
        restore_rng(checkpoint, device)
        del checkpoint

    length = len(train)
    if not 0 <= start <= length or (start % ACCUMULATE and start != length):
        raise ValueError("Checkpoint must be at an optimizer boundary in this dataset")

    def save_progress(epoch, next_doc, score, total, examples, complete):
        atomic_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "epoch": epoch, "best_score": score,
                           "next_epoch": epoch + int(complete), "next_batch": next_doc,
                           "epoch_complete": complete, "loss_sum": total, "seen_batches": examples,
                           "fingerprint": fingerprint, **rng_state(device)}, last)

    for epoch in range(first_epoch, run.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = epoch_order(run.name, epoch, length, seed)
        loader = DataLoader(Subset(train, order[start:]), batch_size=1,
                            generator=torch.Generator().manual_seed(10000 + epoch))
        for position, batch in enumerate(loader, start=start + 1):
            teacher = batch["teacher"].to(device) if run.student else None
            with autocast(device, dtype):
                logits = model(batch["inputs"].to(device), batch["lengths"].to(device))
                loss = objective(run.loss, logits, batch["labels"].to(device), weights, teacher)
            scaler.scale(loss / ACCUMULATE).backward()
            loss_sum += float(loss.detach())
            seen += 1
            if position % ACCUMULATE == 0 or position == length:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if position == length or (save_every and position // ACCUMULATE % save_every == 0):
                    save_progress(epoch, position, best_score, loss_sum, seen, False)
        logits = predict_logits(model, validation, run.eval_batch_size, device, dtype)
        scores = metrics(validation.y, probabilities(logits), .30)
        score = scores["macro_f1"]
        if score > best_score:
            best_score = score
            atomic_checkpoint({"model": model.state_dict(), "config": config,
                               "epoch": epoch, "val_score": score}, best)
        history[str(epoch + 1)] = {"loss": loss_sum / max(1, seen), "best_score": best_score, **scores}
        write_json(history_path, history)
        save_progress(epoch, 0, best_score, 0.0, 0, True)
        print(json.dumps({"epoch": epoch + 1, **history[str(epoch + 1)]}))
        start = seen = 0
        loss_sum = 0.0
