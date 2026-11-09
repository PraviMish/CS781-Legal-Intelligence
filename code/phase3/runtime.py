"""Portable notebook-equivalent training, inference and resumable checkpoint IO.

The original experiments used a fixed seed of 42, three epochs, AdamW,
BF16 autocast on compatible hardware, batch=8 and grad accumulation=2. Training
progress is stored after optimizer-update boundaries. A resumed step never
replays a partially accumulated gradient.
"""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .configs import Experiment, DEFAULT_FOUNDATION_FILES
from .data import make_loader, joined
from .metrics import domain_metrics
from .models import make_classifier, trainable_state, restore_trainable

SEED = 42
ACCUM = 2
BATCH = 8
DEV_BATCH = 16
SAVE_EVERY_UPDATES = 300


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_atomic(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(item, tmp)
    os.replace(tmp, path)


def write_json(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def foundation_path(spec, foundation_root):
    """Allow a user to supply any directory containing original Phase-1 snapshots."""
    return Path(foundation_root) / DEFAULT_FOUNDATION_FILES[spec.foundation]


def build_run(spec, foundation_root, device):
    return make_classifier(spec.architecture, spec.method, foundation_path(spec, foundation_root),
                           device=device, student=spec.student)


def load_best(spec, foundation_root, checkpoint_root, device="cpu"):
    """Recreate pretrained backbone and restore *only* trained task tensors."""
    model = build_run(spec, foundation_root, device)
    path = Path(checkpoint_root) / spec.id / "best.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    archived_id = state.get("experiment_id", state.get("config", {}).get("experiment_id",
                   state.get("config", {}).get("id")))
    if archived_id != spec.id:
        raise ValueError(f"Unexpected experiment ID in checkpoint: {archived_id}")
    restore_trainable(model, state["trainable_state"])
    model.eval()
    return model


def fingerprint(rows):
    hash_obj = hashlib.sha256()
    for row in rows:
        hash_obj.update(repr((row["domain"], row["doc_id"], row["sentence_idx"],
                              row["label_id"], tuple(row["input_ids"]))).encode("utf-8"))
    return hash_obj.hexdigest()


@torch.inference_mode()
def predict(model, rows, *, device="cpu", batch_size=DEV_BATCH, return_logits=False):
    """Predictions keyed by domain/document/position, independent of bucket ordering."""
    device = torch.device(device)
    model.eval()
    outputs = {}
    for batch in make_loader(rows, batch_size=batch_size, shuffle=False):
        x, lengths = batch["input_ids"].to(device), batch["lengths"].to(device)
        with autocast(device):
            logits = model(x, lengths)
        vals = logits.float().cpu().tolist() if return_logits else logits.argmax(-1).cpu().tolist()
        for dom, doc, pos, val in zip(batch["domains"], batch["doc_ids"], batch["sentence_indices"], vals):
            key = (dom, doc, pos)
            if key in outputs:
                raise ValueError(f"Duplicate sentence key {key}")
            outputs[key] = val
    if len(outputs) != len(rows):
        raise ValueError("Output count differs from sentence count")
    return outputs


def validate_on(model, rows, device):
    preds = predict(model, rows, device=device)
    ordered = [preds[(row["domain"], row["doc_id"], row["sentence_idx"])] for row in rows]
    return domain_metrics(rows, ordered)


def _weighted_ce(rows, device):
    counts = np.bincount([row["label_id"] for row in rows], minlength=13)
    weights = 1.0 / np.sqrt(counts + 5.0)
    weights /= np.sum(weights * counts) / len(rows)
    return torch.tensor(np.clip(weights, .2, 5.), dtype=torch.float32, device=device)


def cache_teacher_logits(spec, teacher, train_rows, foundation_root, checkpoint_root, cache_file, device):
    """Cache original B4 teacher logits for E2; no dev/test labels used in KD."""
    cache_file = Path(cache_file)
    signature = fingerprint(train_rows)
    if cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu", weights_only=True)
        if cached.get("teacher") == teacher.id and cached.get("signature") == signature:
            return cached["logits"]
    model = load_best(teacher, foundation_root, checkpoint_root, device=device)
    outputs = predict(model, train_rows, device=device, return_logits=True)
    ordered = torch.tensor([outputs[(r["domain"], r["doc_id"], r["sentence_idx"])]
                            for r in train_rows], dtype=torch.float16)
    save_atomic(cache_file, {"teacher": teacher.id, "signature": signature, "logits": ordered})
    return ordered


def train(spec: Experiment, train_rows, dev_rows, *, foundation_root, checkpoint_root,
          device="auto", teacher_logits=None, epochs=None, save_every=SAVE_EVERY_UPDATES):
    """Train or resume one historically specified experiment.

    Epoch-level best checkpoints are selected by pooled 13-role Macro-F1.
    Resumption is exact at optimizer update boundaries for deterministic CPU
    samplers; reproducing old GPU kernels requires the original GPU stack.
    """
    device = choose_device(device)
    epochs = spec.epochs if epochs is None else epochs
    if spec.loss == "kd":
        if teacher_logits is None or len(teacher_logits) != len(train_rows):
            raise ValueError("Distillation requires logits for every training sentence")
        train_rows = [dict(row, teacher_logits=logits.tolist())
                      for row, logits in zip(train_rows, teacher_logits)]
    if not train_rows or not dev_rows:
        raise ValueError("Training and development must be nonempty")
    seed_everything(SEED)
    model = build_run(spec, foundation_root, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=spec.lr, weight_decay=.01)
    steps_per_epoch = math.ceil(math.ceil(len(train_rows)/BATCH)/ACCUM)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = max(1, round(.05 * total_steps))

    def schedule(step):
        if step < warmup:
            return max(.01, (step+1)/warmup)
        progress = min(1., (step-warmup)/max(1, total_steps-warmup))
        return .1 + .45 * (1 + math.cos(math.pi*progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    experiment_dir = Path(checkpoint_root) / spec.id
    best_file, last_file = experiment_dir / "best.pt", experiment_dir / "last.pt"
    summary_file, history_file = experiment_dir / "summary.json", experiment_dir / "history.json"
    config = dict(spec=asdict(spec), epochs=epochs, max_len=128, effective_batch=BATCH*ACCUM,
                  train_signature=fingerprint(train_rows), dev_signature=fingerprint(dev_rows))
    if summary_file.exists() and best_file.exists():
        prev = json.loads(summary_file.read_text())
        if prev.get("status") == "complete":
            if prev["config"] != config:
                raise ValueError("Completed experiment ID reused with different configuration")
            return prev

    start_epoch = start_batch = updates = 0
    best = -1.; best_epoch = 0; bad_epochs = 0; history = []; loss_sum = 0.; batches_seen = 0
    if last_file.exists():
        checkpoint = torch.load(last_file, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration mismatch; choose new experiment ID")
        restore_trainable(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch, start_batch = checkpoint["epoch"], checkpoint["next_batch"]
        updates, best, best_epoch, bad_epochs = checkpoint["updates"], checkpoint["best"], checkpoint["best_epoch"], checkpoint["bad_epochs"]
        history, loss_sum, batches_seen = checkpoint["history"], checkpoint["loss_sum"], checkpoint["batches_seen"]
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device=device)

    weights = _weighted_ce(train_rows, device) if spec.loss == "balanced" else None
    started = time.perf_counter()

    def save_progress(epoch, next_batch, running, batches):
        save_atomic(last_file, {
            "config": config, "experiment_id": spec.id, "trainable_state": trainable_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "next_batch": next_batch, "updates": updates,
            "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "history": history, "loss_sum": running, "batches_seen": batches,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        })

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_iter = make_loader(train_rows, batch_size=BATCH, shuffle=True, seed=SEED, epoch=epoch)
        running = loss_sum if epoch == start_epoch else 0.
        seen = batches_seen if epoch == start_epoch else 0
        skip = start_batch if epoch == start_epoch else 0
        for batch_idx, batch in enumerate(batch_iter):
            if batch_idx < skip:
                continue
            x, lengths = batch["input_ids"].to(device), batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            with autocast(device):
                logits = model(x, lengths)
                ce = F.cross_entropy(logits.float(), labels, weight=weights)
                if spec.loss == "kd":
                    teacher = batch["teacher_logits"].to(device)
                    t = 2.
                    kd = F.kl_div(F.log_softmax(logits.float()/t, -1),
                                  F.softmax(teacher.float()/t, -1), reduction="batchmean") * (t*t)
                    loss = .5*ce + .5*kd
                else:
                    loss = ce
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            (loss / ACCUM).backward()
            running += float(loss.detach()); seen += 1
            update = ((batch_idx+1) % ACCUM == 0 or batch_idx+1 == len(batch_iter))
            if update:
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates % save_every == 0:
                    save_progress(epoch, batch_idx+1, running, seen)
        metrics = validate_on(model, dev_rows, device)
        history.append({"epoch": epoch+1, "train_loss": running/max(1,seen), "updates": updates, **metrics})
        if metrics["macro_f1"] > best + 1e-6:
            best, best_epoch, bad_epochs = metrics["macro_f1"], epoch+1, 0
            save_atomic(best_file, {"experiment_id": spec.id, "trainable_state": trainable_state(model),
                                    "config": config, "metrics": history[-1]})
        else:
            bad_epochs += 1
        write_json(history_file, history)
        save_progress(epoch+1, 0, 0., 0)
        if bad_epochs >= 2:
            break
    summary = {"status": "complete", "experiment_id": spec.id, "config": config,
               "best_macro_f1": best, "best_epoch": best_epoch, "optimizer_updates": updates,
               "trainable_parameters": sum(p.numel() for p in params),
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "seconds_this_session": time.perf_counter()-started}
    write_json(summary_file, summary)
    return summary
