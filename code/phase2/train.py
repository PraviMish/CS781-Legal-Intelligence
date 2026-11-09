"""Shared, restartable trainer for the original Phase 2 experiment variants."""
import hashlib
import json
import math
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common.artifacts import write_json
from .configs import FULL70_MEMBERS, GQA_FULL, KD_CE_WEIGHT, KD_TEMPERATURE, RUNS
from .data import BIO, IGNORE, WINDOW, build_windows
from .metrics import strict_metrics
from .models import build_ner, load_state, ner_weights, restore_ner
from .runtime import autocast, pad_batch, predict_documents


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_digest(windows, validation_docs, gold):
    digest = hashlib.sha256()
    for window in windows:
        digest.update(json.dumps(window).encode())
    for key, doc in validation_docs.items():
        digest.update(json.dumps([key, doc["ids"], doc["offsets"], gold[key]]).encode())
    return digest.hexdigest()


def supervised_loss(logits, labels, weights=None, teacher=None):
    logits = logits.float()
    if not (labels != IGNORE).any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=weights,
                         ignore_index=IGNORE)
    if teacher is None:
        return ce
    temperature = KD_TEMPERATURE
    token_kl = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        F.softmax(teacher.float() / temperature, dim=-1), reduction="none"
    ).sum(dim=-1)
    kd = token_kl[labels != IGNORE].mean() * temperature**2
    return KD_CE_WEIGHT * ce + (1 - KD_CE_WEIGHT) * kd


def _indexed_batch(batch, pad_id):
    indices, windows = zip(*batch)
    inputs, labels = pad_batch(windows, pad_id, supervised=True)
    return list(indices), inputs, labels


@torch.no_grad()
def _teacher_cache(windows, run, checkpoint_path, output, signature, pad_id, device):
    """Store float16 teacher logits by original window index, in 128-window shards."""
    teacher = None
    teacher_run = RUNS[FULL70_MEMBERS[0] if run.full_data else GQA_FULL]
    batch_size = run.batch if run.full_data else 4
    shards = []
    for start in range(0, len(windows), 128):
        chunk = windows[start:start + 128]
        path = output / f"shard_{start // 128:04d}.pt"
        if path.exists():
            state = load_state(path)
            if state["signature"] != signature or state["start"] != start:
                raise ValueError("Teacher cache does not match this checkpoint or window order.")
            shards.append(state["logits"])
            continue
        if teacher is None:
            teacher = build_ner(teacher_run, pad_id, device=device)
            restore_ner(teacher, load_state(checkpoint_path))
            teacher.eval()
        logits = torch.zeros((len(chunk), WINDOW, len(BIO)), dtype=torch.float16)
        for batch_start in range(0, len(chunk), batch_size):
            batch = chunk[batch_start:batch_start + batch_size]
            inputs, _ = pad_batch(batch, pad_id)
            with autocast(device):
                values = teacher(inputs.to(device)).cpu().half()
            logits[batch_start:batch_start + len(batch), :values.shape[1]] = values
        save_state(path, {"signature": signature, "start": start, "logits": logits})
        shards.append(logits)
    return torch.cat(shards)


def _scheduler(optimizer, run, steps):
    if run.method == "probe":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    warmup = max(1, (round if run.round_warmup else int)(steps * 0.06))

    def rate(step):
        if step < warmup:
            fraction = (step + 1) / warmup
            return max(0.01, fraction) if run.method == "full" and not run.full_data else fraction
        return max(0.0, 1 - (step - warmup) / max(1, steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate)


def train(run_name, run, docs, gold, split, pad_id, seed, output, device, *,
          foundation=None, student_initialization=None, teacher_checkpoint=None):
    output = Path(output)
    windows = build_windows(docs, split["train"])
    if not windows:
        raise ValueError("Training needs at least one supervision window.")
    val_docs = {key: docs[key] for key in split["val"]}
    val_gold = {key: gold[key] for key in val_docs}
    config = {
        "run_name": run_name, "settings": asdict(run), "seed": seed, "pad_id": pad_id,
        "windows": _window_digest(windows, val_docs, val_gold),
        "foundation": file_digest(foundation),
        "student_initialization": file_digest(student_initialization),
        "teacher": file_digest(teacher_checkpoint), "device": str(device),
        "bf16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
    }
    latest = output / "last.pt"
    saved = load_state(latest) if latest.exists() else None
    if saved is not None and saved.get("replay_config") != config:
        raise ValueError("Run inputs changed or last.pt is a notebook checkpoint. Use a new output directory.")
    selected = output / ("final.pt" if run.full_data else "best.pt")
    if saved is not None and saved["complete"]:
        if not selected.is_file():
            raise FileNotFoundError(f"Completed run is missing its selected checkpoint: {selected}")
        write_json(output / "history.json", saved["history"])
        return selected

    targets = None
    if run.method == "distilled":
        if teacher_checkpoint is None:
            raise ValueError("Distillation requires the matching GQA NER checkpoint.")
        cache_signature = [config[key] for key in ("windows", "teacher", "pad_id", "device", "bf16")]
        cache_signature.append(run.batch if run.full_data else 4)
        targets = _teacher_cache(windows, run, teacher_checkpoint, output / "teacher_cache",
                                 cache_signature, pad_id, device)

    # The student control and KD runs share one initialization. In the original
    # experiment it was built at seed + 770, with dropout/shuffle reset for training.
    student = run.method in {"student", "distilled"}
    torch.manual_seed(seed + (770 if student else run.init_offset))
    model = build_ner(run, pad_id, foundation=foundation, student_initialization=student_initialization,
                      seed=seed, device=device)
    if student:
        torch.manual_seed(seed + run.init_offset)

    head = list(model.classifier.parameters())
    backbone = [param for param in model.backbone.parameters() if param.requires_grad]
    groups = [{"params": head, "lr": run.head_lr}]
    if backbone:
        groups.insert(0, {"params": backbone, "lr": run.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    batches = math.ceil(len(windows) / run.batch)
    scheduler = _scheduler(optimizer, run, math.ceil(batches / run.accumulation) * run.epochs)
    weights = None
    if (run.outside_weight, run.entity_weight) != (1.0, 1.0):
        weights = torch.tensor([run.outside_weight] + [run.entity_weight] * (len(BIO) - 1),
                               device=device)

    epoch, next_batch, step, stale = 0, 0, 0, 0
    best, loss_sum, history = -1.0, 0.0, []
    if saved is not None:
        restore_ner(model, saved)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        epoch, next_batch, step, stale = (saved[k] for k in ("epoch", "next_batch", "step", "stale"))
        best, loss_sum, history = (saved[k] for k in ("best", "loss_sum", "history"))
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        del saved

    def save_latest(epoch_index, batch_index, partial_loss, complete=False):
        save_state(latest, {
            **ner_weights(model), "labels": list(BIO), "replay_config": config,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "epoch": epoch_index, "next_batch": batch_index, "step": step, "stale": stale,
            "best": best, "loss_sum": partial_loss, "history": history, "complete": complete,
        })

    for epoch_index in range(epoch, run.epochs):
        loader = DataLoader(
            list(enumerate(windows)), batch_size=run.batch, shuffle=True,
            generator=torch.Generator().manual_seed(seed + run.shuffle_offset + epoch_index),
            collate_fn=partial(_indexed_batch, pad_id=pad_id),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (indices, inputs, labels) in enumerate(loader):
            if batch_index < next_batch:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            teacher = None if targets is None else targets[indices, :inputs.shape[1]].to(device)
            with autocast(device):
                loss = supervised_loss(model(inputs), labels, weights, teacher)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch_index + 1}, batch {batch_index + 1}.")
            loss_sum += loss.item()
            group_start = batch_index // run.accumulation * run.accumulation
            (loss / min(run.accumulation, batches - group_start)).backward()
            if (batch_index + 1) % run.accumulation == 0 or batch_index + 1 == batches:
                torch.nn.utils.clip_grad_norm_(backbone + head, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if run.save_every and step % run.save_every == 0:
                    save_latest(epoch_index, batch_index + 1, loss_sum)

        epoch_number = epoch_index + 1
        row = {"epoch": epoch_number, "train_loss": loss_sum / batches, "optimizer_steps": step}
        complete = epoch_number == run.epochs
        if not run.full_data:
            pred = predict_documents(model, docs, split["val"], run.inference_batch, device)
            metrics, _ = strict_metrics(val_gold, pred)
            row.update(metrics)
            score = metrics["macro_f1"]
            if score > best:
                best, stale = score, 0
                save_state(selected, {**ner_weights(model), "labels": list(BIO),
                                      "epoch": epoch_number, "macro_f1": score})
            else:
                stale += 1
            complete |= bool(run.patience and epoch_number >= run.min_epochs and stale >= run.patience)
        elif complete:
            save_state(selected, {**ner_weights(model), "labels": list(BIO), "epoch": epoch_number})
        history.append(row)
        print(json.dumps(row))
        save_latest(epoch_number, 0, 0.0, complete=complete)
        write_json(output / "history.json", history)
        if complete:
            break
        next_batch, loss_sum = 0, 0.0

    return selected
