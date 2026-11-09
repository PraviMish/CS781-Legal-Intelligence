"""Shared supervised, adapter and distillation training with atomic resume."""

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
from common.artifacts import write_json
from torch.nn import functional as F

from .inference import amp, predict
from .metrics import metrics
from .runtime import (
    ADAPTER_RUNS,
    BEST_FILES,
    STATE_KEYS,
    restore_weights,
    saved_weights,
)
from .variants import allocate_ranks, distillation_loss, update_importance


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def optimizer_for(model, run, config, device):
    weight_decay = config["weight_decay"]
    if run == "E02":
        return torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay
        )
    if run in ADAPTER_RUNS:
        groups = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("classifier.")
                ],
                "lr": config["adapter_lr"],
                "weight_decay": weight_decay,
            },
            {
                "params": list(model.classifier.parameters()),
                "lr": config["head_lr"],
                "weight_decay": weight_decay,
            },
        ]
        betas = (0.9, 0.999)
    else:
        betas = (0.9, 0.95)
        groups = {}
        for name, param in model.named_parameters():
            head = name.startswith("classifier.") and run != "E08"
            no_decay = param.ndim < 2 or "embedding" in name.lower()
            if run != "E01":
                no_decay = no_decay or "norm" in name.lower()
            groups.setdefault((head, no_decay), []).append(param)
        param_groups = []
        for (head, no_decay), params in sorted(groups.items()):
            learning_rate = (
                config["learning_rate"]
                if run == "E08"
                else config["head_lr"]
                if head
                else config["backbone_lr"]
            )
            param_groups.append(
                {
                    "params": params,
                    "lr": learning_rate,
                    "weight_decay": 0.0 if no_decay else weight_decay,
                }
            )
        groups = param_groups
    return torch.optim.AdamW(groups, betas=betas, eps=1e-8, fused=device.type == "cuda")


def rate(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))


def rank_budget(step, total, config, adapters):
    start = int(0.10 * total)
    end = int(0.80 * total)
    progress = np.clip((step - start) / max(1, end - start), 0.0, 1.0)
    fraction = 3 * progress**2 - 2 * progress**3
    return round(
        adapters
        * (config["start_rank"] + (config["target_rank"] - config["start_rank"]) * fraction)
    )


def train(
    model,
    run,
    config,
    training,
    validation,
    out,
    device,
    precision,
    *,
    identity,
    teacher_logits=None,
    batch_size=None,
):
    """Resume this port's checkpoints at an optimizer-step boundary."""
    if training.labels is None or validation.labels is None:
        raise ValueError("Training and internal validation must be labeled")
    if run == "E08":
        if teacher_logits is None or teacher_logits.shape != (len(training.ids), 2):
            raise ValueError(
                "E08 requires two archived teacher logits per training case, in split order"
            )
        if not np.isfinite(teacher_logits).all():
            raise ValueError("Teacher logits must be finite")
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "resume.pt"
    best_file = out / BEST_FILES.get(run, "best_model.pt")
    batch_size = batch_size or config.get(
        "micro_batch_size", config.get("micro_batch", config.get("batch_size"))
    )
    accumulation = config.get("gradient_accumulation", config.get("accumulation", 1))
    epochs = config.get("max_epochs", config.get("epochs"))
    patience = config.get("patience", config.get("early_stopping_patience"))
    interval = config.get("checkpoint_every_updates", config.get("checkpoint_every_steps", 250))
    n = len(training.ids)
    n_micro = math.ceil(n / batch_size)
    updates = math.ceil(n_micro / accumulation) * epochs
    warmup = max(1, int(updates * config.get("warmup_fraction", 0.06)))
    signature = {
        "run": run,
        "config": config,
        "batch_size": batch_size,
        "precision": precision,
        "device": device.type,
        "inputs": identity,
    }
    optimizer = optimizer_for(model, run, config, device)
    scheduler = (
        None
        if run == "E02"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: rate(step, updates, warmup))
    )
    state_key = STATE_KEYS.get(run, "model")
    start_epoch = micro = step = bad_epochs = seen = 0
    loss_sum = 0.0
    best = -1.0
    history = []
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Saved run inputs or settings differ. Use a fresh output directory.")
        restore_weights(model, run, state[state_key])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start_epoch, micro, step, bad_epochs, best, seen, loss_sum = state["cursor"]
        history = state["history"]
    else:
        if best_file.exists():
            raise ValueError("Use a fresh output directory for training")
        torch.manual_seed(config["seed"] + (101 if run == "E01" else 0))

    def save(next_epoch, next_micro, examples, cumulative_loss):
        atomic_save(
            latest,
            {
                "signature": signature,
                state_key: saved_weights(model, run),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "history": history,
                "cursor": [
                    next_epoch,
                    next_micro,
                    step,
                    bad_epochs,
                    best,
                    examples,
                    cumulative_loss,
                ],
            },
        )

    parameters = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(start_epoch, epochs):
        if bad_epochs >= patience:
            break
        model.train()
        seed = config["seed"] + epoch + (0 if run == "E02" else 10000)
        order = np.random.default_rng(seed).permutation(n)
        optimizer.zero_grad(set_to_none=True)
        for group in range(micro, n_micro, accumulation):
            group_end = min(group + accumulation, n_micro)
            for batch in range(group, group_end):
                indices = order[batch * batch_size : (batch + 1) * batch_size]
                labels = torch.from_numpy(training.labels[indices]).to(device)
                with amp(device, "fp32" if run == "E02" else precision):
                    logits = model(*training.batch(indices, device))
                    if run == "E08":
                        teacher = torch.from_numpy(teacher_logits[indices]).to(device)
                        loss, _, _ = distillation_loss(
                            logits, teacher, labels, config["temperature"], config["kd_weight"]
                        )
                    else:
                        loss = F.cross_entropy(logits.float(), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / (group_end - group)).backward()
                loss_sum += loss.item() * len(indices)
                seen += len(indices)
            if run == "E07":
                update_importance(model)
            torch.nn.utils.clip_grad_norm_(
                parameters, config.get("gradient_clip", 1.0), error_if_nonfinite=True
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if run == "E07" and step % 40 == 0:
                allocate_ranks(
                    model,
                    rank_budget(step, updates, config, len(model.adapters)),
                    config["minimum_rank"],
                )
            if run != "E02" and step % interval == 0:
                save(epoch, group_end, seen, loss_sum)

        eval_batch = {"E02": 256, "E06": 12, "E08": 32}.get(run, 16)
        probs = predict(
            model, validation, device, "fp32" if run == "E02" else precision, eval_batch
        )
        measured = metrics(validation.labels, probs)
        score = measured["macro_f1"]
        history.append({"epoch": epoch + 1, "step": step, "loss": loss_sum / seen, **measured})
        print(f"Epoch {epoch + 1} internal macro-F1 {score:.6f}", flush=True)
        if score > best:
            best, bad_epochs = score, 0
            atomic_save(
                best_file,
                {
                    state_key: saved_weights(model, run),
                    "config": config,
                    "epoch": epoch if run == "E01" else epoch + 1,
                    "global_step": step,
                    "metrics": measured,
                },
            )
        else:
            bad_epochs += 1
        micro = seen = 0
        loss_sum = 0.0
        save(epoch + 1, 0, 0, 0.0)
    write_json(out / "history.json", history)
    return best_file
