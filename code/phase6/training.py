"""Versioned target-only training and two historical distillation objectives."""

import hashlib
import json
import math
import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F

from .data import BOS, EOS
from .decoding import amp_context, generate_passage
from .models import VERSIONS, make_model
from .submission import save_json

ACCUMULATION = 4


def schedule(groups, run, seed):
    """Yield the exact data order, including the historical V2 rotation."""
    if run.version == "original":
        rng = np.random.default_rng(seed + int(run.name[1:]))
        required = run.steps * ACCUMULATION
        plan = []
        while len(plan) < required:
            plan.extend((int(i), 0) for i in rng.permutation(len(groups)))
        return plan[:required]
    rng = np.random.default_rng(seed + sum(map(ord, run.name)))
    plan = []
    for epoch in range(run.epochs):
        for i in rng.permutation(len(groups)):
            count = len(groups[i])
            if run.version == "v2":
                plan.append((int(i), (epoch + int(i)) % count))
            else:
                plan.extend((int(i), int(j)) for j in rng.permutation(count))
    return plan


def learning_rate(run, step, total):
    warmup = (min(50, max(1, total // 10)) if run.version == "original"
              else max(20, min(150, int(total * .05))))
    if step <= warmup:
        return run.lr * step / warmup
    progress = min(1, (step - warmup) / max(1, total - warmup))
    floor = .08 if run.version == "v2" else .10
    return run.lr * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress)))


def target_logits(model, architecture, source, target, device):
    if architecture == "encoder_decoder":
        memory = model.encode(torch.tensor([source], device=device))
        logits, _, _ = model.decode(torch.tensor([[BOS] + target[:-1]], device=device), memory)
        return logits[0]
    tokens = torch.tensor([(source + target)[:-1]], device=device)
    prefix = {"prefix_length": len(source)} if architecture == "prefix_lm" else {}
    logits, _ = model(tokens, **prefix)
    return logits[0, len(source) - 1:]


def target_ce(logits, target):
    return F.cross_entropy(logits.float(), torch.tensor(target, device=logits.device))


def distillation_loss(student, teacher, target, temperature=2.0):
    ce = target_ce(student, target)
    kl = F.kl_div(F.log_softmax(student.float() / temperature, dim=-1),
                  F.softmax(teacher.detach().float() / temperature, dim=-1),
                  reduction="batchmean") * temperature ** 2
    return .8 * ce + .2 * kl


def example_loss(model, run, example, tokenizer, device, teacher=None, pseudo=None):
    source, target = example["src"], example["tgt"]
    logits = target_logits(model, run.architecture, source, target, device)
    if teacher is not None:
        with torch.no_grad():
            teacher_logits = target_logits(teacher, "efficient_gqa", source, target, device)
        return distillation_loss(logits, teacher_logits, target)
    gold = target_ce(logits, target)
    text = (pseudo or {}).get(example["key"])
    if not text:
        return gold
    cap, weight = (240, .5) if run.version == "original" else (320, .4)
    synthetic = tokenizer.encode(text).ids[:cap - 1] + [EOS]
    pseudo_logits = target_logits(model, run.architecture, source, synthetic, device)
    return (1 - weight) * gold + weight * target_ce(pseudo_logits, synthetic)


@torch.no_grad()
def validation_nll(model, run, pairs, device, precision, seed):
    indices = np.arange(len(pairs))
    if run.version != "original":
        indices = np.random.default_rng(seed + 7).choice(
            len(pairs), size=min(run.nll_pairs, len(pairs)), replace=False)
    was_training = model.training
    model.eval()
    losses = []
    with amp_context(device, precision):
        for i in indices[:run.nll_pairs]:
            pair = pairs[i]
            logits = target_logits(model, run.architecture, pair["src"], pair["tgt"], device)
            losses.append(float(target_ce(logits, pair["tgt"])))
    model.train(was_training)
    return float(np.mean(losses))


def save_torch(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_teacher(model, run, groups, tokenizer, output, device, precision, seed):
    """Generate E02 pseudo-labels by document in the original/V2 protocols."""
    if run.version == "v3":
        raise ValueError("V3 computes teacher logits online")
    count = 1000 if run.version == "original" else 800
    indices = np.random.default_rng(seed + 600).permutation(len(groups))[:count]
    outputs = {}
    for i in indices:
        for example in groups[i]:
            text = generate_passage(model, run.architecture, example["src"], tokenizer, device,
                                    precision, original=run.version == "original")
            if text:
                outputs[example["key"]] = text
    if not outputs:
        raise ValueError("Teacher produced no usable summaries")
    save_json(output, outputs)
    return len(outputs)


def train(run, groups, val_pairs, tokenizer, output, device, precision, seed, signature,
          foundation=None, warmstart=None, teacher=None, pseudo=None):
    if not groups or not val_pairs:
        raise ValueError("Training and validation pairs are required")
    if (run.version == "v3") != (warmstart is not None):
        raise ValueError("V3 requires the corresponding V2 warm start")
    if run.version != "v3" and foundation is None:
        raise ValueError("Original and V2 training require a Phase 1 foundation")
    if run.method == "distillation" and not (teacher is not None or pseudo):
        raise ValueError("Distillation requires teacher supervision")
    torch.manual_seed(seed)
    random.seed(seed)
    output = Path(output)
    resume = output / "resume.pt"
    best_path = output / "best.pt"
    history_path = output / "history.json"
    plan = schedule(groups, run, seed)
    total = math.ceil(len(plan) / ACCUMULATION)
    signature = hashlib.sha256(
        repr((run, signature, seed, str(device), precision, len(plan))).encode()
    ).hexdigest()

    model = make_model(run, foundation)
    if warmstart is not None:
        state = torch.load(warmstart, map_location="cpu", weights_only=True)
        expected = "E07_gold" if run.name == "E04" else run.name
        if (state.get("version"), state.get("exp_id")) != (VERSIONS["v2"], expected):
            raise ValueError(f"Expected the V2 {expected} warm start")
        model.load_state_dict(state["model"], strict=True)
        del state
    model.to(device)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=run.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    history = []
    best_step = None
    best_nll = math.inf
    start = 0
    cuda = torch.device(device).type == "cuda"

    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Training inputs changed. Use a new output directory.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best_step, best_nll = state["step"], state["best_step"], state["best_nll"]
        history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if cuda:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state

    def save_best(step, loss):
        payload = {"exp_id": run.name, "step": step, "val_nll": loss, "model": model.state_dict()}
        if VERSIONS[run.version] is not None:
            payload["version"] = VERSIONS[run.version]
        else:
            payload["architecture"] = run.architecture
        payload["method"] = run.method
        save_torch(best_path, payload)

    def save_resume(step):
        save_torch(resume, {
            "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "step": step, "best_step": best_step, "best_nll": best_nll,
            "history": history, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if cuda else [],
        })
        save_json(history_path, history)

    if start == 0 and run.version == "v3" and not resume.exists():
        best_step = 0
        best_nll = validation_nll(model, run, val_pairs, device, precision, seed)
        save_best(0, best_nll)
        save_resume(0)

    if best_step is not None:
        if not best_path.exists():
            raise ValueError("The saved best checkpoint is missing")
        # An interruption after a best write but before a resume write leaves a future best.
        saved = torch.load(best_path, map_location="cpu", weights_only=True)
        if saved["step"] < best_step:
            raise ValueError("The best checkpoint is older than the resume metadata")
        if saved["step"] == best_step and not math.isclose(saved["val_nll"], best_nll):
            raise ValueError("Best and resume metadata disagree")
        if saved["step"] > start:
            best_step, best_nll = saved["step"], float(saved["val_nll"])
        del saved

    pseudo = pseudo or {}
    by_key = {ex["key"]: ex for group in groups for ex in group}
    if run.version == "original":
        pseudo_examples = [by_key[key] for key, value in pseudo.items() if key in by_key and value.strip()]
    else:
        pseudo_examples = [ex for key, ex in by_key.items() if key in pseudo]
    for index in range(start, total):
        step = index + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(run, step, total)
        microbatches = plan[index * ACCUMULATION:(index + 1) * ACCUMULATION]
        loss_sum = 0.
        for doc_index, passage_index in microbatches:
            example = groups[doc_index][passage_index]
            if pseudo_examples and random.random() < .5:
                example = random.choice(pseudo_examples)
            with amp_context(device, precision):
                loss = example_loss(model, run, example, tokenizer, device, teacher, pseudo)
                loss = loss / len(microbatches)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at update {step}")
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        scaler.step(optimizer)
        scaler.update()
        validated = step % run.validate_every == 0 or step == total
        if validated:
            nll = validation_nll(model, run, val_pairs, device, precision, seed)
            if nll < best_nll:
                best_nll, best_step = nll, step
                save_best(step, nll)
            history.append({"step": step, "train_loss": loss_sum, "val_nll": nll})
            print(f"{run.name} update {step}/{total}, validation NLL {nll:.5f}", flush=True)
        if validated or step % run.save_every == 0:
            save_resume(step)
    return best_path
