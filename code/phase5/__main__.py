"""Prepare, train, calibrate, submit or validate any historical Phase 5 run."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import (LABELS, CaseDataset, build_arrays, check_disjoint_ids, internal_split,
                   load_tokenizer, read_cases, statute_vocabulary, targets)
from .inference import precision_dtype, predict_logits, submit
from .metrics import calibrate, metrics, probabilities
from .models import RUNS, make_model, restore_model
from .submission import load_thresholds, read_predictions, validate
from .training import fit


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare(args):
    train = read_cases(args.train, labeled=True)
    dev = read_cases(args.dev, labeled=True)
    test = read_cases(args.test, labeled=False)
    check_disjoint_ids(train, dev, test)
    if statute_vocabulary(train, dev) != LABELS:
        raise ValueError("The official labeled splits must span all 100 statute IDs")
    archived = (json.loads(Path(args.core_ids).read_text(encoding="utf-8"))
                if args.core_ids else None)
    core, holdout = internal_split(train, args.seed, archived)
    tokenizer = load_tokenizer(args.tokenizer)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, cases in (("train_core", core), ("val_internal", holdout), ("dev", dev), ("test", test)):
        windows, lengths, counts = build_arrays(cases, tokenizer)
        arrays = {"windows": windows, "lengths": lengths, "counts": counts,
                  "ids": np.asarray([case["id"] for case in cases])}
        if name != "test":
            arrays["labels"] = targets(cases)
        np.savez(output / f"{name}.npz", **arrays)
        print(f"{name}: {len(cases)} cases")


def dataset(cache_path, teacher_path=None, raw_teacher=False):
    with np.load(cache_path, allow_pickle=False) as data:
        teacher = None
        if teacher_path:
            if raw_teacher:
                teacher = np.load(teacher_path, allow_pickle=False)
            else:
                with np.load(teacher_path, allow_pickle=False) as saved:
                    if str(saved["cache_sha256"]) != digest(cache_path):
                        raise ValueError("Teacher logits were generated from a different cache")
                    teacher = saved["logits"]
            if not np.isfinite(teacher).all():
                raise ValueError("Nonfinite teacher logits")
        return CaseDataset(data["windows"], data["lengths"], data["labels"], teacher)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, required, *, run=False, runtime=False):
        action = commands.add_parser(name)
        for key in required.split():
            action.add_argument(f"--{key}", required=True)
        if run:
            action.add_argument("--run", choices=RUNS, required=True)
        if runtime:
            action.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            action.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
        return action

    prep = command("prepare", "train dev test tokenizer output")
    prep.add_argument("--core-ids")
    prep.add_argument("--seed", type=int, default=42)
    train = command("train", "cache output", run=True, runtime=True)
    train.add_argument("--foundation", help="Original Phase 1 weights, required for a fresh run")
    train.add_argument("--teacher")
    train.add_argument("--archived-teacher", action="store_true",
                       help="Accept a historical .npy aligned to the archived train_core case order")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save-every", type=int, default=256, help="Checkpoint interval in updates")
    command("teacher-logits", "cache checkpoint output", run=True, runtime=True)
    command("calibrate", "cache checkpoint output", run=True, runtime=True)
    action = command("evaluate", "cache checkpoint", run=True, runtime=True)
    action.add_argument("--thresholds")
    command("submit", "test tokenizer checkpoint thresholds output", run=True, runtime=True)
    command("validate", "test predictions")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "validate":
        ids = [case["id"] for case in read_cases(args.test, labeled=False)]
        print(f"Validated {validate(read_predictions(args.predictions), ids)} predictions")
        return

    device = torch.device(args.device)
    dtype = precision_dtype(device, args.precision)
    run = RUNS[args.run]
    if args.command == "train":
        if args.save_every < 0:
            parser.error("--save-every must be nonnegative")
        resumed = (Path(args.output) / "last.pt").exists()
        if not resumed and not args.foundation:
            parser.error("--foundation is required without last.pt")
        if run.student != bool(args.teacher):
            parser.error("E06 requires --teacher and other runs do not accept it")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cache = Path(args.cache)
        train_path, val_path = cache / "train_core.npz", cache / "val_internal.npz"
        files = [train_path, val_path] + ([args.teacher] if args.teacher else [])
        signature = ":".join([run.name, str(args.seed), *(digest(path) for path in files)])
        fit(make_model(run, None if resumed else args.foundation), run,
            dataset(train_path, args.teacher, args.archived_teacher), dataset(val_path),
            args.output, device, dtype, args.seed, args.save_every, signature)
        return

    model = restore_model(run, args.checkpoint, device)
    if args.command == "submit":
        count = submit(model, args.tokenizer, args.test, args.thresholds, args.output,
                       run.eval_batch_size, device, dtype)
        print(f"Wrote {count} case predictions")
        return

    cached = dataset(args.cache)
    logits = predict_logits(model, cached, run.eval_batch_size, device, dtype)
    if args.command == "teacher-logits":
        with Path(args.output).open("wb") as stream:
            np.savez(stream, logits=logits, cache_sha256=digest(args.cache))
        return

    probs = probabilities(logits)
    if args.command == "calibrate":
        thresholds, summary = calibrate(cached.y, probs)
        with Path(args.output).open("wb") as stream:
            np.save(stream, thresholds)
    else:
        thresholds = load_thresholds(args.thresholds) if args.thresholds else .30
        summary = {}
    print(json.dumps({**summary, **metrics(cached.y, probs, thresholds)}, indent=2))


if __name__ == "__main__":
    main()
