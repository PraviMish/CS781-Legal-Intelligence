"""Portable entry points for the Phase 4 research experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import (
    load_cases,
    load_tokenizer,
    make_splits,
    read_split,
    save_split,
    tokenize_cases,
)
from .inference import choose_precision, e01_model, predict
from .metrics import blend, metrics, select_blend, select_threshold
from .runtime import Features, create_model, extract_features
from .submission import (
    read_probabilities,
    read_submission,
    write_probabilities,
    write_submission,
)
from .train import digest_file, train

RUNS = tuple(f"E{i:02d}" for i in range(1, 9))
SPLITS = ("train_core", "val_internal", "dev_safe", "test")
CONFIG_DIR = Path(__file__).with_name("configs")
BATCH = {"E02": 256, "E06": 12, "E08": 32}
THRESHOLDS = {
    "E01": 0.51,
    "E02": 0.45,
    "E03": 0.49,
    "E04": 0.50,
    "E05": 0.49,
    "E06": 0.46,
    "E07": 0.44,
    "E08": 0.47,
}


def prepare(args):
    if args.archived_splits:
        if any((args.train, args.dev, args.test)):
            raise ValueError("Choose either archived splits or the three official raw files")
        cases = {
            name: load_cases(args.archived_splits / f"{name}.parquet", labeled=name != "test")
            for name in SPLITS
        }
    else:
        if not all((args.train, args.dev, args.test)):
            raise ValueError("Supply all three official raw files")
        cases = make_splits(
            load_cases(args.train, labeled=True),
            load_cases(args.dev, labeled=True),
            load_cases(args.test, labeled=False),
        )
    tokenizer, special = load_tokenizer(args.tokenizer)
    for name, rows in cases.items():
        path = args.cache / name
        if path.exists():
            raise FileExistsError(f"{path} already exists. Use a fresh cache directory.")
        save_split(path, tokenize_cases(rows, tokenizer, special, args.strategy))
        print(f"{name} {len(rows)} cases", flush=True)


def tensor_split(args, name):
    split = read_split(args.cache / name)
    if args.run == "E02":
        if args.features is None or split.lengths.shape[1] != 5:
            raise ValueError("E02 requires hierarchical --cache and matching --features")
        x = np.load(args.features / f"{name}_features.npy", mmap_mode="r", allow_pickle=False)
        logits = np.load(args.features / f"{name}_logits.npy", mmap_mode="r", allow_pickle=False)
        if x.shape != (len(split.ids), 5, 768) or logits.shape != (len(split.ids), 5, 2):
            raise ValueError("Hierarchical features differ from the split dimensions")
        return Features(split, x, logits)
    if split.strategy != "tail":
        raise ValueError("This run requires a single tail-window cache")
    return split


def save_teacher_artifacts(args, device, precision):
    teacher = e01_model(args.checkpoint).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.splits:
        split = read_split(args.cache / name)
        if args.command == "features":
            generated = extract_features(teacher, split, device, precision)
            np.save(args.output / f"{name}_features.npy", generated.features, allow_pickle=False)
            np.save(args.output / f"{name}_logits.npy", generated.logits, allow_pickle=False)
        else:
            if split.strategy != "tail":
                raise ValueError("Teacher logits require the tail-window cache")
            logits = predict(teacher, split, device, precision, 16, return_logits=True)
            np.save(args.output / f"{name}_logits.npy", logits, allow_pickle=False)
        print(f"{name} {len(split.ids)} cases", flush=True)


def load_logits(directory, name, size):
    values = np.load(directory / f"{name}_logits.npy", allow_pickle=False)
    if values.shape != (size, 2) or not np.isfinite(values).all():
        raise ValueError(f"Expected two finite logits per case for {name}")
    return values.astype(np.float32)


def ensemble(args):
    splits = {name: read_split(args.cache / name) for name in SPLITS[1:]}
    teacher = {}
    other = {}
    prefix = f"{args.member_prefix}_" if args.member_prefix else ""
    for name, split in splits.items():
        teacher[name] = (
            torch.from_numpy(load_logits(args.teacher_logits, name, len(split.ids)))
            .softmax(-1)[:, 1]
            .numpy()
        )
        suffix = "test_probabilities" if name == "test" else name
        other[name] = read_probabilities(
            args.member / f"{prefix}{suffix}.csv", split.ids, split.labels
        )
    weight, threshold, internal = select_blend(
        splits["val_internal"].labels, teacher["val_internal"], other["val_internal"]
    )
    test = blend(teacher["test"], other["test"], weight)
    write_submission(args.output, splits["test"].ids, test, threshold)
    dev = metrics(
        splits["dev_safe"].labels, blend(teacher["dev_safe"], other["dev_safe"], weight), threshold
    )
    print(
        json.dumps(
            {
                "run": args.run,
                "e01_weight": weight,
                "threshold": threshold,
                "internal_macro_f1": internal,
                "dev_macro_f1": dev["macro_f1"],
            },
            indent=2,
        )
    )


def execute_run(args, device, precision):
    if args.run == "E06" and args.tokenizer is None:
        raise ValueError("E06 requires --tokenizer for the BOS query ID")
    bos_id = load_tokenizer(args.tokenizer)[1].bos if args.tokenizer else None
    if args.command == "train":
        config = json.loads((CONFIG_DIR / f"{args.run}.json").read_text(encoding="utf-8"))
        if args.run == "E01" and args.foundation is None:
            raise ValueError("E01 training requires its Phase 1 GQA 700M foundation")
        if args.run == "E08" and args.teacher_logits is None:
            raise ValueError("E08 training requires --teacher-logits")
        if args.run == "E02" and args.features is None:
            raise ValueError("E02 training requires --features")
        torch.manual_seed(config["seed"])
        training = tensor_split(args, "train_core")
        validation = tensor_split(args, "val_internal")
        inputs = [args.foundation, args.tokenizer]
        for name in SPLITS[:2]:
            inputs.extend(
                args.cache / name / file for file in ("tokens.npy", "lengths.npy", "cases.json")
            )
            if args.run == "E02":
                inputs.extend(
                    args.features / f"{name}_{suffix}.npy" for suffix in ("features", "logits")
                )
        teacher = None
        if args.run == "E08":
            teacher = load_logits(args.teacher_logits, "train_core", len(training.ids))
            inputs.append(args.teacher_logits / "train_core_logits.npy")
        model = create_model(args.run, foundation=args.foundation, bos_id=bos_id).to(device)
        best = train(
            model,
            args.run,
            config,
            training,
            validation,
            args.output,
            device,
            precision,
            identity=[digest_file(path) for path in inputs if path is not None],
            teacher_logits=teacher,
            batch_size=args.batch_size,
        )
        print(best)
        return

    model = create_model(
        args.run, checkpoint=args.checkpoint, foundation=args.foundation, bos_id=bos_id
    ).to(device)
    eval_precision = "fp32" if args.run == "E02" else precision
    batch_size = args.batch_size or BATCH.get(args.run, 16)
    threshold = THRESHOLDS[args.run] if args.threshold is None else args.threshold
    if args.command == "evaluate":
        validation = tensor_split(args, "val_internal")
        probabilities = predict(model, validation, device, eval_precision, batch_size)
        if args.threshold is None:
            grid = (
                np.linspace(0.25, 0.75, 51)
                if args.run in ("E01", "E02", "E03")
                else np.linspace(0.20, 0.80, 61)
            )
            threshold = select_threshold(
                validation.labels, probabilities, grid, tie_near_half=True
            )
        dev = tensor_split(args, "dev_safe")
        dev_probs = predict(model, dev, device, eval_precision, batch_size)
        if args.output:
            write_probabilities(
                args.output / "val_internal.csv", validation, probabilities, threshold
            )
            write_probabilities(args.output / "dev_safe.csv", dev, dev_probs, threshold)
        print(
            json.dumps(
                {
                    "threshold": threshold,
                    "val_internal": metrics(validation.labels, probabilities, threshold),
                    "dev_safe": metrics(dev.labels, dev_probs, threshold),
                },
                indent=2,
            )
        )
    else:
        test = tensor_split(args, "test")
        probabilities = predict(model, test, device, eval_precision, batch_size)
        write_submission(args.output, test.ids, probabilities, threshold)
        if args.probabilities:
            write_probabilities(args.probabilities, test, probabilities, threshold)


def compute_arguments(parser):
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")


def positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("Use a positive batch size")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="recreate splits and tokenize documents")
    for name in ("train", "dev", "test", "archived-splits"):
        prep.add_argument(f"--{name}", type=Path)
    prep.add_argument("--cache", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--strategy", choices=("tail", "head_tail", "hierarchical"), default="tail")

    for name in ("features", "teacher-logits"):
        cmd = commands.add_parser(name, help="cache E01 outputs for E02 or E08/E09/E10")
        for arg in ("cache", "checkpoint", "output"):
            cmd.add_argument(f"--{arg}", type=Path, required=True)
        cmd.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
        compute_arguments(cmd)

    for name in ("train", "evaluate", "submit"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run", choices=RUNS, required=True)
        cmd.add_argument("--cache", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=name != "evaluate")
        for arg in ("foundation", "tokenizer", "features"):
            cmd.add_argument(f"--{arg}", type=Path)
        cmd.add_argument("--batch-size", type=positive_int)
        if name == "train":
            cmd.add_argument("--teacher-logits", type=Path)
        else:
            cmd.add_argument("--checkpoint", type=Path, required=True)
            cmd.add_argument("--threshold", type=float)
        if name == "submit":
            cmd.add_argument("--probabilities", type=Path)
        compute_arguments(cmd)

    mix = commands.add_parser("ensemble", help="tune and rebuild E09/E10 from cached outputs")
    mix.add_argument("--run", choices=("E09", "E10"), required=True)
    mix.add_argument(
        "--member-prefix", default="", help="filename prefix used by archived probabilities"
    )
    for arg in ("cache", "teacher-logits", "member", "output"):
        mix.add_argument(f"--{arg}", type=Path, required=True)
    check = commands.add_parser(
        "validate", help="validate a JSON or ZIP against the official test IDs"
    )
    check.add_argument("--test", type=Path, required=True)
    check.add_argument("--predictions", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "validate":
        ids = [row["id"] for row in load_cases(args.test, labeled=False)]
        print(f"Validated {len(read_submission(args.predictions, ids))} case predictions")
    elif args.command == "ensemble":
        ensemble(args)
    else:
        device = torch.device(args.device)
        precision = choose_precision(device, args.precision)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        if getattr(args, "threshold", None) is not None and not 0 <= args.threshold <= 1:
            parser.error("--threshold must be between 0 and 1")
        if args.command in ("features", "teacher-logits"):
            save_teacher_artifacts(args, device, precision)
        else:
            execute_run(args, device, precision)


if __name__ == "__main__":
    main()
