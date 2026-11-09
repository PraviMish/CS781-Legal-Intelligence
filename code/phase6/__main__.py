"""Prepare, train, evaluate, and submit the historical Phase 6 experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer

from .data import load_records, load_tokenizer, split_records
from .decoding import summarize
from .models import METHODS, VERSIONS, configuration, restore_model
from .preparation import input_builder, training_group
from .submission import digest, generate_submission, read_submission, save_json
from .training import generate_teacher, save_torch, train


def file_key(paths):
    return hashlib.sha256(":".join(digest(path) for path in paths if path is not None).encode()).hexdigest()


def make_cache(args):
    tokenizer = load_tokenizer(args.tokenizer)
    ids = json.loads(args.validation_ids.read_text(encoding="utf-8")) if args.validation_ids else None
    train_rows, val_rows = split_records(load_records(args.train, labeled=True), ids, seed=args.seed)
    builder = input_builder(tokenizer, args.version)

    def groups(rows):
        return [group for row in rows if (group := training_group(row, builder, args.version))]

    train_groups, val_groups = groups(train_rows), groups(val_rows)
    save_torch(args.output, {
        "version": args.version, "seed": args.seed, "tokenizer_sha256": digest(args.tokenizer),
        "train_groups": train_groups, "val_groups": val_groups,
    })
    print(f"{len(train_rows)} train and {len(val_rows)} validation documents. "
          f"{sum(map(len, train_groups))} train and {sum(map(len, val_groups))} validation targets.")


def read_cache(args):
    cached = torch.load(args.cache, weights_only=True, map_location="cpu")
    if (cached.get("version"), cached.get("seed"), cached.get("tokenizer_sha256")) != (
        args.version, args.seed, digest(args.tokenizer)
    ):
        raise ValueError("Cache version, seed or tokenizer differs from the requested run")
    return cached["train_groups"], cached["val_groups"]


def rouge_summary(rows):
    if not rows:
        raise ValueError("No labeled predictions to evaluate")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(row["reference"], row["prediction"]) for row in rows]
    return {
        "count": len(rows),
        **{metric: float(np.mean([score[metric].fmeasure for score in scores]))
           for metric in ("rouge1", "rougeL")},
        "mean_generated_words": float(np.mean([len(row["prediction"].split()) for row in rows])),
    }


def evaluate(args, run, model, inputs, strategy):
    train_rows = load_records(args.train, labeled=True)
    ids = json.loads(args.validation_ids.read_text(encoding="utf-8")) if args.validation_ids else None
    _, val = split_records(train_rows, ids, seed=args.seed)
    count = args.documents or run.validation_docs
    if args.version == "original":
        subset = val[:count]
    else:
        indices = np.random.default_rng(args.seed + 42).permutation(len(val))[:count]
        subset = [val[i] for i in indices if val[i]["summary"]]
    samples = []
    for row in subset:
        prediction = summarize(model, run.architecture, inputs, row["document"],
                               args.device, args.precision, strategy, args.policy)
        samples.append({"id": row["id"], "reference": row["summary"], "prediction": prediction})
    result = rouge_summary(samples)
    if args.output:
        save_json(args.output, {"result": result, "samples": samples})
    print(json.dumps(result, indent=2))


def add_options(parser, *, model=False, seed=False):
    parser.add_argument("--version", choices=tuple(VERSIONS), default="v3")
    parser.add_argument("--tokenizer", type=Path, required=True)
    if seed:
        parser.add_argument("--seed", type=int, required=True)
    if model:
        parser.add_argument("--run", choices=tuple(METHODS), required=True)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")


def add_data_options(parser):
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    add_options(prepare, seed=True)
    add_data_options(prepare)
    prepare.add_argument("--output", type=Path, required=True)

    fit = commands.add_parser("train")
    add_options(fit, model=True, seed=True)
    for option in ("cache", "output"):
        fit.add_argument(f"--{option}", type=Path, required=True)
    for option in ("foundation", "warmstart", "teacher", "pseudo"):
        fit.add_argument(f"--{option}", type=Path)

    teacher = commands.add_parser("teacher-targets")
    add_options(teacher, model=True, seed=True)
    for option in ("cache", "checkpoint", "output"):
        teacher.add_argument(f"--{option}", type=Path, required=True)

    score = commands.add_parser("evaluate")
    add_options(score, model=True, seed=True)
    add_data_options(score)
    score.add_argument("--documents", type=int)
    score.add_argument("--output", type=Path)

    submit = commands.add_parser("submit")
    add_options(submit, model=True)
    submit.add_argument("--test", type=Path, required=True)
    submit.add_argument("--output", type=Path, required=True)
    for command in (score, submit):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--strategy", choices=("head", "section", "hierarchical"))
        command.add_argument("--policy", choices=("raw", "grounded"), default="raw")

    validate = commands.add_parser("validate")
    validate.add_argument("--test", type=Path, required=True)
    validate.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "validate":
        print(f"Validated {len(read_submission(args.predictions, load_records(args.test)))} summaries")
        return
    if args.command == "prepare":
        make_cache(args)
        return

    run = configuration(args.version, args.run)
    tokenizer = load_tokenizer(args.tokenizer)
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    if args.command in ("train", "teacher-targets"):
        train_groups, val_groups = read_cache(args)
        if args.command == "teacher-targets":
            model = restore_model(run, args.checkpoint, args.device)
            count = generate_teacher(model, run, train_groups, tokenizer, args.output,
                                     args.device, args.precision, args.seed)
            print(f"Generated {count} nonempty teacher targets")
            return
        distilled = run.method == "distillation"
        if bool(args.teacher) != (distilled and run.version == "v3"):
            parser.error("Use --teacher with the V3 E04 run only")
        if bool(args.pseudo) != (distilled and run.version != "v3"):
            parser.error("Use --pseudo with original/V2 E04 only")
        teacher_model = None
        if args.teacher:
            teacher_model = restore_model(configuration("v3", "E02"), args.teacher, args.device)
            teacher_model.requires_grad_(False)
        pseudo = json.loads(args.pseudo.read_text(encoding="utf-8")) if args.pseudo else None
        signature = file_key((args.cache, args.foundation, args.warmstart, args.teacher, args.pseudo))
        path = train(run, train_groups, [ex for group in val_groups for ex in group], tokenizer,
                     args.output, args.device, args.precision, args.seed, signature,
                     args.foundation, args.warmstart, teacher_model, pseudo)
        print(path)
        return

    model = restore_model(run, args.checkpoint, args.device)
    inputs = input_builder(tokenizer, args.version)
    strategy = args.strategy or ("section" if args.version == "original" else "hierarchical")
    if args.policy == "grounded" and args.version != "v3":
        parser.error("The grounded policy belongs to V3")
    if args.command == "evaluate":
        evaluate(args, run, model, inputs, strategy)
    else:
        fingerprints = (args.version, args.run, strategy, args.policy, args.device, args.precision,
                        file_key((args.test, args.tokenizer, args.checkpoint)))
        key = hashlib.sha256(repr(fingerprints).encode()).hexdigest()
        rows = load_records(args.test)
        generate_submission(model, run.architecture, inputs, rows, args.output, args.device,
                            args.precision, strategy, args.policy, key)
        print(f"Validated {len(rows)} summaries in {args.output}")


if __name__ == "__main__":
    main()
