"""Train, score and package the Phase 2 experiments using local artifacts."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from common.artifacts import write_json
from .analysis.diagnostics import summarize_ensemble
from .configs import DECODER_FULL, RUNS
from .data import load_texts, prepare_training, tokenize, tokenizer_from
from .ensemble import MEMBERS, STRATEGIES, ensemble_documents
from .metrics import strict_metrics
from .models import build_ner, load_state, restore_ner
from .runtime import predict_documents, resolve_device
from .submission import load_predictions, validate_predictions, write_submission
from .train import train


def _model_args(parser, *, checkpoint=False):
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, help="Original Phase 1 best/snapshot .pt.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    if checkpoint:
        parser.add_argument("--checkpoint", required=True, type=Path)
        parser.add_argument("--batch-size", type=int, help="Override the original inference batch.")


def _training_data_args(parser):
    parser.add_argument("--train-dir", type=Path, required=True, help="Official Phase_2_LNER/Train.")
    parser.add_argument("--split", type=Path, help="Archived validation_split.json.")
    parser.add_argument("--seed", type=int, default=42, help="Supply the historical seed privately.")


def _ensemble_args(parser):
    for member in MEMBERS:
        parser.add_argument(f"--{member}", required=True, type=Path, help="Member JSON or ZIP.")


def _restore(args, pad_id, device):
    run = RUNS[args.run]
    if run.method in {"probe", "lora", "lorma"} and args.foundation is None:
        raise ValueError("This classifier/adapter checkpoint also requires its Phase 1 foundation.")
    model = build_ner(run, pad_id, foundation=args.foundation, device=device)
    restore_ner(model, load_state(args.checkpoint))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("train", help="Train or resume one experiment.")
    _model_args(fit)
    _training_data_args(fit)
    fit.add_argument("--output-dir", required=True, type=Path)
    fit.add_argument("--student-initialization", type=Path)
    fit.add_argument("--teacher-checkpoint", type=Path)
    fit.add_argument("--micro-batch", type=int, choices=(2, 4),
                     help="For full fine-tuning, preserve the 16-window effective batch.")

    score = commands.add_parser("evaluate", help="Score a checkpoint on the held-out documents.")
    _model_args(score, checkpoint=True)
    _training_data_args(score)
    score.add_argument("--predictions", type=Path, help="Save validation predictions privately.")

    predict = commands.add_parser("submit", help="Predict fold 3 and write a Codabench ZIP.")
    _model_args(predict, checkpoint=True)
    predict.add_argument("--test-json", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)

    combine = commands.add_parser("ensemble", help="Vote on fold-3 predictions and write a ZIP.")
    _ensemble_args(combine)
    combine.add_argument("--test-json", type=Path, required=True)
    combine.add_argument("--strategy", choices=STRATEGIES, default="majority")
    combine.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare-votes", help="Score the five historical voting rules.")
    _ensemble_args(compare)
    _training_data_args(compare)
    compare.add_argument("--tokenizer", type=Path, required=True)
    compare.add_argument("--diagnostics", action="store_true",
                         help="Also summarize errors and the paired validation bootstrap.")

    validate = commands.add_parser("validate", help="Check an existing prediction JSON or ZIP.")
    validate.add_argument("--test-json", required=True, type=Path)
    validate.add_argument("--predictions", required=True, type=Path)
    validate.add_argument("--nonoverlapping", action="store_true")

    args = parser.parse_args()
    if args.command in {"validate", "ensemble", "submit"}:
        texts = load_texts(args.test_json)
        if len(texts) != 35:
            parser.error("Expected the 35 official fold-3 judgments.")

    if args.command == "validate":
        pred = load_predictions(args.predictions)
        validate_predictions(pred, texts, nonoverlapping=args.nonoverlapping)
        print(f"Validated {len(pred)} documents and {sum(map(len, pred.values()))} spans.")
        return

    if args.command in {"ensemble", "compare-votes"}:
        members = {name: load_predictions(getattr(args, name)) for name in MEMBERS}
        if args.command == "ensemble":
            for pred in members.values():
                validate_predictions(pred, texts)
            pred = ensemble_documents(members, args.strategy)
            write_submission(args.output, pred, texts, nonoverlapping=True)
        else:
            tokenizer = tokenizer_from(args.tokenizer)
            _, gold, split = prepare_training(args.train_dir, tokenizer, full_data=False,
                                              seed=args.seed, split_path=args.split)
            gold = {key: gold[key] for key in split["val"]}
            if members["gqa"].keys() != gold.keys():
                parser.error("Validation prediction IDs must match the selected held-out split.")
            results = {name: strict_metrics(gold, ensemble_documents(members, name))[0]
                       for name in STRATEGIES}
            if args.diagnostics:
                best = max(results, key=lambda name: results[name]["macro_f1"])
                results["diagnostics"] = summarize_ensemble(gold, members, best, args.seed)
            print(json.dumps(results, indent=2))
        return

    run = RUNS[args.run]
    tokenizer = tokenizer_from(args.tokenizer)
    pad_id = tokenizer.token_to_id("<pad>")
    device = resolve_device(args.device)

    if args.command == "train":
        if args.foundation is None and args.student_initialization is None:
            parser.error("Supply the original foundation or the archived student initialization.")
        if args.foundation and args.student_initialization:
            parser.error("Choose one training initialization.")
        if args.teacher_checkpoint and run.method != "distilled":
            parser.error("A teacher checkpoint applies only to distilled students.")
        if args.micro_batch is not None:
            if run.method != "full" and not run.full_data:
                parser.error("Micro-batch override applies to full fine-tuning and full-data runs.")
            inference_batch = run.inference_batch
            if args.run == DECODER_FULL:
                inference_batch = 2 * args.micro_batch
            run = replace(run, batch=args.micro_batch, accumulation=16 // args.micro_batch,
                          inference_batch=inference_batch)
        docs, gold, split = prepare_training(args.train_dir, tokenizer, full_data=run.full_data,
                                             seed=args.seed, split_path=args.split)
        print(train(args.run, run, docs, gold, split, pad_id, args.seed, args.output_dir, device,
                    foundation=args.foundation, student_initialization=args.student_initialization,
                    teacher_checkpoint=args.teacher_checkpoint))
        return

    if args.command == "evaluate" and run.full_data:
        parser.error("Full-data runs used all labeled documents and have no independent local split.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    batch_size = args.batch_size or run.inference_batch
    model = _restore(args, pad_id, device)
    if args.command == "submit":
        docs = tokenize(tokenizer, texts)
        pred = predict_documents(model, docs, texts, batch_size, device)
        write_submission(args.output, pred, texts)
    else:
        docs, gold, split = prepare_training(args.train_dir, tokenizer, full_data=False,
                                             seed=args.seed, split_path=args.split)
        keys = split["val"]
        pred = predict_documents(model, docs, keys, batch_size, device)
        metrics, per_class = strict_metrics({key: gold[key] for key in keys}, pred)
        if args.predictions:
            write_json(args.predictions, pred)
        print(json.dumps({"summary": metrics, "per_class": per_class}, indent=2))


if __name__ == "__main__":
    main()
