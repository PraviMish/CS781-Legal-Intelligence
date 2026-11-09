"""Command-line interface: python -m phase3 {audit,train,predict,validate,viterbi,hybrid}."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .configs import RUNS, POSTPROCESSING, experiment
from .data import load_official, load_tokenizer, make_rows, joined
from .metrics import class_metrics
from .runtime import train, load_best, predict, cache_teacher_logits, choose_device
from .submission import assemble, validate_predictions, write_submission, read_submission, hybrid
from .context import transition_logprobs, viterbi


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit", "train", "predict", "validate", "viterbi", "hybrid"):
        s = sub.add_parser(name)
        s.add_argument("--data", type=Path, required=True,
                       help="Official Phase_3_RR directory containing Train and Test")
        s.add_argument("--output", type=Path, default=Path("artifacts/phase3"))
        if name not in ("audit", "validate", "hybrid"):
            s.add_argument("--tokenizer", type=Path, required=True)
            s.add_argument("--foundations", type=Path, required=True,
                           help="Root containing the archived Phase-1 models/ directory")
            s.add_argument("--checkpoints", type=Path, required=True,
                           help="Directory containing experiment_id/best.pt")
            s.add_argument("--device", default="auto")
        if name in ("train", "predict"):
            s.add_argument("--experiment", choices=RUNS, required=True)
        if name in ("validate", "hybrid"):
            s.add_argument("--submission", type=Path)
        if name == "hybrid":
            s.add_argument("--cl", type=Path, required=True)
            s.add_argument("--it", type=Path, required=True)
        if name == "viterbi":
            s.add_argument("--strength", type=float, default=None)
    return p


def run(argv=None):
    args = parser().parse_args(argv)
    docs, labels = load_official(args.data)
    label_to_id = {label: i for i, label in enumerate(labels)}
    all_test_docs = docs["CL_test"] + docs["IT_test"]
    if args.command == "audit":
        counts = {split: {"documents": len(part), "sentences": sum(len(d["text"]) for d in part)}
                  for split, part in docs.items()}
        classes = {domain: dict(Counter(label for doc in docs[domain+"_train"] for label in doc["labels"]))
                   for domain in ("CL", "IT")}
        print(json.dumps({"splits": counts, "train_labels": classes, "label_order": labels}, indent=2))
        return
    if args.command == "validate":
        if args.submission is None:
            raise ValueError("Supply --submission")
        count = validate_predictions(read_submission(args.submission), all_test_docs, labels)
        print(f"PASS: {len(all_test_docs)} documents and {count} sentence labels")
        return
    if args.command == "hybrid":
        output = hybrid(read_submission(args.cl), read_submission(args.it), docs)
        path = args.output / "D3_domain_specialist_hybrid" / "submission.zip"
        write_submission(path, output, all_test_docs, labels)
        print(path)
        return

    tok = load_tokenizer(args.tokenizer)
    device = choose_device(args.device)

    if args.command in ("train", "predict"):
        spec = experiment(args.experiment)
        rows = make_rows(docs, tok, label_to_id, context=spec.context)
        if args.command == "train":
            train_rows = joined(rows, "train") if spec.domain == "joint" else rows[spec.domain+"_train"]
            dev_rows = joined(rows, "dev")
            teacher_logits = None
            if spec.loss == "kd":
                teacher = experiment(spec.teacher)
                teacher_rows = make_rows(docs, tok, label_to_id, context=teacher.context)
                teacher_logits = cache_teacher_logits(
                    spec, teacher, joined(teacher_rows, "train"), args.foundations, args.checkpoints,
                    args.output / "cache" / f"teacher_logits_{teacher.id}.pt", device)
            result = train(spec, train_rows, dev_rows, foundation_root=args.foundations,
                           checkpoint_root=args.checkpoints, device=device, teacher_logits=teacher_logits)
            print(json.dumps(result, indent=2))
            return
        model = load_best(spec, args.foundations, args.checkpoints, device=device)
        test_rows = joined(rows, "test")
        output = predict(model, test_rows, device=device)
        predictions = assemble(test_rows, output, labels)
        path = args.output / spec.id / "submission.zip"
        write_submission(path, predictions, all_test_docs, labels)
        print(path)
        return

    if args.command == "viterbi":
        baseline = experiment("A5_gqa700_lora")
        model = load_best(baseline, args.foundations, args.checkpoints, device=device)
        rows = make_rows(docs, tok, label_to_id, context="sentence")
        start_log, transition_log = transition_logprobs(
            docs["CL_train"] + docs["IT_train"], label_to_id)

        def decode(rows_to_decode, strength):
            out = predict(model, rows_to_decode, device=device, return_logits=True)
            grouped = {}
            for row in rows_to_decode:
                key = (row["domain"], row["doc_id"])
                grouped.setdefault(key, []).append(
                    (row["sentence_idx"], out[(row["domain"], row["doc_id"], row["sentence_idx"])]) )
            predictions = {}
            for key, items in grouped.items():
                items.sort()
                raw = torch.tensor([v for _, v in items], dtype=torch.float32)
                emissions = F.log_softmax(raw, -1).numpy()
                predictions[key] = viterbi(emissions, start_log, transition_log, strength)
            return predictions

        dev_rows = joined(rows, "dev")
        if args.strength is None:
            best_strength, best_score = 0., -1.
            for strength in POSTPROCESSING["C3_document_viterbi"]["transition_strength_grid"]:
                decoded = decode(dev_rows, strength)
                true = [row["label_id"] for row in dev_rows]
                predicted = [decoded[(row["domain"], row["doc_id"])][row["sentence_idx"]]
                             for row in dev_rows]
                score = class_metrics(true, predicted)["macro_f1"]
                if score > best_score:
                    best_strength, best_score = strength, score
            strength = best_strength
            print(f"Dev-selected strength={strength}, Macro-F1={best_score:.6f}")
        else:
            strength = args.strength
        test_rows = joined(rows, "test")
        decoded = decode(test_rows, strength)
        predictions = {}
        for (domain, doc_id), ids in decoded.items():
            predictions[doc_id] = [labels[i] for i in ids]
        path = args.output / "C3_document_viterbi" / "submission.zip"
        write_submission(path, predictions, all_test_docs, labels)
        print(path)
        return


if __name__ == "__main__":
    run()
