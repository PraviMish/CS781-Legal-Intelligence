"""Recompute Phase 5 aggregates from private archived files."""

import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from phase5.data import LABELS, check_disjoint_ids, read_cases, targets, text_hash
from phase5.metrics import GRID, calibrate, f1, metrics, per_label_counts, probabilities
from phase5.models import RUNS
from phase5.submission import load_thresholds, read_predictions, validate


def quantiles(values, levels):
    return {str(q): float(np.percentile(values, q)) for q in levels}


def coverage(counts):
    return {"tokens": quantiles(counts, [50, 75, 90, 95, 99]),
            "multi_window_share": float((counts > 510).mean()),
            "over_1530_share": float((counts > 1530).mean()),
            "mean_token_coverage": float((np.minimum(counts, 1530) / np.maximum(counts, 1)).mean())}


def lfs_identity(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    pointer = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{digest.hexdigest()}\nsize {path.stat().st_size}\n").encode()
    git_blob = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
    return {"bytes": path.stat().st_size, "lfs_pointer_git_blob": git_blob}


def precision_recall(y, predictions):
    tp, fp, fn = (counts.sum() for counts in per_label_counts(y, predictions))
    return {"micro_precision": float(tp / max(tp + fp, 1)),
            "micro_recall": float(tp / max(tp + fn, 1))}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(source, output):
    dev = read_cases(source / "dev.json", labeled=True)
    test = read_cases(source / "test.json", labeled=False)
    check_disjoint_ids(dev, test)
    y = targets(dev)
    ids = [case["id"] for case in test]
    stats = {split: coverage(np.load(source / f"{split}_counts.npy", allow_pickle=False))
             for split in ("train_core", "val_internal", "dev", "test")}
    for split, cases in (("dev", dev), ("test", test)):
        stats[split].update(documents=len(cases),
                            unique_texts=len({text_hash(row["text"]) for row in cases}),
                            characters=quantiles([len(row["text"]) for row in cases], [50, 95]),
                            **lfs_identity(source / f"{split}.json"))
    pairs = Counter(pair for row in dev for pair in combinations(row["labels"], 2))
    stats["dev"].update(cardinality=float(y.sum(1).mean()), density=float(y.mean()),
                        labels_per_case=dict(sorted(Counter(map(len, (c["labels"] for c in dev))).items())),
                        top_cooccurrences=[{"labels": pair, "count": count}
                                           for pair, count in pairs.most_common(10)])
    with (source / "label_frequencies.csv").open(newline="") as stream:
        frequencies = list(csv.DictReader(stream))
    counts = [int(row["train_count"]) for row in frequencies]
    stats["train"] = {"documents": 42750, "unique_texts": 42315,
                      "characters": {"50": 2717, "95": 23797},
                      "cardinality": sum(counts) / 42750, "density": sum(counts) / 4275000,
                      "min_label_support": min(counts), "max_label_support": max(counts),
                      "shared_exact_texts_with_dev": 81, "shared_exact_texts_with_test": 108,
                      "source": "Executed resume notebook corpus audit and archived label frequencies"}
    for split, size in (("train_core", 38444), ("val_internal", 4306)):
        stats[split]["documents"] = size

    submission_audit = {}
    dev_replay = {}
    threshold_curves = []
    predictions = {}
    errors = []
    length_errors = []
    dev_counts = np.load(source / "dev_counts.npy", allow_pickle=False)
    for code, run in RUNS.items():
        path = source / f"{run.name}.zip"
        rows = read_predictions(path)
        validate(rows, ids)
        predictions[code] = [row["pred"] for row in rows]
        submission_audit[code] = {
            "documents": len(rows), "official_order": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mean_pred_labels": float(np.mean([len(row["pred"]) for row in rows])),
        }
        logits = np.load(source / f"{code}_dev_logits.npy", allow_pickle=False)
        probs = probabilities(logits)
        thresholds = load_thresholds(source / f"{code}_thresholds.npy")
        rebuilt, calibration = calibrate(y, probs)
        if not np.array_equal(rebuilt, thresholds):
            raise ValueError(f"{code}: regenerated thresholds differ from the archive")
        actual = metrics(y, probs, thresholds)
        archived = json.loads((source / f"{code}_dev_metrics.json").read_text(encoding="utf-8"))
        for key, value in {**actual, **calibration}.items():
            if not np.isclose(value, archived[key], atol=1e-10, rtol=0):
                raise ValueError(f"{code}: {key} differs from the archive")
        global_pred = probs >= calibration["global_threshold"]
        local_pred = probs >= thresholds
        dev_replay[code] = {**actual, **calibration, "threshold_array_exact": True,
                            "global_decision": precision_recall(y, global_pred),
                            "per_statute_decision": precision_recall(y, local_pred)}
        threshold_curves.extend({"run": code, "threshold": round(float(t), 3),
                                 "macro_f1": metrics(y, probs, float(t))["macro_f1"]} for t in GRID)
        if code == "E02":
            tp, fp, fn = per_label_counts(y, probs >= thresholds)
            for i, label in enumerate(LABELS):
                errors.append({"statute_id": label, "dev_support": int(y[:, i].sum()),
                               "threshold": float(thresholds[i]), "f1": float(f1(tp[i], fp[i], fn[i])),
                               "true_positives": int(tp[i]), "false_positives": int(fp[i]),
                               "false_negatives": int(fn[i])})
            for name, mask in (("0-510", dev_counts <= 510),
                               ("511-1530", (dev_counts > 510) & (dev_counts <= 1530)),
                               ("1531+", dev_counts > 1530)):
                length_errors.append({"tokens": name, "cases": int(mask.sum()),
                                      "gold_cardinality": float(y[mask].sum(1).mean()),
                                      **metrics(y[mask], probs[mask], thresholds)})

    agreement = sum(a == b for a, b in zip(predictions["E02"], predictions["E06"]))
    result = {"submissions": submission_audit, "dev_calibration_replay": dev_replay,
              "e02_e06_identical_test_sets": agreement, "e02_e06_total_test_cases": len(test),
              "e02_dev_length_groups": length_errors}
    for filename, value in (("dataset_statistics.json", stats), ("archive_verification.json", result)):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "threshold_curves.csv", threshold_curves)
    write_csv(output / "e02_per_statute.csv", errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Private normalized archive directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for aggregate records")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit(args.source, args.output)


if __name__ == "__main__":
    main()
