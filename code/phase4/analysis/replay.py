"""Audit the saved Phase 4 submissions, local metrics and ensemble predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from phase4.metrics import blend, metrics, select_blend
from phase4.submission import read_probabilities, read_submission


def archived_file(directory, run, suffix):
    (path,) = directory.glob(f"P4_{run}_*{suffix}")
    return path


def ordered_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def replay(artifacts, expected_test_ids=None):
    labels = {}
    ids = {}
    probabilities = {}
    for split in ("val_internal", "dev_safe"):
        rows = ordered_rows(archived_file(artifacts, "E01", f"_{split}.csv"))
        ids[split] = [row["id"] for row in rows]
        labels[split] = np.array([int(row["label"]) for row in rows])
    e01_zip = archived_file(artifacts, "E01", ".zip")
    if expected_test_ids is None:
        # Fallback uses E01's archive as the test ID and order reference.
        from zipfile import ZipFile

        with ZipFile(e01_zip) as source:
            expected_test_ids = [row["id"] for row in json.loads(source.read("prediction.json"))]
    ids["test"] = expected_test_ids
    if len(ids["test"]) != 1517:
        raise ValueError("Expected 1,517 official test cases")

    def archived_predictions(run):
        rows = read_submission(archived_file(artifacts, run, ".zip"), ids["test"])
        by_id = {row["id"]: row["pred"] for row in rows}
        return np.array([by_id[case_id] for case_id in ids["test"]])

    results = {}
    test_predictions = {}
    for i in range(1, 11):
        run = f"E{i:02d}"
        summary = json.loads(
            archived_file(artifacts, run, "_summary.json").read_text(encoding="utf-8")
        )
        test_predictions[run] = archived_predictions(run)
        probabilities[run] = {}
        if i <= 8:
            threshold = summary.get("threshold", summary.get("selected_threshold"))
            for split in ("val_internal", "dev_safe"):
                probabilities[run][split] = read_probabilities(
                    archived_file(artifacts, run, f"_{split}.csv"), ids[split], labels[split]
                )
            if i > 1:
                probabilities[run]["test"] = read_probabilities(
                    archived_file(artifacts, run, "_test_probabilities.csv"), ids["test"]
                )
        else:
            threshold = summary["Threshold"]
        results[run] = {"threshold": threshold, "test_class_1": int(test_predictions[run].sum())}

    teacher = {}
    for split, split_ids in ids.items():
        logits = np.load(artifacts / f"{split}_logits.npy", allow_pickle=False)
        if logits.shape != (len(split_ids), 2) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid E01 teacher cache for {split}")
        teacher[split] = torch.from_numpy(logits.astype(np.float32)).softmax(-1)[:, 1].numpy()
    probabilities["E01"]["test"] = teacher["test"]
    for run, other in (("E09", "E05"), ("E10", "E08")):
        w, t, _ = select_blend(
            labels["val_internal"], teacher["val_internal"], probabilities[other]["val_internal"]
        )
        if not np.isclose(w, 0.9) or not np.isclose(t, results[run]["threshold"]):
            raise ValueError(f"{run} did not recover the saved ensemble selection")
        probabilities[run] = {
            split: blend(teacher[split], probabilities[other][split], w) for split in ids
        }

    for run, row in results.items():
        threshold = row["threshold"]
        replayed = probabilities[run]["test"] >= threshold
        if not np.array_equal(replayed, test_predictions[run]):
            raise ValueError(f"{run} did not reproduce the archived test labels")
        for split, targets in labels.items():
            row[split] = metrics(targets, probabilities[run][split], threshold)
    return results


def main():
    from phase4.data import load_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--test", type=Path, help="Independent official JSON used to validate case IDs"
    )
    args = parser.parse_args()
    ids = None if args.test is None else [r["id"] for r in load_cases(args.test, labeled=False)]
    print(json.dumps(replay(args.artifacts, ids), indent=2))


if __name__ == "__main__":
    main()
