"""Validate, align and package Phase 4 binary predictions."""

import csv
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def validate(rows, expected_ids):
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected case IDs are not unique")
    if not isinstance(rows, list):
        raise TypeError("Submission must be an array")
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.keys() != {"id", "pred"}
            or not isinstance(row["id"], str)
            or type(row["pred"]) is not int
            or row["pred"] not in (0, 1)
        ):
            raise ValueError("Expected exactly a string id and a binary integer pred")
        if row["id"] in seen:
            raise ValueError(f"Duplicate case ID {row['id']}")
        seen.add(row["id"])
    if seen != set(expected_ids):
        raise ValueError("Submission IDs differ from the expected case IDs")
    return rows


def read_submission(path, expected_ids):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            if archive.namelist() != ["prediction.json"]:
                raise ValueError("Archive must contain only a root-level prediction.json")
            rows = json.loads(archive.read("prediction.json"))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    return validate(rows, expected_ids)


def write_submission(path, ids, probabilities, threshold):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Submission output path must end in .zip")
    probabilities = np.asarray(probabilities)
    if (
        probabilities.shape != (len(ids),)
        or not np.isfinite(probabilities).all()
        or not ((0 <= probabilities) & (probabilities <= 1)).all()
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid probabilities or threshold")
    rows = [
        {"id": case_id, "pred": int(pred)}
        for case_id, pred in zip(ids, probabilities >= threshold)
    ]
    validate(rows, ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".zip")
    os.close(fd)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.writestr("prediction.json", json.dumps(rows, ensure_ascii=False, indent=2))
        if read_submission(temporary, ids) != rows:
            raise ValueError("Submission round trip changed predictions")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_probabilities(path, ids, labels=None):
    """Align saved probabilities and verify the case set and optional labels."""
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or by_id.keys() != set(ids):
        raise ValueError(f"Probability IDs differ from split IDs in {path}")
    ordered = [by_id[case_id] for case_id in ids]
    if labels is not None and not np.array_equal([int(row["label"]) for row in ordered], labels):
        raise ValueError(f"Probability labels differ from split labels in {path}")
    probs = np.array([row["prob_class_1"] for row in ordered], dtype=np.float32)
    if not (np.isfinite(probs) & (probs >= 0) & (probs <= 1)).all():
        raise ValueError(f"Invalid probabilities in {path}")
    return probs


def write_probabilities(path, split, probabilities, threshold):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "prob_class_1", "pred"]
    if split.labels is not None:
        fields.append("label")
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for i, (case_id, probability) in enumerate(zip(split.ids, probabilities, strict=True)):
            row = {
                "id": case_id,
                "prob_class_1": str(probability),
                "pred": int(probability >= threshold),
            }
            if split.labels is not None:
                row["label"] = int(split.labels[i])
            writer.writerow(row)
