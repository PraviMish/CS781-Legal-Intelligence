"""Phase 5 output schema and ZIP validation."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .data import LABELS


def load_thresholds(path):
    thresholds = np.load(path, allow_pickle=False)
    if (thresholds.shape != (len(LABELS),) or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0) or np.any(thresholds > 1)):
        raise ValueError("Expected one finite probability threshold per statute")
    return thresholds


def decode(ids, probs, thresholds):
    if (probs.shape != (len(ids), len(LABELS)) or not np.isfinite(probs).all()
            or np.shape(thresholds) != (len(LABELS),)):
        raise ValueError("Invalid probabilities, thresholds or case IDs")
    result = []
    for case_id, row in zip(ids, probs):
        selected = np.flatnonzero(row >= thresholds).tolist()
        if not selected:
            selected = [int(row.argmax())]
        result.append({"id": case_id, "pred": [LABELS[i] for i in selected]})
    return result


def validate(records, ids):
    if not isinstance(records, list):
        raise ValueError("Prediction JSON must be a list")
    for row in records:
        if not isinstance(row, dict) or set(row) != {"id", "pred"}:
            raise ValueError("Expected prediction fields id and pred")
        labels = row["pred"]
        if (type(row["id"]) is not str or not isinstance(labels, list) or not labels
                or any(type(x) is not int or x not in LABELS for x in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("Invalid prediction row")
    if [row["id"] for row in records] != list(ids) or len(set(ids)) != len(ids):
        raise ValueError("Predictions must cover each official ID in source order")
    return len(records)


def read_predictions(path):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return json.loads(path.read_text(encoding="utf-8"))
    with ZipFile(path) as archive:
        if archive.namelist() != ["prediction.json"] or archive.testzip() is not None:
            raise ValueError("Expected a valid ZIP with one root prediction.json")
        return json.loads(archive.read("prediction.json"))


def write_predictions(path, records, ids):
    validate(records, ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("prediction.json", payload)
        else:
            temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
