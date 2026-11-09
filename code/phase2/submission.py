"""Codabench submission archive IO with exact bounds and document coverage checks."""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from .data import LABELS, read_json


def validate_predictions(predictions, texts, *, nonoverlapping=False):
    if not isinstance(predictions, dict) or predictions.keys() != texts.keys():
        raise ValueError("Prediction IDs must exactly match the official test IDs.")
    for key, spans in predictions.items():
        if not isinstance(spans, list):
            raise ValueError(f"Expected a list of spans for {key}.")
        previous_end = 0
        for span in spans:
            if not isinstance(span, list) or len(span) != 3:
                raise ValueError(f"Expected [start, end, label] for {key}.")
            start, end, label = span
            if type(start) is not int or type(end) is not int:
                raise ValueError(f"Offsets must be integer character indices for {key}.")
            if not 0 <= start < end <= len(texts[key]) or label not in LABELS:
                raise ValueError(f"Invalid span bounds or label for {key}.")
            if nonoverlapping and start < previous_end:
                raise ValueError(f"Ensemble spans must be sorted and nonoverlapping for {key}.")
            previous_end = end


def load_predictions(path):
    if Path(path).suffix.lower() != ".zip":
        return read_json(path)
    with ZipFile(path) as archive:
        if archive.namelist() != ["prediction.json"]:
            raise ValueError("Expected exactly prediction.json at the archive root.")
        return json.loads(archive.read("prediction.json"))


def write_submission(path, predictions, texts, *, nonoverlapping=False):
    validate_predictions(predictions, texts, nonoverlapping=nonoverlapping)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("prediction.json", json.dumps(predictions, ensure_ascii=False, indent=2))
        if load_predictions(temporary) != predictions:
            raise RuntimeError("Submission archive did not round-trip.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
