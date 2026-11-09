"""Create and validate the list-format Codabench archive."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from .decoding import summarize


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_submission(path, test_rows):
    with ZipFile(path) as archive:
        if archive.namelist() != ["prediction.json"] or archive.testzip() is not None:
            raise ValueError("Archive must contain one valid root-level prediction.json")
        predictions = json.loads(archive.read("prediction.json"))
    if not isinstance(predictions, list):
        raise ValueError("Expected the historical list-format submission")
    for row in predictions:
        if (not isinstance(row, dict) or set(row) != {"id", "summary"}
                or not isinstance(row["id"], str)
                or not isinstance(row["summary"], str) or not row["summary"].strip()):
            raise ValueError("Expected string IDs and nonempty summaries")
    ids = [row["id"] for row in predictions]
    expected = {row["id"] for row in test_rows}
    if not expected or len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("Submission must contain each official test ID exactly once")
    return predictions


def write_submission(path, test_rows, predictions):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Submission output must end in .zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    if set(predictions) != {row["id"] for row in test_rows}:
        raise ValueError("Prediction keys differ from the test ID set")
    rows = [{"id": row["id"], "summary": predictions[row["id"]]} for row in test_rows]
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("prediction.json", json.dumps(rows, ensure_ascii=False, indent=2))
        read_submission(temporary, test_rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp",
                            delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_submission(model, architecture, inputs, rows, path, device, precision,
                        strategy, policy, fingerprint):
    """Resume only predictions tied to the same model, tokenizer, test data and decoder."""
    path = Path(path)
    partial = path.with_name(f"partial_{path.stem}_{fingerprint[:24]}.json")
    required = {row["id"] for row in rows}
    cached = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    predictions = {key: value for key, value in cached.items()
                   if key in required and isinstance(value, str) and value.strip()}
    for index, row in enumerate(rows, 1):
        key = row["id"]
        if key in predictions:
            continue
        prediction = summarize(model, architecture, inputs, row["document"],
                               device, precision, strategy, policy)
        if not prediction.strip():
            save_json(partial, predictions)
            raise ValueError(f"Empty generated summary for {key}")
        predictions[key] = prediction
        if index % 5 == 0:
            save_json(partial, predictions)
    save_json(partial, predictions)
    write_submission(path, rows, predictions)
