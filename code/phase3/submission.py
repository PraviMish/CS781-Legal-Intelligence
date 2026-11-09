"""One exact 13-role label per sentence and archive-root prediction.json."""
from __future__ import annotations
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, ZIP_DEFLATED


def validate_predictions(predictions, documents, labels):
    """documents is the normalized CL+IT test list."""
    expected = {}
    for doc in documents:
        if doc["id"] in expected:
            raise ValueError(f"Duplicate test ID {doc['id']}")
        expected[doc["id"]] = len(doc["text"])
    if not isinstance(predictions, dict) or set(predictions) != set(expected):
        raise ValueError("Submission must contain exactly all test document IDs")
    allowed = set(labels)
    for key, n in expected.items():
        values = predictions[key]
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"Wrong label count for {key}")
        if any(not isinstance(item, str) or item not in allowed for item in values):
            raise ValueError(f"Invalid labels for {key}")
    return sum(expected.values())


def write_submission(path, predictions, documents, labels):
    validate_predictions(predictions, documents, labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as z:
            z.writestr("prediction.json", json.dumps(predictions, ensure_ascii=False, indent=2))
        if read_submission(temporary) != predictions:
            raise ValueError("Submission ZIP round trip failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_submission(path):
    with ZipFile(path) as z:
        if z.namelist() != ["prediction.json"]:
            raise ValueError("Archive must contain only root-level prediction.json")
        return json.loads(z.read("prediction.json"))


def assemble(rows, outputs, labels):
    """Outputs keyed by (domain, doc_id, zero-based sentence index)."""
    predicted = {}
    for row in rows:
        key = (row["domain"], row["doc_id"], row["sentence_idx"])
        if key not in outputs:
            raise ValueError(f"Missing prediction {key}")
        document = predicted.setdefault(row["doc_id"], [])
        document.append((row["sentence_idx"], labels[int(outputs[key])]))
    return {doc: [label for index, label in sorted(items)] for doc, items in predicted.items()}


def hybrid(cl_predictions, it_predictions, documents):
    merged = {}
    for domain, predictions in (("CL", cl_predictions), ("IT", it_predictions)):
        for doc in documents[domain + "_test"]:
            merged[doc["id"]] = predictions[doc["id"]]
    return merged
