"""Prepare exact character offsets and disjoint BIO supervision windows."""
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

LABELS = tuple("APP RESP A.COUNSEL R.COUNSEL JUDGE WIT AUTH COURT STAT PREC DATE CASENO".split())
BIO = ("O",) + tuple(f"{prefix}-{label}" for prefix in "BI" for label in LABELS)
TAG_ID = {tag: i for i, tag in enumerate(BIO)}
IGNORE = -100
OWNED = 256
CONTEXT = 128
WINDOW = OWNED + 2 * CONTEXT


def read_json(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_texts(path):
    texts = read_json(path)
    if not isinstance(texts, dict) or any(not isinstance(text, str) for text in texts.values()):
        raise ValueError("Expected an official mapping of document IDs to unmodified text.")
    return texts


def document_key(fold, doc_id):
    return f"{fold}:{doc_id}"


def tokenizer_from(path):
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size() != 32000 or tokenizer.token_to_id("<pad>") is None:
        raise ValueError("Use the archived Phase 1 legal BPE tokenizer.")
    return tokenizer


def trim_offsets(text, offsets):
    result = []
    for start, end in offsets:
        while start < end and text[start].isspace():
            start += 1
        while start < end and text[end - 1].isspace():
            end -= 1
        result.append((start, end))
    return result


def _first(record, keys):
    return next((record[key] for key in keys.split() if record.get(key) is not None), None)


def _annotation(record):
    if isinstance(record, dict):
        return (
            _first(record, "start start_char start_character start_offset start_idx begin"),
            _first(record, "end end_char end_character end_offset end_idx stop"),
            _first(record, "label entity_type type tag"),
            _first(record, "entity_text text value"),
        )
    if isinstance(record, (list, tuple)) and len(record) >= 3:
        return (*record[:3], record[3] if len(record) >= 4 else None)
    raise ValueError("Unsupported annotation record.")


def normalize_gold(texts, gold):
    """Validate the official end-exclusive convention against supplied mentions."""
    if gold.keys() - texts.keys():
        raise ValueError("Gold contains unknown document IDs.")
    normalized = {}
    for doc_id, text in texts.items():
        records = gold.get(doc_id, [])
        if isinstance(records, dict):
            records = _first(records, "entities annotations spans")
        spans = []
        for record in records or []:
            start, end, label, mention = _annotation(record)
            start, end = int(start), int(end)
            if label not in LABELS or not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid annotation bounds or label in document {doc_id}.")
            if mention is not None and text[start:end] != mention:
                raise ValueError(f"Gold mention disagrees with end-exclusive offsets in {doc_id}.")
            spans.append([start, end, label])
        normalized[doc_id] = spans
    return normalized


def spans_to_bio(offsets, spans):
    tags = [0 if start < end else IGNORE for start, end in offsets]
    # Duplicate offsets can occur in byte-level tokenization. Keep the historical last match.
    starts = {start: i for i, (start, end) in enumerate(offsets) if start < end}
    ends = {end: i for i, (start, end) in enumerate(offsets) if start < end}
    represented = 0
    for start, end, label, *_ in spans:
        first, last = starts.get(start), ends.get(end)
        active = []
        if first is not None and last is not None:
            active = [i for i in range(first, last + 1) if offsets[i][0] < offsets[i][1]]
        if active and all(start <= offsets[i][0] < offsets[i][1] <= end for i in active):
            tags[active[0]] = TAG_ID[f"B-{label}"]
            for i in active[1:]:
                tags[i] = TAG_ID[f"I-{label}"]
            represented += 1
        else:
            for i, (left, right) in enumerate(offsets):
                if left < right and left < end and right > start:
                    tags[i] = IGNORE
    return tags, represented


def decode_bio(tags, offsets):
    if len(tags) != len(offsets):
        raise ValueError("Expected one prediction per token.")
    spans = []
    active = None
    for tag_id, (start, end) in zip(tags, offsets):
        if start >= end:
            continue
        tag = BIO[tag_id]
        if tag == "O":
            active = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or active is None or active[2] != label:
            active = [start, end, label]
            spans.append(active)
        else:
            active[1] = end
    return spans


def tokenize(tokenizer, texts, gold=None):
    docs = {}
    for key, text in texts.items():
        encoding = tokenizer.encode(text, add_special_tokens=False)
        offsets = trim_offsets(text, encoding.offsets)
        item = {"ids": encoding.ids, "offsets": offsets}
        if gold is not None:
            item["labels"], _ = spans_to_bio(offsets, gold[key])
        docs[key] = item
    return docs


def validation_split(keys, gold, seed):
    """Select seven documents per fold with the notebook's 3,000-candidate search."""
    counts = [Counter(span[2] for span in gold[key]) for key in keys]
    matrix = np.array([[count[label] for label in LABELS] for count in counts])
    total = matrix.sum(axis=0)
    folds = [[i for i, key in enumerate(keys) if key.startswith(f"{fold}:")]
             for fold in ("fold1", "fold2")]
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    for _ in range(3000):
        indices = [i for fold in folds for i in rng.choice(fold, size=7, replace=False)]
        validation = matrix[indices].sum(axis=0)
        score = np.abs(validation / np.maximum(total, 1) - 0.2).mean()
        score += np.sum(validation == 0) + np.sum(total - validation == 0)
        if score < best_score:
            best_score, selected = score, set(indices)
    return {
        name: [key for i, key in enumerate(keys) if (i in selected) == is_validation]
        for name, is_validation in (("train", False), ("val", True))
    }


def prepare_training(root, tokenizer, *, full_data, seed, split_path=None):
    """Read official Train files and construct train/validation keys in original order."""
    root = Path(root)
    labels = tuple(root.joinpath("labels.txt").read_text(encoding="utf-8").split())
    if labels != LABELS:
        raise ValueError("Official label order differs from the archived BIO head order.")
    texts, gold = {}, {}
    for fold in ("fold1", "fold2"):
        fold_text = load_texts(root / f"{fold}-text.json")
        fold_gold = normalize_gold(fold_text, read_json(root / f"{fold}-gold.json"))
        texts.update((document_key(fold, key), value) for key, value in fold_text.items())
        gold.update((document_key(fold, key), spans) for key, spans in fold_gold.items())
    if len(texts) != 70:
        raise ValueError("Expected 70 labeled judgments.")
    if full_data:
        split = {"train": list(texts), "val": []}
    elif split_path:
        archived = read_json(split_path)
        split = {name: [document_key(*pair) for pair in archived[name]]
                 for name in ("train", "val")}
        train, val = split["train"], split["val"]
        if len(train) != 56 or len(val) != 14 or set(train + val) != texts.keys():
            raise ValueError("Archived split must partition all 70 documents.")
    else:
        split = validation_split(list(texts), gold, seed)
    return tokenize(tokenizer, texts, gold), gold, split


def build_windows(docs, keys):
    windows = []
    for key in keys:
        doc = docs[key]
        ids = doc["ids"]
        for start in range(0, len(ids), OWNED):
            end = min(start + OWNED, len(ids))
            left, right = max(0, start - CONTEXT), min(len(ids), end + CONTEXT)
            window = {
                "doc_key": key, "input_ids": ids[left:right], "global_start": left,
                "owned_start": start - left, "owned_end": end - left,
            }
            if "labels" in doc:
                window["labels"] = ([IGNORE] * (start - left) + doc["labels"][start:end]
                                    + [IGNORE] * (right - end))
            windows.append(window)
    return windows
