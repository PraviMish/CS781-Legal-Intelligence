"""Case-level decontamination, grouped splitting and token windows."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def load_cases(path, *, labeled):
    """Read an official JSON file or a historical parquet split."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = next(
                (
                    rows[key]
                    for key in ("data", "examples", "records", "cases", "train", "dev", "test")
                    if isinstance(rows.get(key), list)
                ),
                None,
            )
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected nonempty case records in {path}")
    cases = []
    for row in rows:
        if row["id"] is None or row["text"] is None:
            raise ValueError("Case ID and text are required")
        case = {"id": str(row["id"]), "text": str(row["text"])}
        if labeled:
            if str(row["label"]) not in {"0", "1"}:
                raise ValueError("Expected binary CJPE labels")
            case["label"] = int(row["label"])
        cases.append(case)
    return cases


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def representatives(cases):
    """Pick the longest version per ID and retain its original position."""
    selected = {}
    for i, row in enumerate(cases):
        old = selected.get(row["id"])
        if old is None or len(row["text"]) > len(cases[old]["text"]):
            selected[row["id"]] = i
    return [cases[i] for i in sorted(selected.values())]


def make_splits(train, dev, test, seed=42):
    """Recreate notebook case selection and the balanced grouped holdout."""
    from sklearn.model_selection import StratifiedGroupKFold

    seen = {}
    for row in train:
        if seen.setdefault(row["id"], row["label"]) != row["label"]:
            raise ValueError(f"Conflicting labels for training case {row['id']}")
    test_ids = {row["id"] for row in test}
    test_hashes = {text_hash(row["text"]) for row in test}
    safe = [
        row
        for row in dev
        if row["id"] not in test_ids and text_hash(row["text"]) not in test_hashes
    ]
    train = representatives(train)
    y = np.array([row["label"] for row in train])
    groups = [text_hash(row["text"]) for row in train]
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)

    def quality(pair):
        val = pair[1]
        return abs(len(val) / len(y) - 0.10) + abs(y[val].mean() - y.mean())

    training, validation = min(splitter.split(train, y, groups), key=quality)
    return {
        "train_core": [train[i] for i in training],
        "val_internal": [train[i] for i in validation],
        "dev_safe": representatives(safe),
        "test": representatives(test),
    }


@dataclass(frozen=True)
class SpecialTokens:
    pad: int
    bos: int
    eos: int


def load_tokenizer(path):
    tokenizer = Tokenizer.from_file(str(path))
    special = SpecialTokens(
        *(tokenizer.token_to_id(token) for token in ("<pad>", "<bos>", "<eos>"))
    )
    if (
        tokenizer.get_vocab_size() != 32000
        or (special.pad, special.bos, special.eos) != (0, 1, 2)
        or tokenizer.token_to_id("<unk>") != 3
    ):
        raise ValueError("Expected the archived Phase 1 tokenizer with pad, BOS and EOS")
    return tokenizer, special


def windows(tokens, special, *, strategy="tail", width=1024):
    """Wrap the selected text in BOS/EOS, then right-pad."""
    content = width - 2
    if content <= 0:
        raise ValueError("A window must hold BOS, EOS and content")
    if strategy == "tail":
        chunks = [tokens[-content:]]
        slots = 1
    elif strategy == "head_tail":
        head = content // 4
        chosen = tokens if len(tokens) <= content else tokens[:head] + tokens[-(content - head) :]
        chunks = [chosen]
        slots = 1
    elif strategy == "hierarchical":
        slots = 5
        count = min(slots, max(1, math.ceil(len(tokens) / content)))
        starts = np.rint(np.linspace(0, max(0, len(tokens) - content), count)).astype(int)
        chunks = [tokens[start : start + content] for start in starts]
    else:
        raise ValueError(f"Unknown window strategy {strategy}")
    ids = np.full((slots, width), special.pad, dtype=np.uint16)
    lengths = np.zeros(slots, dtype=np.int16)
    for i, chunk in enumerate(chunks):
        sequence = [special.bos, *chunk, special.eos]
        lengths[i] = len(sequence)
        ids[i, : len(sequence)] = sequence
    return ids, lengths


@dataclass
class Split:
    ids: list
    labels: np.ndarray | None
    tokens: np.ndarray
    lengths: np.ndarray
    strategy: str = "tail"

    def batch(self, indices, device):
        import torch

        lengths = self.lengths[indices, 0].astype(np.int64)
        ids = self.tokens[indices, 0, : lengths.max()].astype(np.int64)
        return torch.from_numpy(ids).to(device), torch.from_numpy(lengths).to(device)


def tokenize_cases(cases, tokenizer, special, strategy="tail", *, width=1024):
    slots = 5 if strategy == "hierarchical" else 1
    tokens = np.empty((len(cases), slots, width), dtype=np.uint16)
    lengths = np.empty((len(cases), slots), dtype=np.int16)
    ids = [row["id"] for row in cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected a nonempty split with unique case IDs")
    for i, row in enumerate(cases):
        ids_raw = tokenizer.encode(row["text"], add_special_tokens=False).ids
        tokens[i], lengths[i] = windows(ids_raw, special, strategy=strategy, width=width)
    labels = np.array([row["label"] for row in cases]) if "label" in cases[0] else None
    return Split(ids, labels, tokens, lengths, strategy)


def save_split(path, split):
    """Write metadata last so interrupted preparation is detected."""
    path.mkdir(parents=True, exist_ok=False)
    np.save(path / "tokens.npy", split.tokens, allow_pickle=False)
    np.save(path / "lengths.npy", split.lengths, allow_pickle=False)
    metadata = {
        "ids": split.ids,
        "labels": None if split.labels is None else split.labels.tolist(),
        "strategy": split.strategy,
    }
    (path / "cases.json").write_text(json.dumps(metadata), encoding="utf-8")


def read_split(path):
    metadata = json.loads((path / "cases.json").read_text(encoding="utf-8"))
    labels = metadata["labels"]
    return Split(
        metadata["ids"],
        None if labels is None else np.array(labels, dtype=np.int64),
        np.load(path / "tokens.npy", mmap_mode="r", allow_pickle=False),
        np.load(path / "lengths.npy", mmap_mode="r", allow_pickle=False),
        metadata["strategy"],
    )
