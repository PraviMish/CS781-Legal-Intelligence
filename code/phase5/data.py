"""Official case records, grouped holdout and Phase 1 BPE windows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

REVISION = "93b0059be756c6c461f3a225de25aabda7d5f84a"
LABELS = tuple(range(100))
WINDOWS = 3
WINDOW_SIZE = 512
PAD, BOS, EOS, UNK = range(4)


def read_cases(path, labeled):
    """Read the official JSON array (or JSON lines) in its original order."""
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        rows = ([json.loads(line) for line in source if line.strip()]
                if path.suffix == ".jsonl" else json.load(source))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty array of cases")
    cases = []
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("id")) not in (str, int)
                or not str(row["id"]) or not isinstance(row.get("text"), str)
                or not row["text"]):
            raise ValueError("A case must have a nonempty ID and text")
        case = {"id": str(row["id"]), "text": row["text"]}
        if labeled:
            labels = row.get("labels")
            if (not isinstance(labels, list) or not labels
                    or any(type(label) is not int or label not in LABELS for label in labels)):
                raise ValueError("Expected a nonempty list of official statute IDs")
            case["labels"] = sorted(set(labels))
        cases.append(case)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases


def check_disjoint_ids(*splits):
    ids = [case["id"] for split in splits for case in split]
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs overlap across dataset splits")


def statute_vocabulary(*labeled_splits):
    return tuple(sorted({label for split in labeled_splits for case in split
                         for label in case["labels"]}))


def targets(cases, labels=LABELS):
    columns = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(cases), len(labels)), dtype=np.float32)
    for i, case in enumerate(cases):
        out[i, [columns[label] for label in case["labels"]]] = 1
    return out


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def internal_split(cases, seed=42, core_ids=None):
    """Split on exact-text groups, preserving input order within each partition."""
    hashes = [text_hash(case["text"]) for case in cases]
    if core_ids is None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=.10, random_state=seed)
        core, _ = next(splitter.split(cases, groups=hashes))
        core_ids = {cases[i]["id"] for i in core}
    else:
        core_ids = set(core_ids)
        if not core_ids <= {case["id"] for case in cases}:
            raise ValueError("Archive contains unknown training IDs")
    mask = [case["id"] in core_ids for case in cases]
    core = [case for case, selected in zip(cases, mask) if selected]
    holdout = [case for case, selected in zip(cases, mask) if not selected]
    if (not core or not holdout or
            {h for h, selected in zip(hashes, mask) if selected}
            & {h for h, selected in zip(hashes, mask) if not selected}):
        raise ValueError("Invalid or overlapping grouped split")
    return core, holdout


def load_tokenizer(path):
    tokenizer = Tokenizer.from_file(str(path))
    if (tokenizer.get_vocab_size() != 32000 or
            tuple(tokenizer.token_to_id(token) for token in ("<pad>", "<bos>", "<eos>", "<unk>"))
            != (PAD, BOS, EOS, UNK)):
        raise ValueError("Expected the original Phase 1 32k BPE and special-token IDs")
    return tokenizer


def make_windows(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    budget = WINDOW_SIZE - 2
    starts = [0] if len(tokens) <= budget else np.linspace(
        0, len(tokens) - budget, WINDOWS).round().astype(int)
    windows = np.full((WINDOWS, WINDOW_SIZE), PAD, dtype=np.uint16)
    lengths = np.zeros(WINDOWS, dtype=np.int16)
    for row, start in enumerate(starts):
        packed = [BOS, *tokens[start:start + budget], EOS]
        windows[row, :len(packed)] = packed
        lengths[row] = len(packed)
    return windows, lengths, len(tokens)


class CaseDataset(Dataset):
    def __init__(self, windows, lengths, y=None, teacher=None):
        if (windows.ndim != 3 or windows.shape[1:] != (WINDOWS, WINDOW_SIZE)
                or lengths.shape != windows.shape[:2] or not len(windows)
                or np.any(lengths < 0) or np.any(lengths > WINDOW_SIZE)):
            raise ValueError("Invalid window or length cache")
        for values in (y, teacher):
            if values is not None and values.shape != (len(windows), len(LABELS)):
                raise ValueError("Invalid target or teacher matrix")
        self.windows, self.lengths, self.y, self.teacher = windows, lengths, y, teacher

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        row = {"inputs": torch.from_numpy(self.windows[index].astype(np.int64)),
               "lengths": torch.from_numpy(self.lengths[index].astype(np.int64))}
        for key, array in (("labels", self.y), ("teacher", self.teacher)):
            if array is not None:
                row[key] = torch.tensor(array[index], dtype=torch.float32)
        return row


def build_arrays(cases, tokenizer):
    windows = np.empty((len(cases), WINDOWS, WINDOW_SIZE), dtype=np.uint16)
    lengths = np.empty((len(cases), WINDOWS), dtype=np.int16)
    counts = np.empty(len(cases), dtype=np.int32)
    for i, case in enumerate(cases):
        windows[i], lengths[i], counts[i] = make_windows(case["text"], tokenizer)
    return windows, lengths, counts
