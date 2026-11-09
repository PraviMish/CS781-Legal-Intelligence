"""Official Phase-3 dataset parsing and notebook-equivalent BPE/context processing.

Sentence context is assembled before the target sentence, including the following
sentence in the bidirectional condition. Its final EOS is therefore able to see
both neighbors under the pretrained GQA *causal* attention mask.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

OFFICIAL_LABELS = ("Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute",
                   "Dissent", "PrecedentReliedUpon", "PrecedentNotReliedUpon",
                   "PrecedentOverruled", "RulingByLowerCourt", "RatioOfTheDecision",
                   "RulingByPresentCourt", "None")
SPLITS = ("CL_train", "IT_train", "CL_dev", "IT_dev", "CL_test", "IT_test")
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
MAX_LEN = 128


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_official(root):
    """Return six normalized document lists and the official ordered label strings."""
    root = Path(root)
    mapping = read_json(root / "Train" / "label_vocab.json")
    if not isinstance(mapping, dict) or set(mapping) != set(OFFICIAL_LABELS):
        raise ValueError("Official vocabulary differs from the required 13 labels")
    if sorted(mapping.values()) != list(range(13)):
        raise ValueError("Expected 13 integer label IDs, 0 through 12")
    labels = [name for name, idx in sorted(mapping.items(), key=lambda kv: kv[1])]
    all_docs = {}
    for split in SPLITS:
        part = "Test" if split.endswith("test") else "Train"
        raw = read_json(root / part / f"{split}.json")
        if part == "Test":
            if not isinstance(raw, dict):
                raise ValueError(f"Test split {split} must map document IDs to sentence arrays")
            raw = [{"id": key, "text": value} for key, value in raw.items()]
        elif not isinstance(raw, list):
            raise ValueError(f"Labeled split {split} must be a list")
        seen, normalized = set(), []
        for entry in raw:
            doc_id, text = str(entry["id"]), entry["text"]
            if doc_id in seen or not isinstance(text, list) or not all(isinstance(x, str) for x in text):
                raise ValueError(f"Malformed or repeated document: {split}/{doc_id}")
            seen.add(doc_id)
            record = {"id": doc_id, "text": text}
            if part == "Train":
                ys = entry["labels"]
                if not isinstance(ys, list) or len(ys) != len(text) or any(y not in mapping for y in ys):
                    raise ValueError(f"Label sequence invalid: {split}/{doc_id}")
                record["labels"] = ys
            normalized.append(record)
        all_docs[split] = normalized
    return all_docs, labels


def load_tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size() != 32000:
        raise ValueError("The original 32k Phase-1 BPE file is required")
    for name, num in (("<pad>", 0), ("<bos>", 1), ("<eos>", 2), ("<unk>", 3)):
        if tok.token_to_id(name) != num:
            raise ValueError(f"Original Phase-1 token ID mismatch: {name}")
    return tok


def clip(ids, budget):
    ids = list(ids)
    if len(ids) <= budget:
        return ids
    if budget <= 0:
        return []
    front = max(1, round(.75 * budget))
    return ids[:front] + ids[-(budget-front):] if front < budget else ids[:front]


def sentence_tokens(ids, max_len=MAX_LEN):
    return [BOS_ID] + clip(ids, max_len - 2) + [EOS_ID]


def context_tokens(tokenized, index, context="sentence", max_len=MAX_LEN):
    """Faithfully reproduce notebook cells 18 and 31, including boundary behavior."""
    if context == "sentence":
        return sentence_tokens(tokenized[index], max_len)
    if context == "previous":
        neighbor_indexes = [index - 1]
    elif context == "bidirectional":
        neighbor_indexes = [index - 1, index + 1]
    else:
        raise ValueError(f"Unrecognized context condition {context}")
    neighbor_indexes = [j for j in neighbor_indexes if 0 <= j < len(tokenized)]
    if not neighbor_indexes:
        return sentence_tokens(tokenized[index], max_len)
    target_budget = min(len(tokenized[index]), int(.65 * (max_len - 2)))
    target = clip(tokenized[index], target_budget)
    context_budget = max_len - len(target) - 2 - len(neighbor_indexes)
    per_neighbor = context_budget // len(neighbor_indexes)
    seq = [BOS_ID]
    for j in neighbor_indexes:
        seq += clip(tokenized[j], per_neighbor)
        seq.append(EOS_ID)
    seq += target
    seq.append(EOS_ID)
    assert len(seq) <= max_len
    return seq


def make_rows(documents, tokenizer, label_to_id, context="sentence", max_len=MAX_LEN):
    """Create stable domain/doc/sentence keys, preserving original sentence order."""
    output = {}
    for split, docs in documents.items():
        domain, rows = split.split("_")[0], []
        for doc in docs:
            encodings = [tokenizer.encode(s, add_special_tokens=False).ids for s in doc["text"]]
            gold = doc.get("labels")
            for i in range(len(encodings)):
                rows.append({"domain": domain, "doc_id": doc["id"], "sentence_idx": i,
                             "input_ids": context_tokens(encodings, i, context, max_len),
                             "label_id": label_to_id[gold[i]] if gold is not None else -1})
        output[split] = rows
    return output


class RRSentences(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.lengths = [len(row["input_ids"]) for row in rows]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


class LengthBucketSampler(Sampler):
    def __init__(self, dataset, batch_size=8, shuffle=True, seed=42):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.batch_size * 64):
            bucket = indices[start:start+self.batch_size*64]
            bucket.sort(key=lambda i: self.dataset.lengths[i])
            for i in range(0, len(bucket), self.batch_size):
                batches.append(bucket[i:i+self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


def collate(rows):
    lengths = torch.tensor([len(r["input_ids"]) for r in rows], dtype=torch.long)
    ids = torch.full((len(rows), int(lengths.max())), PAD_ID, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row["input_ids"])] = torch.tensor(row["input_ids"], dtype=torch.long)
    out = {"input_ids": ids, "lengths": lengths,
           "labels": torch.tensor([r["label_id"] for r in rows], dtype=torch.long),
           "domains": [r["domain"] for r in rows], "doc_ids": [r["doc_id"] for r in rows],
           "sentence_indices": [r["sentence_idx"] for r in rows]}
    if rows and "teacher_logits" in rows[0]:
        if not all("teacher_logits" in row for row in rows):
            raise ValueError("Teacher logit missing")
        out["teacher_logits"] = torch.tensor([r["teacher_logits"] for r in rows], dtype=torch.float32)
    return out


def make_loader(rows, batch_size=8, shuffle=False, seed=42, epoch=0):
    ds = RRSentences(rows)
    sampler = LengthBucketSampler(ds, batch_size, shuffle, seed)
    sampler.set_epoch(epoch)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=False)


def joined(rows, suffix):
    return rows["CL_" + suffix] + rows["IT_" + suffix]
