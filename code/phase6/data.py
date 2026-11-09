"""Official Phase 6 record loading, grouped split and revised passage inputs."""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PAD, BOS, EOS, UNK = range(4)
CONTEXT = 1280
TARGET_CAP = 320
HEADER = (
    "Summarize the legal passage. State only facts, legal reasoning and the decision "
    "supported by this passage.\nPassage:\n"
)
FOOTER = "\nSummary:\n"
LEGAL_TERMS = (
    "held", "holding", "appeal", "decision", "judgment", "ordered", "dismissed", "allowed",
    "trial court", "high court", "supreme court", "section", "article", "liable",
    "evidence", "finding", "conclusion", "accordingly",
)
STOP = set(
    "the a an and or but is was were of to in for with by at from this that as it its be "
    "been on such which their these there before after court high supreme case held "
    "appeal petition act section respondent appellant".split()
)


def load_records(path, labeled=False):
    """Read the official list format and the legacy notebook's JSON alternatives."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        container = next(
            (data[key] for key in ("data", "examples", "records", "items")
             if isinstance(data.get(key), list)), None
        )
        data = container if container is not None else [
            dict(value, id=key) if isinstance(value, dict) else {"id": key, "document": value}
            for key, value in data.items()
        ]
    if not isinstance(data, list):
        raise ValueError("Expected a collection of Phase 6 records")
    records = []
    seen = set()
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} is not an object")
        identifier = row.get("id")
        document = row.get("document")
        if document is None:
            document = row.get("text")
        reference = row.get("summary")
        if identifier is None or not isinstance(document, str):
            raise ValueError(f"Invalid ID or document at index {index}")
        if labeled and not isinstance(reference, str):
            raise ValueError(f"Missing summary for record {index}")
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError(f"Duplicate document ID: {identifier}")
        seen.add(identifier)
        records.append({"id": identifier, "document": document.strip(),
                        "summary": reference.strip() if isinstance(reference, str) else None})
    return records


def doc_hash(row):
    return hashlib.sha256(" ".join(row["document"].split()).encode()).hexdigest()


def split_records(rows, validation_ids=None, *, seed=42):
    """Preserve input row order and keep normalized exact duplicates together."""
    groups = {}
    for row in rows:
        groups.setdefault(doc_hash(row), []).append(row["id"])
    if validation_ids is None:
        hashes = sorted(groups)
        np.random.default_rng(seed).shuffle(hashes)
        count = min(max(1, round(len(hashes) * .08)), len(hashes) - 1)
        validation_ids = {identifier for key in hashes[:count] for identifier in groups[key]}
    else:
        validation_ids = {str(identifier) for identifier in validation_ids}
    if not validation_ids <= {row["id"] for row in rows}:
        raise ValueError("Unknown validation IDs")
    train = [row for row in rows if row["id"] not in validation_ids]
    val = [row for row in rows if row["id"] in validation_ids]
    if not train or not val or {doc_hash(row) for row in train} & {doc_hash(row) for row in val}:
        raise ValueError("Empty split or duplicate text across splits")
    return train, val


def load_tokenizer(path):
    tok = Tokenizer.from_file(str(path))
    ids = tuple(tok.token_to_id(name) for name in ("<pad>", "<bos>", "<eos>", "<unk>"))
    if ids != (PAD, BOS, EOS, UNK) or tok.get_vocab_size() != 32000:
        raise ValueError("Expected the original 32k Phase 1 tokenizer and special IDs 0–3")
    return tok


def words(text):
    return {word for word in re.findall(r"[a-z]+|\d+", text.lower())
            if len(word) > 2 and word not in STOP}


def sentences(text):
    normalized = re.sub(r"\s+", " ", text.strip())
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", normalized)
            if part.strip()]


class PassageInputs:
    """The V2/V3 reference-free passage selection and position-labeled prompts."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.header = self.encode(HEADER)
        self.footer = self.encode(FOOTER)
        overhead = len(self.header) + len(self.footer) + len(self.encode("Passage 4 of 4.\n")) + 3
        self.budget = min(840, CONTEXT - TARGET_CAP - overhead, 1024 - overhead)
        if self.budget < 500:
            raise ValueError("Tokenizer leaves too little space for source text")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def passages(self, document, strategy="hierarchical"):
        ids = self.encode(document) or [UNK]
        chunks = [ids[i:i + self.budget] for i in range(0, len(ids), self.budget)]
        if strategy == "head":
            return chunks[:1]
        if strategy != "hierarchical":
            raise ValueError(f"Unknown passage strategy {strategy}")
        chosen = {0, len(chunks) - 1}
        ranked = []
        for index, chunk in enumerate(chunks):
            text = self.tokenizer.decode(chunk).lower()
            score = sum(text.count(term) for term in LEGAL_TERMS)
            if re.search(r"\b(held|therefore|accordingly|we conclude)\b", text):
                score += 2.5
            ranked.append((-score, index))
        for _, index in sorted(ranked):
            if len(chosen) == 4:
                break
            chosen.add(index)
        return [chunks[index] for index in sorted(chosen)]

    def prompt(self, tokens, index, count):
        position = self.encode(f"Passage {index + 1} of {count}.\n")
        source = [BOS] + self.header + position + tokens + self.footer
        if len(source) + TARGET_CAP > CONTEXT or len(source) > 1024:
            raise ValueError("Passage prompt exceeds model capacity")
        return source

    def prompts(self, document, strategy="hierarchical"):
        passages = self.passages(document, strategy)
        return [self.prompt(chunk, i, len(passages)) for i, chunk in enumerate(passages)]


def alignment(tokenizer, passages, reference):
    """Assign each sufficiently overlapping reference sentence to one passage."""
    bags = [words(tokenizer.decode(chunk)) for chunk in passages]
    assigned = [[] for _ in passages]
    for sentence in sentences(reference):
        bag = words(sentence)
        scores = [len(bag & source) / max(1, math.sqrt(len(bag) * len(source))) for source in bags]
        best = int(np.argmax(scores))
        if scores[best] >= .05:
            assigned[best].append((scores[best], sentence))
    return assigned
