"""Original section packing and V2/V3 passage target construction."""

import re

from .data import BOS, CONTEXT, EOS, TARGET_CAP, PassageInputs, alignment, sentences

ORIGINAL_TERMS = (
    "held", "ordered", "judgment", "appeal", "petition", "conviction", "acquitted",
    "dismissed", "allowed", "court finds", "court held", "therefore", "accordingly",
    "conclusion", "final order", "sentence", "liable", "constitutional", "section",
    "article", "evidence", "respondent", "appellant", "issue", "reasoning",
)
ORIGINAL_HEADER = (
    "Summarize the following legal document accurately and concisely.\n"
    "Preserve the important facts, reasoning, and decision.\n\nDocument:\n"
)


class SectionInputs:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.header = tokenizer.encode(ORIGINAL_HEADER).ids
        self.footer = tokenizer.encode("\n\nSummary:\n").ids
        overhead = len(self.header) + len(self.footer)
        self.budget = min(900, CONTEXT - 240 - overhead - 4, 1024 - overhead - 2)

    def prompts(self, document, strategy="section"):
        ids = self.tokenizer.encode(document).ids
        if len(ids) > self.budget and strategy == "section":
            chunks = [ids[i:i + 150] for i in range(0, len(ids), 150)]
            selected = {0, 1, len(chunks) - 2, len(chunks) - 1}

            def importance(index):
                text = self.tokenizer.decode(chunks[index]).lower()
                return sum(3 * text.count(term) for term in ORIGINAL_TERMS) + 2 * index / (len(chunks) - 1)

            for index in sorted(range(len(chunks)), key=importance, reverse=True):
                if len(selected) * 150 >= self.budget:
                    break
                selected.add(index)
            ids = [token for index in sorted(selected) for token in chunks[index]]
        elif strategy not in ("section", "head"):
            raise ValueError(f"Unknown section strategy {strategy}")
        return [[BOS] + self.header + ids[:self.budget] + self.footer]


def v2_target(encode, matches, reference_sentences):
    if sum(len(encode(sentence)) for _, sentence in matches) >= TARGET_CAP:
        matches = sorted(sorted(matches, key=lambda pair: -pair[0])[:8],
                         key=lambda pair: reference_sentences.index(pair[1]))
    kept = []
    used = 0
    for _, sentence in matches:
        size = len(encode(sentence))
        if size and size + used < TARGET_CAP:
            kept.append(sentence)
            used += size
    while kept:
        ids = encode(" ".join(kept))
        if len(ids) < TARGET_CAP:
            return [ids + [EOS]]
        kept.pop()
    return []


def fragments(encode, sentence):
    if len(encode(sentence)) < TARGET_CAP:
        return [sentence]
    chunks = []
    current = []
    for part in re.split(r"(?<=[;:,])\s+", sentence):
        if len(encode(" ".join(current + [part]))) < TARGET_CAP:
            current.append(part)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [part] if len(encode(part)) < TARGET_CAP else []
    if current:
        chunks.append(" ".join(current))
    return chunks


def v3_targets(encode, matches):
    targets = []
    current = []
    for _, sentence in matches:
        for part in fragments(encode, sentence):
            if len(encode(" ".join(current + [part]))) >= TARGET_CAP:
                if current:
                    targets.append(encode(" ".join(current)) + [EOS])
                current = [part]
            else:
                current.append(part)
    if current:
        targets.append(encode(" ".join(current)) + [EOS])
    return targets


def training_group(row, inputs, version):
    identifier = row["id"]
    reference = row["summary"]
    if version == "original":
        return [{"id": identifier, "key": identifier, "src": inputs.prompts(row["document"])[0],
                 "tgt": inputs.tokenizer.encode(reference).ids[:239] + [EOS]}]
    passages = inputs.passages(row["document"])
    matches = alignment(inputs.tokenizer, passages, reference)
    reference_sentences = sentences(reference) if version == "v2" else []
    examples = []
    for index, (passage, assigned) in enumerate(zip(passages, matches)):
        targets = (v2_target(inputs.encode, assigned, reference_sentences) if version == "v2"
                   else v3_targets(inputs.encode, assigned))
        for pack, target in enumerate(targets):
            key = f"{identifier}::{index}"
            if version == "v3":
                key += f"::{pack}"
            examples.append({"id": identifier, "key": key, "passage_index": index,
                             "src": inputs.prompt(passage, index, len(passages)), "tgt": target})
    return examples


def input_builder(tokenizer, version):
    return SectionInputs(tokenizer) if version == "original" else PassageInputs(tokenizer)
