"""Exact span voting and historical overlap resolution."""
from collections import defaultdict

STRATEGIES = ("union", "majority", "unanimous", "teacher_plus_agreement", "teacher_confirmed")
MEMBERS = ("gqa", "decoder", "student")


def accepts(support, strategy):
    if strategy == "union":
        return bool(support)
    if strategy == "majority":
        return len(support) >= 2
    if strategy == "unanimous":
        return len(support) == 3
    if strategy == "teacher_plus_agreement":
        return "gqa" in support or {"decoder", "student"} <= support
    if strategy == "teacher_confirmed":
        return "gqa" in support and len(support) >= 2
    raise ValueError(f"Unknown strategy {strategy}.")


def ensemble_documents(members, strategy="majority"):
    if members.keys() != set(MEMBERS):
        raise ValueError("Supply GQA, decoder, and distilled-student predictions.")
    keys = members["gqa"].keys()
    if any(member.keys() != keys for member in members.values()):
        raise ValueError("Ensemble members must predict the same documents.")

    result = {}
    for key in keys:
        votes = defaultdict(set)
        for model, predictions in members.items():
            for span in predictions[key]:
                votes[tuple(span)].add(model)
        candidates = [span for span, support in votes.items() if accepts(support, strategy)]
        candidates.sort(key=lambda span: (
            -len(votes[span]), -int("gqa" in votes[span]), -(span[1] - span[0]), span
        ))
        chosen = []
        for span in candidates:
            start, end, _ = span
            if not any(start < other[1] and other[0] < end for other in chosen):
                chosen.append(span)
        result[key] = [list(span) for span in sorted(chosen)]
    return result
