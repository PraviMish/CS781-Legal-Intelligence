"""Document-level label-transition smoothing used by experiment C3.

Not a recurrent neural document encoder: estimates Markov transition priors
from training labels and decodes *whole document* emission sequences via Viterbi.
"""
from __future__ import annotations
import numpy as np


def transition_logprobs(documents, label_to_id, smoothing=.5):
    n = len(label_to_id)
    transitions = np.full((n, n), smoothing, dtype=float)
    starts = np.full(n, smoothing, dtype=float)
    for doc in documents:
        indices = [label_to_id[label] for label in doc["labels"]]
        if not indices:
            continue
        starts[indices[0]] += 1
        for a, b in zip(indices[:-1], indices[1:]):
            transitions[a, b] += 1
    return np.log(starts / starts.sum()), np.log(transitions / transitions.sum(1, keepdims=True))


def viterbi(emissions, start_log, transition_log, strength):
    emissions = np.asarray(emissions, dtype=np.float64)
    if emissions.ndim != 2:
        raise ValueError("Emissions must be [sentences, classes]")
    if not len(emissions):
        return []
    if strength == 0:
        return emissions.argmax(1).tolist()
    dp = emissions[0] + strength * start_log
    back = []
    for i in range(1, len(emissions)):
        candidate = dp[:, None] + strength * transition_log
        prev = candidate.argmax(0)
        dp = emissions[i] + candidate.max(0)
        back.append(prev)
    result = [int(dp.argmax())]
    for prev in reversed(back):
        result.append(int(prev[result[-1]]))
    return result[::-1]
