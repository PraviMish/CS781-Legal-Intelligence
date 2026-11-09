"""Binary Macro-F1 and notebook threshold searches."""

import numpy as np


def metrics(labels, probabilities, threshold=0.5):
    predicted = probabilities >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    f0 = 2 * tn / max(1, 2 * tn + fp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "accuracy": (tp + tn) / len(labels),
        "macro_f1": (f0 + f1) / 2,
        "f1_class_0": f0,
        "f1_class_1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def select_threshold(labels, probabilities, grid, *, tie_near_half=False):
    def key(threshold):
        score = metrics(labels, probabilities, threshold)["macro_f1"]
        return score, -abs(float(threshold) - 0.5) if tie_near_half else 0

    return float(max(grid, key=key))


def select_blend(labels, base, other):
    """Search in notebook order, retaining the first maximum."""
    best = None
    for weight in np.linspace(0.55, 0.95, 9):
        blended = blend(base, other, weight)
        threshold = select_threshold(labels, blended, np.linspace(0.25, 0.75, 51))
        score = metrics(labels, blended, threshold)["macro_f1"]
        if best is None or score > best[2]:
            best = (float(weight), threshold, score)
    return best


def blend(base, other, weight):
    # Notebook applies NumPy float64 grid scalars to float32 arrays.
    weight = np.float64(weight)
    return weight * base + (1 - weight) * other
