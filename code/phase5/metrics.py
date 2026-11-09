"""Multilabel scores and the archived development calibration policy."""

import numpy as np

GRID = np.arange(.05, .701, .025)


def probabilities(logits):
    return 1 / (1 + np.exp(-np.clip(np.asarray(logits, dtype=np.float32), -40, 40)))


def per_label_counts(y, predictions):
    y, predictions = np.asarray(y, dtype=bool), np.asarray(predictions, dtype=bool)
    if y.shape != predictions.shape or y.ndim != 2 or not y.size:
        raise ValueError("Expected matching, nonempty document-by-label matrices")
    return ((y & predictions).sum(axis=0), (~y & predictions).sum(axis=0),
            (y & ~predictions).sum(axis=0))


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def metrics(y, probs, thresholds=.30):
    pred = probs >= thresholds
    tp, fp, fn = per_label_counts(y, pred)
    return {"macro_f1": float(f1(tp, fp, fn).mean()),
            "micro_f1": float(f1(tp.sum(), fp.sum(), fn.sum())),
            "subset_accuracy": float((pred == y).all(axis=1).mean()),
            "mean_pred_labels": float(pred.sum(axis=1).mean())}


def calibrate(y, probs):
    global_scores = [metrics(y, probs, float(t))["macro_f1"] for t in GRID]
    best = float(GRID[np.argmax(global_scores)])
    thresholds = np.full(probs.shape[1], best, dtype=np.float32)
    for col in range(probs.shape[1]):
        labels = y[:, col:col + 1].astype(bool)
        positives = int(labels.sum())
        if min(positives, len(y) - positives) < 4:
            continue
        scores = []
        for candidate in GRID:
            counts = per_label_counts(labels, probs[:, col:col + 1] >= candidate)
            penalty = .015 * abs(candidate - best) * np.sqrt(6 / positives)
            scores.append(float(f1(*counts)[0] - penalty))
        optimum = float(GRID[np.argmax(scores)])
        weight = positives / (positives + 12)
        thresholds[col] = (1 - weight) * best + weight * optimum
    return thresholds, {"global_threshold": best, "global_macro_f1": max(global_scores),
                        "regularized_macro_f1": metrics(y, probs, thresholds)["macro_f1"]}
