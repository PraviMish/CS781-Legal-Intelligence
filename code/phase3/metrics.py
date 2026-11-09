"""Macro scoring over all 13 official roles, including absent development roles."""
from __future__ import annotations
from collections import Counter
import numpy as np


def class_metrics(true, predicted, num_classes=13):
    if len(true) != len(predicted):
        raise ValueError("True and predicted label sequences have different lengths")
    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)
    for true_id, pred_id in zip(true, predicted):
        if not (0 <= true_id < num_classes and 0 <= pred_id < num_classes):
            raise ValueError("Label ID outside official vocabulary")
        if true_id == pred_id:
            tp[true_id] += 1
        else:
            fp[pred_id] += 1
            fn[true_id] += 1
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / np.maximum(1, tp + fn)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    return {"macro_f1": float(f1.mean()),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "accuracy": float(tp.sum() / max(1, len(true))),
            "per_class_f1": f1.tolist(),
            "support": (tp + fn).tolist()}


def domain_metrics(rows, predicted):
    if len(rows) != len(predicted):
        raise ValueError("Prediction alignment mismatch")
    pooled = class_metrics([row["label_id"] for row in rows], predicted)
    result = {k: v for k, v in pooled.items() if not isinstance(v, list)}
    for domain in ("CL", "IT"):
        pairs = [(row["label_id"], y) for row, y in zip(rows, predicted) if row["domain"] == domain]
        if pairs:
            result[domain + "_macro_f1"] = class_metrics(
                [a for a, b in pairs], [b for a, b in pairs])["macro_f1"]
    return result
