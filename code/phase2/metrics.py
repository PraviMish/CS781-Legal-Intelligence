"""Strict entity scoring with macro averaging over all twelve legal labels."""
from .data import LABELS


def strict_metrics(gold, predictions):
    counts = {label: [0, 0, 0] for label in LABELS}
    for doc_id in gold.keys() | predictions.keys():
        true = {tuple(span[:3]) for span in gold.get(doc_id, [])}
        pred = {tuple(span[:3]) for span in predictions.get(doc_id, [])}
        for index, spans in enumerate((true & pred, pred - true, true - pred)):
            for _, _, label in spans:
                counts[label][index] += 1
    rows = {}
    for label, (tp, fp, fn) in counts.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        rows[label] = dict(TP=tp, FP=fp, FN=fn, precision=precision, recall=recall,
                           f1=f1, support=tp + fn)
    tp, fp, fn = map(sum, zip(*counts.values()))
    summary = {
        f"macro_{key}": sum(row[key] for row in rows.values()) / len(rows)
        for key in ("precision", "recall", "f1")
    }
    summary.update(micro_f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                   TP=tp, FP=fp, FN=fn)
    return summary, rows
