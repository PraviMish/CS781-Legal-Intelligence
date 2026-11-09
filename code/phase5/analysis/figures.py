"""Render the Phase 5 report figures from released aggregate evidence."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
SOURCE = PHASE / "experiments"
DEST = REPO / "report" / "figures" / "phase5"
RUNS = [f"E{i:02}" for i in range(1, 7)]


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DEST / f"p5_{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    stats = json.loads((SOURCE / "dataset_statistics.json").read_text(encoding="utf-8"))
    audit = json.loads((SOURCE / "archive_verification.json").read_text(encoding="utf-8"))
    official = rows(SOURCE / "official_results.csv")
    frequencies = rows(SOURCE / "label_frequencies.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.65))
    x = np.arange(1, 101)
    axes[0].plot(x, [int(r["train_count"]) for r in frequencies], label="Training")
    axes[0].plot(x, [int(r["dev_count"]) for r in frequencies], label="Development")
    axes[0].set(xlabel="Statute rank by training support", ylabel="Cases", yscale="log")
    axes[0].legend(frameon=False)
    histogram = stats["dev"]["labels_per_case"]
    axes[1].bar([int(k) for k in histogram], [int(v) for v in histogram.values()], color="#32688c")
    axes[1].set(xlabel="Gold labels per development case", ylabel="Cases",
                yscale="log", ylim=(.7, 4000))
    save(fig, "labels")

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.65))
    groups = ["train_core", "val_internal", "dev", "test"]
    for group, label in zip(groups, ("Train core", "Internal val.", "Development", "Test")):
        values = stats[group]["tokens"]
        axes[0].plot([int(key) for key in values], list(values.values()), marker="o", label=label)
    axes[0].set(xlabel="BPE length percentile", ylabel="Tokens", yscale="log")
    axes[0].legend(frameon=False, fontsize=8)
    lengths = audit["e02_dev_length_groups"]
    x = np.arange(len(lengths))
    for offset, metric, label in ((-.18, "macro_f1", "Macro-F1"), (.18, "micro_f1", "Micro-F1")):
        axes[1].bar(x + offset, [row[metric] for row in lengths], width=.34, label=label)
    axes[1].set_xticks(x, [row["tokens"] for row in lengths])
    axes[1].set(xlabel="Development BPE tokens", ylabel="E02 score", ylim=(0, .68))
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "lengths")

    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    x = np.arange(6)
    replay = audit["dev_calibration_replay"]
    ax.bar(x - .26, [replay[k]["global_macro_f1"] for k in RUNS], width=.25, label="Dev global")
    ax.bar(x, [replay[k]["macro_f1"] for k in RUNS], width=.25, label="Dev per statute")
    ax.bar(x + .26, [float(row["official_macro_f1"]) for row in official], width=.25,
           label="Official test")
    ax.set_xticks(x, RUNS)
    ax.set(ylabel="Macro-F1", ylim=(0, .53))
    ax.legend(ncol=3, frameon=False, loc="upper center")
    save(fig, "results")

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.75))
    history = rows(SOURCE / "training_history.csv")
    for run in RUNS:
        group = [r for r in history if r["run"] == run]
        epochs = [int(r["epoch"]) for r in group]
        axes[0].plot(epochs, [float(r["macro_f1"]) for r in group], marker="o", label=run)
        axes[1].plot(epochs, [float(r["loss"]) for r in group], marker="o")
    for ax in axes:
        ax.set(xlabel="Epoch", xticks=[1, 2, 3, 4])
    axes[0].set_ylabel("Internal macro-F1 at 0.30")
    axes[0].legend(ncol=3, frameon=False, fontsize=8, loc="upper center")
    axes[0].set_ylim(.10, .43)
    axes[1].set(ylabel="Training objective", yscale="log")
    save(fig, "history")

    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    curves = rows(SOURCE / "threshold_curves.csv")
    for run in RUNS:
        group = [r for r in curves if r["run"] == run]
        ax.plot([float(r["threshold"]) for r in group], [float(r["macro_f1"]) for r in group],
                label=run)
    ax.set(xlabel="Global threshold", ylabel="Development macro-F1")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    save(fig, "thresholds")

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.7))
    errors = rows(SOURCE / "e02_per_statute.csv")
    axes[0].scatter([int(r["dev_support"]) for r in errors], [float(r["f1"]) for r in errors],
                    s=12, alpha=.75)
    axes[0].set(xscale="log", xlabel="Development support", ylabel="E02 per-statute F1")
    worst = sorted(errors, key=lambda r: float(r["f1"]))[:6][::-1]
    y = np.arange(len(worst))
    axes[1].barh(y - .16, [int(r["false_negatives"]) for r in worst], height=.31, label="Misses")
    axes[1].barh(y + .16, [int(r["false_positives"]) for r in worst], height=.31, label="False positives")
    axes[1].set_yticks(y, [r["statute_id"] for r in worst])
    axes[1].set(xlabel="Development cases", ylabel="Statute ID", xlim=(0, 500))
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "errors")


if __name__ == "__main__":
    main()
