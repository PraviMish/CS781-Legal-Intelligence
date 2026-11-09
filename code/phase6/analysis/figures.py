"""Regenerate Phase 6 vector figures from published aggregate records."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / "code" / "phase6" / "experiments"
FIGURES = ROOT / "report" / "figures" / "phase6"
COLORS = ("#63758C", "#177487", "#A94830")


def read(name):
    with (RECORDS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, filename):
    fig.savefig(FIGURES / f"{filename}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def results():
    rows = read("results.csv")
    labels = [f"E0{number}" for number in range(1, 8)]
    y = np.arange(7)
    fig, (local, official) = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    for color, version, offset in zip(COLORS, ("original", "v2", "v3"), (-.24, 0, .24)):
        selected = [row for row in rows if row["version"] == version]
        positions = [labels.index(row["experiment"][:3]) + offset for row in selected]
        local.barh(positions, [float(row["val_rouge1"]) for row in selected], .22,
                   color=color, label=version.capitalize())
    local.set_yticks(y, labels)
    local.invert_yaxis()
    local.set(xlabel="Held-out ROUGE-1 F1", xlim=(0, .49))
    local.legend(loc="lower right", fontsize=8)
    selected = [row for row in rows if row["version"] == "v3"]
    values = [float(row["official_bertscore_f1"]) for row in selected]
    official.scatter(values, y, c=[COLORS[2] if lab == "E05" else COLORS[1] for lab in labels], s=34)
    for i, value in enumerate(values):
        official.text(value + .0005, i, f"{value:.4f}", va="center", fontsize=8)
    official.set_yticks(y, labels)
    official.invert_yaxis()
    official.set(xlabel="Official BERTScore F1", xlim=(.818, .854))
    for ax in (local, official):
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
    save(fig, "p6_results")


def history():
    rows = read("v3_e05_history.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.5), layout="constrained")
    ax.plot([int(row["step"]) for row in rows], [float(row["val_nll"]) for row in rows],
            color=COLORS[1], marker="o", markersize=3)
    ax.axvline(7750, color=COLORS[2], linestyle="--", label="Selected update 7,750")
    ax.set(xlabel="Optimizer updates", ylabel="Held-out target NLL")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "p6_e05_history")


def pipeline():
    fig, ax = plt.subplots(figsize=(9, 2.15))
    ax.axis("off")
    boxes = [
        ("Official\njudgment", .08),
        ("First, last and\nsalient passages", .29),
        ("Efficient GQA\n+ rank-16 LoRA", .51),
        ("Cached greedy\nper passage", .72),
        ("Ordered merge\nand deduplication", .91),
    ]
    for label, x in boxes:
        ax.text(x, .55, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=.45", "fc": "#EBF3F5", "ec": COLORS[1]})
    for start, end in ((.143, .187), (.386, .416), (.601, .63), (.801, .821)):
        ax.annotate("", xy=(end, .55), xytext=(start, .55),
                    arrowprops={"arrowstyle": "->", "color": COLORS[0]})
    ax.text(.5, .18, "Up to four selected passages, processed separately in document order",
            ha="center", fontsize=9, color="#344055")
    save(fig, "p6_e05_pipeline")


def dataset():
    path = RECORDS / "dataset_statistics.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.8, 3.6), layout="constrained")
    quantiles = ["p50", "p90", "p95", "p99"]
    for color, split in zip(COLORS, ("train", "validation", "test")):
        ax.plot([50, 90, 95, 99],
                [records[split]["document_bpe"][p] for p in quantiles],
                color=color, marker="o", label=split.capitalize())
    ax.axhline(1280, linestyle="--", color="#283646", label="1,280-token context")
    ax.set(xlabel="Document-length percentile", ylabel="BPE tokens per document", yscale="log")
    ax.set_xticks([50, 90, 95, 99])
    ax.grid(alpha=.23)
    ax.legend()
    save(fig, "p6_data_lengths")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    results()
    history()
    pipeline()
    dataset()
