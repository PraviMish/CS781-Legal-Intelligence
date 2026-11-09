"""Regenerate the Phase 4 figures and LaTeX results table from public aggregates."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
EXPERIMENTS = HERE.parents[1] / "experiments"
OUT = HERE.parents[3] / "report" / "figures" / "phase4"


def records(name):
    with (EXPERIMENTS / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(figure, name):
    figure.savefig(
        OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None}
    )
    figure.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def scores(results):
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
    x = np.arange(len(results))
    series = [
        ("internal_macro_f1", "Internal validation", "#648AA5", "o"),
        ("dev_macro_f1", "Clean development", "#C07125", "s"),
        ("codabench_macro_f1", "Official test", "#16715E", "D"),
    ]
    for field, label, color, marker in series:
        ax.scatter(
            x,
            [float(row[field]) for row in results],
            label=label,
            color=color,
            marker=marker,
            s=46,
            zorder=2,
        )
    ax.axvspan(-0.35, 0.35, color="#DDEDE6", alpha=0.8, linewidth=0)
    ax.set(
        xticks=x,
        xticklabels=[row["run"] for row in results],
        ylabel="Macro-F1",
        ylim=(0.57, 0.81),
        xlim=(-0.55, len(results) - 0.45),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=3, frameon=False)
    save(fig, "p4_split_scores")


def trajectories(results, history):
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 4.7), sharey=True, layout="constrained")
    for ax, run in zip(axes.flat, results[:8], strict=True):
        history_rows = [row for row in history if row["run"] == run["run"]]
        epochs = [int(row["epoch"]) for row in history_rows]
        values = [float(row["val_macro_f1_at_half"]) for row in history_rows]
        ax.plot(epochs, values, "o-", color="#306C8C", markersize=4)
        ax.axvline(int(run["selected_epoch"]), color="#16715E", ls="--", lw=1)
        ax.set(title=run["run"], xticks=epochs, ylim=(0.52, 0.80))
        ax.grid(axis="y", alpha=0.2)
    fig.supxlabel("Training epoch")
    fig.supylabel("Internal macro-F1 at threshold 0.50")
    save(fig, "p4_validation_history")


def errors(results):
    names = ("E01", "E02", "E05", "E08", "E09", "E10")
    rows = [row for row in results if row["run"] in names]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.1), layout="constrained")
    for field, offset, label, color in [
        ("dev_fp", -0.18, "Class 0 predicted as 1", "#306C8C"),
        ("dev_fn", 0.18, "Class 1 predicted as 0", "#C07125"),
    ]:
        bars = ax.bar(
            x + offset, [int(row[field]) for row in rows], width=0.34, label=label, color=color
        )
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set(
        xticks=x, xticklabels=[row["run"] for row in rows], ylabel="Clean dev cases", ylim=(0, 345)
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncols=2, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "p4_dev_errors")


def table(results):
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Run & Experiment & Threshold & Internal & Clean dev & Codabench \\",
        r"\midrule",
    ]
    for row in results:
        official = f"{float(row['codabench_macro_f1']):.4f}"
        if row["run"] == "E01":
            official = r"\textbf{" + official + "}"
        lines.append(
            f"{row['run']} & {row['method']} & {float(row['threshold']):.2f} & "
            f"{float(row['internal_macro_f1']):.4f} & {float(row['dev_macro_f1']):.4f} & "
            f"{official} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "p4_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = records("results.csv")
    scores(results)
    trajectories(results, records("training_history.csv"))
    errors(results)
    table(results)


if __name__ == "__main__":
    main()
