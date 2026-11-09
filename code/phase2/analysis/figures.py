"""Regenerate Phase 2 report figures from public aggregate experiment records."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
FIGURES = Path(__file__).resolve().parents[3] / "report" / "figures" / "phase2"
BLUE = "#255775"
TEAL = "#187d70"
ORANGE = "#b95928"


def rows(filename):
    with (EXPERIMENTS / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def result_bars(ax, results, column, title, color):
    data = sorted((float(row[column]), row["display_name"]) for row in results if row[column])
    values, labels = zip(*data)
    ax.barh(labels, values, color=[TEAL if label.startswith(("majority", "Majority")) else color
                                  for label in labels])
    for i, value in enumerate(values):
        ax.text(value + 0.006, i, f"{value:.4f}", va="center", fontsize=8)
    ax.set(xlim=(0, 0.43), xlabel="Strict macro-F1", title=title)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)


def results_figure():
    records = rows("results.csv")
    records = [row for row in records if not row["variant"].startswith("ensemble_v1_")
               or row["variant"] == "ensemble_v1_majority"]
    figure, (validation, official) = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    result_bars(validation, records, "selected_validation_macro_f1",
                "Held-out validation (14 judgments)", BLUE)
    result_bars(official, records, "codabench_strict_macro_f1",
                "Codabench (35 hidden-label judgments)", ORANGE)
    save(figure, "p2_strict_results")


def adapter_figure():
    records = rows("training_history.csv")
    figure, (loss, f1) = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for run, name, color in [
        ("encoder_decoder_100m_lora_v1", "Encoder LoRA", BLUE),
        ("encoder_decoder_100m_lora_balanced_v2", "Encoder LoRA balanced", ORANGE),
        ("prefix_lm_100m_lorma_plus_v1", "Prefix-LM LoRMA+", TEAL),
    ]:
        history = [row for row in records if row["variant"] == run]
        epochs = [int(row["epoch"]) for row in history]
        loss.plot(epochs, [float(row["training_loss"]) for row in history], "o-", label=name, color=color)
        f1.plot(epochs, [float(row["validation_macro_f1"]) for row in history], "o-", color=color)
    loss.set(ylabel="Training cross-entropy", xticks=range(1, 7), xlabel="Epoch")
    f1.set(ylabel="Validation strict macro-F1", xticks=range(1, 7), xlabel="Epoch")
    loss.legend(frameon=False, fontsize=9)
    for axis in (loss, f1):
        axis.grid(alpha=0.2)
    save(figure, "p2_adapter_training")


def label_figure():
    records = rows("class_frequencies.csv")
    figure, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    labels = [row["Label"] for row in records]
    counts = [int(row["Count"]) for row in records]
    ax.bar(labels, counts, color=BLUE)
    for i, value in enumerate(counts):
        ax.text(i, value + 24, f"{value:,}", ha="center", fontsize=8)
    ax.set(ylim=(0, 1750), ylabel="Gold entity spans")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(figure, "p2_class_support")


if __name__ == "__main__":
    results_figure()
    adapter_figure()
    label_figure()
