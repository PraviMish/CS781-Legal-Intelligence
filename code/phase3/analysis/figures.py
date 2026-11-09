"""Regenerate Phase 3 vector-PDF plots solely from archived numeric CSV evidence.

From repository root: python code/phase3/analysis/figures.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DATA = HERE / "experiments"
OUTPUT = ROOT / "report" / "figures" / "phase3"
OUTPUT.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / name, format="pdf", bbox_inches="tight")
    plt.close()


def run():
    official = pd.read_csv(DATA / "official_results.csv")
    local = pd.read_csv(DATA / "local_results.csv")
    observed = official[official["status"] == "Finished"].copy()

    fig, ax = plt.subplots(figsize=(8, 7))
    observed = observed.sort_values("official_macro_f1")
    ax.barh(observed["experiment"], observed["official_macro_f1"])
    ax.axvline(.4249, linestyle="--", linewidth=.8, alpha=.7)
    ax.set(xlabel="Official Codabench Macro-F1", title="Completed Phase-3 Codabench submissions")
    ax.grid(axis="x", alpha=.15)
    save("p3_official_scores.pdf")

    joined = local.merge(observed[["experiment", "official_macro_f1"]], on="experiment")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(joined["macro_f1"], joined["official_macro_f1"])
    for _, row in joined.iterrows():
        if row["experiment"].startswith(("C", "B4")):
            ax.annotate(row["experiment"].split("_")[0],
                        (row["macro_f1"], row["official_macro_f1"]), fontsize=8)
    ax.set(xlabel="Development Macro-F1", ylabel="Official Codabench Macro-F1",
           title="Local versus hidden evaluation (finished submissions)")
    ax.grid(alpha=.15)
    save("p3_local_vs_official.pdf")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    selected = local[local["experiment"].isin([
        "A5_gqa700_lora", "C1_previous_context", "C2_bidirectional_context", "B4_full", "F1_balanced_lora"
    ])].copy().sort_values("macro_f1")
    ax.barh(selected["experiment"], selected["macro_f1"])
    ax.set(xlabel="Development Macro-F1", title="Context and adaptation ablations")
    ax.grid(axis="x", alpha=.15)
    save("p3_context_ablation.pdf")

    classes = pd.read_csv(DATA / "best_per_class.csv", keep_default_na=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = classes.sort_values("f1")
    ax.barh(classes["label"], classes["f1"])
    ax.set(xlim=(0, 1), xlabel="Per-class development F1", title="C2 role-specific performance")
    ax.grid(axis="x", alpha=.15)
    save("p3_per_class_f1.pdf")

    labels = pd.read_csv(DATA / "label_counts.csv", keep_default_na=False).set_index("label")
    fig, ax = plt.subplots(figsize=(9, 6))
    labels[["CL_train", "IT_train"]].plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(xlabel="Training sentences", title="Official Phase-3 label counts by domain")
    ax.grid(axis="x", alpha=.15)
    save("p3_label_distribution.pdf")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    subset=local[local["experiment"].isin(["A5_gqa700_lora", "C1_previous_context", "C2_bidirectional_context"])]
    for _, row in subset.iterrows():
        ax.scatter(row["cl_f1"], row["it_f1"], s=65)
        ax.annotate(row["experiment"].split("_")[0], (row["cl_f1"], row["it_f1"]), xytext=(6,4), textcoords="offset points")
    ax.set(xlabel="CL development Macro-F1",ylabel="IT development Macro-F1",title="Contextual adaptation across domains")
    ax.grid(alpha=.15)
    save("p3_domain_comparison.pdf")

    # Reproducible schematic of the actual C2 inference path. This describes
    # implemented data flow, not an invented learned attention mechanism.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    nodes = [
        (.2, 1.2, 2.2, 1.2, "Prev + next + target\nBOS/EOS, max 128 BPE"),
        (2.85, 1.2, 2.15, 1.2, "Phase-1 700M GQA\n14 causal layers"),
        (5.45, 1.2, 1.95, 1.2, "LoRA adapters\nQ/K/V/O, r = 8"),
        (7.85, 1.2, 1.95, 1.2, "Final EOS hidden\n13-role linear head"),
    ]
    for x, y, w, h, caption in nodes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               linewidth=1.0, facecolor="none")
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, caption, ha="center", va="center", fontsize=10)
    for left,right in [(2.4,2.85),(5.0,5.45),(7.4,7.85)]:
        arrow = FancyArrowPatch((left+0.04,1.8),(right-0.04,1.8),
                                arrowstyle="->", mutation_scale=15, linewidth=1.2)
        ax.add_patch(arrow)
    ax.text(5.0, 2.95, "C2 winning Phase-3 inference (causal attention unchanged)",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.55, "Context is prepended in the token sequence; the target EOS can see both neighbors.",
            ha="center", fontsize=9)
    save("p3_c2_architecture.pdf")

    print("Generated seven vector PDF figures in", OUTPUT)


if __name__ == "__main__":
    run()
