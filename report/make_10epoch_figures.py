"""Figures for the 10-epoch ablation report.

Reads the Colab export (ablation_results (2)/runs.csv + per_class.csv) and the
converged local runs (results/runs.csv) and writes PNGs into report/figs/.

Usage:
    python report/make_10epoch_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLAB = ROOT / "ablation_results (2)"
LOCAL = ROOT / "results"
FIGS = ROOT / "report" / "figs"

CLASS_NAMES = [
    "Null", "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2",
    "Open Fridge", "Close Fridge", "Open Dishwasher", "Close Dishwasher",
    "Open Drawer 1", "Close Drawer 1", "Open Drawer 2", "Close Drawer 2",
    "Open Drawer 3", "Close Drawer 3", "Clean Table", "Drink from Cup",
    "Toggle Switch",
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def bars(ax, labels, means, stds, colors, title, ylabel="Score"):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    colab = pd.read_csv(COLAB / "runs.csv")
    local = pd.read_csv(LOCAL / "runs.csv")
    pc = pd.read_csv(COLAB / "per_class.csv")

    order = ["A_cnn", "A_gru"]
    sub = colab[colab["variant"].isin(order)].set_index("variant").loc[order]
    # single seed -> no error bars
    for base in [c for c in sub.columns if c.startswith("test_") and not c.endswith("_std")]:
        if f"{base}_std" not in sub.columns:
            sub[f"{base}_std"] = 0.0

    # --- Part A: recurrence vs dense ------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    labels = ["Dense CNN\n(2x128, 7.49M)", "GRU\n(128, 2.99M)"]
    bars(
        ax,
        labels,
        sub["test_weighted_f1"].values,
        sub["test_weighted_f1_std"].values,
        ["#4C78A8", "#F58518"],
        "Part A (10 epochs): recurrence vs dense layers",
    )
    ax.text(
        0.5, 0.5, "",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig_a_10ep.png")
    plt.close(fig)

    # --- Part A: both metrics --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(order))
    w = 0.38
    for i, (col, label, color) in enumerate(
        [
            ("test_weighted_f1", "Weighted F1 (Null included)", "#4C78A8"),
            ("test_macro_f1", "Macro F1 (18 classes)", "#F58518"),
        ]
    ):
        vals = sub[col].values
        errs = sub[f"{col}_std"].values
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, label=label, color=color)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["A_cnn", "A_gru"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Part A (10 epochs): weighted vs macro F1")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_a_metrics_10ep.png")
    plt.close(fig)

    # --- Part B: conv depth ----------------------------------------------
    order_b = ["B_conv1", "B_conv2"]
    subb = colab[colab["variant"].isin(order_b)].set_index("variant").loc[order_b]
    for base in [c for c in subb.columns if c.startswith("test_") and not c.endswith("_std")]:
        if f"{base}_std" not in subb.columns:
            subb[f"{base}_std"] = 0.0
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(order_b))
    for i, (col, label, color) in enumerate(
        [
            ("test_weighted_f1", "Weighted F1", "#4C78A8"),
            ("test_macro_f1", "Macro F1", "#F58518"),
        ]
    ):
        vals = subb[col].values
        errs = subb[f"{col}_std"].values
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, label=label, color=color)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["B_conv1\n(1 conv)", "B_conv2\n(2 conv)"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Part B (10 epochs): convolutional depth")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_b_10ep.png")
    plt.close(fig)

    # --- budget effect: 10 epochs vs converged ---------------------------
    pairs = [("A_cnn",), ("A_gru",), ("B_conv1",), ("B_conv2",)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    labels, m10, m80, s10 = [], [], [], []
    for (v,) in pairs:
        if v not in set(local["variant"]):
            continue
        labels.append(v)
        m10.append(float(colab[colab["variant"] == v]["test_macro_f1"].iloc[0]))
        m80.append(float(local[local["variant"] == v]["test_macro_f1"].iloc[0]))
        s10.append(float(colab[colab["variant"] == v]["test_weighted_f1"].iloc[0]))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, m10, 0.38, label="Macro F1 @ 10 epochs", color="#F58518")
    ax.bar(x + 0.2, m80, 0.38, label="Macro F1 @ 80 epochs", color="#54A24B")
    for xi, v in zip(x - 0.2, m10):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, m80):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Effect of the training budget on the macro score")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_budget_effect.png")
    plt.close(fig)

    # --- per-class F1 ----------------------------------------------------
    piv = pc.pivot_table(index="class_name", columns="variant", values="f1")
    piv = piv.reindex(CLASS_NAMES)
    order_cols = [c for c in ["A_cnn", "A_gru", "B_conv1", "B_conv2"] if c in piv.columns]
    mat = piv[order_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    im = ax.imshow(mat.T, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(order_cols)))
    ax.set_yticklabels(order_cols)
    ax.set_xticks(range(18))
    ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_NAMES], rotation=90, fontsize=6)
    ax.set_title("Per-class F1 on the test split (10 epochs, seed 42)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_per_class_10ep.png")
    plt.close(fig)

    print(f"[figures] written to {FIGS}")


if __name__ == "__main__":
    main()
