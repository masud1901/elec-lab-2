"""Collect finished runs into tables and figures.

Produces:
    results/runs.csv          one row per run
    results/per_class.csv     one row per run x class (test)
    results/agg.csv           mean/std per variant
    results/summary.md        markdown tables for the report
    results/tables.tex        LaTeX tables (booktabs)
    results/figures/*.png     ablation figures

Usage:
    python summarize.py
    python summarize.py --runs runs --out results
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils import load_json, save_json

PARTS = ["0", "A", "B", "C", "D", "E", "F"]
PART_TITLES = {
    "0": "Part 0 - Reference",
    "A": "Part A - Is the LSTM what helps?",
    "B": "Part B - Do the convolutions matter?",
    "C": "Part C - How long should the window be?",
    "D": "Part D - How much regularization is needed?",
    "E": "Part E - Which optimizer settings work best?",
    "F": "Part F - Does handling class imbalance help rare gestures?",
}
PART_VARIANTS = {
    "0": ["ref"],
    "A": ["ref", "A_cnn", "A_gru"],
    "B": ["ref", "B_conv0", "B_conv1", "B_conv2"],
    "C": ["ref", "C_w20", "C_w42", "C_w60"],
    "D": ["ref", "D_no_dropout", "D_no_wd", "D_no_aug"],
    "E": ["ref", "E_lr1e-4", "E_lr1e-2", "E_warm1e-4", "E_warm1e-2"],
    "F": ["ref", "F_focal"],
}

RUN_COLUMNS = [
    "run_id", "variant", "part", "seed", "model", "rnn", "n_conv", "window",
    "dropout", "weight_decay", "augment", "lr", "warmup_epochs", "loss",
    "focal_gamma", "params", "best_epoch", "epochs_run", "stopped_early",
    "train_time_s", "val_accuracy", "val_macro_f1", "val_weighted_f1",
    "test_accuracy", "test_macro_f1", "test_weighted_f1",
    "test_gesture_macro_f1", "test_gesture_weighted_f1", "test_loss",
    "n_test",
]

METRIC_COLUMNS = [
    "test_accuracy", "test_macro_f1", "test_weighted_f1",
    "test_gesture_macro_f1", "test_gesture_weighted_f1",
    "val_accuracy", "val_macro_f1", "val_weighted_f1",
]

CLASS_NAMES = [
    "Null", "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2",
    "Open Fridge", "Close Fridge", "Open Dishwasher", "Close Dishwasher",
    "Open Drawer 1", "Close Drawer 1", "Open Drawer 2", "Close Drawer 2",
    "Open Drawer 3", "Close Drawer 3", "Clean Table", "Drink from Cup",
    "Toggle Switch",
]


def variant_from_run_id(run_id: str) -> str:
    return re.sub(r"_s\d+$", "", run_id)


def load_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    rows: List[dict] = []
    per_class_rows: List[dict] = []
    raw: Dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*/run.json")):
        try:
            r = load_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cannot read {path}: {exc}")
            continue
        run_id = r.get("run_id", path.parent.name)
        variant = variant_from_run_id(run_id)
        raw[run_id] = r
        c = r["config"]
        test = r.get("test") or {}
        val = r.get("val") or {}
        if not test:
            print(f"[skip] {run_id}: no test metrics (use --no-test?)")
            continue
        row = {
            "run_id": run_id,
            "variant": variant,
            "part": r.get("part", ""),
            "seed": c.get("seed"),
            "model": c.get("model"),
            "rnn": c.get("rnn"),
            "n_conv": c.get("n_conv"),
            "window": c.get("window"),
            "dropout": c.get("dropout"),
            "weight_decay": c.get("weight_decay"),
            "augment": c.get("augment"),
            "lr": c.get("lr"),
            "warmup_epochs": c.get("warmup_epochs"),
            "loss": c.get("loss"),
            "focal_gamma": c.get("focal_gamma"),
            "params": r.get("param_count"),
            "best_epoch": r.get("best_epoch"),
            "epochs_run": r.get("epochs_run"),
            "stopped_early": r.get("stopped_early"),
            "train_time_s": r.get("train_time_s"),
            "val_accuracy": val.get("accuracy"),
            "val_macro_f1": val.get("macro_f1"),
            "val_weighted_f1": val.get("weighted_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_macro_f1": test.get("macro_f1"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_gesture_macro_f1": test.get("gesture_macro_f1"),
            "test_gesture_weighted_f1": test.get("gesture_weighted_f1"),
            "test_loss": test.get("loss"),
            "n_test": test.get("n_samples"),
        }
        rows.append(row)

        # per-class table: prefer the CSV written by train.py
        csv_path = path.parent / "per_class.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, rec in df.iterrows():
                per_class_rows.append(
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "seed": c.get("seed"),
                        "class_id": int(rec["class_id"]),
                        "class_name": rec["class_name"],
                        "precision": float(rec["precision"]),
                        "recall": float(rec["recall"]),
                        "f1": float(rec["f1"]),
                        "support": int(rec["support"]),
                    }
                )
        else:
            for rec in test.get("per_class", []):
                per_class_rows.append(
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "seed": c.get("seed"),
                        "class_id": rec["class_id"],
                        "class_name": rec["class_name"],
                        "precision": rec["precision"],
                        "recall": rec["recall"],
                        "f1": rec["f1"],
                        "support": rec["support"],
                    }
                )

    df = pd.DataFrame(rows, columns=RUN_COLUMNS)
    pc = pd.DataFrame(per_class_rows)
    return df, pc, raw


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    order = {v: i for i, v in enumerate(sum((PART_VARIANTS[p] for p in PARTS), []))}
    for (part, variant), g in df.groupby(["part", "variant"], dropna=False):
        row = {
            "part": part,
            "variant": variant,
            "n_seeds": len(g),
            "params": g["params"].iloc[0],
            "best_epoch_mean": g["best_epoch"].mean(),
            "train_time_mean_s": g["train_time_s"].mean(),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["variant"].map(lambda v: order.get(v, 999))
    out = out.sort_values(["part", "_order"]).drop(columns="_order")
    return out


def fmt(mean: Optional[float], std: Optional[float], digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_tables(agg: pd.DataFrame) -> str:
    lines = ["# Ablation summary", ""]
    for part in PARTS:
        sub = agg[(agg["part"] == part) & (agg["variant"].isin(PART_VARIANTS[part]))]
        if sub.empty:
            continue
        lines += [f"## {PART_TITLES[part]}", ""]
        lines.append(
            "| Variant | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | Accuracy | Params | Best epoch |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                "| {v} | {n} | {w} | {m} | {g} | {a} | {p:,} | {e} |".format(
                    v=r["variant"],
                    n=int(r["n_seeds"]),
                    w=fmt(r["test_weighted_f1_mean"], r["test_weighted_f1_std"]),
                    m=fmt(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                    g=fmt(r["test_gesture_macro_f1_mean"], r["test_gesture_macro_f1_std"]),
                    a=fmt(r["test_accuracy_mean"], r["test_accuracy_std"]),
                    p=int(r["params"]) if not pd.isna(r["params"]) else 0,
                    e=f"{r['best_epoch_mean']:.1f}" if not pd.isna(r["best_epoch_mean"]) else "-",
                )
            )
        lines.append("")
        lines.append("Values are mean ± std over seeds on the held-out test split.")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `ref` = 4 conv layers, LSTM, window 24 (0.8 s), dropout 0.5, weight decay 1e-4, augmentation.")
    lines.append("- Test predictions are sample-wise (stride 1); train/val windows use stride 12.")
    lines.append("- Weighted F1 includes Null; macro F1 averages all 18 classes; gesture macro F1 excludes Null.")
    return "\n".join(lines) + "\n"


def latex_tables(agg: pd.DataFrame) -> str:
    lines = ["% Auto-generated by summarize.py", ""]
    for part in PARTS:
        sub = agg[(agg["part"] == part) & (agg["variant"].isin(PART_VARIANTS[part]))]
        if sub.empty:
            continue
        lines += [
            "\\begin{table}[ht]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Variant & Weighted F1 & Macro F1 & Gesture macro F1 & Params \\\\",
            "\\midrule",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"{r['variant'].replace('_', '\\_')} & "
                f"{r['test_weighted_f1_mean']:.3f}$\\pm${r['test_weighted_f1_std']:.3f} & "
                f"{r['test_macro_f1_mean']:.3f}$\\pm${r['test_macro_f1_std']:.3f} & "
                f"{r['test_gesture_macro_f1_mean']:.3f}$\\pm${r['test_gesture_macro_f1_std']:.3f} & "
                f"{int(r['params']):,} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    return plt


def _mean_std(df: pd.DataFrame, variant: str, col: str):
    g = df[df["variant"] == variant]
    return g[col].mean(), g[col].std(ddof=1) if len(g) > 1 else 0.0


def _bar_part(plt, df, variants, title, out, metrics=(("test_weighted_f1", "Weighted F1"), ("test_macro_f1", "Macro F1"))):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(variants))
    width = 0.8 / len(metrics)
    for k, (col, label) in enumerate(metrics):
        means, stds = [], []
        for v in variants:
            m, s = _mean_std(df, v, col)
            means.append(m)
            stds.append(s)
        ax.bar(x + k * width - 0.4 + width / 2, means, width, yerr=stds, capsize=2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_figures(df: pd.DataFrame, pc: pd.DataFrame, raw: Dict[str, dict], out_dir: Path) -> None:
    plt = _setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 0: training curves
    ref = df[df["variant"] == "ref"]
    if not ref.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
        for _, r in ref.iterrows():
            hist_path = Path("runs") / r["run_id"] / "history.csv"
            if not hist_path.exists():
                continue
            h = pd.read_csv(hist_path)
            axes[0].plot(h["epoch"], h["train_loss"], alpha=0.6, label=f"{r['run_id']} train")
            axes[0].plot(h["epoch"], h["val_loss"], alpha=0.6, linestyle="--", label=f"{r['run_id']} val")
            axes[1].plot(h["epoch"], h["val_macro_f1"], alpha=0.8, label=r["run_id"])
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Reference: loss")
        axes[0].legend(fontsize=6, frameon=False)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val macro F1")
        axes[1].set_title("Reference: validation F1")
        axes[1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_ref_curves.png")
        plt.close(fig)

    # A
    if "A_cnn" in set(df["variant"]) or "A_gru" in set(df["variant"]):
        _bar_part(plt, df, ["ref", "A_cnn", "A_gru"], "Part A: recurrence vs dense / GRU", out_dir / "fig_part_a.png")

    # B
    b = df[df["variant"].isin(["B_conv0", "B_conv1", "B_conv2", "ref"])]
    if not b.empty:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        xs, labels = [], []
        for v, nconv, label in [
            ("B_conv0", 0, "0 (raw)"), ("B_conv1", 1, "1"), ("B_conv2", 2, "2"),
            ("ref", 4, "4 (ref)"),
        ]:
            sub = df[df["variant"] == v]
            if sub.empty:
                continue
            xs.append(nconv)
            labels.append(label)
            ax.errorbar(
                nconv,
                sub["test_weighted_f1"].mean(),
                yerr=sub["test_weighted_f1"].std(ddof=1) if len(sub) > 1 else 0,
                marker="o", capsize=2, label="Weighted F1" if nconv == 0 else None,
            )
            ax.errorbar(
                nconv,
                sub["test_macro_f1"].mean(),
                yerr=sub["test_macro_f1"].std(ddof=1) if len(sub) > 1 else 0,
                marker="s", capsize=2, label="Macro F1" if nconv == 0 else None,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Number of conv layers")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_title("Part B: conv depth")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_part_b.png")
        plt.close(fig)

    # C
    c = df[df["variant"].isin(["C_w20", "ref", "C_w42", "C_w60"])]
    if not c.empty:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        for v, window, label in [("C_w20", 20, "0.67 s"), ("ref", 24, "0.80 s (ref)"), ("C_w42", 42, "1.40 s"), ("C_w60", 60, "2.00 s")]:
            sub = df[df["variant"] == v]
            if sub.empty:
                continue
            ax.errorbar(
                window,
                sub["test_weighted_f1"].mean(),
                yerr=sub["test_weighted_f1"].std(ddof=1) if len(sub) > 1 else 0,
                marker="o", capsize=2, label="Weighted F1" if window == 20 else None,
            )
            ax.errorbar(
                window,
                sub["test_macro_f1"].mean(),
                yerr=sub["test_macro_f1"].std(ddof=1) if len(sub) > 1 else 0,
                marker="s", capsize=2, label="Macro F1" if window == 20 else None,
            )
        ax.set_xlabel("Window length (samples @ 30 Hz)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_title("Part C: window length")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_part_c.png")
        plt.close(fig)

        # per-gesture heatmap
        order = ["C_w20", "ref", "C_w42", "C_w60"]
        labels = ["0.67 s", "0.80 s", "1.40 s", "2.00 s"]
        sub = pc[pc["variant"].isin(order)]
        if not sub.empty:
            mat = []
            for v in order:
                g = sub[sub["variant"] == v]
                if g.empty:
                    mat.append([np.nan] * 18)
                else:
                    mat.append(g.groupby("class_id")["f1"].mean().reindex(range(18)).values)
            mat = np.array(mat, dtype=float)[:, 1:]  # gestures only
            fig, ax = plt.subplots(figsize=(8.0, 2.9))
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(labels)
            ax.set_xticks(range(17))
            ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_NAMES[1:]], rotation=90, fontsize=6)
            ax.set_title("Part C: per-gesture F1")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_part_c_per_gesture.png")
            plt.close(fig)

    # D
    if set(df["variant"]) & {"D_no_dropout", "D_no_wd", "D_no_aug"}:
        _bar_part(
            plt, df, ["ref", "D_no_dropout", "D_no_wd", "D_no_aug"],
            "Part D: regularization", out_dir / "fig_part_d.png",
        )

    # E: validation curves
    e_variants = ["ref", "E_lr1e-4", "E_lr1e-2", "E_warm1e-4", "E_warm1e-2"]
    e = df[df["variant"].isin(e_variants)]
    if not e.empty:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for v in e_variants:
            sub = df[df["variant"] == v]
            if sub.empty:
                continue
            curves = []
            for _, r in sub.iterrows():
                p = Path("runs") / r["run_id"] / "history.csv"
                if p.exists():
                    h = pd.read_csv(p)
                    curves.append(h.set_index("epoch")["val_macro_f1"])
            if curves:
                m = pd.concat(curves, axis=1).mean(axis=1)
                ax.plot(m.index, m.values, label=v)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val macro F1")
        ax.set_title("Part E: learning-rate stability")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_part_e.png")
        plt.close(fig)

    # F: per-class F1
    if "F_focal" in set(pc["variant"]):
        sub = pc[pc["variant"].isin(["ref", "F_focal"])]
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        x = np.arange(18)
        for i, (v, label) in enumerate([("ref", "Cross-entropy (ref)"), ("F_focal", "Focal loss")]):
            g = sub[sub["variant"] == v].groupby("class_id")["f1"].mean().reindex(range(18))
            ax.bar(x + (i - 0.5) * 0.38, g.values, 0.38, label=label)
        support = (
            sub[sub["variant"] == "ref"].groupby("class_id")["support"].mean().reindex(range(18))
        )
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_NAMES], rotation=90, fontsize=6)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.0)
        ax.set_title("Part F: per-class F1 with and without focal loss")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_part_f.png")
        plt.close(fig)
        if support is not None:
            support.to_csv(out_dir / "fig_part_f_support.csv")

    # Reference confusion matrix
    ref_json = Path("runs") / f"ref_s{int(ref['seed'].min()) if not ref.empty else 42}" / "run.json"
    cm = None
    for _, r in ref.iterrows():
        p = Path("runs") / r["run_id"] / "confusion_test.npy"
        if p.exists():
            cm = np.load(p)
            break
    if cm is not None:
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(18))
        ax.set_yticks(range(18))
        ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(CLASS_NAMES, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Reference: row-normalised confusion")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_confusion_ref.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, pc, raw = load_runs(runs_dir)
    if df.empty:
        raise SystemExit(f"No finished runs with test metrics found in {runs_dir}")
    df = df.sort_values(["part", "variant", "seed"]).reset_index(drop=True)
    df.to_csv(out_dir / "runs.csv", index=False)
    if not pc.empty:
        pc.to_csv(out_dir / "per_class.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out_dir / "agg.csv", index=False)
    (out_dir / "summary.md").write_text(markdown_tables(agg))
    (out_dir / "tables.tex").write_text(latex_tables(agg))
    save_json(
        out_dir / "summarize_manifest.json",
        {
            "runs": sorted(df["run_id"].tolist()),
            "variants": sorted(df["variant"].unique().tolist()),
            "n_runs": int(len(df)),
        },
    )

    if not args.no_figures:
        make_figures(df, pc, raw, out_dir / "figures")

    print(f"[done] {out_dir}: {len(df)} runs, {df['variant'].nunique()} variants")
    print(markdown_tables(agg))


if __name__ == "__main__":
    main()
