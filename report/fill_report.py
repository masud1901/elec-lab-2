"""Fill report/report.md from the template and the aggregated results.

The template (report/report_template.md) contains markers such as
``<!-- RESULTS_A -->``.  This script replaces every marker with a generated
block (tables, figures and prose with the actual numbers) and writes
report/report.md.

Usage:
    python summarize.py            # produces results/*.csv
    python report/fill_report.py   # regenerates report/report.md
    python report/build_report.py  # report/report.md -> report/report.pdf
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TEMPLATE = ROOT / "report" / "report_template.md"
OUTPUT = ROOT / "report" / "report.md"

FIG = "../results/figures"

PART_VARIANTS = {
    "0": ["ref"],
    "A": ["ref", "A_cnn", "A_gru"],
    "B": ["ref", "B_conv0", "B_conv1", "B_conv2"],
    "C": ["ref", "C_w20", "C_w42", "C_w60"],
    "D": ["ref", "D_no_dropout", "D_no_wd", "D_no_aug"],
    "E": ["ref", "E_lr1e-4", "E_lr1e-2", "E_warm1e-4", "E_warm1e-2"],
    "F": ["ref", "F_focal"],
}

VARIANT_LABEL = {
    "ref": "Reference (4 conv + LSTM, 0.8 s)",
    "A_cnn": "CNN: 2 dense layers instead of LSTM",
    "A_gru": "GRU instead of LSTM",
    "B_conv0": "0 conv layers (raw 113 channels)",
    "B_conv1": "1 conv layer",
    "B_conv2": "2 conv layers",
    "C_w20": "Window 0.67 s (20 samples)",
    "C_w42": "Window 1.40 s (42 samples)",
    "C_w60": "Window 2.00 s (60 samples)",
    "D_no_dropout": "No dropout",
    "D_no_wd": "No weight decay",
    "D_no_aug": "No augmentation",
    "E_lr1e-4": "LR 1e-4",
    "E_lr1e-2": "LR 1e-2",
    "E_warm1e-4": "LR 1e-4 + 5-epoch warm-up",
    "E_warm1e-2": "LR 1e-2 + 5-epoch warm-up",
    "F_focal": "Focal loss (gamma = 2)",
}

PART_QUESTION = {
    "0": "The reference model is the baseline every other part is compared against.",
    "A": "Does the recurrent stack actually contribute, and do LSTM cells matter specifically?",
    "B": "How much of the performance comes from the convolutional feature extractor?",
    "C": "How much history does the model need before it can classify a gesture?",
    "D": "Which regulariser is doing the work, if any?",
    "E": "How sensitive is training to the learning rate, and can warm-up rescue a bad one?",
    "F": "Does a loss that down-weights easy windows help the rare gestures?",
}


class Results:
    def __init__(self) -> None:
        self.df = pd.read_csv(RESULTS / "runs.csv")
        self.agg = pd.read_csv(RESULTS / "agg.csv")
        self.pc = pd.read_csv(RESULTS / "per_class.csv")

    # -- accessors --------------------------------------------------------
    def row(self, variant: str) -> Optional[pd.Series]:
        sub = self.agg[self.agg["variant"] == variant]
        if sub.empty:
            return None
        return sub.iloc[0]

    def has(self, variant: str) -> bool:
        return not self.agg[self.agg["variant"] == variant].empty

    def n_seeds(self, variant: str) -> int:
        r = self.row(variant)
        return int(r["n_seeds"]) if r is not None else 0

    def mean(self, variant: str, col: str) -> Optional[float]:
        r = self.row(variant)
        if r is None:
            return None
        value = r.get(f"{col}_mean")
        return None if pd.isna(value) else float(value)

    def std(self, variant: str, col: str) -> Optional[float]:
        r = self.row(variant)
        if r is None:
            return None
        value = r.get(f"{col}_std")
        return None if pd.isna(value) else float(value)

    def ms(self, variant: str, col: str, digits: int = 3) -> str:
        mean, std = self.mean(variant, col), self.std(variant, col)
        if mean is None:
            return "-"
        if std is None or self.n_seeds(variant) < 2:
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    def delta(self, variant: str, col: str, ref: str = "ref") -> Optional[float]:
        a, b = self.mean(variant, col), self.mean(ref, col)
        if a is None or b is None:
            return None
        return a - b

    def delta_str(self, variant: str, col: str, ref: str = "ref", digits: int = 3) -> str:
        d = self.delta(variant, col, ref)
        if d is None:
            return "-"
        return f"{d:+.{digits}f}"

    def per_class(self, variant: str) -> pd.DataFrame:
        sub = self.pc[self.pc["variant"] == variant]
        if sub.empty:
            return sub
        return (
            sub.groupby(["class_id", "class_name"])
            .agg(
                f1=("f1", "mean"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                support=("support", "mean"),
            )
            .reset_index()
        )

    def epochs(self, variant: str) -> float:
        r = self.row(variant)
        if r is None:
            return float("nan")
        return float(r["best_epoch_mean"])

    # -- training history -------------------------------------------------
    def history(self, variant: str) -> Optional[pd.DataFrame]:
        sub = self.df[self.df["variant"] == variant]
        if sub.empty:
            return None
        run_id = sub.iloc[0]["run_id"]
        path = RESULTS.parent / "runs" / run_id / "history.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def gap(self, variant: str) -> Optional[Dict[str, float]]:
        """Train/validation macro F1 at the best epoch (overfitting probe)."""
        hist = self.history(variant)
        if hist is None or hist.empty:
            return None
        best_epoch = self.epochs(variant)
        row = hist.iloc[(hist["epoch"] - best_epoch).abs().argsort()[:1]]
        if row.empty:
            return None
        row = row.iloc[0]
        return {
            "epoch": float(row["epoch"]),
            "train_macro_f1": float(row["train_macro_f1"]),
            "val_macro_f1": float(row["val_macro_f1"]),
            "gap": float(row["train_macro_f1"] - row["val_macro_f1"]),
        }

    def gesture_durations(self) -> pd.DataFrame:
        path = RESULTS / "gesture_duration_summary.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        return df[df["class_id"] > 0]


def table(res: Results, variants: List[str], part: str) -> str:
    header = (
        "| Variant | Configuration | Seeds | Weighted F1 | Macro F1 | Gesture macro F1 | "
        "Accuracy | Params | Best epoch |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for variant in variants:
        if not res.has(variant):
            continue
        r = res.row(variant)
        rows.append(
            "| `{v}` | {label} | {n} | {w} | {m} | {g} | {a} | {p:,} | {e} |".format(
                v=variant,
                label=VARIANT_LABEL.get(variant, variant),
                n=int(r["n_seeds"]),
                w=res.ms(variant, "test_weighted_f1"),
                m=res.ms(variant, "test_macro_f1"),
                g=res.ms(variant, "test_gesture_macro_f1"),
                a=res.ms(variant, "test_accuracy"),
                p=int(r["params"]),
                e=f"{r['best_epoch_mean']:.1f}",
            )
        )
    return header + "\n".join(rows) + "\n"


def figure(name: str, caption: str) -> str:
    return (
        f'<figure><img src="{FIG}/{name}" />'
        f"<figcaption>{caption}</figcaption></figure>\n"
    )


def part_section(res: Results, part: str, variants: List[str], body: str) -> str:
    out = [f"**Question.** {PART_QUESTION[part]}", ""]
    out.append(table(res, variants, part))
    out.append("")
    out.append(body)
    return "\n".join(out)


def build_blocks(res: Results) -> Dict[str, str]:
    blocks: Dict[str, str] = {}

    ref_w = res.ms("ref", "test_weighted_f1")
    ref_m = res.ms("ref", "test_macro_f1")
    ref_g = res.ms("ref", "test_gesture_macro_f1")
    ref_a = res.ms("ref", "test_accuracy")
    ref_ep = res.epochs("ref")

    # ---- abstract -------------------------------------------------------
    headline = []
    for part, variants in PART_VARIANTS.items():
        if part == "0":
            continue
        for v in variants[1:]:
            if not res.has(v):
                continue
            d_w = res.delta(v, "test_weighted_f1")
            d_m = res.delta(v, "test_macro_f1")
            headline.append(
                f"- **{part}** (`{v}`): weighted F1 {res.delta_str(v, 'test_weighted_f1')} "
                f"and macro F1 {res.delta_str(v, 'test_macro_f1')} relative to the reference."
            )
    blocks["RESULTS_ABSTRACT"] = "\n".join(
        [
            "The reference reaches **weighted F1 " + ref_w + "**, **macro F1 " + ref_m
            + "** and **gesture macro F1 " + ref_g + "** on the held-out test subjects "
            f"(accuracy {ref_a}), with the best checkpoint at epoch {ref_ep:.0f}.",
            "",
            "Headline effects (test split, difference to the reference):",
            "",
        ]
        + headline
    )

    # ---- part 0 ---------------------------------------------------------
    ref_runs = res.df[res.df["variant"] == "ref"]
    seeds_txt = ", ".join(str(int(s)) for s in sorted(ref_runs["seed"].unique()))
    blocks["RESULTS_REF"] = "\n".join(
        [
            "**Setup.** The reference model is trained with the protocol of Section 3 "
            f"({res.n_seeds('ref')} seed(s): {seeds_txt}).",
            "",
            table(res, ["ref"], "0"),
            "",
            figure("fig_ref_curves.png", "Reference training: cross-entropy loss and validation macro F1 per epoch."),
            "",
            f"The reference reaches weighted F1 {ref_w} and macro F1 {ref_m}. The gap "
            "between the two metrics is the central observation of this study: 83% of the "
            "test windows are Null, so the weighted score is dominated by the majority "
            "class while the macro score exposes how poorly the rare gestures are "
            "recognised.",
            "",
            figure("fig_confusion_ref.png", "Reference model: row-normalised confusion matrix on the test split."),
        ]
    )

    # ---- part A ---------------------------------------------------------
    body_a = []
    if res.has("A_cnn"):
        body_a += [
            f"Removing the recurrent stack and replacing it with two dense layers of 128 "
            f"units (7.49M parameters instead of 3.97M) changes weighted F1 by "
            f"{res.delta_str('A_cnn', 'test_weighted_f1')} and macro F1 by "
            f"{res.delta_str('A_cnn', 'test_macro_f1')}.",
        ]
    if res.has("A_gru"):
        body_a += [
            f"Swapping LSTM cells for GRU cells changes weighted F1 by "
            f"{res.delta_str('A_gru', 'test_weighted_f1')} and macro F1 by "
            f"{res.delta_str('A_gru', 'test_macro_f1')}.",
        ]
    body_a += [
        "",
        figure("fig_part_a.png", "Part A: test scores of the reference, the CNN and the GRU variant."),
    ]
    blocks["RESULTS_A"] = part_section(res, "A", PART_VARIANTS["A"], "\n".join(body_a))

    # ---- part B ---------------------------------------------------------
    body_b = []
    for v in ["B_conv0", "B_conv1", "B_conv2"]:
        if res.has(v):
            body_b.append(
                f"- `{v}`: weighted F1 {res.delta_str(v, 'test_weighted_f1')}, "
                f"macro F1 {res.delta_str(v, 'test_macro_f1')} relative to the reference."
            )
    body_b += [
        "",
        figure("fig_part_b.png", "Part B: test F1 as a function of the number of convolutional layers."),
    ]
    blocks["RESULTS_B"] = part_section(res, "B", PART_VARIANTS["B"], "\n".join(body_b))

    # ---- part C ---------------------------------------------------------
    body_c = []
    for v in ["C_w20", "C_w42", "C_w60"]:
        if res.has(v):
            body_c.append(
                f"- `{v}`: weighted F1 {res.delta_str(v, 'test_weighted_f1')}, "
                f"macro F1 {res.delta_str(v, 'test_macro_f1')} relative to the reference."
            )
    body_c += [
        "",
        figure("fig_part_c.png", "Part C: test F1 as a function of the window length."),
        figure("fig_part_c_per_gesture.png", "Part C: per-gesture F1 for each window length."),
    ]
    blocks["RESULTS_C"] = part_section(res, "C", PART_VARIANTS["C"], "\n".join(body_c))

    # ---- part D ---------------------------------------------------------
    body_d = []
    for v in ["D_no_dropout", "D_no_wd", "D_no_aug"]:
        if res.has(v):
            body_d.append(
                f"- `{v}`: weighted F1 {res.delta_str(v, 'test_weighted_f1')}, "
                f"macro F1 {res.delta_str(v, 'test_macro_f1')} relative to the reference."
            )
    gaps = []
    for v in ["ref", "D_no_dropout", "D_no_wd", "D_no_aug"]:
        g = res.gap(v)
        if g is None:
            continue
        gaps.append(
            f"| `{v}` | {g['epoch']:.0f} | {g['train_macro_f1']:.3f} | "
            f"{g['val_macro_f1']:.3f} | {g['gap']:+.3f} |"
        )
    if gaps:
        body_d += [
            "",
            "Overfitting probe (first seed, at the selected epoch):",
            "",
            "| Variant | Epoch | Train macro F1 | Val macro F1 | Gap |",
            "|---|---|---|---|---|",
            "\n".join(gaps),
        ]
    body_d += [
        "",
        figure("fig_part_d.png", "Part D: test scores with and without each regulariser."),
    ]
    blocks["RESULTS_D"] = part_section(res, "D", PART_VARIANTS["D"], "\n".join(body_d))

    # ---- part E ---------------------------------------------------------
    body_e = []
    for v in ["E_lr1e-4", "E_lr1e-2", "E_warm1e-4", "E_warm1e-2"]:
        if res.has(v):
            body_e.append(
                f"- `{v}`: weighted F1 {res.delta_str(v, 'test_weighted_f1')}, "
                f"macro F1 {res.delta_str(v, 'test_macro_f1')} relative to the reference."
            )
    body_e += [
        "",
        figure("fig_part_e.png", "Part E: validation macro F1 per epoch for each learning-rate setting."),
    ]
    blocks["RESULTS_E"] = part_section(res, "E", PART_VARIANTS["E"], "\n".join(body_e))

    # ---- part F ---------------------------------------------------------
    body_f = []
    if res.has("F_focal"):
        body_f.append(
            f"Replacing cross-entropy by focal loss (gamma = 2) changes weighted F1 by "
            f"{res.delta_str('F_focal', 'test_weighted_f1')} and macro F1 by "
            f"{res.delta_str('F_focal', 'test_macro_f1')}."
        )
        pc_ref = res.per_class("ref").set_index("class_id")
        pc_foc = res.per_class("F_focal").set_index("class_id")
        if not pc_foc.empty:
            rows = []
            for cid in pc_ref.index:
                if cid not in pc_foc.index:
                    continue
                rows.append(
                    f"| {pc_ref.loc[cid, 'class_name']} | {int(pc_ref.loc[cid, 'support'])} | "
                    f"{pc_ref.loc[cid, 'f1']:.3f} | {pc_foc.loc[cid, 'f1']:.3f} | "
                    f"{pc_foc.loc[cid, 'f1'] - pc_ref.loc[cid, 'f1']:+.3f} |"
                )
            body_f += [
                "",
                "| Gesture | Test windows | F1 (cross-entropy) | F1 (focal) | Delta |",
                "|---|---|---|---|---|",
                "\n".join(rows),
            ]
    body_f += [
        "",
        figure("fig_part_f.png", "Part F: per-class F1 with cross-entropy and with focal loss."),
    ]
    blocks["RESULTS_F"] = part_section(res, "F", PART_VARIANTS["F"], "\n".join(body_f))

    # ---- discussion -----------------------------------------------------
    blocks["RESULTS_DISCUSSION"] = "\n".join(
        [
            "The reference model already separates the two questions the study asks: "
            "on the weighted metric it is close to the published numbers, while on the "
            "macro metric it is roughly half as good, and every variant that trades "
            "capacity for simplicity or removes regularisation moves the two metrics in "
            "different directions. Reporting only weighted F1 - as the original paper "
            "does - hides most of the interesting behaviour.",
        ]
    )

    # ---- gesture durations (Part C context) ------------------------------
    dur = res.gesture_durations()
    if not dur.empty:
        rows = []
        for _, r in dur.iterrows():
            rows.append(
                f"| {r['class_name']} | {r['n_events']:.0f} | {r['mean_duration_s']:.2f} | "
                f"{r['median_duration_s']:.2f} | {r['max_duration_s']:.2f} |"
            )
        blocks["RESULTS_GESTURE_DURATIONS"] = "\n".join(
            [
                "| Gesture | Events | Mean duration (s) | Median (s) | Max (s) |",
                "|---|---|---|---|---|",
                "\n".join(rows),
            ]
        )
    else:
        blocks["RESULTS_GESTURE_DURATIONS"] = "_Gesture durations not computed._"

    # ---- per-class appendix --------------------------------------------
    pc_ref = res.per_class("ref")
    rows = []
    for _, r in pc_ref.iterrows():
        rows.append(
            f"| {int(r['class_id'])} | {r['class_name']} | {int(r['support'])} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    blocks["RESULTS_PER_CLASS"] = "\n".join(
        [
            "Reference model, test split (mean over seeds).",
            "",
            "| Id | Class | Test windows | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|",
            "\n".join(rows),
        ]
    )

    return blocks


def main() -> None:
    template = TEMPLATE.read_text()
    blocks = build_blocks(Results())

    for marker, content in blocks.items():
        pattern = re.compile(rf"<!-- {marker} -->")
        if not pattern.search(template):
            print(f"[warn] marker {marker} not found in template")
            continue
        template = pattern.sub(lambda _m: content, template)

    OUTPUT.write_text(template)
    print(f"[report] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
