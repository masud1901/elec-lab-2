"""Measure gesture durations in the processed recordings.

For Part C of the ablation study we want to know how long each gesture
actually lasts, so that the effect of the observation window can be related to
the duration of the gesture it has to capture.

Usage:
    python scripts/gesture_stats.py --split paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess import CLASS_NAMES  # noqa: E402


def segment_durations(y: np.ndarray) -> dict[int, list[float]]:
    """Return the length (in samples) of every contiguous run of each class."""
    durations: dict[int, list[float]] = {}
    if len(y) == 0:
        return durations
    change = np.flatnonzero(np.diff(y)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(y)]])
    for start, end in zip(starts, ends):
        cls = int(y[start])
        durations.setdefault(cls, []).append(float(end - start))
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--split", default="paper")
    parser.add_argument("--out", default="results/gesture_durations.csv")
    args = parser.parse_args()

    root = Path(args.data) / args.split
    rows = []
    for part in ("train", "val", "test"):
        for path in sorted((root / part).glob("*.npz")):
            y = np.load(path)["y"]
            durations = segment_durations(y)
            for cls, values in durations.items():
                values = np.asarray(values)
                rows.append(
                    {
                        "part": part,
                        "recording": path.stem,
                        "class_id": cls,
                        "class_name": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                        "n_events": len(values),
                        "mean_duration_s": float(values.mean()) / 30.0,
                        "median_duration_s": float(np.median(values)) / 30.0,
                        "max_duration_s": float(values.max()) / 30.0,
                    }
                )
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["class_id", "class_name"])
        .agg(
            n_events=("n_events", "sum"),
            mean_duration_s=("mean_duration_s", "mean"),
            median_duration_s=("median_duration_s", "median"),
            max_duration_s=("max_duration_s", "max"),
        )
        .reset_index()
        .sort_values("class_id")
    )
    summary.to_csv(out.with_name("gesture_duration_summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
