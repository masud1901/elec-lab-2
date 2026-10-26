"""Preprocess OPPORTUNITY: select the 113 challenge channels, fill missing
values, normalise, and write one .npz per recording.

The raw files contain 250 columns:
    0        timestamp
    1..242   sensors (accelerometers, gyroscopes, magnetometers, ...)
    243      locomotion label
    244..248  high-level / low-level labels
    249      ML_Both_Arms gesture label  <- our 18-class target

Following the OPPORTUNITY challenge setup we delete the quaternion channels,
the ambient sensors and the unused label columns, leaving 113 sensor channels.

Splits
------
paper   Train: S1 ADL1-5 + Drill, S2/S3 ADL1-2 + Drill
        Val:   S2/S3 ADL3
        Test:  S2/S3 ADL4-5                  (paper / challenge protocol)
course  Train: S1 Drill + ADL1-4, S2/S3 Drill + ADL1-3
        Val:   S1 ADL5
        Test:  S2/S3 ADL4-5                  (public reference implementation)

Usage:
    python preprocess.py --split paper
    python preprocess.py --split paper --split course
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import save_json

N_SENSOR_CHANNELS = 113
N_CLASSES = 18

# Columns removed by the OPPORTUNITY challenge preprocessing.
FEATURES_DELETE = np.concatenate(
    [
        np.arange(46, 50),   # back quaternions
        np.arange(59, 63),   # right upper arm quaternions
        np.arange(72, 76),   # right lower arm quaternions
        np.arange(85, 89),   # left upper arm quaternions
        np.arange(98, 102),  # left lower arm quaternions
        np.arange(134, 243), # ambient sensors
        np.arange(244, 249), # HL / LL labels
    ]
)

GESTURE_MAP = {
    406516: 1,   # Open Door 1
    406517: 2,   # Open Door 2
    404516: 3,   # Close Door 1
    404517: 4,   # Close Door 2
    406520: 5,   # Open Fridge
    404520: 6,   # Close Fridge
    406505: 7,   # Open Dishwasher
    404505: 8,   # Close Dishwasher
    406519: 9,   # Open Drawer 1
    404519: 10,  # Close Drawer 1
    406511: 11,  # Open Drawer 2
    404511: 12,  # Close Drawer 2
    406508: 13,  # Open Drawer 3
    404508: 14,  # Close Drawer 3
    408512: 15,  # Clean Table
    407521: 16,  # Drink from Cup
    405506: 17,  # Toggle Switch
}

CLASS_NAMES = [
    "Null",
    "Open Door 1",
    "Open Door 2",
    "Close Door 1",
    "Close Door 2",
    "Open Fridge",
    "Close Fridge",
    "Open Dishwasher",
    "Close Dishwasher",
    "Open Drawer 1",
    "Close Drawer 1",
    "Open Drawer 2",
    "Close Drawer 2",
    "Open Drawer 3",
    "Close Drawer 3",
    "Clean Table",
    "Drink from Cup",
    "Toggle Switch",
]

SPLITS: Dict[str, Dict[str, List[str]]] = {
    "paper": {
        "train": [
            "S1-ADL1", "S1-ADL2", "S1-ADL3", "S1-ADL4", "S1-ADL5", "S1-Drill",
            "S2-ADL1", "S2-ADL2", "S2-Drill",
            "S3-ADL1", "S3-ADL2", "S3-Drill",
        ],
        "val": ["S2-ADL3", "S3-ADL3"],
        "test": ["S2-ADL4", "S2-ADL5", "S3-ADL4", "S3-ADL5"],
    },
    "course": {
        "train": [
            "S1-Drill", "S1-ADL1", "S1-ADL2", "S1-ADL3", "S1-ADL4",
            "S2-Drill", "S2-ADL1", "S2-ADL2", "S2-ADL3",
            "S3-Drill", "S3-ADL1", "S3-ADL2", "S3-ADL3",
        ],
        "val": ["S1-ADL5"],
        "test": ["S2-ADL4", "S2-ADL5", "S3-ADL4", "S3-ADL5"],
    },
}


def select_columns(data: np.ndarray) -> np.ndarray:
    """Delete quaternion / ambient / unused-label columns (250 -> 116)."""
    return np.delete(data, FEATURES_DELETE, axis=1)


def divide_x_y(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the 113 sensor channels and the 18-class gesture labels."""
    x = data[:, 1 : N_SENSOR_CHANNELS + 1]
    y = data[:, N_SENSOR_CHANNELS + 2]  # ML_Both_Arms after column deletion
    return x.astype(np.float32), y.astype(np.int64)


def map_labels(y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.int64)
    unknown = set()
    for code, label in GESTURE_MAP.items():
        out[y == code] = label
    known = set(GESTURE_MAP) | {0}
    for value in np.unique(y):
        if int(value) not in known:
            unknown.add(int(value))
    if unknown:
        raise ValueError(f"Unknown gesture codes: {sorted(unknown)}")
    return out


def fill_missing(x: np.ndarray) -> Tuple[np.ndarray, int]:
    """Linearly interpolate NaNs per channel; extrapolate with nearest values.

    Returns the filled array and the number of NaNs that were filled.
    """
    n_missing = int(np.isnan(x).sum())
    if n_missing == 0:
        return x, 0
    out = x.copy()
    n = out.shape[0]
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        valid = np.flatnonzero(~mask)
        if valid.size == 0:
            out[:, j] = 0.0
            continue
        out[:, j] = np.interp(np.arange(n), valid, col[valid])
    return out, n_missing


def load_recording(dataset_dir: Path, run: str) -> Tuple[np.ndarray, np.ndarray, int]:
    path = dataset_dir / f"{run}.dat"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    raw = frame.to_numpy()
    if raw.ndim != 2 or raw.shape[1] != 250:
        raise ValueError(f"{path}: unexpected shape {raw.shape}")
    reduced = select_columns(raw)
    x, y_raw = divide_x_y(reduced)
    y = map_labels(y_raw)
    x, n_missing = fill_missing(x)
    return x, y, n_missing


def compute_stats(arrays: List[np.ndarray]) -> Dict[str, np.ndarray]:
    cat = np.concatenate(arrays, axis=0)
    return {
        "min": cat.min(axis=0),
        "max": cat.max(axis=0),
        "mean": cat.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": cat.std(axis=0, dtype=np.float64).astype(np.float32),
        "count": np.array([cat.shape[0]], dtype=np.int64),
    }


def normalize(x: np.ndarray, stats: Dict[str, np.ndarray], scheme: str) -> np.ndarray:
    if scheme == "minmax":
        lo = stats["min"]
        hi = stats["max"]
        return ((x - lo) / np.maximum(hi - lo, 1e-8)).astype(np.float32)
    if scheme == "zscore":
        return ((x - stats["mean"]) / np.maximum(stats["std"], 1e-8)).astype(np.float32)
    raise ValueError(f"Unknown normalization: {scheme}")


def build_split(
    raw_dir: Path,
    out_root: Path,
    split_name: str,
    norm: str,
    force: bool = False,
) -> None:
    dataset_dir = raw_dir / "OpportunityUCIDataset" / "dataset"
    split = SPLITS[split_name]
    out_dir = out_root / split_name
    if (out_dir / "split.json").exists() and not force:
        print(f"[skip] {out_dir} already exists (use --force to rebuild)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs = split["train"] + split["val"] + split["test"]
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    print(f"[split:{split_name}] loading {len(all_runs)} recordings")
    for i, run in enumerate(all_runs, 1):
        x, y, n_missing = load_recording(dataset_dir, run)
        cache[run] = (x, y, n_missing)
        print(
            f"  [{i:2d}/{len(all_runs)}] {run:9s} x={x.shape} "
            f"null={100*(y == 0).mean():5.1f}% missing={n_missing}"
        )

    stats = compute_stats([cache[r][0] for r in split["train"]])
    np.savez(out_dir / "norm_stats.npz", **stats)
    save_json(
        out_dir / "norm_config.json",
        {"scheme": norm, "source": "train", "n_channels": N_SENSOR_CHANNELS},
    )

    dataset_stats: Dict[str, object] = {
        "split": split_name,
        "normalization": norm,
        "class_names": CLASS_NAMES,
        "runs": {},
    }
    for part in ("train", "val", "test"):
        (out_dir / part).mkdir(exist_ok=True)
        part_counts = np.zeros(N_CLASSES, dtype=np.int64)
        part_samples = 0
        for run in split[part]:
            x, y, n_missing = cache[run]
            x_norm = normalize(x, stats, norm)
            np.savez(out_dir / part / f"{run}.npz", x=x_norm, y=y)
            counts = np.bincount(y, minlength=N_CLASSES)
            part_counts += counts
            part_samples += len(y)
            dataset_stats["runs"][run] = {  # type: ignore[index]
                "part": part,
                "samples": int(len(y)),
                "missing_filled": int(n_missing),
                "class_counts": counts.tolist(),
            }
            print(f"  [write] {part}/{run}.npz {x_norm.shape}")
        dataset_stats[f"{part}_samples"] = int(part_samples)  # type: ignore[index]
        dataset_stats[f"{part}_class_counts"] = part_counts.tolist()  # type: ignore[index]
        dataset_stats[f"{part}_null_fraction"] = float(  # type: ignore[index]
            part_counts[0] / max(part_samples, 1)
        )
        print(
            f"[split:{split_name}] {part}: {part_samples} samples, "
            f"Null={100*part_counts[0]/max(part_samples,1):.1f}%"
        )

    save_json(out_dir / "split.json", {**split, "name": split_name, "norm": norm})
    save_json(out_dir / "class_names.json", CLASS_NAMES)
    save_json(out_dir / "dataset_stats.json", dataset_stats)
    print(f"[done] {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw", help="directory with the extracted dataset")
    parser.add_argument("--out", default="data/processed", help="output directory")
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(SPLITS),
        default=None,
        help="split protocol to build (default: paper)",
    )
    parser.add_argument(
        "--norm",
        choices=["minmax", "zscore"],
        default="minmax",
        help="normalization fitted on the training split",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    splits = args.split or ["paper"]
    for split_name in splits:
        build_split(Path(args.raw), Path(args.out), split_name, args.norm, args.force)


if __name__ == "__main__":
    main()
