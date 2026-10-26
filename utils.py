"""Shared utilities: seeding, metrics, logging, json helpers."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ---------------------------------------------------------------------------
# JSON / files
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def git_info(repo: str | Path = ".") -> Dict[str, Optional[str]]:
    """Return current commit and dirty state, or None if not a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class Logger:
    """Tiny logger that writes to stdout and optionally a file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs)
        if self.path:
            with self.path.open("a") as f:
                f.write(msg + "\n")

    def flush(self) -> None:
        sys.stdout.flush()


class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    n_classes: int = 18,
) -> Dict[str, Any]:
    """Compute accuracy, macro/weighted F1, per-class table and confusion matrix.

    All 18 classes are always scored (classes absent from a split get F1=0),
    which keeps runs with different window lengths comparable.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    labels = list(range(n_classes))
    if class_names is None:
        class_names = [f"class_{i}" for i in labels]
    assert len(class_names) == n_classes, (len(class_names), n_classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = []
    for i, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": i,
                "class_name": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )

    gesture_labels = list(range(1, n_classes))
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "gesture_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=gesture_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "gesture_weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=gesture_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "support": {name: int(support[i]) for i, name in enumerate(class_names)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(y_true.size),
    }
    return metrics


def format_metrics(metrics: Dict[str, Any]) -> str:
    return (
        f"acc={metrics['accuracy']:.4f} "
        f"macroF1={metrics['macro_f1']:.4f} "
        f"weightedF1={metrics['weighted_f1']:.4f} "
        f"gestureMacroF1={metrics['gesture_macro_f1']:.4f}"
    )


def write_per_class_csv(path: str | Path, metrics: Dict[str, Any], run_id: str = "") -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        )
        for row in metrics["per_class"]:
            writer.writerow(
                [
                    run_id,
                    row["class_id"],
                    row["class_name"],
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['f1']:.6f}",
                    row["support"],
                ]
            )
