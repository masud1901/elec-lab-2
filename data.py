"""Windowed OPPORTUNITY datasets and data loaders.

A window of ``window`` samples is labelled with the gesture at its last
sample, following the OPPORTUNITY challenge / DeepConvLSTM protocol.  Train
and validation use a stride of half a window; the test split is evaluated
sample-wise with stride 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import worker_init_fn

PARTS = ("train", "val", "test")


class WindowDataset(Dataset):
    """Sliding-window dataset backed by one .npz file per recording."""

    def __init__(
        self,
        part_dir: str | Path,
        window: int = 24,
        stride: int = 12,
        augment: str = "none",
        drop_null: bool = False,
        limit: Optional[int] = None,
        seed: int = 0,
        training: bool = False,
        class_names: Optional[List[str]] = None,
    ):
        self.part_dir = Path(part_dir)
        self.window = int(window)
        self.stride = int(stride)
        self.augment = augment
        self.training = training
        self.class_names = class_names
        self.rng = np.random.default_rng(seed)

        files = sorted(self.part_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files in {self.part_dir}")
        self.files = files
        self.recording_names = [f.stem for f in files]
        self.data: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.sizes: List[int] = []

        rec_idx: List[np.ndarray] = []
        starts: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for i, path in enumerate(files):
            rec = np.load(path)
            x = np.ascontiguousarray(rec["x"], dtype=np.float32)
            y = np.asarray(rec["y"], dtype=np.int64)
            if x.ndim != 2 or x.shape[1] != 113:
                raise ValueError(f"{path}: expected (n, 113), got {x.shape}")
            if len(x) != len(y):
                raise ValueError(f"{path}: x/y length mismatch")
            if len(x) < self.window:
                raise ValueError(
                    f"{path}: recording has {len(x)} samples < window {self.window}"
                )
            self.data.append(x)
            self.targets.append(y)
            self.sizes.append(len(x))

            s = np.arange(0, len(x) - self.window + 1, self.stride, dtype=np.int64)
            lab = y[s + self.window - 1]
            rec_idx.append(np.full(len(s), i, dtype=np.int64))
            starts.append(s)
            labels.append(lab)

        self.rec_idx = np.concatenate(rec_idx)
        self.starts = np.concatenate(starts)
        self.labels = np.concatenate(labels)

        if drop_null:
            keep = self.labels != 0
            self.rec_idx = self.rec_idx[keep]
            self.starts = self.starts[keep]
            self.labels = self.labels[keep]

        if limit is not None and limit < len(self.labels):
            self.rec_idx = self.rec_idx[:limit]
            self.starts = self.starts[:limit]
            self.labels = self.labels[:limit]

        if self.augment not in ("none", "jitter", "scale", "shift", "full"):
            raise ValueError(f"Unknown augmentation: {self.augment}")

    def __len__(self) -> int:
        return len(self.labels)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.augment == "none":
            return x
        x = x.copy()
        if self.augment in ("jitter", "full"):
            x += self.rng.normal(0.0, 0.01, size=x.shape).astype(np.float32)
        if self.augment in ("scale", "full"):
            x *= np.float32(self.rng.uniform(0.9, 1.1))
        if self.augment in ("shift", "full"):
            shift = int(self.rng.integers(-2, 3))
            if shift:
                if shift > 0:
                    x = np.pad(x, ((shift, 0), (0, 0)), mode="edge")[:-shift]
                else:
                    x = np.pad(x, ((0, -shift), (0, 0)), mode="edge")[-shift:]
        return x

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = int(self.rec_idx[index])
        s = int(self.starts[index])
        x = self.data[r][s : s + self.window]
        if self.training:
            x = self._augment(x)
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = int(self.labels[index])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), torch.tensor(index)

    # -- diagnostics -------------------------------------------------------
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=18)

    def summary(self) -> Dict[str, object]:
        counts = self.class_counts()
        return {
            "n_windows": int(len(self.labels)),
            "n_recordings": len(self.files),
            "recordings": self.recording_names,
            "class_counts": counts.tolist(),
            "null_fraction": float(counts[0] / max(len(self.labels), 1)),
            "window": self.window,
            "stride": self.stride,
            "augment": self.augment,
            "drop_null": bool((self.labels != 0).all() and counts[0] == 0),
        }


def make_datasets(
    data_dir: str | Path,
    split: str = "paper",
    window: int = 24,
    train_stride: int = 12,
    test_stride: int = 1,
    augment: str = "full",
    drop_null: bool = False,
    limit: Optional[int] = None,
    seed: int = 0,
    class_names: Optional[List[str]] = None,
) -> Dict[str, WindowDataset]:
    root = Path(data_dir) / split
    if not (root / "split.json").exists():
        raise FileNotFoundError(
            f"{root}/split.json missing. Run: python preprocess.py --split {split}"
        )
    return {
        "train": WindowDataset(
            root / "train",
            window=window,
            stride=train_stride,
            augment=augment,
            drop_null=drop_null,
            limit=limit,
            seed=seed,
            training=True,
            class_names=class_names,
        ),
        "val": WindowDataset(
            root / "val",
            window=window,
            stride=train_stride,
            augment="none",
            drop_null=drop_null,
            limit=limit,
            seed=seed + 1,
            training=False,
            class_names=class_names,
        ),
        "test": WindowDataset(
            root / "test",
            window=window,
            stride=test_stride,
            augment="none",
            drop_null=drop_null,
            limit=limit,
            seed=seed + 2,
            training=False,
            class_names=class_names,
        ),
    }


def make_loaders(
    datasets: Dict[str, WindowDataset],
    batch_size: int = 100,
    num_workers: int = 0,
    seed: int = 0,
    test_batch_size: int = 256,
) -> Dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=test_batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
