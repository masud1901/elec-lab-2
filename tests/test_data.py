import numpy as np
import pandas as pd
import pytest

from data import WindowDataset


def _write_recording(path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 113), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[50:70] = 1
    y[100:130] = 5
    np.savez(path, x=x, y=y)
    return x, y


def test_window_labels_and_stride(tmp_path):
    x, y = _write_recording(tmp_path / "S1-ADL1.npz")
    ds = WindowDataset(tmp_path, window=24, stride=12)
    assert len(ds) == len(range(0, len(x) - 24 + 1, 12))
    for i in [0, 1, 10, len(ds) - 1]:
        xb, yb, idx = ds[i]
        assert xb.shape == (24, 113)
        start = i * 12
        assert int(yb) == int(y[start + 23])
        assert int(idx) == i
        assert np.allclose(xb.numpy(), x[start : start + 24])


def test_stride_one_and_drop_null(tmp_path):
    x, y = _write_recording(tmp_path / "S1-ADL1.npz")
    ds = WindowDataset(tmp_path, window=24, stride=1)
    assert len(ds) == len(x) - 24 + 1
    ds_no_null = WindowDataset(tmp_path, window=24, stride=1, drop_null=True)
    assert len(ds_no_null) == sum(ds.labels != 0)
    assert (ds_no_null.labels != 0).all()


def test_augmentation_keeps_shape(tmp_path):
    _write_recording(tmp_path / "S1-ADL1.npz")
    ds = WindowDataset(tmp_path, window=24, stride=12, augment="full", training=True, seed=0)
    xb, _, _ = ds[0]
    assert xb.shape == (24, 113)
    assert np.isfinite(xb.numpy()).all()


def test_short_recording_raises(tmp_path):
    _write_recording(tmp_path / "S1-ADL1.npz", n=10)
    with pytest.raises(ValueError):
        WindowDataset(tmp_path, window=24)
