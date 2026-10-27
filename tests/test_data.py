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


def _loader_pair(tmp_path, **kw):
    import torch
    from data import BatchLoader

    _write_recording(tmp_path / "S1-ADL1.npz", n=300, seed=1)
    _write_recording(tmp_path / "S1-ADL2.npz", n=210, seed=2)
    ds = WindowDataset(tmp_path, window=24, stride=12, **kw)
    return ds, BatchLoader(ds, batch_size=7, device=torch.device("cpu"))


def test_batch_loader_matches_dataset(tmp_path):
    ds, loader = _loader_pair(tmp_path)
    seen = 0
    for x, y, idx in loader:
        for k, i in enumerate(idx.tolist()):
            xd, yd, _ = ds[i]
            assert np.allclose(x[k].numpy(), xd.numpy())
            assert int(y[k]) == int(yd)
        seen += len(idx)
    assert seen == len(ds)
    assert len(loader) == -(-len(ds) // 7)


def test_batch_loader_shuffle_covers_all_windows_once(tmp_path):
    import torch
    from data import BatchLoader

    ds, _ = _loader_pair(tmp_path)
    loader = BatchLoader(ds, batch_size=5, shuffle=True, seed=3, device=torch.device("cpu"))
    order = torch.cat([idx for _, _, idx in loader]).tolist()
    assert sorted(order) == list(range(len(ds)))
    assert order != list(range(len(ds)))


def test_batch_loader_jitter_and_scale(tmp_path):
    import torch
    from data import BatchLoader

    for mode in ("jitter", "scale", "full"):
        ds, _ = _loader_pair(tmp_path, augment=mode, training=True)
        clean = WindowDataset(tmp_path, window=24, stride=12)
        x, _, idx = next(iter(BatchLoader(ds, batch_size=8, device=torch.device("cpu"))))
        ref = torch.stack([clean[i][0] for i in idx.tolist()])
        assert x.shape == (8, 24, 113) and torch.isfinite(x).all()
        assert not torch.equal(x, ref)
        if mode == "jitter":
            assert 0.003 < (x - ref).abs().mean() < 0.02
        if mode == "scale":
            ratio = x / ref.clamp_min(1e-6)
            per_window = ratio.flatten(1)
            assert torch.allclose(per_window.min(1).values, per_window.max(1).values, atol=1e-3)
            assert ((per_window[:, 0] > 0.9) & (per_window[:, 0] < 1.1)).all()


def test_batch_loader_shift_matches_numpy_edge_padding(tmp_path):
    import torch
    from data import BatchLoader

    ds, _ = _loader_pair(tmp_path, augment="shift", training=True)
    loader = BatchLoader(ds, batch_size=len(ds), device=torch.device("cpu"))
    x, _, idx = next(iter(loader))
    for k, i in enumerate(idx.tolist()):
        ref = WindowDataset(tmp_path, window=24, stride=12)[i][0].numpy()
        matches = []
        for shift in range(-2, 3):
            if shift > 0:
                cand = np.pad(ref, ((shift, 0), (0, 0)), mode="edge")[:-shift]
            elif shift < 0:
                cand = np.pad(ref, ((0, -shift), (0, 0)), mode="edge")[-shift:]
            else:
                cand = ref
            matches.append(np.allclose(x[k].numpy(), cand))
        assert any(matches)
