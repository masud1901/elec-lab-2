import numpy as np
import pytest
import torch

from models import build_model, count_parameters


def _forward(model, window=24, batch=2):
    x = torch.randn(batch, window, 113)
    y = model(x)
    assert y.shape == (batch, 18)
    assert torch.isfinite(y).all()


def test_reference_param_count():
    m = build_model()
    assert 3.9e6 < count_parameters(m) < 4.1e6
    _forward(m)


def test_cnn_param_count():
    m = build_model(model="cnn")
    assert 7.3e6 < count_parameters(m) < 7.7e6
    _forward(m)


def test_gru_variant():
    m = build_model(rnn="gru")
    assert count_parameters(m) < count_parameters(build_model())
    _forward(m)


@pytest.mark.parametrize("n_conv", [0, 1, 2, 3, 4])
def test_conv_depth(n_conv):
    m = build_model(n_conv=n_conv)
    _forward(m)


@pytest.mark.parametrize("window", [20, 24, 42, 60])
def test_window_lengths(window):
    m = build_model(window=window)
    _forward(m, window=window)


def test_too_short_window_raises():
    with pytest.raises(ValueError):
        build_model(window=16, n_conv=4)
