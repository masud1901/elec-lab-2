import numpy as np
import pytest
import torch

from losses import FocalLoss
from utils import compute_metrics


def test_perfect_metrics():
    y = np.arange(18)
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert m["gesture_macro_f1"] == 1.0


def test_missing_classes_are_scored_zero():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["per_class"][2]["support"] == 0
    assert m["per_class"][2]["f1"] == 0.0
    assert m["macro_f1"] < 0.2


def test_null_excluded_from_gesture_macro():
    y_true = np.array([0] * 10 + [1] * 2)
    y_pred = np.array([0] * 10 + [1] * 2)
    m = compute_metrics(y_true, y_pred)
    # All 17 gesture labels are scored; 16 are absent in this toy example.
    assert m["gesture_macro_f1"] == pytest.approx(1.0 / 17.0)
    assert m["macro_f1"] < 1.0


def test_focal_reduces_to_ce():
    torch.manual_seed(0)
    logits = torch.randn(8, 18)
    target = torch.randint(0, 18, (8,))
    ce = torch.nn.functional.cross_entropy(logits, target)
    fl = FocalLoss(gamma=0.0, alpha=None)(logits, target)
    assert torch.allclose(ce, fl, atol=1e-6)
