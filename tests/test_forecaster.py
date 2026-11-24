"""Tests for the secondary forecasting feature (src/lstm_forecaster.py)."""
import os

import numpy as np
import pytest

from src import lstm_forecaster as lf


def test_default_forecast_is_recent_level_with_prediction_interval():
    rng = np.random.RandomState(0)
    w = list(80 - 0.03 * np.arange(45) + rng.normal(0, 0.3, 45))
    detail = lf.forecast_detail(w)
    assert detail["method_used"] == "level"
    assert lf.forecast(w) == pytest.approx(np.full(7, np.mean(w[-3:])))
    assert "prediction interval" in detail["interval"]
    assert np.all(detail["lo"] < detail["mean"]) and np.all(detail["mean"] < detail["hi"])


def test_linear_method_is_exact_on_a_line_and_is_the_shared_core_code():
    w = 80 + 0.1 * np.arange(30)
    assert lf.forecast(list(w), method="linear") == pytest.approx(80 + 0.1 * np.arange(30, 37), abs=1e-9)
    assert lf.forecast(list(w), method="linear") == pytest.approx(lf.linear_forecast(w[-14:]))


def test_forecast_returns_none_when_too_few_recent_readings():
    assert lf.forecast([80.0, 80.1]) is None


def test_forecast_handles_missed_days():
    w = [80.0] * 30
    w[20] = w[25] = w[28] = None
    assert lf.forecast(w) == pytest.approx(np.full(7, 80.0))


def test_centre_window_is_the_only_normalisation_and_is_invertible():
    win = np.array([80.0, 80.5, 79.5, 80.0])
    centred, mu = lf.centre_window(win)
    assert mu == pytest.approx(80.0) and centred.mean() == pytest.approx(0.0)
    assert centred + mu == pytest.approx(win)


def test_walk_forward_split_has_no_target_overlap():
    w = np.arange(90, dtype=float)                     # value == day index
    Xtr, ytr, Xte, yte = lf.walk_forward_windows(w)
    assert max(y.max() for y in ytr) < lf.TRAIN_DAYS
    assert min(y.min() for y in yte) >= lf.TRAIN_DAYS
    assert len(Xtr) > 0 and len(Xte) > 0
    assert all(len(x) == 14 for x in Xtr + Xte) and all(len(y) == 7 for y in ytr + yte)


def test_lstm_request_falls_back_with_explanation_when_window_has_gaps():
    w = list(80 - 0.05 * np.arange(30))
    w[22] = None
    d = lf.forecast_detail(w, method="lstm")
    assert d["method_used"] == "level" and d["note"]


@pytest.mark.skipif(not (lf.HAS_TENSORFLOW and os.path.exists(lf.MODEL_PATH)),
                    reason="TensorFlow or the optional retrained LSTM model is not available")
def test_optional_lstm_is_shift_equivariant():
    """Window-centred normalisation: adding 10 kg to every input adds 10 kg to every output."""
    rng = np.random.RandomState(1)
    w = 80 - 0.05 * np.arange(30) + rng.normal(0, 0.3, 30)
    a = lf.forecast_detail(list(w), method="lstm")
    b = lf.forecast_detail(list(w + 10.0), method="lstm")
    assert a["method_used"] == "lstm"
    assert np.asarray(b["mean"]) - np.asarray(a["mean"]) == pytest.approx(10.0, abs=1e-3)
