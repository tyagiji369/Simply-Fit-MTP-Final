"""Tests for the compatibility entry point src/ml_engine.run_pipeline (used by the coach)."""
import numpy as np
import pytest

from src.research_estimate import estimate as estimate_energy_balance
from src.ml_engine import KCAL_PER_KG, run_pipeline


def _log(kcal=-500.0, n=45, seed=3, sd=0.3):
    rng = np.random.RandomState(seed)
    return list(82 + (kcal / KCAL_PER_KG) * np.arange(n) + rng.normal(0, sd, n))


def test_output_contract_when_ok():
    out = run_pipeline(_log(), target_weekly_change=-0.5)
    for key in ["status", "message", "kcal_per_day", "kcal_per_day_ci95", "weekly_kg_change", "label",
                "reliability", "reliability_detail", "anomaly_flags", "anomalies_detected", "trend",
                "smoothed", "current_trend_weight", "latest_reading", "target_kcal_per_day", "gap_kcal"]:
        assert key in out, f"missing key: {key}"
    assert out["status"] == "ok"
    lo, hi = out["kcal_per_day_ci95"]
    assert lo < out["kcal_per_day"] < hi
    assert out["target_kcal_per_day"] == pytest.approx(-0.5 * KCAL_PER_KG / 7, abs=0.05)
    assert out["gap_kcal"] == pytest.approx(out["target_kcal_per_day"] - out["kcal_per_day"], abs=0.2)


def test_removed_unsupported_outputs_are_gone():
    out = run_pipeline(_log(), target_weekly_change=-0.5)
    for key in ["r_squared", "food_adjustment_kcal", "activity_adjustment_kcal"]:
        assert key not in out


def test_run_pipeline_agrees_with_shared_core():
    log = _log()
    out = run_pipeline(log)
    core = estimate_energy_balance(log, window_days=28)
    assert out["kcal_per_day"] == pytest.approx(core["kcal_per_day"], abs=0.05)
    assert out["weekly_kg_change"] == pytest.approx(core["weekly_kg_change"], abs=0.0005)
    assert out["label"] == core["label"]


def test_short_history_returns_no_number_but_does_not_crash():
    out = run_pipeline([80.0, 79.8, 79.9], target_weekly_change=-0.5)
    assert out["status"] == "insufficient"
    assert out["kcal_per_day"] is None and out["kcal_per_day_ci95"] is None
    assert out["gap_kcal"] is None
    assert out["current_trend_weight"] == pytest.approx(79.9)
    assert out["anomalies_detected"] == 0


def test_missing_days_supported():
    log = _log(n=40)
    for i in (5, 6, 17, 30):
        log[i] = None
    assert run_pipeline(log)["status"] == "ok"


def test_invalid_input_is_reported():
    out = run_pipeline([80, 900, 80] + [80] * 20)
    assert out["status"] == "invalid" and out["kcal_per_day"] is None
