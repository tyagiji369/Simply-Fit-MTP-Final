"""
The internship (v1) estimator is kept frozen in src/legacy/ for comparison.
These tests (a) prove the frozen copies are byte-identical to the files in
the original ZIP, and (b) keep the original v1 unit tests running against
them, so the 'original' side of every comparison is the real original.
"""
import hashlib
import os

import numpy as np
import pytest

from src.legacy import app_estimator_v1 as app_v1
from src.legacy.ml_engine_v1 import (KCAL_PER_KG, detect_anomalies, estimate_calorie_balance,
                                     ewma_filter, run_pipeline)

LEGACY = os.path.join(os.path.dirname(__file__), "..", "src", "legacy")
ORIGINAL_SHA256 = {       # sha256 of the files as delivered in Simply-Fit-main-2.zip
    "ml_engine_v1.py": "88be47752c4e0cde392ba4b04090a47575d7023109c22ec527296a4ffad58964",
    "data_generator_v1.py": "0aa76eeae4de646ac1473367b276dcd22039d9c8d39782ddde1ffb4c6076e30d",
}


@pytest.mark.parametrize("name,digest", sorted(ORIGINAL_SHA256.items()))
def test_frozen_copy_is_byte_identical_to_original(name, digest):
    with open(os.path.join(LEGACY, name), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == digest


# ---- original v1 tests, unchanged except for the import path ----
def test_ewma_smooths_and_converges():
    rng = np.random.RandomState(0)
    log = 80 + rng.normal(0, 0.4, 60).cumsum() * 0
    smoothed = ewma_filter(log)
    assert len(smoothed) == len(log)
    assert np.std(np.diff(smoothed)) <= np.std(np.diff(log))
    assert smoothed[0] == pytest.approx(log[0], abs=1e-9)


def test_anomaly_flags_shape_and_rate():
    rng = np.random.RandomState(1)
    log = list(80 + rng.normal(0, 0.3, 50))
    log[10] += 3.0
    log[30] -= 2.5
    flags, smoothed = detect_anomalies(log)
    assert len(flags) == len(log)
    assert flags.dtype == bool
    assert 1 <= flags.sum() <= 10
    assert flags[10] and flags[30]


def test_calorie_balance_recovers_known_deficit():
    kg_per_day = -0.5 / 7
    days = np.arange(90)
    trend = 85.0 + kg_per_day * days
    rng = np.random.RandomState(2)
    log = trend + rng.normal(0, 0.05, 90)
    res = estimate_calorie_balance(log)
    expected = kg_per_day * KCAL_PER_KG
    assert res["kcal_per_day"] == pytest.approx(expected, abs=60)
    ci_lo, ci_hi = res["kcal_per_day_ci95"]
    assert ci_lo < res["kcal_per_day"] < ci_hi
    assert 0.0 <= res["r_squared"] <= 1.0


def test_run_pipeline_output_contract():
    rng = np.random.RandomState(3)
    log = list(82 + rng.normal(0, 0.3, 45).cumsum() * 0.01)
    out = run_pipeline(log, target_weekly_change=-0.5)
    for key in ["smoothed", "anomaly_flags", "anomalies_detected", "kcal_per_day",
                "kcal_per_day_ci95", "weekly_kg_change", "r_squared",
                "target_kcal_per_day", "gap_kcal", "food_adjustment_kcal",
                "activity_adjustment_kcal"]:
        assert key in out, f"missing key: {key}"


# ---- documented v1 defects, pinned so the comparison baseline cannot drift ----
def test_v1_shrinks_a_noise_free_two_week_trend():
    w = list(85.0 + (-550.0 / 7700) * np.arange(14))
    assert run_pipeline(w)["kcal_per_day"] == pytest.approx(-448.0, abs=1.0)   # truth is -550


def test_v1_research_and_app_paths_disagree():
    rng = np.random.RandomState(4)
    w = list(np.round(80 - 0.05 * np.arange(28) + rng.normal(0, 0.4, 28), 1))
    assert abs(run_pipeline(w)["kcal_per_day"] - app_v1.estimate_balance(w)[0]) > 1.0


def test_v1_treats_skipped_days_as_consecutive():
    w = 85.0 + (-550.0 / 7700) * np.arange(42)
    every_other = list(w[::2])                       # half the days missing
    assert app_v1.estimate_balance(every_other)[0] < -1000     # roughly double the true -550
