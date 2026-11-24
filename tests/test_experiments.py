"""Reproducibility and fairness checks for the experiment pipeline (experiments/)."""
import numpy as np
import pandas as pd

from experiments import methods as M
from experiments import run_experiments as R
from src.legacy import app_estimator_v1 as app_v1
from src.legacy import ml_engine_v1 as ml_v1
from src.simulation import simulate_series


def _task(seed=R.EVAL_SEED, legacy=True):
    return (seed, "B", "deficit_to_maintenance", "base", 1, ("last28", "last56"), legacy)


def test_experiment_rows_are_reproducible_with_fixed_seeds():
    a = pd.DataFrame(R._series_task(_task()))
    b = pd.DataFrame(R._series_task(_task()))
    pd.testing.assert_frame_equal(a, b)
    assert set(a.window) == {"last28", "last56"}


def test_development_and_evaluation_seed_sets_are_disjoint():
    assert R.DEV_SEED != R.EVAL_SEED
    a = pd.DataFrame(R._series_task(_task(R.DEV_SEED, False)))
    b = pd.DataFrame(R._series_task(_task(R.EVAL_SEED, False)))
    assert not np.allclose(a.truth_balance, b.truth_balance)
    assert not any(c.startswith("est_legacy") for c in a.columns)     # legacy code is never run on dev data


def test_all_methods_receive_identical_readings_and_truth():
    """The 'original' columns are the frozen original functions applied to the same window."""
    row = R._series_task(_task())[0]
    sim = simulate_series("B", "deficit_to_maintenance", "base", 1, R.EVAL_SEED, R.CFG)
    a, b = R.WINDOWS[row["window"]]
    days, w = R._window_data(sim, a, b)
    assert row["n_obs"] == len(w)
    assert row["est_legacy_research"] == ml_v1.run_pipeline(list(w))["kcal_per_day"]
    assert row["est_legacy_app"] == app_v1.estimate_balance(list(w))[0]
    assert row["est_huber"] == M.huber(days, w)
    assert row["truth_balance"] == sim.true_mean_balance(a, b)


def test_attenuation_factor_of_original_smoothing():
    """Shrinkage of a noise-free slope by EWMA(span 7)-then-regression, by window length."""
    k14, k28, k56 = (M.ewma_attenuation(np.arange(n, dtype=float)) for n in (14, 28, 56))
    assert k14 < k28 < k56 < 1.0
    assert abs(k14 - 0.806) < 0.005 and abs(k28 - 0.931) < 0.005


def test_tracking_rows_use_only_data_up_to_each_day():
    rows = R._tracking_task((R.EVAL_SEED, "A", "deficit_to_surplus", "base", 2, False))
    sim = simulate_series("A", "deficit_to_surplus", "base", 2, R.EVAL_SEED, R.CFG)
    r = rows[5]
    t = r["day"]
    days, w = R._window_data(sim, 0, t)
    keep = days > t - 28
    assert r["est_huber_trailing_28"] == M.huber(days[keep], w[keep])
    assert r["truth_recent"] == sim.true_mean_balance(t - R.TRACK_TRUTH_DAYS, t)
