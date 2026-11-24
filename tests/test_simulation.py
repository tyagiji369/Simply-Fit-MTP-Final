"""Tests for the scenario simulator (src/simulation.py)."""
import numpy as np
import pytest

from src.simulation import MECHANISMS, demo_series, load_config, simulate_series

CFG = load_config()
SCENARIOS = sorted(CFG["scenarios"])


@pytest.mark.parametrize("mech", MECHANISMS)
@pytest.mark.parametrize("scen", SCENARIOS)
def test_reproducible_with_fixed_seed(mech, scen):
    a = simulate_series(mech, scen, "base", user=3, seed_base=123)
    b = simulate_series(mech, scen, "base", user=3, seed_base=123)
    np.testing.assert_array_equal(a.observed, b.observed)
    np.testing.assert_array_equal(a.balance, b.balance)
    c = simulate_series(mech, scen, "base", user=4, seed_base=123)
    assert not np.array_equal(a.tissue, c.tissue)


def test_development_and_evaluation_seeds_give_different_users():
    dev, ev = CFG["seeds"]["development_base"], CFG["seeds"]["evaluation_base"]
    assert dev != ev
    a = simulate_series("A", "stable", "base", 0, dev)
    b = simulate_series("A", "stable", "base", 0, ev)
    assert not np.array_equal(a.tissue, b.tissue)


def test_mechanism_A_ground_truth_is_internally_consistent():
    """In A, tissue change x 7700 must equal the summed true balance exactly."""
    s = simulate_series("A", "sustained_deficit", "clean", 1, 5)
    assert s.true_mean_balance(10, 60) == pytest.approx(s.tissue_trend_equivalent(10, 60), rel=1e-9)
    assert np.isnan(s.balance[0])
    assert s.true_mean_balance(0, 89) < -150


def test_mechanism_B_does_not_follow_the_7700_rule():
    """B is deliberately not built on the estimator's assumption."""
    ratios = []
    for u in range(40):
        s = simulate_series("B", "sustained_deficit", "clean", u, 5)
        ratios.append(s.tissue_trend_equivalent(30, 89) / s.true_mean_balance(30, 89))
    ratios = np.array(ratios)
    assert np.all(ratios > 0)
    assert np.abs(ratios - 1).max() > 0.10          # individuals differ from 7700 by >10%
    assert ratios.std() > 0.02                      # and the factor varies between people


def test_mechanism_B_balance_shrinks_as_the_body_adapts():
    first, last = [], []
    for u in range(40):
        s = simulate_series("B", "sustained_deficit", "clean", u, 5)
        first.append(s.true_mean_balance(0, 20))
        last.append(s.true_mean_balance(69, 89))
    assert np.mean(last) > np.mean(first)            # deficit becomes smaller in magnitude
    assert np.mean(last) < 0


def test_conditions_share_the_same_underlying_person():
    a = simulate_series("B", "gradual_drift", "clean", 2, 9)
    b = simulate_series("B", "gradual_drift", "outliers_10", 2, 9)
    np.testing.assert_array_equal(a.tissue, b.tissue)
    np.testing.assert_array_equal(a.balance, b.balance)
    assert not np.array_equal(a.observed, b.observed, equal_nan=True)


def test_clean_condition_has_no_gaps_or_gross_errors():
    for mech in MECHANISMS:
        s = simulate_series(mech, "stable", "clean", 0, 1)
        assert not np.isnan(s.observed).any()
        assert not s.is_outlier.any()


@pytest.mark.parametrize("mech", MECHANISMS)
def test_missing_and_outlier_rates_match_configuration(mech):
    miss, out = [], []
    for u in range(60):
        s = simulate_series(mech, "stable", "missing_25", u, 2)
        miss.append(np.isnan(s.observed).mean())
        s2 = simulate_series(mech, "stable", "outliers_10", u, 2)
        out.append(s2.is_outlier.sum() / (~np.isnan(s2.observed)).sum())
    assert np.mean(miss) == pytest.approx(0.25, abs=0.05)
    assert np.mean(out) == pytest.approx(0.10, abs=0.03)


def test_gross_errors_are_large_and_marked():
    s = simulate_series("A", "stable", "outliers_10", 0, 3)
    clean = simulate_series("A", "stable", "outliers_0", 0, 3)
    idx = np.where(s.is_outlier)[0]
    assert len(idx) > 0
    both = idx[~np.isnan(clean.observed[idx])]
    assert np.all(np.abs(s.observed[both] - clean.observed[both]) >= 0.9)


def test_readings_are_rounded_to_scale_resolution_and_plausible():
    s = simulate_series("B", "sustained_surplus", "base", 5, 4)
    obs = s.observed[~np.isnan(s.observed)]
    assert np.allclose(obs * 10, np.round(obs * 10), atol=1e-6)
    assert obs.min() > 30 and obs.max() < 200


def test_step_scenarios_change_sign_of_planned_balance():
    s = simulate_series("A", "deficit_to_surplus", "clean", 0, 6)
    cd = s.meta["change_day"]
    assert 35 <= cd <= 75
    assert s.true_mean_balance(1, cd - 1) < -200
    assert s.true_mean_balance(cd, 89) > 100


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        simulate_series("C", "stable")
    with pytest.raises(ValueError):
        simulate_series("A", "no_such_scenario")


def test_demo_series_is_shifted_to_requested_start_and_has_gaps_marked_none():
    d = demo_series(start_weight=78.0)
    vals = [v for v in d if v is not None]
    assert len(d) == 42 and abs(vals[0] - 78.0) < 2.5
    assert demo_series(start_weight=78.0) == d
