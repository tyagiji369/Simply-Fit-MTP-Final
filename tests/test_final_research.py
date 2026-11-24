import numpy as np
import pytest
from research.world import simulate
from src.research_estimate import estimate, rolling
from src.state_space import state_trend
from src.ml_engine import run_pipeline


def test_matching_observations_can_hide_different_balances():
    days = np.arange(28)
    tissue_a = 76 - 0.04 * days
    tissue_b = 76 - 0.02 * days
    np.testing.assert_allclose(tissue_a, tissue_b - 0.02 * days)
    assert 7700 * (-0.04) != 5500 * (-0.02)


def test_irregular_state_covariance_and_translation():
    x = np.r_[0, 1, 2, np.arange(4, 55, 2)]
    y = 78 + 0.025 * x
    slope, se, _ = state_trend(x, y, 0.25)
    assert 0 < se < 0.05
    assert slope == pytest.approx(0.025, abs=0.002)
    assert state_trend(x, y + 10, 0.25)[:2] == pytest.approx((slope, se))
    y[10] += 4
    assert abs(state_trend(x, y, 0.25)[0] - 0.025) < 0.01


def test_simulated_split_identity_and_rerun():
    a = simulate("development", 1)
    assert np.array_equal(a[1], simulate("development", 1)[1])
    assert not np.array_equal(a[1], simulate("validation", 1)[1])


def test_gate_consistency_and_invalid_dates():
    y = np.tile([77.0, 80.0, 79.0, 74.0], 7)
    assert estimate(y)["status"] == "withheld"
    assert run_pipeline(y)["kcal_per_day"] is None
    assert np.isnan(rolling(y, 28)[-1]["kcal_per_day"])
    assert estimate([70, 71], [0, np.inf])["status"] == "invalid"
    assert estimate(np.full(90, 78))["status"] == "withheld"


def test_no_lookahead_in_rolling_output():
    y = 80 - 0.03 * np.arange(42)
    a = rolling(y, 28)
    b = rolling(np.r_[y, 95], 28)
    np.testing.assert_allclose(
        [r["kcal_per_day"] for r in a],
        [r["kcal_per_day"] for r in b[:-1]],
        equal_nan=True,
    )
