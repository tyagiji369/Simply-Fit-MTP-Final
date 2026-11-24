"""Unit tests for the shared estimation core (src/energy_balance.py)."""
import numpy as np
import pytest
import statsmodels.api as sm

from src import energy_balance as eb

KCAL = eb.KCAL_PER_KG


def line(kcal_per_day, n=28, start=80.0):
    return start + (kcal_per_day / KCAL) * np.arange(n)


def ar1(rng, n, phi, sd):
    e = np.empty(n)
    e[0] = rng.normal(0, sd)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
    return e


# ---------------------------------------------------------------- units / sign
def test_unit_conversion_one_kg_per_week_is_1100_kcal_per_day():
    w = 80.0 - (1.0 / 7.0) * np.arange(28)          # exactly -1 kg/week
    res = eb.estimate_energy_balance(w)
    assert res["weekly_kg_change"] == pytest.approx(-1.0, abs=1e-9)
    assert res["kcal_per_day"] == pytest.approx(-7700 / 7, abs=1e-6)
    assert res["kg_per_day"] * KCAL == pytest.approx(res["kcal_per_day"])
    lo, hi = res["kg_per_day_ci"]
    assert res["kcal_ci"] == pytest.approx((lo * KCAL, hi * KCAL))


def test_custom_conversion_factor_scales_linearly():
    w = line(-500)
    a = eb.estimate_energy_balance(w)["kcal_per_day"]
    b = eb.estimate_energy_balance(w, kcal_per_kg=3850.0)["kcal_per_day"]
    assert b == pytest.approx(a / 2)


@pytest.mark.parametrize("kcal,label", [(-500.0, "deficit"), (400.0, "surplus")])
def test_sign_convention_and_labels(kcal, label):
    rng = np.random.RandomState(0)
    res = eb.estimate_energy_balance(line(kcal, 42) + rng.normal(0, 0.15, 42))
    assert np.sign(res["kcal_per_day"]) == np.sign(kcal)
    assert np.sign(res["weekly_kg_change"]) == np.sign(kcal)
    assert res["point_label"] == label
    assert res["label"] == label


def test_constant_weight_is_zero_balance_and_maintenance():
    res = eb.estimate_energy_balance([75.0] * 30)
    assert res["status"] == "ok"
    assert res["kcal_per_day"] == pytest.approx(0.0, abs=1e-9)
    assert res["label"] == "maintenance"
    assert np.isfinite(res["kcal_ci"]).all()


@pytest.mark.parametrize("true_kcal", [-770.0, -385.0, 0.0, 385.0])
@pytest.mark.parametrize("n", [14, 28, 56])
def test_known_linear_trend_recovered_exactly_without_noise(true_kcal, n):
    """No smoothing-induced shrinkage: a noise-free line is recovered at any length."""
    res = eb.estimate_energy_balance(line(true_kcal, n))
    assert res["kcal_per_day"] == pytest.approx(true_kcal, abs=1e-6)


def test_classification_rules():
    assert eb.classify_point(-101) == "deficit"
    assert eb.classify_point(100) == "maintenance"
    assert eb.classify_point(101) == "surplus"
    assert eb.classify_with_uncertainty(-400, -600, -200) == "deficit"
    assert eb.classify_with_uncertainty(-400, -900, 100) == "inconclusive"   # CI crosses zero
    assert eb.classify_with_uncertainty(300, 50, 550) == "surplus"
    assert eb.classify_with_uncertainty(20, -150, 190) == "maintenance"
    assert eb.classify_with_uncertainty(20, -400, 440) == "inconclusive"     # too wide to call
    assert eb.classify_with_uncertainty(-400, -600, -200, n_calendar_days=20) == "inconclusive"
    assert eb.classify_with_uncertainty(-400, -600, -200, n_calendar_days=25) == "deficit"


# ---------------------------------------------------------------- anomalies
def test_gross_errors_do_not_move_robust_estimate_much():
    rng = np.random.RandomState(1)
    w = line(-400, 28) + rng.normal(0, 0.2, 28)
    clean = eb.estimate_energy_balance(w)
    bad = w.copy()
    bad[3] += 3.0
    bad[25] -= 2.5                                    # two gross errors near the ends
    rob = eb.estimate_energy_balance(bad)
    ols = eb.fit_line(np.arange(28), bad, "ols")["slope"] * KCAL
    shift_robust = abs(rob["kcal_per_day"] - clean["kcal_per_day"])
    shift_ols = abs(ols - clean["kcal_per_day"])
    assert shift_ols > 150                             # plain least squares is dragged a long way
    assert shift_robust < shift_ols / 3                # Huber limits (does not eliminate) the damage
    assert rob["outlier_mask"][3] and rob["outlier_mask"][25]
    assert rob["n_outliers"] == 2
    assert rob["outlier_sensitivity_kcal"] > 100


def test_no_reading_is_ever_replaced_or_removed():
    w = list(line(-300, 28))
    w[10] += 4.0
    res = eb.estimate_energy_balance(w)
    assert len(res["fit_weights"]) == 28
    assert res["fit_weights"][10] == pytest.approx(w[10])   # the reading itself is untouched
    assert res["robust_weights"][10] < 0.5                  # it is only given less weight


def test_clean_gaussian_data_flags_few_readings():
    """The original flagged ~10% of readings on error-free data by construction."""
    rng = np.random.RandomState(2)
    frac = [eb.estimate_energy_balance(80 + rng.normal(0, 0.3, 28))["outlier_mask"].mean()
            for _ in range(200)]
    assert np.mean(frac) < 0.04


def test_huber_matches_statsmodels_rlm():
    rng = np.random.RandomState(3)
    x = np.arange(40.0)
    y = 80 - 0.05 * x + rng.normal(0, 0.3, 40)
    y[[5, 17, 30]] += [2.5, -3.0, 2.0]
    mine = eb.fit_line(x, y, "huber")
    ref = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT(eb.HUBER_K)).fit(scale_est="mad")
    assert mine["slope"] == pytest.approx(ref.params[1], rel=1e-5)
    assert mine["intercept"] == pytest.approx(ref.params[0], rel=1e-6)


# ---------------------------------------------------------------- missing / irregular
def test_missing_days_do_not_bias_the_slope():
    true = -550.0
    w = list(line(true, 42))
    rng = np.random.RandomState(4)
    for i in rng.choice(np.arange(1, 41), 16, replace=False):
        w[i] = None                                   # 38% of days missing
    res = eb.estimate_energy_balance(w)
    assert res["kcal_per_day"] == pytest.approx(true, abs=1e-6)
    assert res["n_missing_days"] == 16
    assert res["missing_fraction"] == pytest.approx(16 / 42)


def test_explicit_irregular_days_equal_grid_with_gaps():
    days = np.array([0, 1, 2, 5, 6, 9, 10, 11, 15, 16, 20, 21, 22, 27])
    w = 80 - 0.07 * days
    grid = [None] * 28
    for d, v in zip(days, w):
        grid[d] = v
    a = eb.estimate_energy_balance(w, days)
    b = eb.estimate_energy_balance(grid)
    assert a["kcal_per_day"] == pytest.approx(b["kcal_per_day"])
    assert a["kcal_per_day"] == pytest.approx(-0.07 * KCAL, abs=1e-6)


def test_unsorted_days_are_sorted_with_warning():
    days = np.arange(20)[::-1]
    w = 80 - 0.05 * days
    log = eb.validate_weight_log(w, days)
    assert log.ok and any("sorted" in m for m in log.warnings)
    assert eb.estimate_energy_balance(w, days)["kcal_per_day"] == pytest.approx(-0.05 * KCAL, abs=1e-6)


# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("bad", [
    [80, 81, "abc", 80] + [80] * 20,
    [80, 800, 80] + [80] * 20,                        # wrong units / typo
    [80, 5, 80] + [80] * 20,
    [80, float("inf")] + [80] * 20,
    [],
    [None, None, None],
])
def test_invalid_inputs_are_rejected_without_a_number(bad):
    res = eb.estimate_energy_balance(bad)
    assert res["status"] == "invalid"
    assert res["kcal_per_day"] is None and res["kcal_ci"] is None
    assert res["errors"]


def test_duplicate_day_and_length_mismatch_are_errors():
    assert not eb.validate_weight_log([80, 81, 82], [0, 1, 1]).ok
    assert not eb.validate_weight_log([80, 81, 82], [0, 1]).ok
    assert not eb.validate_weight_log([80, 81], [0, 1.5]).ok


def test_large_jump_is_warned_not_rejected():
    log = eb.validate_weight_log([80, 80.2, 84.5, 80.1])
    assert log.ok and any("3 kg" in m for m in log.warnings)


# ---------------------------------------------------------------- short histories
@pytest.mark.parametrize("n", [1, 5, 9, 13])
def test_short_history_gives_no_number(n):
    res = eb.estimate_energy_balance(list(line(-500, n)))
    assert res["status"] == "insufficient"
    assert res["kcal_per_day"] is None and res["label"] is None
    assert "Not enough data" in res["message"]
    assert eb.assess_reliability(res)["level"] == "insufficient"


def test_enough_readings_but_too_short_a_span_is_insufficient():
    assert eb.estimate_energy_balance([80.0] * 12)["status"] == "insufficient"       # 12 days
    assert eb.estimate_energy_balance([80.0] * 14)["status"] == "ok"                 # 14 days
    sparse = [80.0 if i % 3 == 0 else None for i in range(22)]                       # 8 readings
    assert eb.estimate_energy_balance(sparse)["status"] == "insufficient"


def test_window_uses_only_the_last_days():
    w = np.concatenate([line(-700, 40), line(-700, 40)[-1] + (300 / KCAL) * np.arange(1, 29)])
    recent = eb.estimate_energy_balance(w, window_days=28)
    whole = eb.estimate_energy_balance(w)
    assert recent["kcal_per_day"] == pytest.approx(300, abs=1e-6)
    assert whole["kcal_per_day"] < 0
    assert recent["n_obs"] == 28


# ---------------------------------------------------------------- intervals
def test_classical_se_matches_statsmodels_ols():
    rng = np.random.RandomState(5)
    x = np.arange(30.0)
    y = 80 - 0.04 * x + rng.normal(0, 0.3, 30)
    fit = eb.fit_line(x, y, "ols")
    se, dof, _ = eb.slope_se(x, y, fit, "ols", "classical")
    ref = sm.OLS(y, sm.add_constant(x)).fit()
    assert fit["slope"] == pytest.approx(ref.params[1])
    assert se == pytest.approx(ref.bse[1], rel=1e-9)
    assert dof == 28


def test_hac_se_matches_statsmodels_newey_west():
    rng = np.random.RandomState(6)
    x = np.arange(60.0)
    y = 80 - 0.04 * x + ar1(rng, 60, 0.6, 0.4)
    fit = eb.fit_line(x, y, "ols")
    se, _, _ = eb.slope_se(x, y, fit, "ols", "hac")
    L = int(np.floor(4 * (60 / 100.0) ** (2.0 / 9.0)))
    ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})
    assert se == pytest.approx(ref.bse[1], rel=1e-6)


def test_ar1_interval_reduces_to_classical_when_rho_is_zero():
    x = np.arange(30.0)
    y = 80 - 0.04 * x + np.tile([0.3, -0.3], 15)     # strongly negative lag-1 correlation -> rho clipped to 0
    fit = eb.fit_line(x, y, "ols")
    se_c, _, _ = eb.slope_se(x, y, fit, "ols", "classical")
    se_a, _, rho = eb.slope_se(x, y, fit, "ols", "ar1")
    assert rho == 0.0
    assert se_a == pytest.approx(se_c)


def test_ar1_interval_is_wider_than_classical_under_positive_autocorrelation():
    rng = np.random.RandomState(7)
    wider = 0
    for _ in range(200):
        y = 80 + ar1(rng, 28, 0.7, 0.4)
        x = np.arange(28.0)
        fit = eb.fit_line(x, y, "ols")
        wider += eb.slope_se(x, y, fit, "ols", "ar1")[0] > eb.slope_se(x, y, fit, "ols", "classical")[0]
    assert wider > 190


def test_interval_coverage_on_autocorrelated_noise_beats_classical():
    """Empirical check with known truth: AR(1)-aware interval covers far more often."""
    rng = np.random.RandomState(8)
    x = np.arange(28.0)
    hit_ar1 = hit_cls = 0
    n_rep = 600
    from scipy import stats
    for _ in range(n_rep):
        true = rng.uniform(-600, 400)
        y = 80 + (true / KCAL) * x + ar1(rng, 28, 0.6, 0.4)
        res = eb.estimate_energy_balance(y)
        hit_ar1 += res["kcal_ci"][0] <= true <= res["kcal_ci"][1]
        fit = eb.fit_line(x, y, "ols")
        se, dof, _ = eb.slope_se(x, y, fit, "ols", "classical")
        hw = stats.t.ppf(0.975, dof) * se * KCAL
        hit_cls += abs(fit["slope"] * KCAL - true) <= hw
    assert hit_cls / n_rep < 0.80          # classical interval is badly anti-conservative here
    assert hit_ar1 / n_rep > 0.86          # corrected interval is much closer to nominal
    assert hit_ar1 > hit_cls


def test_estimate_lies_inside_its_interval_and_halfwidth_is_consistent():
    rng = np.random.RandomState(9)
    res = eb.estimate_energy_balance(line(-300, 35) + rng.normal(0, 0.3, 35))
    lo, hi = res["kcal_ci"]
    assert lo < res["kcal_per_day"] < hi
    assert res["kcal_ci_halfwidth"] == pytest.approx((hi - lo) / 2)


def test_block_bootstrap_interval_is_reproducible_and_contains_estimate():
    rng = np.random.RandomState(10)
    x = np.arange(40.0)
    y = 80 - 0.05 * x + rng.normal(0, 0.3, 40)
    a = eb.block_bootstrap_slope_ci(x, y, seed=3)
    b = eb.block_bootstrap_slope_ci(x, y, seed=3)
    assert a == b
    assert a[0] < eb.fit_line(x, y, "huber")["slope"] < a[1]


# ---------------------------------------------------------------- real-time use
def test_rolling_estimates_never_use_future_readings():
    rng = np.random.RandomState(11)
    w = list(line(-400, 60) + rng.normal(0, 0.3, 60))
    first = eb.rolling_energy_balance(w[:45], window_days=28)
    later = eb.rolling_energy_balance(w, window_days=28)
    for a, b in zip(first, later[:45]):
        assert a == b                    # adding days 46-60 changes nothing already reported
    assert first[10]["kcal_per_day"] is None and first[40]["kcal_per_day"] is not None


# ---------------------------------------------------------------- reliability
def test_reliability_levels_follow_interval_width():
    rng = np.random.RandomState(12)
    long_clean = eb.estimate_energy_balance(line(-400, 56) + rng.normal(0, 0.15, 56))
    short_noisy = eb.estimate_energy_balance(line(-400, 28) + ar1(rng, 28, 0.7, 0.9))
    r1, r2 = eb.assess_reliability(long_clean), eb.assess_reliability(short_noisy)
    assert r1["level"] == "high" and long_clean["kcal_ci_halfwidth"] <= eb.RELIABILITY_HIGH_HALFWIDTH
    assert r2["level"] == "low" and short_noisy["kcal_ci_halfwidth"] > eb.RELIABILITY_MODERATE_HALFWIDTH
    names = {f["name"] for f in r1["factors"]}
    assert {"Estimated uncertainty", "Readings used", "Missed days", "Day-to-day scatter",
            "Sensitivity to unusual readings"} <= names


def test_reliability_is_downgraded_by_many_missing_days():
    w = list(line(-400, 56))
    full = eb.assess_reliability(eb.estimate_energy_balance(w))
    for i in range(1, 55):
        if i % 2:
            w[i] = None
    gappy_res = eb.estimate_energy_balance(w)
    gappy = eb.assess_reliability(gappy_res)
    assert gappy_res["missing_fraction"] > eb.MISSING_WARN_FRACTION
    assert full["level"] == "high" and gappy["level"] == "moderate"
    assert [f for f in gappy["factors"] if f["name"] == "Missed days"][0]["status"] == "warn"


def test_reliability_has_no_percentage_score():
    rel = eb.assess_reliability(eb.estimate_energy_balance(line(-400, 30)))
    assert set(rel) == {"level", "summary", "factors"}
    assert rel["level"] in {"low", "moderate", "high"}


# ---------------------------------------------------------------- minimum history for a label
def test_no_direction_label_below_25_days_even_with_a_clear_trend():
    w = line(-900, 24)                                    # steep, noise-free, but only 24 days
    res = eb.estimate_energy_balance(w)
    assert res["status"] == "ok" and res["kcal_per_day"] == pytest.approx(-900, abs=1e-6)
    assert res["label"] == "inconclusive" and res["too_short_for_label"]
    assert res["label_ungated"] == "deficit"
    rel = eb.assess_reliability(res)
    assert rel["level"] == "low"
    assert [f for f in rel["factors"] if f["name"] == "Readings used"][0]["status"] == "bad"
    assert eb.estimate_energy_balance(line(-900, 25))["label"] == "deficit"


# ---------------------------------------------------------------- forecast (secondary)
def test_level_forecast_is_mean_of_last_three_readings():
    w = [80.0] * 20 + [79.0, 80.0, 81.0]
    fc = eb.forecast_weight(w)
    assert fc["method"] == "level"
    assert fc["mean"] == pytest.approx(np.full(7, 80.0))
    assert list(fc["days"]) == list(range(23, 30))


def test_linear_forecast_extends_an_exact_line_exactly():
    w = 80 + 0.1 * np.arange(30)
    fc = eb.forecast_weight(w, method="linear")
    assert fc["mean"] == pytest.approx(80 + 0.1 * np.arange(30, 37), abs=1e-9)


def test_prediction_interval_reflects_scatter_and_is_wider_than_trend_ci():
    rng = np.random.RandomState(13)
    w = line(-300, 42) + rng.normal(0, 0.4, 42)
    fc = eb.forecast_weight(w)
    assert fc["lo"] is not None
    width = fc["hi"] - fc["lo"]
    assert np.all(width > 2 * 0.4)                               # at least +-1 SD of day-to-day scatter
    assert np.all((fc["lo"] < fc["mean"]) & (fc["mean"] < fc["hi"]))
    res = eb.estimate_energy_balance(w, window_days=28)
    trend_ci_width_kg_over_week = (res["kg_per_day_ci"][1] - res["kg_per_day_ci"][0]) * 7
    assert width.min() > trend_ci_width_kg_over_week             # prediction interval != confidence interval


def test_prediction_interval_is_empirically_calibrated_on_ar1_noise():
    rng = np.random.RandomState(14)
    hits = total = 0
    for _ in range(300):
        w = line(rng.uniform(-500, 300), 67) + ar1(rng, 67, 0.6, 0.4)
        fc = eb.forecast_weight(w[:60])
        hits += np.sum((fc["lo"] <= w[60:67]) & (w[60:67] <= fc["hi"]))
        total += 7
    assert 0.72 < hits / total < 0.90                            # nominal 0.80


def test_forecast_without_enough_history_has_no_band_and_stale_logs_give_none():
    short = eb.forecast_weight([80.0, 80.2, 79.9, 80.1, 80.0])
    assert short is not None and short["lo"] is None and short["hi"] is None
    assert eb.forecast_weight([80.0, 80.1]) is None
    stale = [80.0] * 20 + [None] * 12 + [80.0]                   # last 3 readings span > 7 days
    assert eb.forecast_weight(stale) is None
    assert eb.forecast_weight([80.0] * 5, method="linear") is None
