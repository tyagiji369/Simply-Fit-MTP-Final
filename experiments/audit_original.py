"""
Audit of the ORIGINAL (internship, v1) Simply-Fit code.

Every check below runs the frozen original code in src/legacy/ on small,
fully specified inputs and prints/saves a measured number. Nothing here is a
claim about the corrected code; it is the evidence base for
CORRECTIONS_REPORT.md.

Run from the repository root:
    python -m experiments.audit_original
Writes results/audit/audit_original.json
"""

import json
import os

import numpy as np

from src.legacy import app_estimator_v1 as app_v1
from src.legacy import ml_engine_v1 as ml_v1
from src.legacy import data_generator_v1 as gen_v1
from src.legacy import lstm_forecaster_v1 as lstm_v1

KCAL = 7700
OUT = os.path.join("results", "audit", "audit_original.json")


def _ar1(rng, n, phi, sd):
    """Stationary AR(1) noise with marginal standard deviation `sd`."""
    e = np.empty(n)
    e[0] = rng.normal(0, sd)
    innov = sd * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0, innov)
    return e


def check_ewma_attenuation():
    """A1. Noise-free straight line: any error is pure method bias."""
    true_kcal = -550.0
    rows = []
    for n in (14, 21, 28, 56, 90):
        w = 85.0 + (true_kcal / KCAL) * np.arange(n)
        research = ml_v1.run_pipeline(list(w))["kcal_per_day"]
        app = app_v1.estimate_balance(list(w))[0]
        rows.append({
            "n_days": n, "true_kcal": true_kcal,
            "v1_research_kcal": float(research),
            "v1_research_bias_pct": round(100 * (research - true_kcal) / abs(true_kcal), 1),
            "v1_app_kcal": round(float(app), 1),
            "v1_app_bias_pct": round(100 * (app - true_kcal) / abs(true_kcal), 1),
        })
    return rows


def check_two_code_paths(n_series=300, n=28):
    """A2. The research module and the app compute different numbers for the same log."""
    rng = np.random.RandomState(101)
    diffs = []
    for _ in range(n_series):
        slope = rng.uniform(-600, 400) / KCAL
        w = np.round(80 + slope * np.arange(n) + _ar1(rng, n, 0.5, 0.4), 1)
        a = ml_v1.run_pipeline(list(w))["kcal_per_day"]
        b = app_v1.estimate_balance(list(w))[0]
        diffs.append(a - b)
    diffs = np.array(diffs)
    return {"n_series": n_series, "n_days": n,
            "mean_abs_disagreement_kcal": round(float(np.mean(np.abs(diffs))), 1),
            "p95_abs_disagreement_kcal": round(float(np.percentile(np.abs(diffs), 95)), 1),
            "max_abs_disagreement_kcal": round(float(np.max(np.abs(diffs))), 1)}


def check_missing_days():
    """A3. The original takes a bare list: skipped days silently compress time."""
    true_kcal = -550.0
    rng = np.random.RandomState(7)
    rows = []
    for miss in (0.0, 0.2, 0.4):
        est = []
        for _ in range(200):
            n = 42
            w = 85.0 + (true_kcal / KCAL) * np.arange(n)      # noise-free on purpose
            keep = rng.rand(n) >= miss
            keep[0] = keep[-1] = True
            est.append(app_v1.estimate_balance(list(w[keep]))[0])
        rows.append({"missing_rate": miss, "true_kcal": true_kcal,
                     "v1_app_mean_kcal": round(float(np.mean(est)), 1),
                     "bias_pct": round(100 * (np.mean(est) - true_kcal) / abs(true_kcal), 1)})
    return rows


def check_if_flags_clean_data(n_series=300, n=28):
    """A4. contamination=0.1 flags ~10% of readings even when there are NO errors."""
    rng = np.random.RandomState(11)
    frac, stable = [], []
    for _ in range(n_series):
        w = 80 + rng.normal(0, 0.3, n)                        # pure Gaussian scatter
        flags, _ = ml_v1.detect_anomalies(list(w))
        frac.append(flags.mean())
        # are yesterday's flags still the same after one more reading arrives?
        w2 = np.append(w, 80 + rng.normal(0, 0.3))
        flags2, _ = ml_v1.detect_anomalies(list(w2))
        stable.append(np.array_equal(flags, flags2[:-1]))
    return {"n_series": n_series, "n_days": n,
            "mean_fraction_flagged_on_error_free_data": round(float(np.mean(frac)), 3),
            "share_of_logs_whose_past_flags_change_after_one_new_reading":
                round(1 - float(np.mean(stable)), 3)}


def check_ci_coverage(n_series=2000, n=28):
    """A5. Empirical coverage of the original 'kcal_per_day_ci95'."""
    out = {}
    for label, phi in (("white_noise", 0.0), ("ar1_phi_0.6", 0.6)):
        rng = np.random.RandomState(21)
        hit_true, hit_own_mean, width, ests = 0, 0, [], []
        res = []
        for _ in range(n_series):
            true_kcal = rng.uniform(-700, 500)
            w = 80 + (true_kcal / KCAL) * np.arange(n) + _ar1(rng, n, phi, 0.4)
            r = ml_v1.estimate_calorie_balance(w)
            lo, hi = r["kcal_per_day_ci95"]
            res.append((true_kcal, r["kcal_per_day"], lo, hi))
        res = np.array(res)
        out[label] = {
            "n_series": n_series, "n_days": n, "nominal": 0.95,
            "empirical_coverage_of_truth": round(float(np.mean((res[:, 2] <= res[:, 0]) & (res[:, 0] <= res[:, 3]))), 3),
            "mean_ci_width_kcal": round(float(np.mean(res[:, 3] - res[:, 2])), 1),
            "actual_rmse_kcal": round(float(np.sqrt(np.mean((res[:, 1] - res[:, 0]) ** 2))), 1),
        }
    return out


def check_rolling_5day(n_series=500, n=30):
    """A6. The 5-day endpoint 'daily balance' chart on a person at exact maintenance."""
    rng = np.random.RandomState(31)
    vals = []
    for _ in range(n_series):
        w = np.round(80 + _ar1(rng, n, 0.5, 0.4), 1)          # true balance = 0 every day
        vals.extend(app_v1.rolling_5day_balance(list(w)))
    vals = np.array(vals)
    return {"true_kcal_every_day": 0.0,
            "sd_of_displayed_bars_kcal": round(float(vals.std()), 0),
            "share_of_bars_beyond_plus_minus_500_kcal": round(float(np.mean(np.abs(vals) > 500)), 3)}


def check_on_track_message():
    """A7. Replicates the tracker-page branch logic of the original app verbatim."""
    def original_message(weekly_kg, target_weekly, goal):
        on_track = abs(weekly_kg - target_weekly) < 0.15
        if on_track:
            return "on track"
        elif weekly_kg < target_weekly and goal == "lose":
            return "Slightly behind target. Try reducing kcal"
        else:
            return "Ahead of schedule. Maintain your current pace."
    return [
        {"goal": "lose", "target_kg_per_week": -0.5, "actual_kg_per_week": -0.9,
         "situation": "losing FASTER than planned",
         "original_message": original_message(-0.9, -0.5, "lose")},
        {"goal": "lose", "target_kg_per_week": -0.5, "actual_kg_per_week": +0.2,
         "situation": "GAINING weight on a loss goal",
         "original_message": original_message(0.2, -0.5, "lose")},
    ]


def check_fallback_forecast():
    """A9. 'OLS on the last 7 readings' fallback is not OLS and mis-scales the slope."""
    slope = 0.10
    w = list(80 + slope * np.arange(30))                      # exact line, no noise
    pred = lstm_v1._linear_extrapolate(w)
    true_next7 = 80 + slope * np.arange(30, 37)
    evaluated_baseline = lstm_v1._linear_baseline(np.array(w[-14:]))
    return {"true_slope_kg_per_day": slope,
            "fallback_implied_slope": round(float(pred[1] - pred[0]), 4),
            "fallback_day7_error_kg": round(float(pred[-1] - true_next7[-1]), 3),
            "evaluated_linear_baseline_day7_error_kg": round(float(evaluated_baseline[-1] - true_next7[-1]), 6),
            "note": "fallback divides a 6-day difference by 7 and uses only 2 of the 7 points; "
                    "the baseline reported in results/model_evaluation.json is a 14-day OLS fit"}


def _instrumented_generate_user(user_id, random_state, days=90):
    """
    Line-for-line copy of the original generate_user() that ALSO returns the
    hidden true-weight path, so the true mean energy balance is known. The
    caller verifies that it reproduces the original weight_log exactly.
    """
    rng = np.random.RandomState(random_state)
    F = gen_v1.NHANES_FIT
    gender = "Female" if rng.rand() < F["p_female"] else "Male"
    age = int(np.clip(rng.normal(F["age_mean"], F["age_std"]), 18, 80))
    h_mean, h_std = F["height"][gender]
    height = rng.normal(h_mean, h_std)
    w_mu, w_sigma = F["weight_lognormal"][gender]
    start_weight = np.clip(rng.lognormal(w_mu, w_sigma), 32, 220)
    rng.choice(["sedentary", "light", "moderate", "active", "very_active"])
    goal = rng.choice(["lose", "gain", "maintain"], p=[0.55, 0.25, 0.20])
    disease = rng.choice(gen_v1.DISEASE_OPTIONS, p=[0.40, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08])
    adherence = rng.uniform(0.4, 1.0)
    adapt = rng.uniform(0.02, 0.10)
    if goal == "lose":
        target = rng.uniform(-600, -200)
    elif goal == "gain":
        target = rng.uniform(200, 500)
    else:
        target = 0.0
    true_w, log, deltas = start_weight, [], []
    for day in range(days):
        adapted = target * (1 - adapt * (day / days))
        actual = adapted if rng.rand() < adherence else rng.uniform(-100, 200)
        true_w += actual / KCAL
        deltas.append(actual)
        g, s, m = rng.normal(0, 0.30), rng.normal(0, 0.20), rng.normal(0, 0.12)
        if disease in ["Hypertension", "CKD", "NAFLD"]:
            s *= 1.5
        if disease in ["PCOS", "Hypothyroidism"]:
            g *= 1.3
        log.append(round(true_w + g + s + m, 1))
    return log, np.array(deltas), target


def check_circular_validation(n_users=200):
    """
    A11. README: 'recovers realized energy balance with median error 8 kcal/day'.
    Notebook 02 defined 'realized' as np.polyfit(observed weights) * 7700, i.e.
    another estimate from the SAME noisy readings, not the simulator's truth.
    Here the simulator's actual mean daily delta is used instead.
    """
    err_true, err_circular, identical = [], [], True
    for i in range(n_users):
        seed = 5000 + i
        log, deltas, _ = _instrumented_generate_user(i, seed)
        identical &= (log == gen_v1.generate_user(i, random_state=seed)["weight_log"])
        est = ml_v1.run_pipeline(log)["kcal_per_day"]
        err_true.append(est - deltas[1:].mean())
        err_circular.append(est - np.polyfit(range(len(log)), log, 1)[0] * KCAL)
    err_true, err_circular = np.array(err_true), np.array(err_circular)
    return {"n_users": n_users, "n_days": 90,
            "instrumented_copy_reproduces_original_logs_exactly": bool(identical),
            "median_abs_error_vs_polyfit_of_same_data_kcal": round(float(np.median(np.abs(err_circular))), 1),
            "median_abs_error_vs_TRUE_simulated_balance_kcal": round(float(np.median(np.abs(err_true))), 1),
            "p90_abs_error_vs_TRUE_simulated_balance_kcal": round(float(np.percentile(np.abs(err_true), 90)), 1),
            "bias_vs_TRUE_kcal": round(float(err_true.mean()), 1)}


def check_generator_noise_structure(n_users=200):
    """A12. The original simulator's day-to-day scatter is white (no water-weight memory)."""
    r1 = []
    for i in range(n_users):
        log = np.array(gen_v1.generate_user(i, random_state=900 + i)["weight_log"])
        t = np.arange(len(log))
        resid = log - np.polyval(np.polyfit(t, log, 1), t)
        r1.append(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return {"n_users": n_users,
            "mean_lag1_autocorrelation_of_detrended_readings": round(float(np.mean(r1)), 3),
            "note": "about 0 => scatter is independent day to day; no missing days, no gross errors, "
                    "one near-linear trajectory shape"}


def main():
    report = {
        "A1_ewma_attenuation_noise_free_line": check_ewma_attenuation(),
        "A2_two_code_paths_disagree": check_two_code_paths(),
        "A3_missing_days_time_compression": check_missing_days(),
        "A4_isolation_forest_on_error_free_data": check_if_flags_clean_data(),
        "A5_original_ci_coverage": check_ci_coverage(),
        "A6_rolling_5day_bars_at_true_maintenance": check_rolling_5day(),
        "A7_on_track_message_logic": check_on_track_message(),
        "A9_fallback_forecast": check_fallback_forecast(),
        "A11_circular_validation_of_readme_claim": check_circular_validation(),
        "A12_original_generator_noise": check_generator_noise_structure(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
