"""
Estimation entry point used by the coach (and kept for backward
compatibility with the internship code's `run_pipeline` name).

All mathematics lives in src/energy_balance.py. This file only reshapes
that result into the dictionary the coach expects.

What changed from the internship version (kept verbatim in
src/legacy/ml_engine_v1.py for comparison):

  * EWMA smoothing before regression            -> removed (it shrank the slope
                                                   and broke the interval).
  * Isolation Forest with contamination=0.1     -> removed (flagged ~10% of
    and replacement of flagged readings            readings even on error-free
                                                   data). Replaced by Huber
                                                   regression, which down-weights
                                                   unusual readings and never
                                                   overwrites a measurement.
  * list position used as the day index         -> real day index; missed days
                                                   are supported.
  * "95% CI" from residuals of smoothed data    -> autocorrelation-aware
                                                   interval on raw readings.
  * a number was returned for any input         -> "insufficient" status when
                                                   the history is too short.

Breaking changes in the returned dictionary: `r_squared`,
`food_adjustment_kcal` and `activity_adjustment_kcal` are gone (R^2 of a
smoothed series said nothing about accuracy; the 60/40 food/activity split
had no evidential basis). `smoothed` is retained as an alias of the fitted
trend line. `kcal_per_day` is None when `status` is not "ok".
"""

import numpy as np

from src.energy_balance import (DEFAULT_WINDOW_DAYS, KCAL_PER_KG, assess_reliability,
                                validate_weight_log)
from src.research_estimate import estimate as estimate_energy_balance

__all__ = ["KCAL_PER_KG", "run_pipeline"]


def run_pipeline(weight_log, target_weekly_change=None, days=None,
                 window_days=DEFAULT_WINDOW_DAYS):
    """
    weight_log : readings in kg on a daily grid (None/NaN = missed day), or
                 observed readings with matching `days`.
    target_weekly_change : optional planned change in kg/week (negative = loss).
    window_days : trailing window for the "current" estimate (None = all data).

    Returns a dict; check `status` before using any number.
    """
    log = validate_weight_log(weight_log, days)
    res = estimate_energy_balance(log, window_days=window_days)
    rel = assess_reliability(res)
    ok = res["status"] == "ok"

    output = {
        "status": res["status"],
        "message": res["message"],
        "n_obs": res["n_obs"],
        "window_days": window_days,
        "kcal_per_day": round(res["kcal_per_day"], 1) if ok else None,
        "kcal_per_day_ci95": (round(res["kcal_ci"][0], 1), round(res["kcal_ci"][1], 1)) if ok else None,
        "weekly_kg_change": round(res["weekly_kg_change"], 3) if ok else None,
        "label": res["label"],
        "reliability": rel["level"],
        "reliability_detail": rel,
        "anomaly_flags": res["outlier_mask"] if ok else np.zeros(0, dtype=bool),
        "anomalies_detected": res["n_outliers"] if ok else 0,
        "trend": res["trend_line"] if ok else np.zeros(0),
        "trend_days": res["fit_days"] if ok else np.zeros(0),
        "current_trend_weight": float(res["trend_line"][-1]) if ok else
                                (float(log.weights[-1]) if log.n_obs else None),
        "latest_reading": float(log.weights[-1]) if log.n_obs else None,
    }
    output["smoothed"] = output["trend"]          # legacy alias

    if target_weekly_change is not None:
        target_kcal = target_weekly_change * KCAL_PER_KG / 7.0
        output["target_kcal_per_day"] = round(target_kcal, 1)
        output["gap_kcal"] = round(target_kcal - res["kcal_per_day"], 1) if ok else None
    return output
