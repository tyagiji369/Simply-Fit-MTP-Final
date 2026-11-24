"""Shared features and deployed estimate. The archived MTP API remains available."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from . import energy_balance as eb
from .state_space import state_trend

POLICY = Path(__file__).resolve().parents[1] / "research" / "policy.json"


@lru_cache(maxsize=1)
def model_settings():
    design = json.loads(POLICY.with_name("design.json").read_text())
    return design["process_sd"], design["innovation_cutoff"]


def candidates(res):
    x, y = res["fit_days"], res["fit_weights"]
    scale = max(0.16, res["residual_scale_kg"])
    out = {}
    for name, robust in [("state", True), ("gaussian", False)]:
        slope, se, start = state_trend(x, y, scale, robust, *model_settings())
        out[name] = (7700 * slope, 7700 * se, start - slope * x[0])
    h, o = res["kcal_per_day"], res["ols_kcal_per_day"]
    span = res["span_days"]
    scatter = 7700 * scale / max(1, np.linalg.norm(x - x.mean()))
    # Window-length floor models unresolved water; gaps, scatter, disagreement
    # and uncertainty in the fitted trend each increase the empirical risk scale.
    risk = (
        np.hypot(4500 / span, scatter * np.sqrt(np.max(np.diff(x))))
        + 0.2 * abs(h - o)
        + 0.2 * abs(h - out["state"][0])
    )
    risk += 0.25 * min(res["kcal_ci_halfwidth"], 2000)
    return out, float(risk)


@lru_cache(maxsize=1)
def policy():
    return json.loads(POLICY.read_text())


def withheld(n, span, window, message, warnings):
    return dict(
        status="withheld",
        n_obs=n,
        span_days=span,
        window_days=window,
        errors=[],
        warnings=warnings,
        message="Estimate withheld. " + message,
        kcal_per_day=None,
        kcal_ci=None,
        kg_per_day=None,
        kg_per_day_ci=None,
        weekly_kg_change=None,
        label=None,
    )


def estimate(weights, days=None, window_days=None):
    log = (
        weights
        if isinstance(weights, eb.WeightLog)
        else eb.validate_weight_log(weights, days)
    )
    if log.ok:
        selected = (
            log.days
            if window_days is None
            else log.days[log.days > log.days[-1] - window_days]
        )
        if len(selected) and selected[-1] - selected[0] > 83:
            return withheld(
                len(selected),
                int(selected[-1] - selected[0]),
                window_days,
                "The observed history exceeds the tested 84-day range. Choose a shorter window.",
                log.warnings,
            )
    res = eb.estimate_energy_balance(log, window_days=window_days)
    if res["status"] != "ok":
        return res
    p = policy()
    states, risk = candidates(res)
    if risk > p["cutoff"]:
        return withheld(
            res["n_obs"],
            res["span_days"],
            window_days,
            f"The simulation-tested risk score is {risk:.0f}, above the reporting cutoff of {p['cutoff']:.0f}. "
            "The score combines history length, spacing, scatter, interval width and estimator disagreement.",
            res["warnings"],
        )
    name = p["estimator"]
    point = {
        "huber": res["kcal_per_day"],
        "ols": res["ols_kcal_per_day"],
        **{k: v[0] for k, v in states.items()},
    }[name]
    x, y = res["fit_days"], res["fit_weights"]
    slope = point / 7700
    intercept = (
        states[name][2]
        if name in states
        else (np.mean(y - slope * x) if name == "ols" else res["intercept"])
    )
    hw = p["accepted_quantiles"]["0.95"] * risk
    lo, hi = point - hw, point + hw
    res.update(
        method=name,
        kcal_per_day=point,
        kcal_ci=(lo, hi),
        kg_per_day=slope,
        weekly_kg_change=7 * slope,
        kg_per_day_ci=(lo / 7700, hi / 7700),
        kcal_ci_halfwidth=hw,
        intercept=intercept,
        trend_line=intercept + slope * x,
        label=eb.classify_with_uncertainty(
            point, lo, hi, n_calendar_days=res["n_calendar_days"]
        ),
        point_label=eb.classify_point(point),
        risk_score=risk,
        se_method="simulation calibration",
    )
    return res


def rolling(weights, window_days):
    log = eb.validate_weight_log(weights)
    rows = []
    if not log.ok:
        return rows
    for i, day in enumerate(log.days):
        r = estimate(log.weights[: i + 1], log.days[: i + 1], window_days=window_days)
        ok = r["status"] == "ok"
        lo, hi = r["kcal_ci"] if ok else (np.nan, np.nan)
        rows.append(
            dict(
                day=day,
                kcal_per_day=r["kcal_per_day"] if ok else np.nan,
                kcal_lo=lo,
                kcal_hi=hi,
            )
        )
    return rows
