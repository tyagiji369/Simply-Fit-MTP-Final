"""
Shared estimation core for Simply-Fit (M.Tech version).

ONE implementation of every statistical calculation used by the Streamlit
app, the coach, the experiments and the tests. Nothing else in the project
re-implements these formulas.

What is estimated
-----------------
From dated body-weight readings we estimate the *slope of the underlying
weight trend* (kg/day) and convert it to an approximate
"energy-balance equivalent" in kcal/day:

    kcal/day  =  slope [kg/day]  x  KCAL_PER_KG [kcal/kg]

KCAL_PER_KG = 7700 is a SIMPLIFYING ASSUMPTION (the classic Wishnofsky
figure for the energy content of mixed adipose tissue). The true energy
content of a kilogram of weight change depends on its fat / lean / water
composition and differs between people and over time. The output is
therefore a weight-trend-derived *equivalent*, not a measurement of
physiological energy balance, calorie intake or energy expenditure.

Units used everywhere in this module
------------------------------------
    time       : days (calendar day index, integers; gaps allowed)
    weight     : kg
    slope      : kg/day
    energy     : kcal/day   (negative = deficit, positive = surplus)

Three different uncertainties (never to be confused)
----------------------------------------------------
    1. trend slope CI         - uncertainty of the fitted slope (kg/day)
    2. energy-equivalent CI   - (1) multiplied by KCAL_PER_KG. It does NOT
                                include uncertainty in KCAL_PER_KG itself.
    3. prediction interval    - range for a FUTURE scale reading; it adds
                                day-to-day scatter on top of trend
                                uncertainty and is much wider than (1).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Documented constants (all in one place; see module docstring for units)
# --------------------------------------------------------------------------
KCAL_PER_KG = 7700.0            # simplifying assumption, kcal per kg of weight change
MIN_PLAUSIBLE_KG = 20.0         # input validation bounds for an adult scale reading
MAX_PLAUSIBLE_KG = 400.0
MIN_OBS_FOR_ESTIMATE = 10       # fewer readings -> no kcal number is reported
MIN_CALENDAR_DAYS_FOR_ESTIMATE = 14  # first-to-last reading must cover at least this many calendar days
MIN_CALENDAR_DAYS_FOR_LABEL = 25      # below this a deficit/surplus/maintenance label is never stated.
                                      # Chosen on development simulations: the shortest history at which
                                      # definite direction calls were wrong no more than 5% of the time.
HUBER_K = 1.345                 # standard Huber tuning constant (95% Gaussian efficiency)
MIN_SCALE_KG = 0.05             # floor on the robust residual scale (scales display 0.1 kg)
CLASS_TOLERANCE_KCAL = 100.0    # |estimate| <= this is called "approximate maintenance"
MAINTENANCE_CI_LIMIT_KCAL = 250.0   # CI must lie inside +/- this to call maintenance with confidence
DEFAULT_WINDOW_DAYS = 28        # default trailing window for the "current" estimate
DEFAULT_CONFIDENCE = 0.95


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
@dataclass
class WeightLog:
    """Validated readings. `days`/`weights` contain observed readings only."""
    days: np.ndarray
    weights: np.ndarray
    n_calendar_days: int
    n_missing_days: int
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def n_obs(self):
        return int(len(self.weights))

    @property
    def span_days(self):
        return int(self.days[-1] - self.days[0]) if self.n_obs else 0

    @property
    def ok(self):
        return not self.errors


def validate_weight_log(weights, days=None):
    """
    Validate user-supplied readings.

    weights : sequence of numbers; None / NaN / "" mark a day with no reading.
    days    : optional sequence of integer day indices (same length). When
              omitted, position i is calendar day i (a daily grid in which
              missed days are given as None/NaN).

    Returns a WeightLog. Problems that make the data unusable are listed in
    `.errors` (the caller must not estimate anything); recoverable oddities
    are listed in `.warnings`. No value is ever invented or filled in.
    """
    errors, warnings = [], []
    try:
        raw = list(weights)
    except TypeError:
        return WeightLog(np.array([]), np.array([]), 0, 0, ["Weight log is not a sequence."], [])

    vals = []
    for i, v in enumerate(raw):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            vals.append(np.nan)
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            errors.append(f"Entry {i + 1} ('{v}') is not a number.")
            vals.append(np.nan)
    vals = np.asarray(vals, dtype=float)
    if np.isinf(vals).any():
        errors.append("Infinite values are not valid weights.")
        vals[np.isinf(vals)] = np.nan

    if days is None:
        d = np.arange(len(vals), dtype=float)
    else:
        try:
            d = np.asarray(list(days), dtype=float)
        except (TypeError, ValueError):
            return WeightLog(np.array([]), np.array([]), 0, 0, ["Day indices must be numbers."], [])
        if len(d) != len(vals):
            return WeightLog(np.array([]), np.array([]), 0, 0,
                             ["`days` and `weights` must have the same length."], [])
        if d.ndim != 1 or np.isinf(d).any():
            return WeightLog(np.array([]), np.array([]), 0, 0, ["Day indices must be finite scalar values."], [])
        if np.isnan(d).any():
            errors.append("Day indices contain missing values.")
            keep = ~np.isnan(d)
            d, vals = d[keep], vals[keep]
        if np.any(d != np.round(d)):
            errors.append("Day indices must be whole days.")
        order = np.argsort(d, kind="stable")
        if np.any(order != np.arange(len(d))):
            warnings.append("Readings were not in date order; they have been sorted.")
            d, vals = d[order], vals[order]

    obs = ~np.isnan(vals)
    bad = obs & ((vals < MIN_PLAUSIBLE_KG) | (vals > MAX_PLAUSIBLE_KG))
    if bad.any():
        errors.append(
            f"{int(bad.sum())} reading(s) outside the plausible range "
            f"{MIN_PLAUSIBLE_KG:.0f}-{MAX_PLAUSIBLE_KG:.0f} kg: "
            + ", ".join(f"{v:g}" for v in vals[bad][:5]) + ". Check the units (kg) and typing."
        )
    d_obs, w_obs = d[obs], vals[obs]
    if len(d_obs) and len(np.unique(d_obs)) != len(d_obs):
        errors.append("More than one reading for the same day; keep one reading per day.")
    if len(d_obs) == 0:
        errors.append("No weight readings were provided.")
        return WeightLog(np.array([]), np.array([]), 0, 0, errors, warnings)

    n_cal = int(d_obs[-1] - d_obs[0]) + 1
    n_missing = n_cal - len(d_obs)
    if len(w_obs) >= 2 and not errors:
        gaps = np.diff(d_obs)
        jumps = np.abs(np.diff(w_obs)) / np.maximum(gaps, 1)
        if np.any(jumps > 3.0):
            warnings.append(
                f"{int(np.sum(jumps > 3.0))} day-to-day change(s) larger than 3 kg. "
                "These are kept, but check them for typing or scale errors."
            )
    return WeightLog(d_obs.astype(float), w_obs.astype(float), n_cal, int(n_missing), errors, warnings)


# --------------------------------------------------------------------------
# Trend fitting
# --------------------------------------------------------------------------
def _wls_batch(x, Y, Wt):
    """Closed-form weighted straight-line fit for each row of Y (B x n)."""
    sw = Wt.sum(axis=1)
    xm = (Wt * x).sum(axis=1) / sw
    ym = (Wt * Y).sum(axis=1) / sw
    dx = x[None, :] - xm[:, None]
    sxx = (Wt * dx * dx).sum(axis=1)
    sxy = (Wt * dx * (Y - ym[:, None])).sum(axis=1)
    slope = np.where(sxx > 0, sxy / np.where(sxx > 0, sxx, 1.0), 0.0)
    return slope, ym - slope * xm


def _mad_scale(R):
    """Row-wise normalised MAD about zero (as in standard robust regression), floored."""
    return np.maximum(np.median(np.abs(R), axis=1) / 0.6745, MIN_SCALE_KG)


def fit_line_batch(x, Y, method="huber", k=HUBER_K, max_iter=50, tol=1e-9):
    """
    Fit y = intercept + slope * x to every row of Y.

    method = "ols"   : ordinary least squares.
    method = "huber" : Huber M-estimator by iteratively reweighted least
                       squares; residual scale = MAD / 0.6745 (re-estimated
                       each iteration, floored at MIN_SCALE_KG). Readings
                       within k*scale of the line get full weight; beyond
                       that the weight decays as k*scale/|residual|, so a
                       gross error is down-weighted, never deleted or
                       replaced by an invented value.
    Returns (slope, intercept, weights, scale), arrays over rows.
    """
    x = np.asarray(x, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    Wt = np.ones_like(Y)
    slope, icpt = _wls_batch(x, Y, Wt)
    R = Y - (icpt[:, None] + slope[:, None] * x[None, :])
    if method == "ols":
        dof = max(Y.shape[1] - 2, 1)
        return slope, icpt, Wt, np.sqrt((R ** 2).sum(axis=1) / dof)
    if method != "huber":
        raise ValueError(f"unknown method: {method}")
    scale = _mad_scale(R)
    for _ in range(max_iter):
        c = k * scale[:, None]
        Wt = np.where(np.abs(R) <= c, 1.0, c / np.maximum(np.abs(R), 1e-12))
        new_slope, new_icpt = _wls_batch(x, Y, Wt)
        R = Y - (new_icpt[:, None] + new_slope[:, None] * x[None, :])
        converged = np.max(np.abs(new_slope - slope)) < tol
        slope, icpt = new_slope, new_icpt
        scale = _mad_scale(R)
        if converged:
            break
    c = k * scale[:, None]
    Wt = np.where(np.abs(R) <= c, 1.0, c / np.maximum(np.abs(R), 1e-12))
    return slope, icpt, Wt, scale


def fit_line(days, weights, method="huber", k=HUBER_K):
    """Single-series convenience wrapper. Returns dict(slope, intercept, weights, scale)."""
    s, a, wt, sc = fit_line_batch(days, np.asarray(weights, dtype=float)[None, :], method, k)
    return {"slope": float(s[0]), "intercept": float(a[0]), "weights": wt[0], "scale": float(sc[0])}


def theil_sen_slope(days, weights):
    """Median of all pairwise slopes (kg/day). Used as a comparison method."""
    res = stats.theilslopes(np.asarray(weights, float), np.asarray(days, float))
    return float(res[0]), float(res[1])


# --------------------------------------------------------------------------
# Slope uncertainty
# --------------------------------------------------------------------------
def estimate_ar1(days, resid, bias_correct=True):
    """
    Lag-1 autocorrelation of residuals, using only pairs of readings exactly
    one day apart (so gaps do not distort it). With fewer than 6 such pairs,
    all consecutive pairs are used and the correlation is converted to a
    per-day value via the mean gap.

    bias_correct adds the first-order small-sample correction for a
    detrended series, rho + (1 + 4*rho)/n  (the raw estimate is biased
    towards zero, strongly so for 2-8 weeks of data). Result is limited to
    [0, 0.9]: negative day-to-day correlation is not physically expected for
    water weight, and treating it as zero is the conservative choice.
    """
    days = np.asarray(days, float)
    r = np.asarray(resid, float)
    n = len(r)
    if n < 5:
        return 0.0
    gaps = np.diff(days)
    one = gaps == 1
    denom = float(np.sum(r ** 2) / n)
    if denom <= 0:
        return 0.0
    if one.sum() >= 6:
        rho = float(np.mean(r[:-1][one] * r[1:][one]) / denom)
    else:
        rho_g = float(np.mean(r[:-1] * r[1:]) / denom)
        rho = np.sign(rho_g) * abs(rho_g) ** (1.0 / max(float(np.mean(gaps)), 1.0))
    if bias_correct:
        rho = rho + (1.0 + 4.0 * rho) / n
    return float(np.clip(rho, 0.0, 0.9))


def slope_se(days, weights, fit, method="huber", se_method="ar1", k=HUBER_K, bias_correct=True):
    """
    Standard error of the fitted slope (kg/day) and the degrees of freedom
    for the t critical value.

    se_method
      "classical" : textbook OLS formula; assumes independent errors.
      "hac"       : Newey-West (Bartlett kernel, lag = floor(4*(n/100)^(2/9)),
                    small-sample factor n/(n-2)), in reading order.
      "ar1"       : sandwich estimator with an AR(1) working correlation
                    rho^|dt| evaluated on the ACTUAL day gaps (handles
                    missing days exactly), small-sample bias-corrected rho,
                    t distribution with effective sample size
                    n*(1-rho)/(1+rho) - 2 degrees of freedom.
    For method="huber" the sandwich uses the Huber score (residuals clipped
    at k*scale) and the share of readings inside the clipping band, which is
    the standard M-estimator covariance.
    """
    x = np.asarray(days, float)
    y = np.asarray(weights, float)
    n = len(x)
    if n < 3:
        return float("nan"), 1.0, 0.0
    resid = y - (fit["intercept"] + fit["slope"] * x)
    if method == "huber":
        c = k * max(fit["scale"], MIN_SCALE_KG)
        psi = np.clip(resid, -c, c)
        dpsi = (np.abs(resid) <= c).astype(float)
        if dpsi.sum() < 3:
            dpsi = np.ones(n)
    else:
        psi, dpsi = resid, np.ones(n)

    Z = np.column_stack([np.ones(n), x - x.mean()])
    A = Z.T @ (Z * dpsi[:, None])
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("nan"), 1.0, 0.0
    sigma2 = float(np.sum(psi ** 2) / max(n - 2, 1))

    if se_method == "classical":
        B = sigma2 * (Z.T @ Z)
        rho, dof = 0.0, max(n - 2, 1)
    elif se_method == "hac":
        L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        U = Z * psi[:, None]
        B = U.T @ U
        for lag in range(1, L + 1):
            wgt = 1.0 - lag / (L + 1.0)
            G = U[lag:].T @ U[:-lag]
            B += wgt * (G + G.T)
        B *= n / max(n - 2, 1)
        rho, dof = 0.0, max(n - 2, 1)
    elif se_method == "ar1":
        rho = estimate_ar1(x, psi, bias_correct=bias_correct)
        Rm = rho ** np.abs(x[:, None] - x[None, :])
        B = sigma2 * (Z.T @ Rm @ Z)
        n_eff = n * (1.0 - rho) / (1.0 + rho)
        dof = max(n_eff - 2.0, 2.0)
    else:
        raise ValueError(f"unknown se_method: {se_method}")

    cov = A_inv @ B @ A_inv
    return float(np.sqrt(max(cov[1, 1], 0.0))), float(dof), float(rho)


def block_bootstrap_slope_ci(days, weights, method="huber", n_boot=400, block=None,
                             confidence=DEFAULT_CONFIDENCE, seed=0):
    """
    Moving-block bootstrap of residuals (comparison method for the
    experiments). Blocks of consecutive residuals are resampled so short-range
    serial dependence is retained; the line is re-fitted to each resample.
    Returns (lo, hi) for the slope in kg/day by the basic percentile method.
    """
    x = np.asarray(days, float)
    y = np.asarray(weights, float)
    n = len(x)
    f = fit_line(x, y, method)
    fitted = f["intercept"] + f["slope"] * x
    resid = y - fitted
    b = block or max(2, int(np.ceil(n ** (1.0 / 3.0))) + 1)
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(n / b))
    starts = rng.randint(0, n - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]
    Yb = fitted[None, :] + resid[idx]
    slopes, _, _, _ = fit_line_batch(x, Yb, method)
    a = (1 - confidence) / 2
    return float(np.quantile(slopes, a)), float(np.quantile(slopes, 1 - a))


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classify_point(kcal, tolerance=CLASS_TOLERANCE_KCAL):
    """Three-way label from a point estimate alone (used to score every method alike)."""
    if kcal < -tolerance:
        return "deficit"
    if kcal > tolerance:
        return "surplus"
    return "maintenance"


def classify_with_uncertainty(kcal, lo, hi, tolerance=CLASS_TOLERANCE_KCAL,
                              maintenance_limit=MAINTENANCE_CI_LIMIT_KCAL, n_calendar_days=None):
    """
    Label shown to the user. A direction is only stated when the interval
    supports it AND (when `n_calendar_days` is given) the readings cover at
    least MIN_CALENDAR_DAYS_FOR_LABEL days:
        deficit       : estimate < -tolerance and the whole CI is below 0
        surplus       : estimate > +tolerance and the whole CI is above 0
        maintenance   : |estimate| <= tolerance and the CI lies inside
                        +/- maintenance_limit
        inconclusive  : anything else (the data cannot yet tell)
    """
    if n_calendar_days is not None and n_calendar_days < MIN_CALENDAR_DAYS_FOR_LABEL:
        return "inconclusive"
    if kcal < -tolerance and hi < 0:
        return "deficit"
    if kcal > tolerance and lo > 0:
        return "surplus"
    if abs(kcal) <= tolerance and lo > -maintenance_limit and hi < maintenance_limit:
        return "maintenance"
    return "inconclusive"


# --------------------------------------------------------------------------
# Main estimator
# --------------------------------------------------------------------------
def estimate_energy_balance(weights, days=None, window_days=None, method="huber",
                            se_method="ar1", confidence=DEFAULT_CONFIDENCE,
                            kcal_per_kg=KCAL_PER_KG):
    """
    Estimate the weight-trend slope and its energy-balance equivalent.

    weights, days : see validate_weight_log().
    window_days   : if given, only readings from the last `window_days`
                    calendar days (ending at the latest reading) are used.
                    Only past readings are ever used - there is no smoothing
                    that looks ahead.

    Returns a dict. `status` is one of
        "ok"            - numbers are present
        "insufficient"  - too few readings / too short a span; all numeric
                          fields are None and `message` explains why
        "invalid"       - input failed validation (`errors` lists why)
    """
    log = weights if isinstance(weights, WeightLog) else validate_weight_log(weights, days)
    base = {"status": "invalid", "errors": list(log.errors), "warnings": list(log.warnings),
            "kcal_per_day": None, "kcal_ci": None, "kg_per_day": None, "kg_per_day_ci": None,
            "weekly_kg_change": None, "label": None, "point_label": None,
            "n_obs": log.n_obs, "span_days": log.span_days, "method": method,
            "se_method": se_method, "confidence": confidence, "kcal_per_kg": kcal_per_kg,
            "window_days": window_days, "message": ""}
    if not log.ok:
        base["message"] = " ".join(log.errors)
        return base

    d, w = log.days, log.weights
    if window_days is not None:
        keep = d > d[-1] - window_days
        d, w = d[keep], w[keep]
    n = len(w)
    span = int(d[-1] - d[0]) if n else 0
    n_cal = span + 1
    base.update(n_obs=n, span_days=span, n_missing_days=n_cal - n,
                missing_fraction=(n_cal - n) / n_cal if n_cal else 0.0)

    if n < MIN_OBS_FOR_ESTIMATE or n_cal < MIN_CALENDAR_DAYS_FOR_ESTIMATE:
        base["status"] = "insufficient"
        base["message"] = (
            f"Not enough data for an energy-balance estimate: {n} reading(s) covering "
            f"{n_cal} calendar day(s). At least {MIN_OBS_FOR_ESTIMATE} readings covering "
            f"{MIN_CALENDAR_DAYS_FOR_ESTIMATE} days are required, because day-to-day water-weight "
            "changes are larger than the fat change that accumulates over a shorter period."
        )
        return base

    fit = fit_line(d, w, method)
    se, dof, rho = slope_se(d, w, fit, method, se_method)
    tcrit = float(stats.t.ppf(0.5 + confidence / 2, dof))
    slope = fit["slope"]
    lo, hi = slope - tcrit * se, slope + tcrit * se
    kcal, kcal_lo, kcal_hi = slope * kcal_per_kg, lo * kcal_per_kg, hi * kcal_per_kg

    ols = fit_line(d, w, "ols")
    resid = w - (fit["intercept"] + slope * d)
    # "unusual" = more than 2.5 robust standard deviations away from the trend line
    downweighted = np.abs(resid) > 2.5 * max(fit["scale"], MIN_SCALE_KG)

    base.update({
        "status": "ok",
        "kg_per_day": slope,
        "kg_per_day_ci": (lo, hi),
        "weekly_kg_change": slope * 7.0,
        "kcal_per_day": kcal,
        "kcal_ci": (kcal_lo, kcal_hi),
        "kcal_ci_halfwidth": (kcal_hi - kcal_lo) / 2.0,
        "slope_se": se,
        "dof": dof,
        "rho": rho,
        "intercept": fit["intercept"],
        "residual_scale_kg": fit["scale"],
        "fit_days": d,
        "fit_weights": w,
        "trend_line": fit["intercept"] + slope * d,
        "robust_weights": fit["weights"],
        "outlier_mask": downweighted,
        "n_outliers": int(downweighted.sum()),
        "ols_kcal_per_day": ols["slope"] * kcal_per_kg,
        "outlier_sensitivity_kcal": abs(ols["slope"] - slope) * kcal_per_kg,
        "point_label": classify_point(kcal),
        "label": classify_with_uncertainty(kcal, kcal_lo, kcal_hi, n_calendar_days=n_cal),
        "label_ungated": classify_with_uncertainty(kcal, kcal_lo, kcal_hi),
        "n_calendar_days": n_cal,
        "too_short_for_label": n_cal < MIN_CALENDAR_DAYS_FOR_LABEL,
    })
    return base


def rolling_energy_balance(weights, days=None, window_days=DEFAULT_WINDOW_DAYS, step=1,
                           method="huber", se_method="ar1"):
    """
    Trailing-window ("as of each day") estimates for the dashboard. The
    estimate reported for day t uses readings from days (t - window, t]
    only, i.e. exactly what would have been known on day t.
    Returns a list of dicts: day, kcal_per_day, kcal_lo, kcal_hi (None when
    the window does not hold enough data).
    """
    log = weights if isinstance(weights, WeightLog) else validate_weight_log(weights, days)
    out = []
    if not log.ok or log.n_obs == 0:
        return out
    for i in range(0, log.n_obs, step):
        t = log.days[i]
        res = estimate_energy_balance(log.weights[: i + 1], log.days[: i + 1],
                                      window_days=window_days, method=method, se_method=se_method)
        if res["status"] == "ok":
            out.append({"day": float(t), "kcal_per_day": res["kcal_per_day"],
                        "kcal_lo": res["kcal_ci"][0], "kcal_hi": res["kcal_ci"][1]})
        else:
            out.append({"day": float(t), "kcal_per_day": None, "kcal_lo": None, "kcal_hi": None})
    return out


# --------------------------------------------------------------------------
# Reliability indicator
# --------------------------------------------------------------------------
# Thresholds on the CI half-width (kcal/day). They are tied to the
# classification tolerance, not chosen for appearance, and are checked
# against realised error in EXPERIMENT_RESULTS.md.
RELIABILITY_HIGH_HALFWIDTH = 150.0
RELIABILITY_MODERATE_HALFWIDTH = 300.0
MISSING_WARN_FRACTION = 0.30
OUTLIER_SENSITIVITY_WARN_KCAL = 75.0
NOISE_WARN_KG = 0.60


def assess_reliability(result):
    """
    Transparent data-quality indicator built from measurable factors.

    The level is driven by the width of the confidence interval (which
    already reflects the number of readings, their spread over time, their
    scatter and its day-to-day persistence). It is held at "low" when the
    readings cover fewer than MIN_CALENDAR_DAYS_FOR_LABEL days. Other
    factors can only LOWER the level by one step and are always listed with
    their measured value,
    so the user can see why. No percentage "confidence score" is produced:
    nothing in the available evidence would calibrate one.

    Returns dict(level, summary, factors=[{name, value, status, message}]).
    level is "insufficient", "low", "moderate" or "high".
    """
    if result["status"] == "withheld":
        return {"level": "insufficient", "summary": result["message"], "factors": [
            {"name": "Selective reporting", "value": "Withheld", "status": "warn",
             "message": "The log falls outside the simulation-tested reporting rule."}]}
    if result["status"] != "ok":
        return {"level": "insufficient",
                "summary": result.get("message") or "No estimate could be produced.",
                "factors": [{"name": "Readings available", "value": f"{result.get('n_obs', 0)}",
                             "status": "bad",
                             "message": f"Need at least {MIN_OBS_FOR_ESTIMATE} readings covering "
                                        f"{MIN_CALENDAR_DAYS_FOR_ESTIMATE} calendar days."}]}

    hw = result["kcal_ci_halfwidth"]
    if hw <= RELIABILITY_HIGH_HALFWIDTH:
        level = 2
    elif hw <= RELIABILITY_MODERATE_HALFWIDTH:
        level = 1
    else:
        level = 0

    too_short = result["span_days"] + 1 < MIN_CALENDAR_DAYS_FOR_LABEL
    if too_short:
        level = 0
    factors = [{
        "name": "Estimated uncertainty",
        "value": f"±{hw:.0f} kcal/day",
        "status": ["bad", "warn", "good"][level],
        "message": "95% interval half-width of the energy-balance equivalent.",
    }, {
        "name": "Readings used",
        "value": f"{result['n_obs']} over {result['span_days'] + 1} days",
        "status": "bad" if too_short else "good",
        "message": (f"Fewer than {MIN_CALENDAR_DAYS_FOR_LABEL} days of history: reliability is held at "
                    "'low' and no deficit/surplus label is given, because in simulation tests "
                    "direction calls made on less data were wrong too often.") if too_short else
                   "Enough history to attempt a deficit/surplus call.",
    }]

    downgrade = False
    mf = result["missing_fraction"]
    miss_bad = mf > MISSING_WARN_FRACTION
    factors.append({
        "name": "Missed days",
        "value": f"{result['n_missing_days']} ({mf:.0%})",
        "status": "warn" if miss_bad else "good",
        "message": "Many skipped days." if miss_bad else "Few or no skipped days.",
    })
    downgrade |= miss_bad

    noise = result["residual_scale_kg"]
    noisy = noise > NOISE_WARN_KG
    factors.append({
        "name": "Day-to-day scatter",
        "value": f"{noise:.2f} kg",
        "status": "warn" if noisy else "good",
        "message": "Readings vary a lot around the trend; weigh at a consistent time of day."
                   if noisy else "Typical scatter around the trend.",
    })
    downgrade |= noisy

    sens = result["outlier_sensitivity_kcal"]
    sensitive = sens > OUTLIER_SENSITIVITY_WARN_KCAL
    factors.append({
        "name": "Sensitivity to unusual readings",
        "value": f"{sens:.0f} kcal/day ({result['n_outliers']} unusual reading(s))",
        "status": "warn" if sensitive else "good",
        "message": "The answer changes noticeably depending on how a few unusual readings are "
                   "treated." if sensitive else
                   "The robust and ordinary fits agree, so no single reading drives the result.",
    })
    downgrade |= sensitive

    if downgrade and level > 0:
        level -= 1
    name = ["low", "moderate", "high"][level]
    summary = {
        "high": "The trend is well determined by the data. The estimate is still an "
                "approximation that depends on the 7,700 kcal/kg assumption.",
        "moderate": "The direction of the trend is probably right, but its size is uncertain.",
        "low": "The data do not pin down the trend. Treat the number as a rough indication only.",
    }[name]
    return {"level": name, "summary": summary, "factors": factors}


# --------------------------------------------------------------------------
# Secondary feature: short-term weight forecast
# --------------------------------------------------------------------------
FORECAST_FIT_DAYS = 14          # window of the linear comparison method
FORECAST_HORIZON = 7
FORECAST_MIN_OBS = 7            # linear method: readings needed in the last 14 days
FORECAST_LEVEL_READINGS = 3     # level method: average of this many most recent readings ...
FORECAST_LEVEL_MAX_AGE = 7      # ... which must all fall within the last 7 days
FORECAST_BACKTEST_DAYS = 42     # history replayed to size the prediction interval
FORECAST_BACKTEST_MIN = 8       # minimum number of replayed forecasts for an interval


def _level_backtest_errors(days, weights, horizon_h, k, history_days):
    """Errors the level rule would have made h days ahead on the user's own recent history."""
    errs = []
    start = days[-1] - history_days
    for i in range(k - 1, len(days)):
        if days[i] < start or days[i] - days[i - k + 1] >= FORECAST_LEVEL_MAX_AGE:
            continue
        hit = np.nonzero(days == days[i] + horizon_h)[0]
        if len(hit):
            errs.append(weights[hit[0]] - weights[i - k + 1: i + 1].mean())
    return np.asarray(errs)


def forecast_weight(weights, days=None, horizon=FORECAST_HORIZON, method="level",
                    fit_days=FORECAST_FIT_DAYS, confidence=0.80):
    """
    Forecast of future SCALE READINGS (not of tissue weight).

    method="level" (default)
        The average of the 3 most recent readings, carried forward. On the
        scenario simulator this had the lowest 7-day error of the simple
        methods tried on development data, because day-to-day water weight is
        larger than a week of trend. It ignores the trend, so it reads
        slightly high during sustained loss and low during sustained gain.
        The band is a PREDICTION interval obtained by replaying the same rule
        on the user's own last 42 days and taking the spread of the errors it
        would have made at each horizon (None if there is too little history).
    method="linear"
        Least squares on the last `fit_days` days, extended forward, with the
        classical regression prediction interval. Kept for comparison.

    A prediction interval describes a single future reading. It is not the
    confidence interval of the trend and is much wider.

    Returns None when there are too few recent readings, otherwise
    dict(days, mean, lo, hi, confidence, n_obs, method).
    """
    log = weights if isinstance(weights, WeightLog) else validate_weight_log(weights, days)
    if not log.ok or log.n_obs == 0:
        return None
    future = log.days[-1] + np.arange(1, horizon + 1)

    if method == "level":
        k = FORECAST_LEVEL_READINGS
        if log.n_obs < k or log.days[-1] - log.days[-k] >= FORECAST_LEVEL_MAX_AGE:
            return None
        level = float(log.weights[-k:].mean())
        mean = np.full(horizon, level)
        lo, hi = np.full(horizon, np.nan), np.full(horizon, np.nan)
        for h in range(1, horizon + 1):
            e = _level_backtest_errors(log.days, log.weights, h, k, FORECAST_BACKTEST_DAYS)
            if len(e) >= FORECAST_BACKTEST_MIN:
                half = (float(stats.t.ppf(0.5 + confidence / 2, len(e) - 1))
                        * float(np.sqrt(np.mean(e ** 2) * (1.0 + 1.0 / len(e)))))
                lo[h - 1], hi[h - 1] = level - half, level + half
        has_band = not np.isnan(lo).any()
        return {"days": future, "mean": mean, "lo": lo if has_band else None,
                "hi": hi if has_band else None, "confidence": confidence, "n_obs": k,
                "method": "level"}

    if method != "linear":
        raise ValueError(f"unknown forecast method: {method}")
    keep = log.days > log.days[-1] - fit_days
    x, y = log.days[keep], log.weights[keep]
    n = len(x)
    if n < FORECAST_MIN_OBS:
        return None
    f = fit_line(x, y, "ols")
    resid = y - (f["intercept"] + f["slope"] * x)
    s2 = float(np.sum(resid ** 2) / max(n - 2, 1))
    xm, sxx = x.mean(), float(np.sum((x - x.mean()) ** 2))
    mean = f["intercept"] + f["slope"] * future
    se_pred = np.sqrt(s2 * (1.0 + 1.0 / n + (future - xm) ** 2 / sxx))
    tcrit = float(stats.t.ppf(0.5 + confidence / 2, max(n - 2, 1)))
    return {"days": future, "mean": mean, "lo": mean - tcrit * se_pred,
            "hi": mean + tcrit * se_pred, "confidence": confidence, "n_obs": n,
            "method": "linear", "slope_kg_per_day": f["slope"]}
