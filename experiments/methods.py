"""
Every estimation method compared in the experiments, behind one interface:

    method(days, weights) -> kcal/day        (days, weights = observed readings only)

"legacy_*" methods call the frozen internship code in src/legacy/ unchanged.
They are given exactly what the original application had: a bare list of
readings with no dates.
"""

import numpy as np

from src import energy_balance as eb
from src.legacy import app_estimator_v1 as app_v1
from src.legacy import ml_engine_v1 as ml_v1

KCAL = eb.KCAL_PER_KG


# ---- original pipeline, as shipped --------------------------------------
def legacy_research_full(days, w):
    """src/ml_engine.py::run_pipeline as shipped. Returns the whole output dict."""
    return ml_v1.run_pipeline(list(w))


def legacy_app(days, w):
    """The estimator duplicated inside the original Streamlit app (the number users saw)."""
    return float(app_v1.estimate_balance(list(w))[0])


def legacy_timefix_from(days, w, legacy_out):
    """
    Original EWMA + Isolation Forest steps, but regressed against the REAL day
    index. Isolates the effect of the smoothing/anomaly steps from the
    effect of ignoring missed days.
    """
    clean = np.where(legacy_out["anomaly_flags"], legacy_out["smoothed"], w)
    return float(np.polyfit(days, ml_v1.ewma_filter(clean), 1)[0] * KCAL)


# ---- alternatives -------------------------------------------------------
def ewma_ols(days, w):
    """EWMA (span 7) then least squares on real days; no anomaly step."""
    return float(np.polyfit(days, ml_v1.ewma_filter(w), 1)[0] * KCAL)


def ewma_attenuation(days):
    """
    Factor by which EWMA-then-regression shrinks the slope of a noise-free
    straight line observed on exactly these days (1.0 = no shrinkage).
    """
    return float(np.polyfit(days, ml_v1.ewma_filter(np.asarray(days, float)), 1)[0])


def ols_raw(days, w):
    return eb.fit_line(days, w, "ols")["slope"] * KCAL


def huber(days, w):
    return eb.fit_line(days, w, "huber")["slope"] * KCAL


def theil_sen(days, w):
    return eb.theil_sen_slope(days, w)[0] * KCAL


def all_point_estimates(days, w, include_legacy=True):
    out = {"ols_raw": ols_raw(days, w), "huber": huber(days, w),
           "theil_sen": theil_sen(days, w), "ewma_ols": ewma_ols(days, w)}
    # Diagnostic controls (not candidate methods): do the EWMA results differ
    # from the robust fit only by shrinking the slope towards zero?
    k = ewma_attenuation(days)
    out["attenuation_factor"] = k
    out["ewma_ols_unshrunk"] = out["ewma_ols"] / k
    out["huber_shrunk_like_ewma"] = out["huber"] * k
    if include_legacy:
        leg = legacy_research_full(days, w)
        out["legacy_research"] = float(leg["kcal_per_day"])
        out["legacy_research_ci_lo"], out["legacy_research_ci_hi"] = leg["kcal_per_day_ci95"]
        out["legacy_n_flagged"] = int(leg["anomalies_detected"])
        out["legacy_app"] = legacy_app(days, w)
        out["legacy_timefix"] = legacy_timefix_from(days, w, leg)
    return out


# ---- interval candidates ------------------------------------------------
def interval_candidates(days, w, with_bootstrap=True, seed=0):
    """95% intervals (kcal/day) from each candidate method. Returns {name: (lo, hi)}."""
    from scipy import stats
    out = {}
    f_ols = eb.fit_line(days, w, "ols")
    f_hub = eb.fit_line(days, w, "huber")
    specs = {
        "ols_classical": (f_ols, "ols", "classical", True),
        "ols_hac":       (f_ols, "ols", "hac", True),
        "ols_ar1":       (f_ols, "ols", "ar1", True),
        "huber_classical": (f_hub, "huber", "classical", True),
        "huber_ar1_nobc": (f_hub, "huber", "ar1", False),
        "huber_ar1":     (f_hub, "huber", "ar1", True),
    }
    for name, (fit, method, se_method, bc) in specs.items():
        se, dof, _ = eb.slope_se(days, w, fit, method, se_method, bias_correct=bc)
        t = stats.t.ppf(0.975, dof)
        out[name] = ((fit["slope"] - t * se) * KCAL, (fit["slope"] + t * se) * KCAL)
    if with_bootstrap:
        lo, hi = eb.block_bootstrap_slope_ci(days, w, "huber", n_boot=300, seed=seed)
        out["huber_block_bootstrap"] = (lo * KCAL, hi * KCAL)
    return out
