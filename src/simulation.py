"""
Scenario simulator with known ground truth (M.Tech extension).

Everything produced here is SIMULATED. No series from this module is, or
may be described as, a real person's measurements.

Why a second generator exists
-----------------------------
src/data_generator.py (internship version) produces one trajectory shape
with independent day-to-day noise, no missed days, no gross errors, and it
does not keep the true energy balance. That is too easy and cannot be
scored. This module adds configurable scenarios, harder observation
conditions and - crucially - the true daily energy balance of every
simulated day.

Two deliberately different data-generating mechanisms
-----------------------------------------------------
Mechanism A ("ledger")
    Tissue weight changes by  balance / 7700  each day. This is exactly the
    conversion the estimator assumes, so results on A show how well a
    method copes with noise, gaps and errors when its model is right.

Mechanism B ("two-compartment, adaptive")
    Separate fat and lean compartments with different energy densities,
    Forbes partitioning, expenditure that responds to body composition and
    intake, a fast glycogen/water pool, a weekly eating rhythm, heavy-tailed
    water noise and (for some users) a slow monthly water cycle. A kilogram
    of weight change is NOT worth 7700 kcal here. Results on B show what
    happens when the estimator's assumptions are wrong.

Scenario kcal values mean different things in the two mechanisms: in A they
are the planned energy balance itself; in B they are the planned change in
INTAKE relative to initial maintenance, and the realised balance then drifts
as the simulated body adapts.

All parameters live in config/scenarios.json. Equations, parameter sources
and limitations are in SYNTHETIC_DATA_METHODOLOGY.md.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

from src.data_generator import NHANES_FIT, calculate_bmr

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "scenarios.json")
MECHANISMS = ("A", "B")
REFERENCE_KCAL_PER_KG = 7700.0


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        return json.load(f)


@dataclass
class SimulatedSeries:
    """
    days[t]        calendar day index 0..n-1
    observed[t]    scale reading in kg (NaN = no reading that day)
    tissue[t]      true tissue weight in kg (fat + lean; excludes water noise)
    balance[t]     TRUE energy balance during day t in kcal/day; it is what
                   moved tissue[t-1] to tissue[t]. balance[0] is NaN.
    is_outlier[t]  True where a gross measurement error was injected
    """
    days: np.ndarray
    observed: np.ndarray
    tissue: np.ndarray
    balance: np.ndarray
    is_outlier: np.ndarray
    meta: dict = field(default_factory=dict)

    def true_mean_balance(self, first_day, last_day):
        """Mean true energy balance (kcal/day) over the changes between two days."""
        return float(np.mean(self.balance[first_day + 1: last_day + 1]))

    def tissue_trend_equivalent(self, first_day, last_day, kcal_per_kg=REFERENCE_KCAL_PER_KG):
        """True tissue-weight change per day x 7700: the best any weight-only method could report."""
        return float((self.tissue[last_day] - self.tissue[first_day]) / (last_day - first_day) * kcal_per_kg)


# --------------------------------------------------------------------------
def _u(rng, rng_pair):
    lo, hi = rng_pair
    return float(lo if lo == hi else rng.uniform(lo, hi))


def _planned_path(scenario, n_days, rng):
    """Planned kcal/day for each day from a scenario definition."""
    plan = np.zeros(n_days)
    info = {}
    if "ramp" in scenario:
        a, b = _u(rng, scenario["ramp"]["start_kcal"]), _u(rng, scenario["ramp"]["end_kcal"])
        plan = np.linspace(a, b, n_days)
        info.update(ramp_start=a, ramp_end=b)
        return plan, info
    change = None
    if "change_day" in scenario:
        change = int(round(_u(rng, scenario["change_day"])))
        info["change_day"] = change
    start = 0
    for seg in scenario["segments"]:
        end = change if seg["until_frac"] == "change" else int(round(seg["until_frac"] * n_days))
        plan[start:end] = _u(rng, seg["kcal"])
        start = end
    return plan, info


def _demographics(rng):
    sex = "Female" if rng.random() < NHANES_FIT["p_female"] else "Male"
    age = float(np.clip(rng.normal(NHANES_FIT["age_mean"], NHANES_FIT["age_std"]), 18, 80))
    h_mu, h_sd = NHANES_FIT["height"][sex]
    height = float(np.clip(rng.normal(h_mu, h_sd), 140, 205))
    w_mu, w_sd = NHANES_FIT["weight_lognormal"][sex]
    weight = float(np.clip(rng.lognormal(w_mu, w_sd), 40, 180))
    return sex, age, height, weight


def _ar1(rng, n, phi, sd, t_dof=None):
    """Stationary AR(1) series with marginal SD `sd`; optional Student-t innovations."""
    innov_sd = sd * np.sqrt(1 - phi ** 2)
    if t_dof:
        e = rng.standard_t(t_dof, size=n) * np.sqrt((t_dof - 2) / t_dof) * innov_sd
    else:
        e = rng.normal(0, innov_sd, size=n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _missing_mask(rng, n, rate, bursty, run_len=(2, 6)):
    """True = reading missing. Independent days (A) or runs of consecutive days (B)."""
    miss = np.zeros(n, dtype=bool)
    if rate <= 0:
        return miss
    if not bursty:
        return rng.random(n) < rate
    target = int(round(rate * n))
    guard = 0
    while miss.sum() < target and guard < 10 * n:
        guard += 1
        length = int(rng.integers(run_len[0], run_len[1] + 1))
        start = int(rng.integers(0, n))
        miss[start:start + length] = True
    return miss


# --------------------------------------------------------------------------
def _simulate_A(cfg, scenario, n_days, rng):
    p = cfg["mechanism_A"]
    sex, age, height, w0 = _demographics(rng)
    plan, info = _planned_path(scenario, n_days, rng)
    daily_sd = _u(rng, p["daily_balance_sd_kcal"])
    balance = plan + rng.normal(0, daily_sd, n_days)          # what actually happened each day
    tissue = np.empty(n_days)
    tissue[0] = w0
    for t in range(1, n_days):
        tissue[t] = tissue[t - 1] + balance[t] / p["kcal_per_kg"]
    balance[0] = np.nan
    water = _ar1(rng, n_days, _u(rng, p["water_ar1_phi"]), _u(rng, p["water_sd_kg"]))
    scale_err = rng.normal(0, _u(rng, p["scale_error_sd_kg"]), n_days)
    meta = dict(sex=sex, age=age, height_cm=height, start_weight_kg=w0,
                daily_balance_sd=daily_sd, **info)
    return tissue, balance, water, scale_err, meta


def _simulate_B(cfg, scenario, n_days, rng):
    p = cfg["mechanism_B"]
    sex, age, height, w0 = _demographics(rng)
    bmi = w0 / (height / 100) ** 2
    # Deurenberg et al. (1991) body-fat prediction, clipped to a plausible range.
    fat_frac = np.clip((1.20 * bmi + 0.23 * age - 10.8 * (sex == "Male") - 5.4) / 100, 0.10, 0.50)
    F, L = w0 * fat_frac, w0 * (1 - fat_frac)
    F0, L0 = F, L
    maintenance = calculate_bmr(w0, height, age, sex) * _u(rng, (1.4, 1.7))

    plan, info = _planned_path(scenario, n_days, rng)        # planned intake offset
    weekend = _u(rng, p["weekend_extra_intake_kcal"])
    dow0 = int(rng.integers(0, 7))
    dow = (dow0 + np.arange(n_days)) % 7
    weekly = np.where(dow >= 5, weekend, -2.0 * weekend / 5.0)   # zero mean over a week
    feast_mean = p["feast_day_probability"] * np.mean(p["feast_day_extra_kcal"])
    feast = np.where(rng.random(n_days) < p["feast_day_probability"],
                     rng.uniform(*p["feast_day_extra_kcal"], n_days), 0.0) - feast_mean
    intake_off = plan + weekly + feast + rng.normal(0, _u(rng, p["daily_intake_sd_kcal"]), n_days)

    rho_F, rho_L = p["fat_energy_density_kcal_per_kg"], p["lean_energy_density_kcal_per_kg"]
    C = p["forbes_constant_kg"] * rho_L / rho_F
    g_k = _u(rng, p["glycogen_water_kg_per_1000kcal"])
    tau_g, tau_at = p["glycogen_tau_days"], p["adaptive_thermogenesis_tau_days"]

    tissue, balance, glyc = np.empty(n_days), np.full(n_days, np.nan), np.zeros(n_days)
    tissue[0] = F + L
    AT, G = 0.0, 0.0
    for t in range(1, n_days):
        AT += (p["adaptive_thermogenesis_fraction"] * plan[t] - AT) / tau_at
        d_expenditure = (p["expenditure_per_kg_fat_kcal"] * (F - F0)
                         + p["expenditure_per_kg_lean_kcal"] * (L - L0)
                         + p["activity_per_kg_body_kcal"] * ((F + L) - (F0 + L0))
                         + p["thermic_effect_fraction"] * intake_off[t]
                         + AT)
        eb = intake_off[t] - d_expenditure                    # intake minus expenditure
        lean_share = C / (C + F)                              # Forbes partition of the imbalance
        F += (1 - lean_share) * eb / rho_F
        L += lean_share * eb / rho_L
        G += (g_k * intake_off[t] / 1000.0 - G) / tau_g       # fast glycogen + bound water pool
        tissue[t], balance[t], glyc[t] = F + L, eb, G

    water = _ar1(rng, n_days, _u(rng, p["water_ar1_phi"]), _u(rng, p["water_sd_kg"]),
                 t_dof=p["water_innovation_t_dof"])
    has_cycle = rng.random() < p["slow_cycle_probability"]
    amp, period = _u(rng, p["slow_cycle_amplitude_kg"]), _u(rng, p["slow_cycle_period_days"])
    phase = rng.uniform(0, 2 * np.pi)
    cycle = amp * np.sin(2 * np.pi * np.arange(n_days) / period + phase) if has_cycle else np.zeros(n_days)
    scale_err = rng.normal(0, _u(rng, p["scale_error_sd_kg"]), n_days)
    meta = dict(sex=sex, age=age, height_cm=height, start_weight_kg=w0,
                start_fat_fraction=float(fat_frac), maintenance_kcal=float(maintenance),
                glycogen_gain=g_k, has_slow_cycle=bool(has_cycle), planned_intake_offset=plan,
                glycogen_water=glyc, **info)
    # water-like terms that scale with the noise condition are returned separately
    return tissue, balance, water + cycle, scale_err, meta, glyc


def simulate_series(mechanism, scenario, condition="base", user=0, seed_base=0,
                    config=None, n_days=None):
    """
    Simulate one user.

    mechanism : "A" or "B"        scenario / condition : keys of config/scenarios.json
    user, seed_base : together fix the random stream. The same (seed_base,
        mechanism, scenario, user) gives the SAME underlying person and true
        trajectory under every condition; only noise level, gaps and gross
        errors differ, so conditions can be compared like-for-like.
    """
    cfg = config or load_config()
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    scen_names = sorted(cfg["scenarios"])
    if scenario not in cfg["scenarios"]:
        raise ValueError(f"unknown scenario '{scenario}'")
    cond = cfg["conditions"][condition] if isinstance(condition, str) else dict(condition)
    n = int(n_days or cfg["n_days"])

    key = [int(seed_base), MECHANISMS.index(mechanism), scen_names.index(scenario), int(user)]
    latent_rng = np.random.default_rng(np.random.SeedSequence(key + [0]))
    obs_rng = np.random.default_rng(np.random.SeedSequence(key + [1]))

    if mechanism == "A":
        tissue, balance, water, scale_err, meta = _simulate_A(cfg, cfg["scenarios"][scenario], n, latent_rng)
        fast = np.zeros(n)
        mp = cfg["mechanism_A"]
        bursty, pos_share = False, 0.5
    else:
        tissue, balance, water, scale_err, meta, fast = _simulate_B(cfg, cfg["scenarios"][scenario], n, latent_rng)
        mp = cfg["mechanism_B"]
        bursty, pos_share = True, 0.7       # real gross errors skew upward (clothes, after a meal)

    ns = float(cond["noise_scale"])
    reading = tissue + fast + ns * water + ns * scale_err

    is_out = obs_rng.random(n) < cond["outlier_rate"]
    sign = np.where(obs_rng.random(n) < pos_share, 1.0, -1.0)
    mag = obs_rng.uniform(*mp["outlier_magnitude_kg"], n)
    reading = reading + is_out * sign * mag

    step = mp["rounding_kg"]
    reading = np.round(reading / step) * step
    miss = _missing_mask(obs_rng, n, cond["missing_rate"], bursty,
                         tuple(mp.get("missing_run_length_days", (2, 6))))
    observed = np.where(miss, np.nan, reading)
    is_out = is_out & ~miss

    meta.update(mechanism=mechanism, scenario=scenario,
                condition=condition if isinstance(condition, str) else "custom",
                user=int(user), seed_base=int(seed_base), **{f"cond_{k}": v for k, v in cond.items()})
    return SimulatedSeries(np.arange(n, dtype=float), observed, tissue, balance, is_out, meta)


def demo_series(start_weight=None, seed=7):
    """
    A single clearly-labelled SIMULATED 6-week log for the app's optional demo
    button. If `start_weight` is given the whole series is shifted so that it
    starts there (so the demo matches the profile the user typed).
    """
    sim = simulate_series("A", "sustained_deficit", "base", user=seed, seed_base=424242, n_days=42)
    shift = 0.0 if start_weight is None else float(start_weight) - float(sim.tissue[0])
    return [None if np.isnan(v) else round(float(v) + shift, 1) for v in sim.observed]
