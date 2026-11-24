"""
Simulation experiments for Simply-Fit.

Stages
------
dev       Development seeds only. New candidate methods and interval methods
          (no legacy code is tuned). Every design decision is taken from
          these files.
eval      Evaluation seeds only (disjoint users). All methods including the
          frozen original pipeline, on identical series.
tracking  Evaluation seeds. "As of day t" estimates over time (real-time use).
app       Development and evaluation seeds. What the application would show
          (estimate / no estimate, label, reliability) for trailing windows
          of 14-56 days. No legacy code.
(The 7-day forecast experiment is separate: experiments/forecast_experiment.py)

Usage (from the repository root)
    python -m experiments.run_experiments --stage all
    python -m experiments.run_experiments --stage all --quick     # ~1 minute smoke run
Raw per-series results are written to results/experiments/raw/*.csv
(or results/experiments/quick/ with --quick). Summaries and figures are
produced by   python -m experiments.summarize
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiments import methods as M
from src import energy_balance as eb
from src.legacy import app_estimator_v1 as app_v1
from src.simulation import MECHANISMS, load_config, simulate_series

CFG = load_config()
SCENARIOS = sorted(CFG["scenarios"])
N_DAYS = CFG["n_days"]
DEV_SEED = CFG["seeds"]["development_base"]
EVAL_SEED = CFG["seeds"]["evaluation_base"]

# calendar windows (first_day, last_day), inclusive
WINDOWS = {"last14": (N_DAYS - 14, N_DAYS - 1), "last28": (N_DAYS - 28, N_DAYS - 1),
           "last56": (N_DAYS - 56, N_DAYS - 1), "first28": (0, 27)}
MAIN_CONDITIONS = ("base", "clean")
SWEEP_CONDITIONS = ("noise_low", "noise_high", "missing_0", "missing_25", "missing_45",
                    "outliers_0", "outliers_5", "outliers_10")
MIN_OBS_TO_SCORE = 7          # a window with fewer readings is skipped for ALL methods alike
TRACK_DAYS = list(range(27, N_DAYS, 3))
TRACK_WINDOWS = (14, 21, 28, 42)
TRACK_TRUTH_DAYS = 14         # "current balance" target = mean true balance of the last 14 days
APP_WINDOWS = (14, 21, 28, 42, 56)
APP_CONDITIONS = ("base", "noise_low", "noise_high", "missing_45", "outliers_10")

SIZES = {   # users per (mechanism x scenario) cell
    "full":  {"dev_main": 100, "dev_sweep": 30, "eval_main": 150, "eval_clean": 100,
              "eval_sweep": 40, "tracking": 40, "app": 100},
    "quick": {"dev_main": 6, "dev_sweep": 3, "eval_main": 6, "eval_clean": 4,
              "eval_sweep": 3, "tracking": 3, "app": 4},
}


def _window_data(sim, a, b):
    obs = sim.observed[a:b + 1]
    m = ~np.isnan(obs)
    return sim.days[a:b + 1][m], obs[m]


def _series_task(task):
    seed_base, mech, scen, cond, user, windows, include_legacy = task
    sim = simulate_series(mech, scen, cond, user, seed_base, CFG)
    rows = []
    for wname in windows:
        a, b = WINDOWS[wname]
        days, w = _window_data(sim, a, b)
        if len(w) < MIN_OBS_TO_SCORE:
            continue
        row = {"seed_base": seed_base, "mechanism": mech, "scenario": scen, "condition": cond,
               "user": user, "window": wname, "n_obs": len(w), "n_calendar": b - a + 1,
               "n_gross_errors": int(sim.is_outlier[a:b + 1].sum()),
               "truth_balance": sim.true_mean_balance(a, b),
               "truth_tissue_equiv": sim.tissue_trend_equivalent(a, b),
               "change_day": sim.meta.get("change_day", np.nan)}
        for k, v in M.all_point_estimates(days, w, include_legacy).items():
            row[f"est_{k}"] = v
        for k, (lo, hi) in M.interval_candidates(days, w, seed=user).items():
            row[f"ci_{k}_lo"], row[f"ci_{k}_hi"] = lo, hi
        rows.append(row)
    return rows


def _app_task(task):
    """
    What the APPLICATION would show (status, label, reliability) for trailing
    windows of several lengths ending on the last day. No legacy code here.
    `label_ungated` is the interval-based label before the minimum-history
    rule is applied, so that rule can be chosen on development data.
    """
    seed_base, mech, scen, cond, user = task
    sim = simulate_series(mech, scen, cond, user, seed_base, CFG)
    rows = []
    for W in APP_WINDOWS:
        a, b = N_DAYS - W, N_DAYS - 1
        days, w = _window_data(sim, a, b)
        res = eb.estimate_energy_balance(w, days)
        row = {"seed_base": seed_base, "mechanism": mech, "scenario": scen, "condition": cond, "user": user,
               "window_days": W, "n_obs": len(w), "status": res["status"],
               "truth_balance": sim.true_mean_balance(a, b),
               "truth_tissue_equiv": sim.tissue_trend_equivalent(a, b)}
        if res["status"] == "ok":
            lo, hi = res["kcal_ci"]
            row.update(est=res["kcal_per_day"], ci_lo=lo, ci_hi=hi, halfwidth=res["kcal_ci_halfwidth"],
                       label=res["label"], reliability=eb.assess_reliability(res)["level"],
                       label_ungated=eb.classify_with_uncertainty(res["kcal_per_day"], lo, hi))
        rows.append(row)
    return rows


def _tracking_task(task):
    seed_base, mech, scen, cond, user, include_legacy = task
    sim = simulate_series(mech, scen, cond, user, seed_base, CFG)
    rows = []
    for t in TRACK_DAYS:
        days, w = _window_data(sim, 0, t)
        if len(w) < MIN_OBS_TO_SCORE:
            continue
        row = {"mechanism": mech, "scenario": scen, "condition": cond, "user": user, "day": t,
               "truth_recent": sim.true_mean_balance(t - TRACK_TRUTH_DAYS, t),
               "change_day": sim.meta.get("change_day", np.nan)}
        # original app: headline number used the whole list; bar chart used 5-entry endpoints
        if include_legacy:
            row["est_legacy_app_full_history"] = float(app_v1.estimate_balance(list(w))[0])
            row["est_legacy_5day_bars"] = float(app_v1.rolling_5day_balance(list(w))[-1])
        row["est_huber_full_history"] = M.huber(days, w)
        for W in TRACK_WINDOWS:
            keep = days > t - W
            if keep.sum() >= MIN_OBS_TO_SCORE:
                row[f"est_huber_trailing_{W}"] = M.huber(days[keep], w[keep])
                row[f"est_ols_trailing_{W}"] = M.ols_raw(days[keep], w[keep])
        rows.append(row)
    return rows


def _run(pool, fn, tasks, label):
    t0 = time.time()
    rows = []
    for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=8)):
        rows.extend(r)
        if (i + 1) % 500 == 0:
            print(f"  {label}: {i + 1}/{len(tasks)} series, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("mechanism", "scenario", "condition", "user", "window", "window_days", "day")
                 if c in df.columns]
    print(f"  {label}: done, {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    return df.sort_values(sort_cols).reset_index(drop=True)


def _cells():
    return [(m, s) for m in MECHANISMS for s in SCENARIOS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["dev", "eval", "tracking", "app", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    size = SIZES["quick" if args.quick else "full"]
    out = args.out or os.path.join("results", "experiments", "quick" if args.quick else "raw")
    os.makedirs(out, exist_ok=True)
    all_w = tuple(WINDOWS)

    with Pool(args.workers) as pool:
        if args.stage in ("dev", "all"):
            tasks = [(DEV_SEED, m, s, c, u, all_w, False)
                     for m, s in _cells() for c in MAIN_CONDITIONS for u in range(size["dev_main"])]
            tasks += [(DEV_SEED, m, s, c, u, ("last28",), False)
                      for m, s in _cells() for c in SWEEP_CONDITIONS for u in range(size["dev_sweep"])]
            _run(pool, _series_task, tasks, "dev").to_csv(os.path.join(out, "dev_series.csv"), index=False)
            ttasks = [(DEV_SEED, m, s, "base", u, False) for m, s in _cells() for u in range(size["tracking"])]
            _run(pool, _tracking_task, ttasks, "dev-tracking").to_csv(
                os.path.join(out, "dev_tracking.csv"), index=False)

        if args.stage in ("eval", "all"):
            tasks = [(EVAL_SEED, m, s, "base", u, all_w, True)
                     for m, s in _cells() for u in range(size["eval_main"])]
            tasks += [(EVAL_SEED, m, s, "clean", u, all_w, True)
                      for m, s in _cells() for u in range(size["eval_clean"])]
            tasks += [(EVAL_SEED, m, s, c, u, ("last28",), True)
                      for m, s in _cells() for c in SWEEP_CONDITIONS for u in range(size["eval_sweep"])]
            _run(pool, _series_task, tasks, "eval").to_csv(os.path.join(out, "eval_series.csv"), index=False)

        if args.stage in ("tracking", "all"):
            ttasks = [(EVAL_SEED, m, s, "base", u, True) for m, s in _cells() for u in range(size["tracking"])]
            _run(pool, _tracking_task, ttasks, "eval-tracking").to_csv(
                os.path.join(out, "eval_tracking.csv"), index=False)

        if args.stage in ("app", "all"):
            for name, seed in (("dev", DEV_SEED), ("eval", EVAL_SEED)):
                atasks = [(seed, m, s, c, u) for m, s in _cells() for c in APP_CONDITIONS
                          for u in range(size["app"])]
                _run(pool, _app_task, atasks, f"{name}-app").to_csv(
                    os.path.join(out, f"{name}_app.csv"), index=False)
    print("raw results in", out)


if __name__ == "__main__":
    main()
