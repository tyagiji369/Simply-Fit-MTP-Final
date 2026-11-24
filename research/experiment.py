"""Run selection, calibration and evaluation in separate invocations."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from src import energy_balance as eb
from src.research_estimate import candidates
from src.legacy.app_estimator_v1 import estimate_balance
from .world import ROOT, DESIGN, simulate

RESULTS = ROOT / "results"
METHODS = ["huber", "ols", "state", "gaussian"]
LEVELS = DESIGN["coverage"]
SWEEPS = {
    "duration": [14, 28, 56, 84],
    "noise": [0.12, 0.5],
    "cadence": [1, 3],
    "missing": ["random", "informative"],
    "outliers": [0, 0.08],
    "density": [5200, 9200],
    "pattern": ["steady", "step", "ramp"],
    "adaptation": [0, 0.35],
}
REFERENCE = dict(
    duration=56,
    noise=0.28,
    cadence=1,
    missing="random",
    outliers=0.025,
    density=7700,
    pattern="step",
    adaptation=0.18,
)


def one(job):
    split, i, override, factor, setting = job
    d, w, truth, equiv, params = simulate(split, i, **override)
    row = dict(
        person=i,
        **params,
        factor=factor,
        setting=setting,
        n=len(w),
        truth=truth,
        target=equiv,
    )
    r = eb.estimate_energy_balance(w, d)
    row["eligible"] = r["status"] == "ok"
    if not row["eligible"]:
        return row
    states, risk = candidates(r)
    row.update(huber=r["kcal_per_day"], ols=r["ols_kcal_per_day"], risk=risk)
    for name, (point, se, _) in states.items():
        row[name] = point
        row[name + "_se"] = se
    if split == "final":
        row["btp"] = estimate_balance(w)[0]
    # One common block sample for all three quantiles.
    fit = r["trend_line"]
    b = int(np.ceil(len(d) ** (1 / 3))) + 1
    rng = np.random.default_rng([DESIGN["splits"][split][0], i, 42])
    starts = rng.integers(
        0, len(d) - b + 1, (DESIGN["bootstrap_draws"], int(np.ceil(len(d) / b)))
    )
    idx = (starts[:, :, None] + np.arange(b)).reshape(len(starts), -1)[:, : len(d)]
    boot = 7700 * eb.fit_line_batch(d, fit + (w - fit)[idx])[0]
    for level in LEVELS:
        a = (1 - level) / 2
        for name, point, hw in [
            ("ar", row["huber"], 7700 * t.ppf(1 - a, r["dof"]) * r["slope_se"]),
            ("state", row["state"], norm.ppf(1 - a) * row["state_se"]),
        ]:
            row[f"{name}_lo{level}"], row[f"{name}_hi{level}"] = point - hw, point + hw
        row[f"block_lo{level}"], row[f"block_hi{level}"] = np.quantile(boot, [a, 1 - a])
    return row


def panel(split):
    n = DESIGN["splits"][split][1]
    settings = (
        [({}, "mix", "mix")]
        if split != "sensitivity"
        else [(REFERENCE, "reference", "base")]
        + [
            (REFERENCE | {key: value}, key, str(value))
            for key, values in SWEEPS.items()
            for value in values
        ]
    )
    jobs = [
        (split, i, override, factor, level)
        for override, factor, level in settings
        for i in range(n)
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        df = pd.DataFrame(pool.map(one, jobs, chunksize=8))
    df.to_csv(RESULTS / f"{split}.csv", index=False)
    return df[df.eligible].copy()


def quantiles(values):
    values = np.sort(values)
    return {
        str(v): float(values[min(len(values), int(np.ceil((len(values) + 1) * v))) - 1])
        for v in LEVELS
    }


def run(stage):
    if stage == "select":
        df = panel("development")
        losses = df[METHODS].sub(df.target, axis=0).abs()
        winner = losses.mean().idxmin()
        # Prefer the incumbent when its paired difference is within one standard error.
        diff = losses.huber - losses[winner]
        chosen = "huber" if diff.mean() <= diff.sem() else winner
        config = dict(
            estimator=chosen,
            dev_mae=losses.mean().to_dict(),
            cutoff=float(df.risk.quantile(DESIGN["desired_eligible_retention"])),
        )
        (ROOT / "selection.json").write_text(json.dumps(config, indent=2))
    elif stage == "calibrate":
        config = json.loads((ROOT / "selection.json").read_text())
        df = panel("validation")
        error = (df[config["estimator"]] - df.target).abs() / df.risk
        config.update(
            all_quantiles=quantiles(error),
            accepted_quantiles=quantiles(error[df.risk <= config["cutoff"]]),
        )
        (ROOT / "policy.json").write_text(json.dumps(config, indent=2))
    else:
        # Validate availability of the frozen policy before generating final logs.
        json.loads((ROOT / "policy.json").read_text())
        panel(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["select", "calibrate", "final", "sensitivity", "shift"]
    )
    run(parser.parse_args().stage)
