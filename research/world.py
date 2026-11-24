"""Event-driven stress world. No estimator imports. Each ID has independent random streams."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DESIGN = json.loads((ROOT / "design.json").read_text())


def simulate(split, person, **overrides):
    seed = DESIGN["splits"][split][0]
    rng = np.random.default_rng([seed, person])
    # Generate a fixed horizon before cropping, so one-factor experiments reuse innovations.
    t = np.arange(84)
    params = dict(
        duration=int(rng.choice(DESIGN["durations"])),
        cadence=int(rng.choice(DESIGN["cadences"])),
        noise=float(rng.choice(DESIGN["noise"])),
        outliers=float(rng.choice(DESIGN["outliers"])),
        density=rng.uniform(*DESIGN["density"]),
        missing=rng.choice(["random", "informative"]),
        pattern=rng.choice(["steady", "step", "ramp"]),
        adaptation=rng.uniform(0.0, 0.35),
    )
    params.update(overrides)
    initial, change = rng.uniform(-550, 550, 2)
    switch = int(rng.integers(7, 35))
    step = (t >= switch).astype(float)
    ramp = np.clip((t - switch) / 18, 0, 1)
    demand = (
        initial + change * {"steady": 0, "step": step, "ramp": ramp}[params["pattern"]]
    )
    demand = np.broadcast_to(demand, t.shape)
    response = np.zeros(len(t))
    tau = rng.uniform(10, 35)
    for i in range(1, len(t)):
        response[i] = response[i - 1] + (demand[i - 1] - response[i - 1]) / tau
    energy = demand - params["adaptation"] * response
    tissue = rng.uniform(58, 105) + np.r_[0, np.cumsum(energy[:-1] / params["density"])]
    # Random pulses represent water/gut contents, including events before the study.
    events = rng.random(114) < 0.16
    amplitudes = rng.uniform(0.18, 1.25, 114) * rng.choice([-1, 1], 114)
    decay = rng.uniform(*DESIGN["water_decay_days"][split])
    water = np.convolve(events * amplitudes, np.exp(-np.arange(114) / decay))[:114][
        -84:
    ]
    offset = rng.choice([-1, 1]) * rng.uniform(0.35, 1.15)
    start, length = rng.integers(3, 42), rng.integers(3, 14)
    water += offset * ((t >= start) & (t < start + length))
    sigma = params["noise"]
    weight = np.round(
        tissue
        + water
        + sigma * rng.standard_t(5, 84) * np.sqrt(3 / 5)
        + (rng.random(84) < params["outliers"])
        * rng.choice([-1, 1], 84)
        * rng.uniform(1.2, 3, 84),
        1,
    )
    # Missingness depends on today's unobserved water deviation; same uniform variates for paired sweeps.
    risk = (
        0.14
        if params["missing"] == "random"
        else 0.05 + 0.55 / (1 + np.exp(-3 * (water - 0.35)))
    )
    phase = int(rng.integers(0, 6)) % params["cadence"]
    take = (
        (t < params["duration"])
        & ((t - phase) % params["cadence"] == 0)
        & (rng.random(84) >= risk)
    )
    days = t[take]
    truth = equivalent = np.nan
    if len(days) > 1:
        a, b = days[[0, -1]]
        truth = energy[a:b].mean()
        equivalent = (tissue[b] - tissue[a]) * 7700 / (b - a)
    return days, weight[take], truth, equivalent, params
