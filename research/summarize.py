"""Derive tables and figures only from the archived simulation output and frozen policy."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from .experiment import RESULTS, ROOT, LEVELS, METHODS

P = json.loads((ROOT / "policy.json").read_text())
SELECTED = P["estimator"]


def ci_mean(values):
    arr = np.array(values)
    rng = np.random.default_rng(1671)
    means = np.mean(rng.choice(arr, (2000, len(arr))), axis=1)
    return np.quantile(means, [0.025, 0.975])


def point_stats(error):
    low, high = ci_mean(abs(error))
    return dict(
        mae=abs(error).mean(),
        rmse=np.sqrt(np.mean(error**2)),
        bias=error.mean(),
        mae_low=low,
        mae_high=high,
    )


def interval_table(df, scope, group, setting):
    rows = []
    for level in LEVELS:
        for method in ["ar", "block", "state", "calibrated"]:
            if method == "calibrated":
                hw = (
                    P["accepted_quantiles" if scope == "reported" else "all_quantiles"][
                        str(level)
                    ]
                    * df.risk
                )
                lo, hi = df[SELECTED] - hw, df[SELECTED] + hw
            else:
                lo, hi = df[f"{method}_lo{level}"], df[f"{method}_hi{level}"]
            hit = (lo <= df.target) & (df.target <= hi)
            interval = binomtest(int(hit.sum()), len(df)).proportion_ci(method="wilson")
            rows.append(
                dict(
                    scope=scope,
                    group=group,
                    setting=setting,
                    method=method,
                    level=level,
                    n=len(df),
                    coverage=hit.mean(),
                    lower=interval.low,
                    upper=interval.high,
                    width=(hi - lo).mean(),
                    physical_coverage=((lo <= df.truth) & (df.truth <= hi)).mean(),
                )
            )
    return rows


def main():
    full = pd.read_csv(RESULTS / "final.csv")
    df = full[full.eligible].copy()
    df["density_band"] = pd.cut(df.density, [5000, 6500, 8000, 9500])
    accept = df.risk <= P["cutoff"]
    pts = []
    paired = []
    intervals = []
    for scope, sub in [("eligible", df), ("reported", df[accept])]:
        for target in ["target", "truth"]:
            for method in ["btp"] + METHODS:
                e = sub[method] - sub[target]
                delta = abs(e) - abs(sub.huber - sub[target])
                lo, hi = ci_mean(delta)
                pts.append(
                    dict(
                        scope=scope,
                        target=target,
                        method=method,
                        n=len(sub),
                        **point_stats(e),
                    )
                )
                paired.append(
                    dict(
                        scope=scope,
                        target=target,
                        method=method,
                        n=len(sub),
                        delta=delta.mean(),
                        lower=lo,
                        upper=hi,
                    )
                )
        intervals += interval_table(sub, scope, "all", "all")
        for group in [
            "duration",
            "noise",
            "missing",
            "outliers",
            "cadence",
            "pattern",
            "density_band",
        ]:
            for key, cell in sub.groupby(group, observed=True):
                intervals += interval_table(cell, scope, group, str(key))
    s = pd.read_csv(RESULTS / "shift.csv")
    for scope, sub in [
        ("eligible", s[s.eligible]),
        ("reported", s[s.eligible & (s.risk <= P["cutoff"])]),
    ]:
        intervals += interval_table(sub, scope, "long_water", "8–14 days")
        pts.append(
            dict(
                scope="shift_" + scope,
                target="target",
                method=SELECTED,
                n=len(sub),
                **point_stats(sub[SELECTED] - sub.target),
            )
        )
    dev = pd.read_csv(RESULTS / "development.csv").dropna(subset=["risk"])
    curves = []
    for retain in [0.1, 0.25, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0]:
        cutoff = dev.risk.quantile(retain) if retain < 1 else np.inf
        cell = df[df.risk <= cutoff]
        curves.append(
            dict(
                dev_fraction=retain,
                cutoff=cutoff,
                n=len(cell),
                all_fraction=len(cell) / len(full),
                eligible_fraction=len(cell) / len(df),
                **point_stats(cell[SELECTED] - cell.target),
            )
        )
    sens = pd.read_csv(RESULTS / "sensitivity.csv")
    ref = sens[sens.factor.eq("reference") & sens.eligible].set_index("person")
    sweeps = []
    for (group, key), cell in sens.groupby(["factor", "setting"]):
        eligible = cell[cell.eligible].set_index("person")
        for method in METHODS:
            e = eligible[method] - eligible.target
            shared = e.index.intersection(ref.index)
            delta = e.loc[shared].abs() - (ref[method] - ref.target).loc[shared].abs()
            low, high = ci_mean(delta)
            sweeps.append(
                dict(
                    factor=group,
                    setting=key,
                    method=method,
                    n=len(e),
                    n_attempt=len(cell),
                    physical_mae=abs(eligible[method] - eligible.truth).mean(),
                    **point_stats(e),
                    delta_vs_ref=delta.mean(),
                    delta_low=low,
                    delta_high=high,
                    n_pair=len(delta),
                )
            )
        intervals += interval_table(eligible, "eligible", "sweep_" + group, key)
    tables = dict(
        points=pts,
        paired=paired,
        intervals=intervals,
        selection_curve=curves,
        ablations=sweeps,
    )
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)
    plot(tables)


def plot(tables):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    pts = (
        pd.DataFrame(tables["points"])
        .query("scope=='eligible' and target=='target'")
        .sort_values("mae")
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    ax.barh(
        pts.method,
        pts.mae,
        xerr=[pts.mae - pts.mae_low, pts.mae_high - pts.mae],
        color="#28786b",
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Mean absolute error (kcal/day), with 95% bootstrap intervals",
        title="Estimating average tissue change on the same eligible logs",
    )
    fig.savefig(RESULTS / "estimators.png")
    plt.close(fig)
    tab = pd.DataFrame(tables["intervals"]).query("scope=='eligible' and group=='all'")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for method, c in tab.groupby("method"):
        axes[0].plot(100 * c.level, 100 * c.coverage, "o-", label=method)
        axes[1].plot(100 * c.level, c.width, "o-", label=method)
    axes[0].plot([80, 95], [80, 95], "k:", label="nominal")
    axes[0].set(ylabel="Empirical coverage (%)")
    axes[1].set(ylabel="Mean interval width (kcal/day)")
    for ax in axes:
        ax.set_xlabel("Nominal coverage (%)")
    axes[0].legend(fontsize=8)
    fig.savefig(RESULTS / "intervals.png")
    plt.close(fig)
    c = pd.DataFrame(tables["selection_curve"])
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    ax.fill_between(100 * c.all_fraction, c.mae_low, c.mae_high, color="#b6d9ce")
    ax.plot(100 * c.all_fraction, c.mae, "o-", color="#175a50")
    best = c[c.dev_fraction.eq(0.75)].iloc[0]
    ax.scatter(
        [100 * best.all_fraction],
        [best.mae],
        color="#ad472b",
        zorder=5,
        label="Deployed cutoff",
    )
    ax.set(
        xlabel="Percentage of all submitted logs receiving an estimate",
        ylabel="MAE among reported logs (kcal/day)",
        title="Error as the reporting rule admits more logs",
    )
    ax.legend()
    fig.savefig(RESULTS / "selective_risk.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
