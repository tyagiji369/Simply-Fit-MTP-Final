"""
Turn the raw per-series results into tables, figures and summary.json.

    python -m experiments.summarize                 # full results
    python -m experiments.summarize --quick         # the smoke-run results

Outputs (results/experiments/ or results/experiments/quick/):
    tables/*.md      markdown tables (also embedded into EXPERIMENT_RESULTS.md)
    figures/*.png
    summary.json     key numbers used by the reports and by the app
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments import methods as M
from experiments.run_experiments import EVAL_SEED, WINDOWS
from src import energy_balance as eb
from src.legacy import ml_engine_v1 as ml_v1
from src.simulation import load_config, simulate_series

TOL = eb.CLASS_TOLERANCE_KCAL
POINT_METHODS = ["legacy_research", "legacy_app", "legacy_timefix", "ewma_ols", "ols_raw", "theil_sen", "huber"]
NAMES = {
    "legacy_research": "Original: research module (EWMA + IF + EWMA)",
    "legacy_app": "Original: app (IF-replaced raw)",
    "legacy_timefix": "Original steps, real day index",
    "ewma_ols": "EWMA then regression (no IF)",
    "ols_raw": "Least squares on raw readings",
    "theil_sen": "Theil-Sen",
    "huber": "Huber regression (recommended)",
}
SHORT = {"legacy_research": "Original (research)", "legacy_app": "Original (app)",
         "legacy_timefix": "Original + real days", "ewma_ols": "EWMA + OLS", "ols_raw": "OLS raw",
         "theil_sen": "Theil-Sen", "huber": "Huber"}
# fixed entity -> colour assignment (validated categorical order; colour follows the method)
COLOR = {"huber": "#2a78d6", "legacy_research": "#eb6834", "legacy_app": "#1baf7a", "ols_raw": "#eda100",
         "ewma_ols": "#e87ba4", "theil_sen": "#008300", "legacy_timefix": "#6250d6"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
CI_METHODS = ["ols_classical", "ols_hac", "ols_ar1", "huber_classical", "huber_ar1_nobc", "huber_ar1",
              "huber_block_bootstrap"]
CI_NAMES = {"ols_classical": "Classical OLS", "ols_hac": "Newey-West HAC (OLS)", "ols_ar1": "AR(1) sandwich (OLS)",
            "huber_classical": "Huber, independence", "huber_ar1_nobc": "Huber AR(1), no small-sample correction",
            "huber_ar1": "Huber AR(1) sandwich (recommended)", "huber_block_bootstrap": "Huber, block bootstrap",
            "legacy": "Original 'kcal_per_day_ci95'"}
WIN_ORDER = ["last14", "last28", "last56", "first28"]
WIN_NAMES = {"last14": "last 14 d", "last28": "last 28 d", "last56": "last 56 d", "first28": "first 28 d"}


# ------------------------------------------------------------------ helpers
def md_table(df, floatfmt="{:.0f}", index=True):
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    count_cols = {"n", "n users", "days of data", "horizon", "n readings"}
    for _, r in d.iterrows():
        cells = []
        for c, v in zip(cols, r):
            if isinstance(v, (float, np.floating)) and not np.isnan(v) and c in count_cols:
                cells.append(str(int(round(v))))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def err(df, m, truth="truth_balance"):
    return df[f"est_{m}"] - df[truth]


def metrics(df, methods, truth="truth_balance"):
    rows = {}
    for m in methods:
        e = err(df, m, truth).to_numpy()
        rows[NAMES.get(m, m)] = {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
                                 "Bias": np.mean(e), "Median AE": np.median(np.abs(e))}
    return pd.DataFrame(rows).T


def label3(x):
    return np.where(x < -TOL, "deficit", np.where(x > TOL, "surplus", "maintenance"))


def coverage(df, lo, hi, truth):
    return float(((df[lo] <= df[truth]) & (df[truth] <= df[hi])).mean())


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new_fig(w, h, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=150, **kw)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        style_ax(ax)
    return fig, axes


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    base = os.path.join("results", "experiments", "quick" if args.quick else "")
    raw = os.path.join("results", "experiments", "quick" if args.quick else "raw")
    tdir, fdir = os.path.join(base, "tables"), os.path.join(base, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    dev = pd.read_csv(os.path.join(raw, "dev_series.csv"))
    devt = pd.read_csv(os.path.join(raw, "dev_tracking.csv"))
    ev = pd.read_csv(os.path.join(raw, "eval_series.csv"))
    evt = pd.read_csv(os.path.join(raw, "eval_tracking.csv"))
    fc = pd.read_csv(os.path.join(raw, "eval_forecast.csv"))
    dfc = pd.read_csv(os.path.join(raw, "dev_forecast.csv"))
    dapp = pd.read_csv(os.path.join(raw, "dev_app.csv"))
    eapp = pd.read_csv(os.path.join(raw, "eval_app.csv"))
    T, S = {}, {}

    def put(name, text):
        T[name] = text
        with open(os.path.join(tdir, name + ".md"), "w") as f:
            f.write(text + "\n")

    S["n_series"] = {"dev_rows": int(len(dev)), "eval_rows": int(len(ev)),
                     "dev_users": int(dev.groupby(["mechanism", "scenario", "condition", "user"]).ngroups),
                     "eval_users": int(ev.groupby(["mechanism", "scenario", "condition", "user"]).ngroups),
                     "tracking_rows": int(len(evt)), "forecast_rows": int(len(fc)),
                     "app_rows_eval": int(len(eapp)), "app_users_eval": int(len(eapp) // 5)}

    # ---------------- DEV: decisions -------------------------------------
    cand = ["ols_raw", "huber", "theil_sen", "ewma_ols"]
    d_main = dev[dev.condition.isin(["base", "clean"])]
    t = d_main.groupby(["condition", "mechanism", "window"]).apply(
        lambda g: pd.Series({SHORT[m]: np.abs(err(g, m)).mean() for m in cand} | {"n": len(g)}),
        include_groups=False)
    put("dev_point_mae", md_table(t))
    d28 = dev[dev.window == "last28"]
    t = d28.groupby("condition").apply(
        lambda g: pd.Series({SHORT[m]: np.abs(err(g, m)).mean() for m in cand} | {"n": len(g)}),
        include_groups=False)
    put("dev_sweep_mae", md_table(t))
    rows = []
    for (mech, win), g in d_main.groupby(["mechanism", "window"]):
        for c in CI_METHODS:
            rows.append({"mechanism": mech, "window": WIN_NAMES[win], "interval": CI_NAMES[c],
                         "coverage of tissue-trend equivalent": coverage(g, f"ci_{c}_lo", f"ci_{c}_hi", "truth_tissue_equiv"),
                         "coverage of true balance": coverage(g, f"ci_{c}_lo", f"ci_{c}_hi", "truth_balance"),
                         "mean width (kcal/day)": float((g[f"ci_{c}_hi"] - g[f"ci_{c}_lo"]).mean())})
    dci = pd.DataFrame(rows)
    piv = dci.pivot_table(index="interval", columns=["mechanism", "window"],
                          values="coverage of tissue-trend equivalent").reindex([CI_NAMES[c] for c in CI_METHODS])
    piv.columns = [f"{a} {b}" for a, b in piv.columns]
    put("dev_interval_coverage", md_table(piv, "{:.2f}"))
    S["dev_interval_pooled"] = {c: round(coverage(d_main, f"ci_{c}_lo", f"ci_{c}_hi", "truth_tissue_equiv"), 3)
                                for c in CI_METHODS}

    def phase(df):
        return np.where(df.change_day.isna(), "no change in scenario",
                        np.where(df.day < df.change_day, "before the change",
                                 np.where(df.day < df.change_day + 28, "0-27 days after change",
                                          "28+ days after change")))
    devt["phase"] = phase(devt)
    evt["phase"] = phase(evt)
    tcols = ["huber_trailing_14", "huber_trailing_21", "huber_trailing_28", "huber_trailing_42", "huber_full_history"]
    t = devt.groupby("phase").apply(
        lambda g: pd.Series({c.replace("huber_", "").replace("_", " "): np.abs(g[f"est_{c}"] - g.truth_recent).mean()
                             for c in tcols}), include_groups=False)
    t.loc["worst phase"] = t.max(axis=0)
    put("dev_tracking_window", md_table(t))
    S["dev_tracking_worst_phase"] = {k: round(float(v)) for k, v in t.loc["worst phase"].items()}

    # ---------------- EVAL: main comparison ------------------------------
    for cond in ("base", "clean"):
        e = ev[ev.condition == cond]
        rows = []
        for (mech, win), g in e.groupby(["mechanism", "window"]):
            mt = metrics(g, POINT_METHODS)
            for name, r in mt.iterrows():
                rows.append({"mechanism": mech, "window": WIN_NAMES[win], "method": name, **r.to_dict(), "n": len(g)})
        full = pd.DataFrame(rows)
        full.to_csv(os.path.join(tdir, f"eval_{cond}_full_metrics.csv"), index=False)
        piv = full.pivot_table(index="method", columns=["mechanism", "window"], values="MAE")
        piv = piv.reindex([NAMES[m] for m in POINT_METHODS])
        piv = piv[[(m, WIN_NAMES[w]) for m in ("A", "B") for w in WIN_ORDER]]
        piv.columns = [f"{a} {b}" for a, b in piv.columns]
        put(f"eval_{cond}_mae", md_table(piv))
        pivb = full.pivot_table(index="method", columns=["mechanism", "window"], values="Bias")
        pivb = pivb.reindex([NAMES[m] for m in POINT_METHODS])[[(m, WIN_NAMES[w]) for m in ("A", "B") for w in WIN_ORDER]]
        pivb.columns = [f"{a} {b}" for a, b in pivb.columns]
        put(f"eval_{cond}_bias", md_table(pivb, "{:+.0f}"))
        pivr = full.pivot_table(index="method", columns=["mechanism", "window"], values="RMSE")
        pivr = pivr.reindex([NAMES[m] for m in POINT_METHODS])[[(m, WIN_NAMES[w]) for m in ("A", "B") for w in WIN_ORDER]]
        pivr.columns = [f"{a} {b}" for a, b in pivr.columns]
        put(f"eval_{cond}_rmse", md_table(pivr))
        S[f"eval_{cond}_mae"] = {f"{mech}_{win}": {m: round(float(np.abs(err(g, m)).mean()), 4) for m in POINT_METHODS}
                                 for (mech, win), g in e.groupby(["mechanism", "window"])}

    eb28 = ev[(ev.condition == "base") & (ev.window == "last28")]
    t = eb28.groupby(["mechanism", "scenario"]).apply(
        lambda g: pd.Series({SHORT[m]: np.abs(err(g, m)).mean() for m in ["legacy_research", "legacy_app", "ols_raw", "huber"]}
                            | {"Huber bias": err(g, "huber").mean(), "n": len(g)}), include_groups=False)
    put("eval_by_scenario", md_table(t))

    # sweeps
    sweeps = {"noise": ["noise_low", "base", "noise_high"], "missing": ["missing_0", "base", "missing_25", "missing_45"],
              "outliers": ["outliers_0", "base", "outliers_5", "outliers_10"]}
    e28 = ev[ev.window == "last28"]
    # like-for-like: restrict every condition to the users present in the sweep cells
    sweep_users = e28[e28.condition == "noise_low"][["mechanism", "scenario", "user"]].drop_duplicates()
    e28s = e28.merge(sweep_users, on=["mechanism", "scenario", "user"])
    for name, conds in sweeps.items():
        t = e28s[e28s.condition.isin(conds)].groupby("condition").apply(
            lambda g: pd.Series({SHORT[m]: np.abs(err(g, m)).mean() for m in POINT_METHODS}
                                | {"bias original (app)": err(g, "legacy_app").mean(),
                                   "bias Huber": err(g, "huber").mean(), "n": len(g)}),
            include_groups=False).reindex(conds)
        put(f"eval_sweep_{name}", md_table(t))
        S[f"sweep_{name}"] = {c: {m: round(float(np.abs(err(g, m)).mean()), 4) for m in POINT_METHODS}
                              for c, g in e28s[e28s.condition.isin(conds)].groupby("condition")}

    # paired comparisons (each row = an independent simulated user)
    rows = []
    groups = {"base, last 28 d": ev[(ev.condition == "base") & (ev.window == "last28")],
              "base, last 56 d": ev[(ev.condition == "base") & (ev.window == "last56")],
              "base, last 14 d": ev[(ev.condition == "base") & (ev.window == "last14")],
              "clean, last 28 d": ev[(ev.condition == "clean") & (ev.window == "last28")],
              "clean, last 56 d": ev[(ev.condition == "clean") & (ev.window == "last56")],
              "missing 45%, last 28 d": e28[e28.condition == "missing_45"],
              "gross errors 10%, last 28 d": e28[e28.condition == "outliers_10"],
              "high noise, last 28 d": e28[e28.condition == "noise_high"]}
    S["paired"] = {}
    for gname, g in groups.items():
        for ref in ("legacy_research", "legacy_app", "ewma_ols", "ols_raw"):
            diff = (np.abs(err(g, "huber")) - np.abs(err(g, ref))).to_numpy()
            lo, hi = boot_ci(diff)
            try:
                p = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                p = float("nan")
            rows.append({"data": gname, "Huber compared with": SHORT[ref], "n users": len(g),
                         "mean change in abs. error (kcal/day)": diff.mean(), "95% CI low": lo, "95% CI high": hi,
                         "Wilcoxon p": p, "share of users where Huber is closer": float((diff < 0).mean())})
            S["paired"][f"{gname} vs {ref}"] = {"diff": round(float(diff.mean()), 4), "lo": round(lo, 4),
                                             "hi": round(hi, 4), "p": p, "n": int(len(g))}
    pt = pd.DataFrame(rows)
    pt.to_csv(os.path.join(tdir, "eval_paired.csv"), index=False)
    disp = pt.copy()
    disp["mean change in abs. error (kcal/day)"] = disp.apply(
        lambda r: f"{r['mean change in abs. error (kcal/day)']:+.1f} ({r['95% CI low']:+.1f} to {r['95% CI high']:+.1f})", axis=1)
    disp["Wilcoxon p"] = disp["Wilcoxon p"].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    disp["share of users where Huber is closer"] = disp["share of users where Huber is closer"].map("{:.0%}".format)
    put("eval_paired", md_table(disp.drop(columns=["95% CI low", "95% CI high"]), index=False))

    # classification
    rows = []
    for gname, g in {"base, last 28 d": groups["base, last 28 d"], "base, last 56 d": groups["base, last 56 d"],
                     "base, last 14 d": groups["base, last 14 d"]}.items():
        truth = label3(g.truth_balance)
        row = {"data": gname, "n": len(g)}
        for m in POINT_METHODS:
            row[SHORT[m]] = float((label3(g[f"est_{m}"]) == truth).mean())
        rows.append(row)
    put("eval_classification", md_table(pd.DataFrame(rows).set_index("data"), "{:.3f}"))
    def wrong_sign(g, col):
        return ((g[col] == "deficit") & (g.truth_balance > 0)) | ((g[col] == "surplus") & (g.truth_balance < 0))

    def label_table(app, col):
        rows_ = []
        for W, g in app[(app.condition == "base")].groupby("window_days"):
            ok = g[g.status == "ok"]
            dec = ok[col] != "inconclusive"
            rows_.append({"days of data": W, "n users": len(g),
                          "no estimate shown": float((g.status != "ok").mean()),
                          "definite label given": float(dec.sum() / len(g)),
                          "definite labels equal to the true class": float((ok[col][dec] == label3(ok.truth_balance)[dec]).mean()) if dec.any() else np.nan,
                          "definite labels pointing the WRONG WAY": float(wrong_sign(ok, col)[dec].mean()) if dec.any() else np.nan})
        return pd.DataFrame(rows_).set_index("days of data")
    put("dev_labels_ungated", md_table(label_table(dapp, "label_ungated"), "{:.3f}"))
    put("eval_labels_ungated", md_table(label_table(eapp, "label_ungated"), "{:.3f}"))
    lt = label_table(eapp, "label")
    put("eval_decisive_labels", md_table(lt, "{:.3f}"))
    S["decisive"] = {str(W): {k: round(float(v), 3) for k, v in r.items()} for W, r in lt.iterrows()}
    # the original app always printed a signed number; how often was its sign wrong?
    rows_ = []
    for win in ("last14", "last28", "last56"):
        g_ = ev[(ev.condition == "base") & (ev.window == win)]
        lab = label3(g_.est_legacy_app)
        ws = ((lab == "deficit") & (g_.truth_balance > 0)) | ((lab == "surplus") & (g_.truth_balance < 0))
        rows_.append({"days of data": WIN_NAMES[win], "n users": len(g_),
                      "original app: share of deficit/surplus outputs": float((lab != "maintenance").mean()),
                      "of which pointing the WRONG WAY": float(ws[lab != "maintenance"].mean())})
    put("eval_legacy_wrong_sign", md_table(pd.DataFrame(rows_).set_index("days of data"), "{:.3f}"))
    S["legacy_wrong_sign"] = {r["days of data"]: round(r["of which pointing the WRONG WAY"], 3) for r in rows_}

    # intervals on evaluation data
    rows = []
    for cond in ("base", "clean"):
        for (mech, win), g in ev[ev.condition == cond].groupby(["mechanism", "window"]):
            for c in CI_METHODS + ["legacy"]:
                lo, hi = (("est_legacy_research_ci_lo", "est_legacy_research_ci_hi") if c == "legacy"
                          else (f"ci_{c}_lo", f"ci_{c}_hi"))
                rows.append({"condition": cond, "mechanism": mech, "window": WIN_NAMES[win], "interval": CI_NAMES[c],
                             "cov_equiv": coverage(g, lo, hi, "truth_tissue_equiv"),
                             "cov_true": coverage(g, lo, hi, "truth_balance"),
                             "width": float((g[hi] - g[lo]).mean()), "n": len(g)})
    eci = pd.DataFrame(rows)
    eci.to_csv(os.path.join(tdir, "eval_interval_full.csv"), index=False)
    order = [CI_NAMES["legacy"]] + [CI_NAMES[c] for c in CI_METHODS]
    b = eci[eci.condition == "base"]
    for val, nm, fmt in (("cov_equiv", "eval_interval_coverage", "{:.2f}"), ("cov_true", "eval_interval_coverage_true", "{:.2f}"),
                         ("width", "eval_interval_width", "{:.0f}")):
        piv = b.pivot_table(index="interval", columns=["mechanism", "window"], values=val).reindex(order)
        piv = piv[[(m, WIN_NAMES[w]) for m in ("A", "B") for w in WIN_ORDER]]
        piv.columns = [f"{a} {b_}" for a, b_ in piv.columns]
        put(nm, md_table(piv, fmt))
    S["eval_coverage"] = {f"{r.mechanism}_{r.window}|{r.interval}": {"equiv": round(r.cov_equiv, 3), "true": round(r.cov_true, 3),
                                                                   "width": round(r.width)} for r in b.itertuples()}
    # inflation that WOULD have been needed (reported as a limitation, not applied)
    rows = []
    for (mech, win), g in ev[ev.condition == "base"].groupby(["mechanism", "window"]):
        hw = (g.ci_huber_ar1_hi - g.ci_huber_ar1_lo) / 2
        rows.append({"mechanism": mech, "window": WIN_NAMES[win],
                     "interval width multiplier needed for 95% coverage": float(np.quantile(np.abs(g.est_huber - g.truth_tissue_equiv) / hw, 0.95))})
    put("eval_interval_inflation", md_table(pd.DataFrame(rows).pivot(index="window", columns="mechanism",
                                                                    values="interval width multiplier needed for 95% coverage").reindex([WIN_NAMES[w] for w in WIN_ORDER]), "{:.2f}"))

    # reliability indicator validation (application output, all five conditions and window lengths)
    relg = eapp[eapp.status == "ok"]
    relg = relg.assign(abs_err=np.abs(relg.est - relg.truth_balance),
                       covered=(relg.ci_lo <= relg.truth_tissue_equiv) & (relg.truth_tissue_equiv <= relg.ci_hi))
    t = relg.groupby("reliability").apply(
        lambda x: pd.Series({"n": len(x), "MAE (kcal/day)": x.abs_err.mean(), "median abs. error": x.abs_err.median(),
                             "90th percentile abs. error": x.abs_err.quantile(0.9),
                             "share within 150 kcal/day": (x.abs_err <= 150).mean(),
                             "interval coverage": x.covered.mean()}), include_groups=False).reindex(["high", "moderate", "low"])
    put("eval_reliability", md_table(t, "{:.2f}"))
    S["reliability"] = {lvl: {"n": int(r["n"]), "mae": round(float(r["MAE (kcal/day)"]), 4),
                              "median": round(float(r["median abs. error"]), 4),
                              "p90": round(float(r["90th percentile abs. error"]), 4),
                              "within150": round(float(r["share within 150 kcal/day"]), 3),
                              "coverage": round(float(r["interval coverage"]), 3)}
                        for lvl, r in t.iterrows() if not np.isnan(r["n"])}
    b_app = eapp[eapp.condition == "base"]
    t2 = b_app.groupby("window_days").apply(
        lambda x: pd.Series({"no estimate (insufficient data)": (x.status != "ok").mean(),
                             "high": (x.reliability == "high").mean(), "moderate": (x.reliability == "moderate").mean(),
                             "low": (x.reliability == "low").mean()}), include_groups=False)
    t2.index = [f"{w} days" for w in t2.index]
    t2.index.name = "days of data in the window"
    put("eval_reliability_by_window", md_table(t2, "{:.2f}"))
    t3 = eapp.groupby("condition").apply(
        lambda x: pd.Series({"no estimate": (x.status != "ok").mean(), "high": (x.reliability == "high").mean(),
                             "moderate": (x.reliability == "moderate").mean(), "low": (x.reliability == "low").mean()}),
        include_groups=False)
    put("eval_reliability_by_condition", md_table(t3, "{:.2f}"))

    # shrinkage diagnostic
    rows = []
    for cond in ("base", "clean"):
        for win in ("last14", "last28", "last56"):
            g = ev[(ev.condition == cond) & (ev.window == win)]
            sus = g[g.scenario.isin(["sustained_deficit", "sustained_surplus"])]
            row = {"data": f"{cond}, {WIN_NAMES[win]}", "mean shrink factor k": g.est_attenuation_factor.mean()}
            for m, nm in (("huber", "Huber"), ("huber_shrunk_like_ewma", "Huber x k"), ("ewma_ols", "EWMA + OLS"),
                          ("ewma_ols_unshrunk", "(EWMA + OLS) / k"), ("legacy_research", "Original (research)")):
                row[f"MAE {nm}"] = np.abs(err(g, m)).mean()
            for m, nm in (("huber", "Huber"), ("ewma_ols", "EWMA + OLS"), ("legacy_research", "Original (research)")):
                row[f"size ratio {nm}"] = float(np.sum(sus[f"est_{m}"] * sus.truth_balance) / np.sum(sus.truth_balance ** 2))
            rows.append(row)
    sh = pd.DataFrame(rows).set_index("data")
    put("eval_shrinkage", md_table(sh, "{:.2f}"))
    S["shrinkage"] = {i: {k: round(float(v), 3) for k, v in r.items()} for i, r in sh.iterrows()}

    # mechanism B: conversion-factor error
    gb = ev[(ev.condition == "clean") & (ev.mechanism == "B") & (ev.window == "last56")
            & ev.scenario.isin(["sustained_deficit", "sustained_surplus"])]
    ratio = gb.truth_tissue_equiv / gb.truth_balance
    S["mechB_conversion"] = {"n": int(len(gb)), "median_ratio": round(float(ratio.median()), 3),
                             "p5": round(float(ratio.quantile(0.05)), 3), "p95": round(float(ratio.quantile(0.95)), 3),
                             "median_excess": round(float(ratio.median() - 1), 3),
                             "p5_excess": round(float(ratio.quantile(0.05) - 1), 3),
                             "p95_excess": round(float(ratio.quantile(0.95) - 1), 3),
                             "mae_vs_true": round(float(np.abs(gb.est_huber - gb.truth_balance).mean()), 4),
                             "mae_vs_equiv": round(float(np.abs(gb.est_huber - gb.truth_tissue_equiv).mean()), 4)}
    t = ev[(ev.condition == "base")].groupby(["mechanism", "window"]).apply(
        lambda x: pd.Series({"Huber MAE vs true balance": np.abs(x.est_huber - x.truth_balance).mean(),
                             "Huber MAE vs tissue-trend equivalent": np.abs(x.est_huber - x.truth_tissue_equiv).mean(),
                             "bias vs true balance": (x.est_huber - x.truth_balance).mean()}), include_groups=False)
    put("eval_two_truths", md_table(t, "{:+.0f}"))

    # tracking
    tm = {"legacy_app_full_history": "Original app headline (whole list)", "legacy_5day_bars": "Original app 5-entry bar chart",
          "huber_full_history": "Huber, whole history", "huber_trailing_14": "Huber, trailing 14 d",
          "huber_trailing_28": "Huber, trailing 28 d (default)", "huber_trailing_42": "Huber, trailing 42 d"}
    t = evt.groupby("phase").apply(
        lambda g_: pd.Series({nm: np.abs(g_[f"est_{c}"] - g_.truth_recent).mean() for c, nm in tm.items()}),
        include_groups=False).T
    t["all days"] = [np.abs(evt[f"est_{c}"] - evt.truth_recent).mean() for c in tm]
    t.index.name = "method"
    put("eval_tracking", md_table(t))
    S["tracking"] = {c: {ph: round(float(np.abs(g_[f"est_{c}"] - g_.truth_recent).mean()), 4) for ph, g_ in evt.groupby("phase")}
                     | {"all": round(float(np.abs(evt[f"est_{c}"] - evt.truth_recent).mean()), 4)} for c in tm}

    # forecast (secondary feature)
    fm = {"err_level": "Recent level: mean of last 3 readings (app default)", "err_persistence": "Persistence (last reading)",
          "err_linear14": "Linear trend, last 14 days", "err_legacy_fallback": "Original fallback (2-point)"}

    def fc_table(d_):
        sub = d_[d_.complete_14 & d_.err_lstm_v2.notna()]
        out_ = {}
        for c, nm in fm.items():
            out_[nm] = {"MAE (kg), all forecasts": d_[c].abs().mean(),
                        "MAE (kg), model A": d_[d_.mechanism == "A"][c].abs().mean(),
                        "MAE (kg), model B": d_[d_.mechanism == "B"][c].abs().mean(),
                        "MAE (kg), subset with 14 complete days": sub[c].abs().mean() if len(sub) else np.nan}
        if len(sub):
            out_["Optional LSTM, window-centred (needs 14 complete days)"] = {
                "MAE (kg), all forecasts": np.nan, "MAE (kg), model A": np.nan, "MAE (kg), model B": np.nan,
                "MAE (kg), subset with 14 complete days": sub.err_lstm_v2.abs().mean()}
        o = pd.DataFrame(out_).T
        o.index.name = "method"
        return o, len(d_), len(sub)
    tdev, n_dev, n_dev_sub = fc_table(dfc)
    tev, n_ev, n_ev_sub = fc_table(fc)
    put("forecast_dev", md_table(tdev, "{:.3f}"))
    put("forecast_eval", md_table(tev, "{:.3f}"))
    hp = fc[fc.level_has_pi]
    t = pd.DataFrame({
        "Recent level, personal-backtest band (app default)": {"coverage (nominal 0.80)": hp.level_in_pi80.mean(),
                                                               "mean width (kg)": hp.level_pi80_width.mean(), "n": len(hp)},
        "Linear trend, classical regression band": {"coverage (nominal 0.80)": fc.linear_in_pi80.mean(),
                                                    "mean width (kg)": fc.linear_pi80_width.mean(), "n": len(fc)}}).T
    t.index.name = "prediction band"
    put("forecast_pi", md_table(t, "{:.3f}"))
    put("forecast_pi_by_horizon", md_table(hp.groupby("horizon").apply(
        lambda x: pd.Series({"level-band coverage": x.level_in_pi80.mean(), "linear-band coverage": x.linear_in_pi80.mean()}),
        include_groups=False), "{:.3f}"))
    S["forecast"] = {"n_eval": n_ev, "n_eval_complete": n_ev_sub, "n_dev": n_dev,
                     "mae_level": round(float(fc.err_level.abs().mean()), 3),
                     "mae_persistence": round(float(fc.err_persistence.abs().mean()), 3),
                     "mae_linear14": round(float(fc.err_linear14.abs().mean()), 3),
                     "mae_legacy_fallback": round(float(fc.err_legacy_fallback.abs().mean()), 3),
                     "dev_mae_level": round(float(dfc.err_level.abs().mean()), 3),
                     "dev_mae_persistence": round(float(dfc.err_persistence.abs().mean()), 3),
                     "dev_mae_linear14": round(float(dfc.err_linear14.abs().mean()), 3),
                     "pi80_level": round(float(hp.level_in_pi80.mean()), 3),
                     "pi80_linear": round(float(fc.linear_in_pi80.mean()), 3),
                     "bias_level_sustained_deficit": round(float(fc[fc.scenario == "sustained_deficit"].err_level.mean()), 3)}
    sub = fc[fc.complete_14 & fc.err_lstm_v2.notna()]
    if len(sub):
        S["forecast"]["mae_lstm_subset"] = round(float(sub.err_lstm_v2.abs().mean()), 3)
        S["forecast"]["mae_level_subset"] = round(float(sub.err_level.abs().mean()), 3)

    # headline before/after table
    def cell(g, m):
        e = err(g, m)
        return f"{np.abs(e).mean():.0f}"
    hb = []
    for gname, g in groups.items():
        hb.append({"evaluation data (simulated, unseen users)": gname, "n users": len(g),
                   "Original app MAE": cell(g, "legacy_app"), "Original research-module MAE": cell(g, "legacy_research"),
                   "Corrected (Huber) MAE": cell(g, "huber")})
    put("headline_before_after", md_table(pd.DataFrame(hb), index=False))
    S["headline"] = {gname: {m: round(float(np.abs(err(g, m)).mean()), 4) for m in ("legacy_app", "legacy_research", "huber", "ols_raw", "ewma_ols")}
                     | {"n": int(len(g))} for gname, g in groups.items()}
    eb_all = ev[ev.condition == "base"]
    S["coverage_pooled_base"] = {
        "original": round(coverage(eb_all, "est_legacy_research_ci_lo", "est_legacy_research_ci_hi", "truth_tissue_equiv"), 3),
        **{c: round(coverage(eb_all, f"ci_{c}_lo", f"ci_{c}_hi", "truth_tissue_equiv"), 3) for c in CI_METHODS},
        "original_mean_width": round(float((eb_all.est_legacy_research_ci_hi - eb_all.est_legacy_research_ci_lo).mean())),
        "n": int(len(eb_all))}
    S["legacy_flag_rate_clean"] = round(float((ev[ev.condition == "clean"].est_legacy_n_flagged / ev[ev.condition == "clean"].n_obs).mean()), 3)

    # app headline (shown inside the app's validation expander)
    S["app_headline"] = {
        f"{WIN_NAMES[w]} of data": {
            "typical error (MAE, kcal/day)": int(round(np.abs(err(ev[(ev.condition == 'base') & (ev.window == w)], 'huber')).mean())),
            "interval contained the truth": f"{coverage(ev[(ev.condition == 'base') & (ev.window == w)], 'ci_huber_ar1_lo', 'ci_huber_ar1_hi', 'truth_tissue_equiv'):.0%}"}
        for w in ("last14", "last28", "last56")}
    S["app_headline_caption"] = ("Simulated users only (two simulation models pooled, 10% missed days, 2% gross errors). "
                                 "'Interval contained the truth' is for a nominal 95% interval.")

    # ---------------- figures --------------------------------------------
    e = ev[ev.condition == "base"]
    fig, axes = new_fig(10, 3.8, 2, sharey=True)
    show = ["legacy_research", "legacy_app", "ols_raw", "ewma_ols", "huber"]
    for ax, mech in zip(axes, ("A", "B")):
        for i, m in enumerate(show):
            vals = [np.abs(err(e[(e.mechanism == mech) & (e.window == w)], m)).mean() for w in WIN_ORDER]
            xs = np.arange(len(WIN_ORDER)) + (i - 2) * 0.16
            ax.bar(xs, vals, width=0.14, color=COLOR[m], label=SHORT[m])
            if m in ("legacy_app", "huber"):
                for x_, v in zip(xs, vals):
                    ax.text(x_, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(np.arange(len(WIN_ORDER)))
        ax.set_xticklabels([WIN_NAMES[w] for w in WIN_ORDER])
        ax.set_title(f"Simulation model {mech}", fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("Mean absolute error (kcal/day)", fontsize=8.5, color=INK2)
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Error of the energy-balance estimate by amount of data (base condition, unseen simulated users)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    save(fig, os.path.join(fdir, "fig1_mae_by_window.png"))

    fig, axes = new_fig(11, 3.6, 3, sharey=True)
    labels = {"noise": ["0.5x", "1x", "1.6x"], "missing": ["0%", "10%", "25%", "45%"], "outliers": ["0%", "2%", "5%", "10%"]}
    titles = {"noise": "Noise level", "missing": "Share of days with no reading", "outliers": "Share of gross errors"}
    for ax, (name, conds) in zip(axes, sweeps.items()):
        for m in ["legacy_research", "legacy_app", "ols_raw", "huber"]:
            vals = [np.abs(err(e28s[e28s.condition == c], m)).mean() for c in conds]
            ax.plot(range(len(conds)), vals, color=COLOR[m], linewidth=2, marker="o", markersize=5, label=SHORT[m])
            if m in ("legacy_app", "huber"):
                ax.text(len(conds) - 1 + 0.08, vals[-1], f"{vals[-1]:.0f}", fontsize=7, color=INK2, va="center")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels[name])
        ax.set_xlim(-0.3, len(conds) - 0.45)
        ax.set_title(titles[name], fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("Mean absolute error (kcal/day), last 28 days", fontsize=8.5, color=INK2)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    save(fig, os.path.join(fdir, "fig2_stress_sweeps.png"))

    fig, axes = new_fig(10, 3.6, 2, sharey=True)
    cis = [("legacy", "#eb6834"), ("ols_classical", "#eda100"), ("ols_hac", "#1baf7a"), ("huber_block_bootstrap", "#e87ba4"),
           ("huber_ar1", "#2a78d6")]
    wins3 = ["last14", "last28", "last56"]
    for ax, mech in zip(axes, ("A", "B")):
        for c, col in cis:
            vals = [b[(b.mechanism == mech) & (b.window == WIN_NAMES[w]) & (b.interval == CI_NAMES[c])].cov_equiv.iloc[0] for w in wins3]
            ax.plot(range(3), vals, color=col, linewidth=2, marker="o", markersize=5, label=CI_NAMES[c])
            ax.text(2.07, vals[-1], f"{vals[-1]:.2f}", fontsize=7, color=INK2, va="center")
        ax.axhline(0.95, color=INK2, linewidth=1, linestyle="--")
        ax.text(-0.25, 0.957, "nominal 0.95", fontsize=7, color=INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([WIN_NAMES[w] for w in wins3])
        ax.set_xlim(-0.3, 2.45)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Simulation model {mech}", fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("Share of intervals containing the truth", fontsize=8.5, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, os.path.join(fdir, "fig3_interval_coverage.png"))

    # tracking around a change in behaviour
    ch = evt[evt.change_day.notna()].assign(rel=lambda d_: ((d_.day - d_.change_day) // 6) * 6)
    ch = ch[(ch.rel >= -30) & (ch.rel <= 42)]
    fig, ax = new_fig(7.5, 3.8)
    for c, col, nm in (("legacy_app_full_history", COLOR["legacy_app"], "Original app (whole list)"),
                       ("huber_full_history", COLOR["ols_raw"], "Huber, whole history"),
                       ("huber_trailing_28", COLOR["huber"], "Huber, trailing 28 days (default)")):
        mae = ch.groupby("rel").apply(lambda g_: np.abs(g_[f"est_{c}"] - g_.truth_recent).mean(), include_groups=False)
        ax.plot(mae.index, mae.values, color=col, linewidth=2, marker="o", markersize=4.5, label=nm)
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.text(1, ax.get_ylim()[1] * 0.96, "behaviour changes", fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel("Days relative to the change in energy balance", fontsize=8.5, color=INK2)
    ax.set_ylabel("MAE vs. true recent balance (kcal/day)", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Following a change: 'as of today' estimates (step-change scenarios)", fontsize=9.5, color=INK, loc="left")
    save(fig, os.path.join(fdir, "fig4_tracking_after_change.png"))

    # reliability levels
    fig, ax = new_fig(6, 3.6)
    data = [relg[relg.reliability == lvl].abs_err.to_numpy() for lvl in ("high", "moderate", "low")]
    bp = ax.boxplot(data, tick_labels=[f"{lvl}\n(n={len(d_)})" for lvl, d_ in zip(("high", "moderate", "low"), data)],
                    showfliers=False, patch_artist=True, widths=0.5, medianprops=dict(color=INK, linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set(facecolor="#2a78d6", alpha=0.55, edgecolor=INK2)
    ax.set_ylabel("Absolute error (kcal/day)", fontsize=8.5, color=INK2)
    ax.set_xlabel("Reliability level shown by the app", fontsize=8.5, color=INK2)
    ax.set_title("Does the reliability indicator track real error?", fontsize=9.5, color=INK, loc="left")
    save(fig, os.path.join(fdir, "fig5_reliability_vs_error.png"))

    # missing-days scatter
    gm = e28[e28.condition == "missing_45"]
    fig, axes = new_fig(8.5, 4, 2, sharex=True, sharey=True)
    for ax, m in zip(axes, ("legacy_app", "huber")):
        ax.scatter(gm.truth_balance, gm[f"est_{m}"], s=9, color=COLOR[m], alpha=0.55, linewidths=0)
        lim = [-1500, 1500]
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_title(f"{SHORT[m]}  (MAE {np.abs(err(gm, m)).mean():.0f})", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("True energy balance (kcal/day)", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("Estimated (kcal/day)", fontsize=8.5, color=INK2)
    fig.suptitle("45% of days without a reading, last 28 days (dashed line = perfect)", fontsize=10, color=INK, x=0.01, ha="left")
    save(fig, os.path.join(fdir, "fig6_missing_days_scatter.png"))

    # one worked example
    cfg = load_config()
    sim = simulate_series("A", "sustained_deficit", "outliers_10", 4, EVAL_SEED, cfg)
    a, b_ = WINDOWS["last28"]
    obs = sim.observed[a:b_ + 1]
    mk = ~np.isnan(obs)
    d, w = sim.days[a:b_ + 1][mk], obs[mk]
    leg = ml_v1.run_pipeline(list(w))
    hub = eb.fit_line(d, w, "huber")
    fig, ax = new_fig(8, 4)
    out_m = sim.is_outlier[a:b_ + 1][mk]
    ax.plot(d[~out_m], w[~out_m], linestyle="none", marker="o", markersize=4.5, color="#9a948c", label="Simulated reading")
    ax.plot(d[out_m], w[out_m], linestyle="none", marker="X", markersize=8, color="#0b0b0b", label="Injected gross error")
    ax.plot(sim.days[a:b_ + 1], sim.tissue[a:b_ + 1], color=INK2, linewidth=1.2, linestyle="--", label="True tissue weight")
    ax.plot(d, ml_v1.ewma_filter(np.where(leg["anomaly_flags"], leg["smoothed"], w)), color=COLOR["legacy_research"],
            linewidth=2, label=f"Original smoothed series -> {leg['kcal_per_day']:+.0f} kcal/day")
    ax.plot(d, hub["intercept"] + hub["slope"] * d, color=COLOR["huber"], linewidth=2,
            label=f"Huber trend -> {hub['slope'] * 7700:+.0f} kcal/day")
    ax.set_xlabel("Day", fontsize=8.5, color=INK2)
    ax.set_ylabel("Weight (kg)", fontsize=8.5, color=INK2)
    ax.set_title(f"One simulated user (true balance {sim.true_mean_balance(a, b_):+.0f} kcal/day over these 28 days)",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    save(fig, os.path.join(fdir, "fig7_worked_example.png"))

    def clean(o):                                   # strict JSON: NaN -> null
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(clean(S), f, indent=2)
    with open(os.path.join(base, "tables.json"), "w") as f:
        json.dump(T, f, indent=1)
    print("tables:", len(T), "| figures in", fdir, "| summary:", os.path.join(base, "summary.json"))


if __name__ == "__main__":
    main()
