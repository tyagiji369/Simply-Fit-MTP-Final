"""
7-day forecast experiment on the scenario simulator (secondary feature).

For every simulated user (base condition) and three forecast origins, each
method predicts the next 7 scale readings from the readings up to the
origin. Development seeds are used to choose the app's default method;
evaluation seeds give the reported numbers.

Methods
    level        mean of the last 3 readings carried forward (app default)
    persistence  last reading carried forward
    linear14     least squares on the last 14 days (internship baseline)
    legacy_fallback  the original 2-point `_linear_extrapolate`
    lstm_v2      optional retrained LSTM; only scored when TensorFlow and the
                 model are present AND the last 14 days have no missed reading
Prediction intervals (80%): level (personal backtest) and linear14 (classical).

    python -m experiments.forecast_experiment [--quick]
Writes results/experiments/raw/{dev,eval}_forecast.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from src import energy_balance as eb
from src import lstm_forecaster as lf
from src.legacy import lstm_forecaster_v1 as lstm_v1
from src.simulation import MECHANISMS, load_config, simulate_series

CFG = load_config()
ORIGINS = (40, 60, 75)


def run(seed_base, n_users, model):
    rows, lstm_inputs, lstm_rows = [], [], []
    for mech in MECHANISMS:
        for scen in sorted(CFG["scenarios"]):
            for u in range(n_users):
                sim = simulate_series(mech, scen, "base", u, seed_base, CFG)
                for t0 in ORIGINS:
                    obs = sim.observed[: t0 + 1]
                    m = ~np.isnan(obs)
                    days, w = sim.days[: t0 + 1][m], obs[m]
                    level = eb.forecast_weight(w, days, method="level")
                    linear = eb.forecast_weight(w, days, method="linear")
                    if level is None or linear is None or len(w) < 14:
                        continue
                    legacy = lstm_v1._linear_extrapolate(list(w))
                    last14 = days > t0 - 14
                    complete = bool(last14.sum() == 14 and days[-1] == t0)
                    for h in range(1, 8):
                        actual = sim.observed[t0 + h]
                        if np.isnan(actual):
                            continue
                        rows.append({
                            "mechanism": mech, "scenario": scen, "user": u, "origin": t0, "horizon": h,
                            "is_gross_error": bool(sim.is_outlier[t0 + h]), "complete_14": complete,
                            "err_level": level["mean"][h - 1] - actual,
                            "err_persistence": w[-1] - actual,
                            "err_linear14": linear["mean"][h - 1] - actual,
                            "err_legacy_fallback": legacy[h - 1] - actual,
                            "level_has_pi": level["lo"] is not None,
                            "level_in_pi80": bool(level["lo"] is not None
                                                  and level["lo"][h - 1] <= actual <= level["hi"][h - 1]),
                            "level_pi80_width": (level["hi"][h - 1] - level["lo"][h - 1]) if level["lo"] is not None else np.nan,
                            "linear_in_pi80": bool(linear["lo"][h - 1] <= actual <= linear["hi"][h - 1]),
                            "linear_pi80_width": linear["hi"][h - 1] - linear["lo"][h - 1],
                            "actual": actual, "err_lstm_v2": np.nan})
                        if complete and model is not None:
                            lstm_inputs.append(w[last14])
                            lstm_rows.append((len(rows) - 1, h))
    df = pd.DataFrame(rows)
    if lstm_inputs:
        P = lf.lstm_predict(model, np.array(lstm_inputs))
        idx = [i for i, _ in lstm_rows]
        df.loc[idx, "err_lstm_v2"] = [P[j, h - 1] for j, (_, h) in enumerate(lstm_rows)] - df.loc[idx, "actual"].to_numpy()
    return df.drop(columns=["actual"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = os.path.join("results", "experiments", "quick" if args.quick else "raw")
    os.makedirs(out, exist_ok=True)
    model = None
    if lf.HAS_TENSORFLOW and os.path.exists(lf.MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(lf.MODEL_PATH, compile=False)
    else:
        print("optional LSTM not available: lstm_v2 column will be empty")
    n = 4 if args.quick else 60
    for name, seed in (("dev", CFG["seeds"]["development_base"]), ("eval", CFG["seeds"]["evaluation_base"])):
        df = run(seed, n, model)
        df.to_csv(os.path.join(out, f"{name}_forecast.csv"), index=False)
        cols = ["err_level", "err_persistence", "err_linear14", "err_legacy_fallback"]
        print(name, len(df), "rows | MAE kg:", df[cols].abs().mean().round(3).to_dict(),
              "| level PI80 coverage:", round(df[df.level_has_pi].level_in_pi80.mean(), 3))


if __name__ == "__main__":
    main()
