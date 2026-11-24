"""
Re-evaluation of the forecasting models (secondary feature).

1. HISTORICAL: reads results/model_evaluation.json (saved by the internship
   version on 2026-08-25) and reports it as-is.
2. REPRODUCTION: regenerates the same synthetic cohort (original generator,
   500 users, seed 42), applies the same walk-forward split and re-scores
   - the two baselines (deterministic, should match the saved file), and
   - the SAVED internship LSTM file models/simply_fit_lstm.keras, without
     retraining it.
3. DEPLOYMENT MISMATCH: scores the same saved LSTM the way the original
   `forecast()` actually called it, i.e. with a MinMaxScaler fitted on the
   log it was given instead of on the user's first 72 days:
     (a) log = only the 14 input days  (a new user)
     (b) log = everything up to the forecast origin
   and the original 2-point fallback.
4. CORRECTED OPTIONAL LSTM: retrains with window-centred normalisation
   (src/lstm_forecaster.py) and scores it on the same windows, together
   with the app's default "recent level" rule.

Needs TensorFlow.   python -m experiments.forecast_reevaluation
Writes results/forecast_reevaluation.json
"""

import json
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src import lstm_forecaster as v2
from src.legacy import lstm_forecaster_v1 as v1
from src.legacy.data_generator_v1 import generate_dataset

OUT = os.path.join("results", "forecast_reevaluation.json")
OLD_MODEL = os.path.join("models", "simply_fit_lstm.keras")


def main():
    with open(os.path.join("results", "model_evaluation.json")) as f:
        historical = json.load(f)
    report = {"historical_saved_file": {"source": "results/model_evaluation.json (not re-run)",
                                        "trained_at": historical["trained_at"],
                                        "mae_kg": historical["mae_kg"],
                                        "n_test_windows": historical["n_test_windows"]}}
    if not v2.HAS_TENSORFLOW:
        report["error"] = "TensorFlow not installed; sections 2-4 not run."
        json.dump(report, open(OUT, "w"), indent=2)
        print(json.dumps(report, indent=2))
        return
    from tensorflow.keras.models import load_model

    df = generate_dataset(n_users=500, seed=42)
    cases = []          # (input_raw, target_raw, scaled input by v1 train scaler, scaler, history up to origin)
    for uid, g in df.groupby("user_id"):
        w = g["weight"].values.astype(float)
        _, _, test_cases = v1._walk_forward_split(w)
        i0 = len(w) - v1.SEQUENCE_LENGTH - v1.FORECAST_LENGTH + 1 - len(test_cases)
        for k, c in enumerate(test_cases):
            start = i0 + k
            assert np.array_equal(w[start:start + 14], c["input_raw"])
            c["history"] = w[:start + 14]
            cases.append(c)
    X_raw = np.array([c["input_raw"] for c in cases])
    Y = np.array([c["target_raw"] for c in cases])
    mae = lambda P: float(np.mean(np.abs(P - Y)))

    base = {"persistence": mae(np.array([v2.persistence_forecast(x) for x in X_raw])),
            "linear_14day_ols": mae(np.array([v1._linear_baseline(x) for x in X_raw])),
            "linear_shared_core": mae(np.array([v2.linear_forecast(x) for x in X_raw])),
            "level_mean_of_last_3 (app default)": mae(np.array([v2.level_forecast(x) for x in X_raw]))}

    old = load_model(OLD_MODEL, compile=False)

    def old_predict(scaled_inputs, scalers):
        P = old.predict(np.array(scaled_inputs).reshape(-1, 14, 1), verbose=0, batch_size=512)
        return np.array([s.inverse_transform(p.reshape(-1, 1)).ravel() for p, s in zip(P, scalers)])

    # (2) as evaluated originally: scaler fitted on the user's first 72 days
    p_train = old_predict([c["input_scaled"] for c in cases], [c["scaler"] for c in cases])
    # (3a) as deployed for a user with exactly 14 days of data
    sc14 = [MinMaxScaler().fit(x.reshape(-1, 1)) for x in X_raw]
    p_14 = old_predict([s.transform(x.reshape(-1, 1)).ravel() for s, x in zip(sc14, X_raw)], sc14)
    # (3b) as deployed for a user with the whole history so far
    sch = [MinMaxScaler().fit(c["history"].reshape(-1, 1)) for c in cases]
    p_h = old_predict([s.transform(x.reshape(-1, 1)).ravel() for s, x in zip(sch, X_raw)], sch)
    fallback = np.array([v1._linear_extrapolate(list(x)) for x in X_raw])

    report["reproduction_same_cohort_and_split"] = {
        "n_test_windows": int(len(cases)),
        "mae_kg": {**base, "saved_lstm_with_training_scaler": mae(p_train)},
        "baselines_match_saved_file": bool(abs(base["persistence"] - historical["mae_kg"]["persistence"]) < 1e-9
                                           and abs(base["linear_14day_ols"] - historical["mae_kg"]["linear"]) < 1e-9),
    }
    report["deployment_mismatch_of_saved_lstm"] = {
        "mae_kg": {"as_evaluated (scaler from first 72 days)": mae(p_train),
                   "as_deployed, 14-day log (scaler from the 14 inputs)": mae(p_14),
                   "as_deployed, full history (scaler from all days so far)": mae(p_h),
                   "original_fallback_2_point": mae(fallback)},
    }

    rep = v2.train_model(df, verbose=0)
    report["corrected_optional_lstm_window_centred"] = {
        "mae_kg": rep["mae_kg"], "epochs_run": rep["epochs_run"], "n_test_windows": rep["n_test_windows"],
        "model_file": "models/simply_fit_lstm_v2.keras",
    }
    report["conclusion"] = (
        "On THIS cohort (the internship generator: independent day-to-day noise, near-linear "
        "trajectories) the corrected LSTM has the lowest error, followed by the recent-level rule and "
        "the linear trend, which are close to each other; persistence is last. On the "
        "scenario simulator the ranking is different (see results/experiments/tables/forecast_*.md). "
        "The saved internship figure for the LSTM (0.537 kg) was measured with a different "
        "normalisation from the one the app applied, so it did not describe the deployed model. "
        "All data are SYNTHETIC.")
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
