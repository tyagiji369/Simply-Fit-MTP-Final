"""
7-day weight forecasting (SECONDARY feature of Simply-Fit).

Deployed default: "recent level"
--------------------------------
`forecast()` carries forward the average of the last 3 readings
(src/energy_balance.forecast_weight, method="level") with a prediction
interval obtained by replaying that rule on the user's own recent history.

Why not the LSTM, and why not the linear trend: the ranking of forecasters
depends on the noise in the data, and no method wins everywhere
(results/forecast_reevaluation.json, results/experiments/tables/forecast_*.md):

  * internship cohort (independent day-to-day noise): corrected LSTM best,
    recent level and linear trend close behind, persistence last;
  * scenario simulator (persistent water-weight noise, gaps, gross errors):
    recent level and the corrected LSTM about equal, linear trend clearly
    worse than simply repeating the last reading.

The level rule needs no TensorFlow, works with missed days, and was chosen
on development simulations. All differences are small next to the scatter
of single readings (about 0.5 kg).

What was wrong in the internship version (src/legacy/lstm_forecaster_v1.py)
---------------------------------------------------------------------------
1. Train/predict normalisation mismatch. Training fitted a MinMaxScaler on
   a user's first 72 days; `forecast()` fitted a new scaler on whatever log
   it was handed. The saved evaluation therefore did not measure the model
   as the app actually ran it (the re-evaluation shows the deployed
   configuration scored differently from the evaluated one).
2. The fallback `_linear_extrapolate` was described as "OLS on the last 7
   readings" but was a two-point difference (6 days apart) divided by 7. It
   was not the linear baseline that had been evaluated, and it was the
   least accurate method in every comparison.
3. The forecast was shown as a single line with no interval.

Corrections
-----------
* One forecast function in the shared core is used by the app, the coach and
  the experiments.
* The optional LSTM (method="lstm") is retrained with a normalisation that
  is identical by construction at training and prediction time: each
  14-day input window is centred on its own mean (kg), and the network
  predicts the next 7 days relative to that mean. No scaler object exists,
  so nothing can be fitted differently later.
* `forecast_detail()` reports which method actually produced the numbers.

Run `python -m src.lstm_forecaster` to retrain the optional LSTM and rewrite
models/simply_fit_lstm_v2.keras and results/model_evaluation_v2.json.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

from src.energy_balance import (FORECAST_FIT_DAYS, FORECAST_HORIZON, forecast_weight,
                                validate_weight_log)

HAS_TENSORFLOW = False
try:                                    # the app must run without TensorFlow
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential, load_model

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

SEQUENCE_LENGTH = FORECAST_FIT_DAYS      # 14 input days
FORECAST_LENGTH = FORECAST_HORIZON       # 7 output days
TRAIN_DAYS = 72                          # per-user walk-forward boundary (same as v1)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(_ROOT, "models", "simply_fit_lstm_v2.keras")
RESULTS_PATH = os.path.join(_ROOT, "results", "model_evaluation_v2.json")


# ---- normalisation: the single definition used for training AND prediction
def centre_window(window):
    """Return (window - mean, mean). Units stay in kg."""
    window = np.asarray(window, dtype=float)
    mu = float(window.mean())
    return window - mu, mu


def linear_forecast(input_window, forecast_len=FORECAST_LENGTH):
    """OLS on a complete input window, extended forward (shared-core implementation)."""
    fc = forecast_weight(list(input_window), horizon=forecast_len, method="linear",
                         fit_days=len(input_window))
    return fc["mean"]


def level_forecast(input_window, forecast_len=FORECAST_LENGTH):
    """Average of the last 3 readings carried forward: the app's default method."""
    return forecast_weight(list(input_window), horizon=forecast_len, method="level")["mean"]


def persistence_forecast(input_window, forecast_len=FORECAST_LENGTH):
    return np.full(forecast_len, float(input_window[-1]))


def build_model():
    if not HAS_TENSORFLOW:
        return None
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(FORECAST_LENGTH),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def walk_forward_windows(weights, seq_len=SEQUENCE_LENGTH, forecast_len=FORECAST_LENGTH,
                         train_days=TRAIN_DAYS):
    """
    Leakage-free split of one user's series (same rule as v1): windows whose
    targets end on or before `train_days` are training windows; windows whose
    targets start at or after it are test windows; straddling windows are
    dropped. Returns (train_inputs, train_targets, test_inputs, test_targets)
    in raw kg.
    """
    w = np.asarray(weights, dtype=float)
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(len(w) - seq_len - forecast_len + 1):
        ts, te = i + seq_len, i + seq_len + forecast_len
        if te <= train_days:
            Xtr.append(w[i:ts]); ytr.append(w[ts:te])
        elif ts >= train_days:
            Xte.append(w[i:ts]); yte.append(w[ts:te])
    return Xtr, ytr, Xte, yte


def lstm_predict(model, input_windows):
    """Predict in kg for an array of complete 14-day windows (n, 14)."""
    X = np.asarray(input_windows, dtype=float)
    mu = X.mean(axis=1, keepdims=True)
    pred = model.predict((X - mu).reshape(-1, SEQUENCE_LENGTH, 1), verbose=0)
    return pred + mu


def train_model(df, epochs=50, batch_size=64, seed=42, model_path=MODEL_PATH,
                results_path=RESULTS_PATH, verbose=0):
    """
    Train the optional LSTM on a one-row-per-user-day DataFrame and evaluate
    it against persistence and linear extrapolation on the same held-out
    windows. Writes the model and a metrics JSON; returns the metrics dict.
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow not installed. Skipping training.")
        return None
    tf.keras.utils.set_random_seed(seed)

    Xtr, ytr, Xte, yte = [], [], [], []
    for _, g in df.groupby("user_id"):
        w = g["weight"].values
        if len(w) < TRAIN_DAYS:
            continue
        a, b, c, d = walk_forward_windows(w)
        Xtr += a; ytr += b; Xte += c; yte += d
    Xtr, ytr, Xte, yte = map(np.asarray, (Xtr, ytr, Xte, yte))

    mu = Xtr.mean(axis=1, keepdims=True)
    model = build_model()
    hist = model.fit((Xtr - mu).reshape(-1, SEQUENCE_LENGTH, 1), ytr - mu,
                     epochs=epochs, batch_size=batch_size, validation_split=0.1,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                              restore_best_weights=True)],
                     verbose=verbose)

    preds = {
        "lstm": lstm_predict(model, Xte),
        "persistence": np.array([persistence_forecast(x) for x in Xte]),
        "linear": np.array([linear_forecast(x) for x in Xte]),
        "level": np.array([level_forecast(x) for x in Xte]),
    }
    mae = {k: float(np.mean(np.abs(p - yte))) for k, p in preds.items()}
    mae_by_day = {k: [float(v) for v in np.mean(np.abs(p - yte), axis=0)] for k, p in preds.items()}
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": "SYNTHETIC cohort from src/data_generator.py (not real users)",
        "normalisation": "each 14-day input window centred on its own mean (kg); "
                         "identical at training and prediction time",
        "n_train_users": int(df["user_id"].nunique()),
        "n_train_windows": int(len(Xtr)),
        "n_test_windows": int(len(Xte)),
        "split": "per-user walk-forward: targets <= day 72 train, targets > day 72 test",
        "mae_kg": mae,
        "mae_kg_by_horizon_day": mae_by_day,
        "epochs_run": len(hist.history["loss"]),
        "seed": seed,
    }
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


def forecast_detail(weight_log, days=None, method="level", model_path=MODEL_PATH):
    """
    Forecast the next 7 scale readings.

    method="level" (default): average of the last 3 readings carried forward,
        with an 80% prediction interval when there is enough history.
    method="linear": least squares on the last 14 days (comparison method).
    method="lstm": the optional retrained network. It needs TensorFlow, the
        model file and 14 consecutive days with no missed reading; otherwise
        the level method is used and `note` says why.

    Returns None when there are too few recent readings, else a dict with
    `mean`, `lo`, `hi` (lo/hi may be None), `method_used`, `note`.
    """
    log = validate_weight_log(weight_log, days)
    basic = forecast_weight(log, method="linear" if method == "linear" else "level")
    if basic is None:
        return None
    out = {"days": basic["days"], "mean": basic["mean"], "lo": basic["lo"], "hi": basic["hi"],
           "interval": "80% prediction interval for a single future reading" if basic["lo"] is not None else None,
           "method_used": basic["method"], "note": ""}
    if method != "lstm":
        return out
    last14 = log.days > log.days[-1] - SEQUENCE_LENGTH
    if not HAS_TENSORFLOW:
        out["note"] = "TensorFlow is not installed; level forecast used."
    elif not os.path.exists(model_path):
        out["note"] = "LSTM model file not found; level forecast used."
    elif last14.sum() != SEQUENCE_LENGTH:
        out["note"] = "The LSTM needs 14 consecutive daily readings; level forecast used."
    else:
        try:
            model = load_model(model_path, compile=False)
            pred = lstm_predict(model, log.weights[last14][None, :])[0]
            out.update(mean=pred, lo=None, hi=None, interval=None, method_used="lstm")
        except Exception as exc:
            out["note"] = f"LSTM could not be used ({type(exc).__name__}); level forecast used."
    return out


def forecast(weight_log, days=None, method="level", model_path=MODEL_PATH):
    """Backward-compatible wrapper: returns just the 7 forecast values (kg) or None."""
    detail = forecast_detail(weight_log, days, method, model_path)
    return None if detail is None else np.asarray(detail["mean"], dtype=float)


if __name__ == "__main__":
    from src.data_generator import generate_dataset

    print("Generating SYNTHETIC cohort (500 users x 90 days, seed=42)...")
    data = generate_dataset(n_users=500, seed=42)
    print("Training optional LSTM with window-centred normalisation...")
    rep = train_model(data, verbose=2)
    if rep:
        for name in ("linear", "level", "persistence", "lstm"):
            print(f"  MAE ({name:12s}): {rep['mae_kg'][name]:.3f} kg")
