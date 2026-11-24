"""
Verbatim copy of the estimator functions that lived inside the original
app/streamlit_app.py (lines 402-420 of the internship version). The original
application did NOT call src/ml_engine.py for the number it displayed; it used
this second, slightly different implementation (regression on the
anomaly-replaced RAW series, not on the EWMA-smoothed series).

Frozen for comparison only. Do not import from application code.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

KCAL_PER_KG = 7700


def ewma(arr, span=7):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values

def detect_anomalies(log):
    smoothed  = ewma(log)
    residuals = np.array(log) - smoothed
    iso       = IsolationForest(contamination=0.1, random_state=42)
    flags     = iso.fit_predict(residuals.reshape(-1, 1)) == -1
    return flags, smoothed

def estimate_balance(log):
    flags, smoothed = detect_anomalies(log)
    clean           = np.where(flags, smoothed, log)
    days            = np.arange(len(clean)).reshape(-1, 1)
    m               = LinearRegression().fit(days, clean)
    kg_per_day      = m.coef_[0]
    kcal_per_day    = kg_per_day * KCAL_PER_KG
    r2              = round(m.score(days, clean), 2)
    return kcal_per_day, kg_per_day * 7, r2


def rolling_5day_balance(log, wb=5):
    """The 'Inferred daily calorie balance' bar chart of the original report page."""
    out = []
    for i in range(wb, len(log)):
        chunk = log[i - wb:i + 1]
        out.append((chunk[-1] - chunk[0]) / wb * KCAL_PER_KG)
    return np.array(out)
