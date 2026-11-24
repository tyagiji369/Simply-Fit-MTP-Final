"""
Parsing of user-entered weight logs (text box or CSV upload) into the daily
grid used everywhere else: a list with one entry per calendar day, where a
missed day is None. Nothing is filled in or guessed.
"""

import io

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "-", "na", "nan", "n/a", "none", "x", "?"}


def parse_weight_text(text):
    """
    Parse readings typed one per day, separated by commas, semicolons or new
    lines. A skipped day is written as '-' (or NA, or left empty between two
    commas). Returns (values, errors).
    """
    if text is None or not str(text).strip():
        return [], []
    tokens = [t.strip() for t in str(text).replace("\n", ",").replace(";", ",").split(",")]
    while tokens and tokens[-1] == "":            # trailing comma / newline is not a missed day
        tokens.pop()
    values, errors = [], []
    for i, tok in enumerate(tokens, 1):
        if tok.lower() in MISSING_TOKENS:
            values.append(None)
            continue
        try:
            values.append(float(tok))
        except ValueError:
            errors.append(f"Entry {i} ('{tok}') is not a number.")
            values.append(None)
    return values, errors


def parse_weight_csv(file_or_buffer):
    """
    Parse an uploaded CSV with a weight column and, optionally, a date (or
    day-number) column. Recognised weight headers: weight, weight_kg, kg.
    Recognised time headers: date, day. Without a time column, rows are
    taken as consecutive days. Returns (values on a daily grid, errors).
    """
    try:
        if isinstance(file_or_buffer, (bytes, bytearray)):
            file_or_buffer = io.BytesIO(file_or_buffer)
        df = pd.read_csv(file_or_buffer)
    except Exception as exc:                       # malformed upload
        return [], [f"Could not read the CSV file: {exc}"]
    cols = {c.strip().lower(): c for c in df.columns}
    wcol = next((cols[c] for c in ("weight", "weight_kg", "kg") if c in cols), None)
    if wcol is None:
        return [], ["The CSV needs a column named 'weight' (in kg)."]
    w = pd.to_numeric(df[wcol], errors="coerce")
    errors = []
    bad = int((w.isna() & df[wcol].notna() & (df[wcol].astype(str).str.strip() != "")).sum())
    if bad:
        errors.append(f"{bad} weight value(s) in the CSV are not numbers.")

    if "date" in cols:
        d = pd.to_datetime(df[cols["date"]], errors="coerce")
        if d.isna().any():
            return [], ["Some dates in the CSV could not be read. Use YYYY-MM-DD."]
        day = (d.dt.normalize() - d.dt.normalize().min()).dt.days.to_numpy()
    elif "day" in cols:
        dn = pd.to_numeric(df[cols["day"]], errors="coerce")
        if dn.isna().any() or (dn != dn.round()).any():
            return [], ["The 'day' column must contain whole numbers."]
        day = (dn - dn.min()).astype(int).to_numpy()
    else:
        day = np.arange(len(df))

    if len(day) == 0:
        return [], ["The CSV has no rows."]
    if len(np.unique(day)) != len(day):
        return [], ["The CSV has more than one row for the same day; keep one reading per day."]
    if day.max() > 3650:
        return [], ["The CSV spans more than 10 years; check the date column."]
    grid = [None] * (int(day.max()) + 1)
    for dd, ww in zip(day, w):
        if not np.isnan(ww):
            grid[int(dd)] = float(ww)
    return grid, errors
