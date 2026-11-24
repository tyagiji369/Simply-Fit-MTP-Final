"""
Integration tests: the complete Streamlit workflow driven headlessly with
streamlit.testing (no browser). They check that what the user is SHOWN is
exactly what the shared backend computes.
"""
import os
import re

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest          # noqa: E402

from src.energy_balance import assess_reliability
from src.research_estimate import estimate as estimate_energy_balance   # noqa: E402

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "streamlit_app.py"))


def click(at, label):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button '{label}' not found among {[b.label for b in at.button]}"
    return matches[0].click().run()


def page_text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def to_weight_log_page():
    at = AppTest.from_file(APP, default_timeout=120).run()
    for label in ("Continue", "Set my goal", "Continue", "Enter weight log"):
        at = click(at, label)
        assert not at.exception
    return at


def enter_log(at, values):
    text = ", ".join("-" if v is None else f"{v:.1f}" for v in values)
    at.text_area[0].set_value(text).run()
    return at


def make_log(kcal=-500.0, n=35, seed=0, sd=0.25, start=78.0):
    rng = np.random.RandomState(seed)
    return [round(float(v), 1) for v in start + (kcal / 7700) * np.arange(n) + rng.normal(0, sd, n)]


def test_no_fictional_measurements_are_prefilled():
    at = to_weight_log_page()
    assert at.session_state.weight_log == []
    assert at.text_area[0].value == ""
    assert at.session_state.is_demo is False
    assert "No readings entered yet" in page_text(at)
    assert [b for b in at.button if b.label == "Start daily tracker"][0].disabled


def test_full_workflow_displays_exactly_the_backend_estimate():
    log = make_log()
    at = enter_log(to_weight_log_page(), log)
    assert "35</strong> reading(s)" in page_text(at)
    at = click(at, "Start daily tracker")
    assert not at.exception

    backend = estimate_energy_balance(log, window_days=28)
    shown = at.session_state["last_result"]
    assert shown["kcal_per_day"] == pytest.approx(backend["kcal_per_day"])
    assert shown["kcal_ci"] == pytest.approx(backend["kcal_ci"])
    assert at.session_state["last_reliability"]["level"] == assess_reliability(backend)["level"]

    text = page_text(at)
    r10 = lambda v: int(round(v / 10.0) * 10)
    assert f"≈ {r10(backend['kcal_per_day']):+d}" in text
    assert f"{r10(backend['kcal_ci'][0]):+d} to {r10(backend['kcal_ci'][1]):+d}" in text
    assert f"{backend['weekly_kg_change']:+.2f}" in text
    assert "Calorie deficit" in text
    assert f"Reliability: {assess_reliability(backend)['level']}" in text

    at = click(at, "View full report")
    assert not at.exception
    assert f"≈ {r10(backend['kcal_per_day']):+d}" in page_text(at)
    assert "prediction interval" in page_text(at)


def test_short_history_shows_no_number():
    at = enter_log(to_weight_log_page(), make_log(n=8))
    at = click(at, "Start daily tracker")
    assert not at.exception
    text = page_text(at)
    assert "No estimate yet" in text
    assert "kcal/day</span>" not in text                 # no estimate card rendered
    assert at.session_state["last_result"]["status"] == "insufficient"
    assert at.session_state["last_reliability"]["level"] == "insufficient"


def test_invalid_entries_block_progress():
    at = to_weight_log_page()
    at.text_area[0].set_value("78, 77.9, abc, 780, 77.5").run()
    text = page_text(at)
    assert "is not a number" in text
    assert [b for b in at.button if b.label == "Start daily tracker"][0].disabled
    at.text_area[0].set_value("78, 77.9, 780, 77.5").run()
    assert "outside the plausible range" in page_text(at)
    assert [b for b in at.button if b.label == "Start daily tracker"][0].disabled


def test_missed_days_are_accepted_and_counted():
    log = make_log(n=40)
    for i in (4, 5, 19, 33):
        log[i] = None
    at = enter_log(to_weight_log_page(), log)
    assert "4 day(s) without a reading" in page_text(at)
    at = click(at, "Start daily tracker")
    res = at.session_state["last_result"]
    assert res["status"] == "ok"
    assert res["kcal_per_day"] == pytest.approx(estimate_energy_balance(log, window_days=28)["kcal_per_day"])


def test_demo_data_is_explicit_and_labelled():
    at = click(to_weight_log_page(), "Load simulated demo data")
    assert at.session_state.is_demo is True
    assert "Simulated demo data" in page_text(at)
    at = click(at, "Start daily tracker")
    assert "Simulated demo data" in page_text(at)


def test_add_reading_and_missed_day_buttons_update_the_log():
    log = make_log(n=30)
    at = click(enter_log(to_weight_log_page(), log), "Start daily tracker")
    at = click(at, "No reading today")
    assert at.session_state.weight_log[-1] is None and len(at.session_state.weight_log) == 31
    at = click(at, "Add reading")
    assert len(at.session_state.weight_log) == 32 and at.session_state.weight_log[-1] is not None
    assert at.session_state["last_result"]["status"] == "ok"


def test_plan_comparison_direction_is_correct():
    """v1 told users losing faster than planned that they were 'behind' (and vice versa)."""
    fast = make_log(kcal=-1000.0, n=42, sd=0.1)          # plan set by the target page: -0.5 kg/week (-550 kcal/day)
    at = click(enter_log(to_weight_log_page(), fast), "Start daily tracker")
    assert "losing faster than planned" in page_text(at)
    gaining = make_log(kcal=+400.0, n=42, sd=0.1)
    at = click(enter_log(to_weight_log_page(), gaining), "Start daily tracker")
    text = page_text(at)
    assert "behind your plan" in text and "faster than planned" not in text


def test_coach_runs_inside_the_app_without_an_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = click(enter_log(to_weight_log_page(), make_log()), "Start daily tracker")
    question = [t for t in at.text_input if t.label.startswith("Ask anything")][0]
    question.set_value("How long until I reach my goal?")
    at = click(at, "Ask coach")
    assert not at.exception
    box = [m.value for m in at.markdown if "coach-box" in m.value and "<style>" not in m.value]
    assert box and "73.0 kg" in box[0]                   # goal = 78 kg start - 5 kg planned loss
    assert re.search(r"is \d\.\d kg\.", box[0])          # a few kg remaining, not ~70


def test_app_has_no_private_copies_of_statistical_code():
    src = open(APP).read()
    for forbidden in ("IsolationForest", "LinearRegression", "np.polyfit", ".ewm(", "7700", "import random"):
        assert forbidden not in src, f"app still contains '{forbidden}'"


def test_noisy_history_withholds_numeric_estimate_in_tracker_and_report():
    at = enter_log(to_weight_log_page(), list(np.tile([77., 80., 79., 74.], 7)))
    at = click(at, "Start daily tracker")
    assert not at.exception
    assert at.session_state["last_result"]["status"] == "withheld"
    assert "Estimate withheld" in page_text(at)
    assert "Simulation-calibrated 95% range" not in page_text(at)
    at = click(at, "View full report")
    assert not at.exception
    assert "Estimate withheld" in page_text(at)
    assert "Simulation-calibrated 95% range" not in page_text(at)


def test_hydration_reply_names_offline_mode_and_links_source(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = click(enter_log(to_weight_log_page(), make_log()), "Start daily tracker")
    field = next(t for t in at.text_input if t.label.startswith("Ask anything"))
    field.set_value("How much water should I drink per day?")
    at = click(at, "Ask coach")
    assert not at.exception
    assert any("No Gemini API key" in item.value for item in at.caption)
    assert any("https://www.nhs.uk/" in item.value for item in at.markdown)
