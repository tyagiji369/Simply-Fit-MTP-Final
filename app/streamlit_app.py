import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import re
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.agent_coach import agent_coach
from src.rag_pipeline import rag_engine
from src.calibration import run_nhanes_calibration_test
# Single source of truth for every calculation shown in this app:
from src.energy_balance import (
    CLASS_TOLERANCE_KCAL, DEFAULT_WINDOW_DAYS, KCAL_PER_KG, MAX_PLAUSIBLE_KG,
    MIN_CALENDAR_DAYS_FOR_ESTIMATE, MIN_CALENDAR_DAYS_FOR_LABEL, MIN_OBS_FOR_ESTIMATE, MIN_PLAUSIBLE_KG,
    assess_reliability, validate_weight_log,
)
from src.research_estimate import estimate as estimate_energy_balance, rolling as rolling_energy_balance
from src.input_parsing import parse_weight_csv, parse_weight_text
from src.lstm_forecaster import forecast_detail
from src.simulation import demo_series

# Streamlit renamed `use_container_width=True` to `width="stretch"`; support both.
import inspect as _inspect
BTN_WIDE = ({"width": "stretch"} if "width" in _inspect.signature(st.button).parameters
            else {"use_container_width": True})
PLOT_WIDE = ({"width": "stretch"} if "width" in _inspect.signature(st.pyplot).parameters
             else {"use_container_width": True})

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Simply-Fit",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F5F1;
    color: #1A1A1A;
}
.block-container { padding: 3rem 2rem 6rem 2rem; max-width: 820px; }

.brand-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.5rem 0 1.3rem 0;
    border-bottom: 1px solid #E8E4DC;
    margin-bottom: 2.2rem;
}
.brand-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 600;
    color: #1A1A1A; letter-spacing: -0.2px;
}
.brand-logo em { font-style: italic; color: #2D6A4F; }
.brand-pill {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #fff; background: #1A1A1A;
    padding: 3px 10px; border-radius: 20px;
}
.prog-wrap { margin-bottom: 2.4rem; }
.prog-meta { display: flex; justify-content: space-between; margin-bottom: 8px; }
.prog-name { font-size: 0.68rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #1A1A1A; }
.prog-frac { font-size: 0.68rem; color: #B8B0A8; }
.prog-track { display: flex; gap: 4px; }
.prog-seg { height: 3px; flex: 1; border-radius: 3px; background: #E4E0D8; }
.prog-seg.done { background: #2D6A4F; }
.prog-seg.active { background: #95C4AA; }
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; font-weight: 400;
    color: #1A1A1A; letter-spacing: -1px; line-height: 1.13;
    margin-bottom: 0.4rem;
}
.page-title em { font-style: italic; color: #2D6A4F; }
.page-sub { font-size: 0.9rem; color: #909090; font-weight: 300; line-height: 1.6; margin-bottom: 1.8rem; }
.slabel {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #B8B0A8; margin: 2rem 0 0.7rem 0;
    display: flex; align-items: center; gap: 8px;
}
.slabel::after { content: ''; flex: 1; height: 1px; background: #E8E4DC; }
.cards { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.6rem 0 1.6rem 0; }
.card {
    flex: 1; min-width: 130px;
    background: #fff; border: 1px solid #E8E4DC;
    border-radius: 16px; padding: 1.15rem 1.3rem 1.1rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0; background: #E4E0D8;
}
.card.green::before  { background: #2D6A4F; }
.card.amber::before  { background: #B57800; }
.card.red::before    { background: #B92D2D; }
.card.blue::before   { background: #1A56B0; }
.card.dark::before   { background: #1A1A1A; }
.c-lbl { font-size: 0.61rem; font-weight: 700; letter-spacing: 2.2px; text-transform: uppercase; color: #C0B8B0; margin-bottom: 0.45rem; }
.c-val { font-family: 'Playfair Display', serif; font-size: 2.15rem; font-weight: 600; color: #1A1A1A; line-height: 1; letter-spacing: -1px; }
.c-unit { font-family: 'Inter', sans-serif; font-size: 0.76rem; font-weight: 400; letter-spacing: 0; color: #A8A098; margin-left: 5px; }
.c-sub  { font-size: 0.74rem; color: #A8A098; margin-top: 0.32rem; line-height: 1.4; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.69rem; font-weight: 700; margin-top: 0.4rem; }
.p-green  { background: #DCFCE7; color: #14532D; }
.p-yellow { background: #FEF9C3; color: #713F12; }
.p-orange { background: #FFEDD5; color: #7C2D12; }
.p-red    { background: #FEE2E2; color: #7F1D1D; }
.alert { padding: 0.9rem 1.1rem; border-radius: 11px; font-size: 0.87rem; line-height: 1.65; margin: 0.9rem 0; border: 1px solid transparent; }
.a-info    { background: #EFF6FF; color: #1E3A8A; border-color: #93C5FD; }
.a-success { background: #F0FDF4; color: #14532D; border-color: #86EFAC; }
.a-warn    { background: #FFFBEB; color: #713F12; border-color: #FCD34D; }
.a-danger  { background: #FFF1F2; color: #7F1D1D; border-color: #FCA5A5; }
.a-neutral { background: #FAFAF8; color: #3C3C3C; border-color: #D6D3D1; }
.callout {
    background: #1A1A1A; color: #F5F5F3;
    border-radius: 16px; padding: 1.4rem 1.6rem;
    margin: 1rem 0 1.6rem 0;
    display: flex; align-items: center; gap: 1.4rem;
}
.callout-big {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 600;
    letter-spacing: -2px; line-height: 1; white-space: nowrap;
}
.callout-detail { font-size: 0.82rem; color: #999; line-height: 1.6; }
.callout-detail strong { color: #E0DDD8; }
.chart-wrap {
    background: #fff; border: 1px solid #E8E4DC;
    border-radius: 14px; padding: 1.3rem 1.4rem 0.7rem;
    margin: 0.6rem 0 1.3rem 0;
}
.chart-lbl { font-size: 0.67rem; font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase; color: #C0B8B0; margin-bottom: 0.5rem; }
.coach-box {
    background: #1A1A1A; color: #F5F5F3;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    margin: 0.8rem 0; font-size: 0.88rem; line-height: 1.75;
}
.rec {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 0.9rem 1.1rem; background: #fff;
    border: 1px solid #E8E4DC; border-radius: 11px;
    margin-bottom: 8px; font-size: 0.86rem; color: #444; line-height: 1.6;
}
.rec-n { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 600; color: #D4CFC8; min-width: 22px; }
.rec.flagged { border-color: #FCA5A5; background: #FFF8F8; }
.rec.flagged .rec-n { color: #FCA5A5; }
.meal-card { background: #fff; border: 1px solid #E8E4DC; border-radius: 13px; padding: 1rem 1.2rem; margin-bottom: 9px; }
.meal-head { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #B0A898; margin-bottom: 0.6rem; }
.meal-item { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.86rem; color: #333; padding: 0.22rem 0; border-bottom: 1px dashed #F0EDE8; }
.meal-item:last-child { border-bottom: none; }
.meal-qty { font-size: 0.78rem; color: #AAA; white-space: nowrap; margin-left: 8px; }
.meal-kcal { font-size: 0.72rem; color: #2D6A4F; font-weight: 600; min-width: 48px; text-align: right; }
.div { border: none; border-top: 1px solid #E8E4DC; margin: 2.2rem 0; }
.footer { text-align: center; font-size: 0.69rem; color: #C8C0B8; margin-top: 4rem; padding-top: 1.2rem; border-top: 1px solid #E8E4DC; }
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.87rem !important; font-weight: 500 !important;
    border-radius: 10px !important; padding: 0.55rem 1.2rem !important;
}
div[data-testid="column"]:last-child .stButton > button {
    background: #1A1A1A !important; color: #F7F5F1 !important;
    border: 1.5px solid #1A1A1A !important;
}
div[data-testid="column"]:first-child .stButton > button {
    background: transparent !important; color: #1A1A1A !important;
    border: 1.5px solid #D0CBC3 !important;
}
.stNumberInput input, .stTextArea textarea {
    border-radius: 10px !important; border: 1.5px solid #D8D4CC !important;
    background: #fff !important;
}
.stSelectbox > div > div {
    border-radius: 10px !important; border: 1.5px solid #D8D4CC !important;
    background: #fff !important;
}
.stRadio label {
    font-size: 0.9rem !important; background: #fff;
    border: 1.5px solid #E4E0D8; border-radius: 10px;
    padding: 0.65rem 1rem !important; width: 100%; cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for Gemini API Key setup
with st.sidebar:
    st.markdown("### 🔑 API Settings")
    gemini_key_input = st.text_input(
        "Gemini API Key (Optional):",
        type="password",
        value="",
        help="Paste a Google Gemini API key from https://aistudio.google.com to enable real-time Gemini LLM coaching."
    )
    # Keep a server-side secret out of the browser widget, even on a shared deployment.
    server_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_server_key = server_key and st.checkbox("Use the configured server key", value=True)
    st.session_state["gemini_api_key"] = gemini_key_input.strip() or (server_key if use_server_key else "")
    st.caption("When a key is set, your question, selected conditions and summarized progress are sent to Google Gemini. With no active key, the coach gives offline general recommendations.")

    if st.button("Test Gemini connection", disabled=not st.session_state["gemini_api_key"]):
        from src import grounded_coach as grounded
        try:
            # Same two-stage generation path as the coach. No profile or log is sent.
            with st.spinner("Testing Gemini with a sample question…"):
                answer, _ = grounded.online(st.session_state["gemini_api_key"],
                                            "What is two plus two? Answer with one digit.", [], {})
            if answer.strip() == "4":
                st.success("Connected. Gemini answered the sample question and completed the review.")
            else:
                st.warning("Gemini responded, but the sample answer did not pass the expected-output check.")
        except Exception as exc:
            st.error(grounded.failure_notice(exc))

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
PAGES      = ["profile", "metrics", "goal", "target", "baseline", "tracker", "report"]
TOTAL      = len(PAGES)
PAGE_NAMES = ["Profile", "Metrics", "Goal", "Target", "Weight log", "Dashboard", "Report"]

ACTIVITY_MULT = {
    "Sedentary (desk job, little exercise)": 1.2,
    "Lightly active (1–3 days/week)":        1.375,
    "Moderately active (3–5 days/week)":     1.55,
    "Very active (6–7 days/week)":           1.725,
    "Athlete / physical job":                1.9,
}

DISEASES = [
    "Type 2 Diabetes",
    "Hypertension (High Blood Pressure)",
    "High Cholesterol",
    "Hypothyroidism",
    "PCOS",
    "NAFLD",
    "Insulin Resistance / Pre-Diabetes",
    "Chronic Kidney Disease (CKD)",
]

MEAL_PLANS = {
    "lose": {
        "Breakfast (7–9 AM)": [
            ("Oats porridge", "40g dry", 150),
            ("Boiled eggs", "2 whole", 140),
            ("Almonds", "10 nuts", 70),
        ],
        "Lunch (1–2 PM)": [
            ("Brown rice / chapati", "½ cup / 1 roti", 160),
            ("Moong dal", "150g cooked", 120),
            ("Stir-fried vegetables", "1.5 cups", 80),
            ("Green salad", "freely", 30),
        ],
        "Snacks (4–5 PM)": [
            ("Buttermilk / chaas", "1 glass", 35),
            ("Cucumber + carrot sticks", "freely", 40),
            ("Roasted chana", "20g", 75),
        ],
        "Dinner (7–8 PM)": [
            ("Moong dal soup", "1 bowl", 100),
            ("Stir-fried vegetables", "2 cups", 80),
            ("Grilled paneer / tofu", "75g", 90),
        ],
    },
    "gain": {
        "Breakfast (7–9 AM)": [
            ("Whole wheat toast", "3 slices", 240),
            ("Peanut butter", "2 tbsp", 190),
            ("Banana", "1 large", 105),
            ("Whole milk", "1 glass", 150),
        ],
        "Lunch (1–2 PM)": [
            ("Steamed rice / chapati", "1 cup / 2 rotis", 300),
            ("Chicken curry / Paneer", "150g", 250),
            ("Mixed dal", "1 cup", 180),
            ("Curd / dahi", "1 bowl", 100),
        ],
        "Snacks (4–5 PM)": [
            ("Mixed nuts & raisins", "30g", 180),
            ("Protein smoothie", "1 glass", 250),
        ],
        "Dinner (7–8 PM)": [
            ("Chapati / Rice", "2 rotis / 1 cup", 240),
            ("Fish / Paneer / Tofu", "150g", 220),
            ("Cooked vegetables", "1 cup", 100),
        ],
    },
    "maintain": {
        "Breakfast (7–9 AM)": [
            ("Oats / Poha", "1 bowl", 200),
            ("Boiled egg / Tofu", "1 serving", 120),
            ("Fresh fruit", "1 piece", 80),
        ],
        "Lunch (1–2 PM)": [
            ("Chapati / Rice", "1.5 rotis / ¾ cup", 200),
            ("Dal / Rajma / Chole", "1 bowl", 160),
            ("Cooked sabzi", "1 cup", 90),
            ("Salad", "1 plate", 30),
        ],
        "Snacks (4–5 PM)": [
            ("Green tea + makhana", "1 cup / 20g", 80),
        ],
        "Dinner (7–8 PM)": [
            ("Lauki / Torai sabzi", "1 cup", 70),
            ("Chapati / Soup", "1 roti / 1 bowl", 120),
            ("Paneer / Dal", "100g / 1 bowl", 140),
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "profile"

defaults = {
    "age": 28,
    "gender": "male",
    "height": 175.0,
    "weight": 78.0,
    "activity": "Lightly active (1–3 days/week)",
    "diseases": [],
    "goal": "lose",
    "target_change": 5.0,
    "target_weeks": 10,
    "target_calories": -385.0,
    "tdee": 2200.0,
    "bmr": 1700.0,
    "weight_log": [],          # daily grid: float kg, or None for a day with no reading
    "is_demo": False,          # True only after the user explicitly loads simulated demo data
    "window_days": DEFAULT_WINDOW_DAYS,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def go(target_page):
    st.session_state.page = target_page

def brand():
    st.markdown("""
    <div class="brand-header">
      <div class="brand-logo">Simply<em>-Fit</em></div>
      <div class="brand-pill">M.Tech Project</div>
    </div>
    """, unsafe_allow_html=True)

def progress_bar(current_page):
    idx  = PAGES.index(current_page)
    frac = f"{idx + 1} OF {TOTAL}"
    name = PAGE_NAMES[idx].upper()

    segs = ""
    for i in range(TOTAL):
        if i < idx:
            segs += '<div class="prog-seg done"></div>'
        elif i == idx:
            segs += '<div class="prog-seg active"></div>'
        else:
            segs += '<div class="prog-seg"></div>'

    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-meta">
        <span class="prog-name">{name}</span>
        <span class="prog-frac">{frac}</span>
      </div>
      <div class="prog-track">{segs}</div>
    </div>
    """, unsafe_allow_html=True)

def title(heading, subheading):
    st.markdown(f'<div class="page-title">{heading}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subheading}</div>', unsafe_allow_html=True)

def slabel(text):
    st.markdown(f'<div class="slabel">{text}</div>', unsafe_allow_html=True)

def alert(text, kind="neutral"):
    st.markdown(f'<div class="alert a-{kind}">{text}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

def nav(prev_page, next_label, disabled=False):
    col_l, col_r = st.columns([1, 1])
    clicked = False
    with col_l:
        if prev_page:
            if st.button("← Back", key=f"back_{st.session_state.page}"):
                go(prev_page)
                st.rerun()
    with col_r:
        if st.button(next_label, disabled=disabled, key=f"next_{st.session_state.page}"):
            clicked = True
    return clicked

def calc_bmr(w, h, a, g):
    if g == "male":
        return 10 * w + 6.25 * h - 5 * a + 5
    return 10 * w + 6.25 * h - 5 * a - 161

def bmi_cat(bmi):
    if bmi < 18.5:   return "Underweight", "p-orange", "amber"
    elif bmi < 25.0: return "Normal weight", "p-green", "green"
    elif bmi < 30.0: return "Overweight", "p-yellow", "amber"
    else:            return "Obese", "p-red", "red"

def smart_ylim(vals, pad=0.2):
    lo, hi = min(vals), max(vals)
    span   = max(hi - lo, 1.0)
    return lo - span * pad, hi + span * pad

def make_fig(w=7, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor("#fff")
    ax.set_facecolor("#fff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8E4DC")
    ax.spines["bottom"].set_color("#E8E4DC")
    ax.tick_params(colors="#1A1A1A", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4, color="#D0CBC3")
    return fig, ax

def chart_wrap(lbl=""):
    if lbl:
        st.markdown(f'<div class="chart-wrap"><div class="chart-lbl">{lbl}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)

def chart_end():
    st.markdown('</div>', unsafe_allow_html=True)

# ── Shared-estimator helpers. The app contains NO statistical formulas of its
#    own: every number below comes from src/energy_balance.py. ─────────────
LABEL_TEXT = {
    "deficit": "calorie deficit",
    "surplus": "calorie surplus",
    "maintenance": "approximately at maintenance",
    "inconclusive": "inconclusive",
}
WINDOW_OPTIONS = [14, 21, 28, 42, 56, None]


def round10(x):
    """Displayed kcal values are rounded to the nearest 10: finer digits are not meaningful."""
    return int(round(float(x) / 10.0) * 10)


def log_to_text(log):
    return ", ".join("-" if v is None else f"{v:g}" for v in log)


def latest_reading(log):
    for v in reversed(log):
        if v is not None:
            return v
    return None


def load_weight_log(readings, is_demo):
    """Synchronize the editable text and saved readings before widgets render."""
    st.session_state.weight_log = readings
    st.session_state.is_demo = is_demo
    st.session_state.log_text_area = log_to_text(readings)


def load_demo_log():
    load_weight_log(demo_series(start_weight=st.session_state.weight), True)


def demo_banner():
    if st.session_state.get("is_demo"):
        alert("<strong>Simulated demo data.</strong> The readings shown were generated by a computer "
              "simulation, not measured on a person.", "warn")


def window_selector():
    current = st.session_state.get("window_days", DEFAULT_WINDOW_DAYS)
    choice = st.selectbox(
        "Analysis window",
        WINDOW_OPTIONS,
        index=WINDOW_OPTIONS.index(current) if current in WINDOW_OPTIONS else 2,
        format_func=lambda v: "All data" if v is None else f"Last {v} days",
        help="How far back to look when estimating your current trend. Short windows react "
             "faster to a change in habits but are much less certain.",
        key=f"window_select_{st.session_state.page}")
    st.session_state.window_days = choice
    return choice


def analyse(log, window):
    """Single call site for the estimator; the result is kept for tests and the coach."""
    res = estimate_energy_balance(log, window_days=window)
    rel = assess_reliability(res)
    st.session_state["last_result"] = res
    st.session_state["last_reliability"] = rel
    return res, rel


def render_verdict(res, rel):
    slabel("Approximate energy balance from your weight trend")
    if res["status"] != "ok":
        alert(f"<strong>No estimate yet.</strong> {res['message']}", "info")
        return
    lo, hi = res["kcal_ci"]
    wlo, whi = (v * 7 for v in res["kg_per_day_ci"])
    label = res["label"]
    headline = {
        "deficit": "Calorie deficit",
        "surplus": "Calorie surplus",
        "maintenance": "About maintenance",
        "inconclusive": "Not yet clear",
    }[label]
    explain = {
        "deficit": "Your weight trend is downward and the whole uncertainty range is below zero.",
        "surplus": "Your weight trend is upward and the whole uncertainty range is above zero.",
        "maintenance": "Your weight trend is flat and the uncertainty range is narrow.",
        "inconclusive": "The uncertainty range is too wide to say whether you are in a deficit, "
                        "a surplus or at maintenance. More days of data will narrow it.",
    }[label]
    if res["too_short_for_label"]:
        headline = "Too early to tell"
        explain = (f"The readings in this window cover {res['n_calendar_days']} days. No deficit or surplus "
                   f"call is made on fewer than {MIN_CALENDAR_DAYS_FOR_LABEL} days, because in simulation "
                   f"tests such early calls pointed the wrong way too often. The figures below are a rough "
                   f"early indication only.")
    rel_col = {"high": "green", "moderate": "amber", "low": "red"}[rel["level"]]
    st.markdown(
        f'<div class="callout">'
        f'<div class="callout-big">{headline}</div>'
        f'<div class="callout-detail">{explain}</div></div>',
        unsafe_allow_html=True)
    st.markdown(f"""
    <div class="cards">
      <div class="card dark">
        <div class="c-lbl">Energy-balance equivalent</div>
        <div class="c-val">≈ {round10(res['kcal_per_day']):+d}<span class="c-unit">kcal/day</span></div>
        <div class="c-sub">Simulation-calibrated 95% range {round10(lo):+d} to {round10(hi):+d}</div>
      </div>
      <div class="card">
        <div class="c-lbl">Weight trend</div>
        <div class="c-val">{res['weekly_kg_change']:+.2f}<span class="c-unit">kg/week</span></div>
        <div class="c-sub">Range {wlo:+.2f} to {whi:+.2f}</div>
      </div>
      <div class="card {rel_col}">
        <div class="c-lbl">Data reliability</div>
        <div class="c-val">{rel['level'].capitalize()}</div>
        <div class="c-sub">{res['n_obs']} readings over {res['span_days'] + 1} days</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_reliability(rel):
    kind = {"high": "success", "moderate": "warn", "low": "danger", "insufficient": "info"}[rel["level"]]
    alert(f"<strong>Reliability: {rel['level']}.</strong> {rel['summary']}", kind)
    with st.expander("Why this reliability level? (data-quality checks)"):
        mark = {"good": "OK", "warn": "Check", "bad": "Poor"}
        st.table(pd.DataFrame(
            [{"Check": f["name"], "Measured": f["value"], "Status": mark[f["status"]],
              "What it means": f["message"]} for f in rel["factors"]]).set_index("Check"))
        st.caption("The level is set by the width of the uncertainty range and lowered one step if "
                   "any other check is flagged. It is a description of data quality, not a "
                   "probability that the estimate is correct.")


def render_plan_comparison(res, target_weekly, goal):
    """Compare the measured trend with the plan, using the interval (corrected sign logic)."""
    if goal == "maintain":
        if res["label"] == "maintenance":
            alert("Your trend is consistent with your goal of maintaining weight.", "success")
        elif res["label"] in ("deficit", "surplus"):
            alert(f"Your goal is to maintain, but the trend indicates a {LABEL_TEXT[res['label']]}.", "warn")
        return
    wlo, whi = (v * 7 for v in res["kg_per_day_ci"])
    rate = res["weekly_kg_change"]
    if res["label"] == "inconclusive":
        alert(f"Your trend is too uncertain (range {wlo:+.2f} to {whi:+.2f} kg/week) to say whether "
              f"you are on your plan of {target_weekly:+.2f} kg/week.", "neutral")
        return
    if wlo <= target_weekly <= whi:
        alert(f"Your trend ({rate:+.2f} kg/week, range {wlo:+.2f} to {whi:+.2f}) is "
              f"<strong>consistent with your plan</strong> of {target_weekly:+.2f} kg/week.", "success")
        return
    faster = (rate < target_weekly) if goal == "lose" else (rate > target_weekly)
    verb = "losing" if goal == "lose" else "gaining"
    gap_kcal = round10(abs(rate - target_weekly) * KCAL_PER_KG / 7)
    if faster:
        alert(f"You are <strong>{verb} faster than planned</strong>: {rate:+.2f} kg/week against a plan of "
              f"{target_weekly:+.2f} kg/week (roughly {gap_kcal} kcal/day beyond the plan).", "info")
    else:
        alert(f"You are <strong>behind your plan</strong>: {rate:+.2f} kg/week against a plan of "
              f"{target_weekly:+.2f} kg/week (roughly {gap_kcal} kcal/day short of it).", "warn")


def render_trend_chart(log, res):
    checked = validate_weight_log(log)
    if not checked.ok or checked.n_obs == 0:
        return
    slabel("Weight readings and underlying trend")
    chart_wrap("Daily readings · fitted trend · range of trends consistent with the data")
    d, w = checked.days + 1, checked.weights
    fig_t, ax_t = make_fig(7, 3.4)
    ax_t.plot(d, w, linestyle="none", marker="o", markersize=4.2, markerfacecolor="#9A948C",
              markeredgecolor="#9A948C", label="Daily reading", zorder=3)
    vals = list(w)
    if res["status"] == "ok":
        fd = res["fit_days"]
        mid = fd.mean()
        centre = res["intercept"] + res["kg_per_day"] * mid
        lo_s, hi_s = res["kg_per_day_ci"]
        band_lo = centre + np.minimum(lo_s * (fd - mid), hi_s * (fd - mid))
        band_hi = centre + np.maximum(lo_s * (fd - mid), hi_s * (fd - mid))
        ax_t.fill_between(fd + 1, band_lo, band_hi, color="#2D6A4F", alpha=0.16,
                          label="Slopes in calibrated range, anchored at midpoint", zorder=2)
        ax_t.plot(fd + 1, res["trend_line"], color="#1A1A1A", linewidth=2.4,
                  label="Underlying trend", zorder=4)
        om = res["outlier_mask"]
        if om.any():
            ax_t.scatter(fd[om] + 1, res["fit_weights"][om], facecolors="none", edgecolors="#B92D2D",
                         s=90, linewidths=1.6, zorder=5, label="Unusual reading (down-weighted)")
        vals += list(band_lo) + list(band_hi)
    ylo, yhi = smart_ylim(vals)
    ax_t.set_ylim(ylo, yhi)
    ax_t.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_t.set_xlabel("Day", fontsize=8.5)
    ax_t.set_ylabel("Weight (kg)", fontsize=8.5)
    ax_t.legend(loc="best", fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig_t, **PLOT_WIDE)
    plt.close()
    chart_end()
    if res["status"] == "ok" and res["n_outliers"]:
        st.caption(f"{res['n_outliers']} reading(s) lie unusually far from the trend. They are kept in "
                   "the data and given less influence; nothing is deleted or replaced.")


def render_rolling_chart(log, window, target):
    roll = rolling_energy_balance(log, window_days=window)
    if sum(np.isfinite(r["kcal_per_day"]) for r in roll) < 2:
        return
    slabel("How the estimate has moved over time")
    span_txt = "all data up to each day" if window is None else f"the {window} days up to each day"
    chart_wrap(f"Estimate using only {span_txt} · shaded band = approx. 95% range")
    x = np.array([r["day"] for r in roll]) + 1
    y = np.array([r["kcal_per_day"] for r in roll])
    lo = np.array([r["kcal_lo"] for r in roll])
    hi = np.array([r["kcal_hi"] for r in roll])
    fig2, ax2 = make_fig(7, 3.2)
    ax2.axhspan(-CLASS_TOLERANCE_KCAL, CLASS_TOLERANCE_KCAL, color="#E8E4DC", alpha=0.55,
                label=f"Maintenance zone (±{CLASS_TOLERANCE_KCAL:.0f})", zorder=1)
    ax2.axhline(0, color="#B8B0A8", linewidth=0.9, zorder=2)
    ax2.fill_between(x, lo, hi, color="#2D6A4F", alpha=0.18, zorder=2)
    ax2.plot(x, y, color="#1A1A1A", linewidth=2.2, label="Energy-balance equivalent", zorder=4)
    if target != 0:
        ax2.axhline(target, color="#B57800", linewidth=1.4, linestyle="--",
                    label=f"Plan ({round10(target):+d} kcal/day)", zorder=3)
    ax2.set_xlabel("Day", fontsize=8.5)
    ax2.set_ylabel("kcal / day  (− deficit, + surplus)", fontsize=8.5)
    ax2.legend(loc="best", fontsize=7.5)
    plt.tight_layout()
    st.pyplot(fig2, **PLOT_WIDE)
    plt.close()
    chart_end()


def render_assumptions():
    with st.expander("Assumptions and limitations — please read"):
        st.markdown(f"""
- **This is an estimate from your weight trend, not a measurement.** Weight alone cannot reveal how
  many calories you ate or burned.
- **7,700 kcal per kg is a simplifying assumption.** The energy in a kilogram of weight change depends
  on how much of it is fat, lean tissue or water, and differs between people and over time. The
  uncertainty range does *not* include this.
- **The range targets the average tissue-weight trend equivalent over the observed dates.** It is calibrated on separate simulated logs passing the reporting rule. At the nominal 95% level, held-out coverage was 95.2% overall among reported logs, 87.7% for reported 28-day logs, and 79.4% with longer-lasting simulated water shifts.
- **Changes in body water look like changes in energy balance.** Starting or stopping a diet, salty
  meals, illness, travel, hormonal cycles and new exercise routines can shift weight by a kilogram or
  more for days or weeks.
- **At least {MIN_OBS_FOR_ESTIMATE} readings covering {MIN_CALENDAR_DAYS_FOR_ESTIMATE} days** are required
  before any number is shown, and **{MIN_CALENDAR_DAYS_FOR_LABEL} days** before a deficit or surplus is named.
- **Tested on simulated data only.** No clinical or real-world validation has been done.
- Not medical advice.
""")


def coach_html(text):
    """Minimal, safe rendering of the coach's **bold** and line breaks inside the dark box."""
    import html as _html
    safe = _html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    return safe.replace("\n", "<br>")


def render_coach_reply(reply):
    st.caption(reply["notice"])
    st.markdown(f'<div class="coach-box">{coach_html(reply["response"])}</div>', unsafe_allow_html=True)
    for source in reply["sources"]:
        st.markdown(f'[{source["title"]}]({source["url"]})')


def get_coach_response(profile, weight_log, question, target_weekly):
    goal = st.session_state.get("goal", "lose")
    change = st.session_state.get("target_change", 0.0)
    start_w = st.session_state.get("weight", 0.0)
    goal_weight = start_w + (-change if goal == "lose" else change if goal == "gain" else 0.0)
    prof_dict = {
        "diseases": profile.get("diseases", []),
        "is_demo": st.session_state.is_demo,
        "target_weight": round(goal_weight, 1),
        "tdee": st.session_state.get("tdee", 2200),
    }
    api_key = st.session_state.get("gemini_api_key", os.getenv("GEMINI_API_KEY"))
    try:
        res = agent_coach.run_agent(prof_dict, weight_log, question, target_weekly, gemini_api_key=api_key, window_days=st.session_state.window_days)
        return res
    except Exception as exc:
        # Expose only project source location and class, never exception text, credentials or health data.
        from pathlib import Path
        import traceback
        root = Path(__file__).resolve().parents[1]
        frames = [f for f in traceback.extract_tb(exc.__traceback__)
                  if Path(f.filename).resolve().is_relative_to(root)]
        location = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else "coach"
        return dict(notice=f"Coach error · {type(exc).__name__} · {location}",
                    response="Open API Settings and choose Test Gemini connection. If you need help, share the error label above.",
                    sources=[])


def render_meal_plan(goal, tdee, target):
    plan = MEAL_PLANS.get(goal, MEAL_PLANS["lose"])
    rec_target = int(tdee + target)
    slabel(f"Suggested daily meal plan (~{rec_target} kcal target)")

    for meal_name, items in plan.items():
        st.markdown(f'<div class="meal-card"><div class="meal-head">{meal_name}</div>', unsafe_allow_html=True)
        items_html = ""
        for name, qty, kcal in items:
            items_html += (f'<div class="meal-item"><span>{name}<span class="meal-qty">({qty})</span></span>'
                           f'<span class="meal-kcal">{kcal} kcal</span></div>')
        st.markdown(items_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PROFILE
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "profile":
    brand()
    progress_bar("profile")
    title("Let's set up your<br><em>profile.</em>",
          "Used for BMI and a formula-based maintenance estimate. The energy-balance estimate itself uses only your weight log.")

    col1, col2 = st.columns(2)
    with col1:
        age    = st.number_input("Age", 18, 90, int(st.session_state.age))
        gender = st.selectbox("Biological sex", ["male", "female"],
                              index=0 if st.session_state.gender=="male" else 1,
                              format_func=lambda x: x.capitalize())
        height = st.number_input("Height (cm)", 120.0, 220.0,
                                 float(st.session_state.height), step=0.5)

    with col2:
        weight   = st.number_input("Starting weight (kg)", 35.0, 200.0,
                                   float(st.session_state.weight), step=0.5)
        activity = st.selectbox("Activity level", list(ACTIVITY_MULT.keys()),
                                index=list(ACTIVITY_MULT.keys()).index(st.session_state.activity))

    divider()
    if nav(None, "Continue"):
        st.session_state.age      = age
        st.session_state.gender   = gender
        st.session_state.height   = height
        st.session_state.weight   = weight
        st.session_state.activity = activity

        bmr  = calc_bmr(weight, height, age, gender)
        tdee = bmr * ACTIVITY_MULT[activity]
        st.session_state.bmr  = bmr
        st.session_state.tdee = tdee

        go("metrics")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — METRICS
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "metrics":
    brand()
    progress_bar("metrics")
    title("Your physiological<br><em>baseline.</em>",
          "Calculated from standard population formulas (Mifflin-St Jeor); individual values can differ.")

    w = st.session_state.weight
    h = st.session_state.height
    a = st.session_state.age
    g = st.session_state.gender

    bmi                     = w / (h / 100) ** 2
    bmi_lbl, bmi_pill, card_col = bmi_cat(bmi)

    bmr  = st.session_state.bmr
    tdee = st.session_state.tdee

    slabel("Key indicators")
    st.markdown(f"""
    <div class="cards">
      <div class="card {card_col}">
        <div class="c-lbl">BMI</div>
        <div class="c-val">{bmi:.1f}</div>
        <span class="pill {bmi_pill}">{bmi_lbl}</span>
      </div>
      <div class="card">
        <div class="c-lbl">BMR</div>
        <div class="c-val">{int(bmr)}<span class="c-unit">kcal</span></div>
        <div class="c-sub">Basal metabolic rate</div>
      </div>
      <div class="card dark">
        <div class="c-lbl">TDEE</div>
        <div class="c-val">{int(tdee)}<span class="c-unit">kcal</span></div>
        <div class="c-sub">Maintenance calories</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    slabel("Medical conditions")
    diseases = st.multiselect("Select any relevant medical conditions", DISEASES,
                              default=st.session_state.diseases)
    st.session_state.diseases = diseases

    if diseases:
        alert(f"Selected: <strong>{', '.join(diseases)}</strong>. "
              f"The app will show a short reference note for each; it does not adjust any "
              f"calculation for medical conditions.", "warn")

    # BMI Spectrum Chart
    slabel("BMI Spectrum Analysis")
    chart_wrap("BMI Categories")
    fig_b, ax_b = make_fig(7, 1.8)
    ax_b.axvspan(15, 18.5, color="#FEF3C7", alpha=0.6)
    ax_b.axvspan(18.5, 25, color="#DCFCE7", alpha=0.6)
    ax_b.axvspan(25, 30, color="#FEF3C7", alpha=0.6)
    ax_b.axvspan(30, 40, color="#FEE2E2", alpha=0.6)
    ax_b.axvline(bmi, color="#1A1A1A", linewidth=2.5, linestyle="-")
    ax_b.text(bmi, 0.1, f" You ({bmi:.1f})",
              va="bottom", fontsize=9.5, fontweight="bold", color="#1A1A1A")
    ax_b.set_xlim(15, 40); ax_b.set_ylim(-0.72, 0.6)
    ax_b.set_yticks([]); ax_b.set_xticks([18.5, 25, 30])
    ax_b.set_xticklabels(["18.5", "25", "30"]); ax_b.grid(False)
    for sp in ax_b.spines.values(): sp.set_visible(False)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig_b, **PLOT_WIDE)
    plt.close()
    chart_end()

    divider()
    if nav("profile", "Set my goal"):
        go("goal")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — GOAL
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "goal":
    brand()
    progress_bar("goal")
    title("What are you<br><em>working towards?</em>",
          "Your goal shapes every calorie target and recommendation.")

    goal = st.radio("Select your primary goal:", ["lose", "gain", "maintain"],
                    format_func=lambda x: {
                        "lose": "Lose weight",
                        "gain": "Gain weight",
                        "maintain": "Maintain weight"
                    }[x],
                    index=["lose","gain","maintain"].index(st.session_state.goal))

    descs = {
        "lose":     ("success", "Achieve a calorie deficit through diet and movement. Safe limit: 1 kg/week."),
        "gain":     ("warn",    "Build mass through a controlled surplus. Safe limit: 1.0 kg/week."),
        "maintain": ("info",    "Keep your current weight stable. We will track fluctuations and alert you if the trend drifts."),
    }
    alert(descs[goal][1], descs[goal][0])

    divider()
    if nav("metrics", "Continue"):
        st.session_state.goal = goal
        go("target")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — TARGET
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "target":
    brand()
    progress_bar("target")
    goal = st.session_state.goal
    title("Define your<br><em>target.</em>",
          "How much do you want to change, and in how long?")

    ok = True
    target_calories = 0.0

    if goal == "maintain":
        alert(f"No change target needed. Your daily goal equals your TDEE: "
              f"<strong>{int(st.session_state.tdee)} kcal/day</strong>.", "info")
    else:
        slabel("Weight change goal")
        c1, c2 = st.columns(2)
        with c1:
            change = st.number_input("Weight change (kg)", 0.5, 50.0,
                                     float(st.session_state.target_change),
                                     step=0.5, format="%.1f")
        with c2:
            weeks = st.number_input("Time frame (weeks)", 1, 104,
                                    int(st.session_state.target_weeks), step=1)

        rate  = change / weeks
        max_r = 1.0 if goal == "lose" else 1.0

        if rate > max_r:
            alert(f"Rate of <strong>{rate:.2f} kg/week</strong> exceeds safe limit of "
                  f"{max_r} kg/week. Please adjust.", "danger")
            ok = False
        else:
            daily_kcal      = (KCAL_PER_KG * change) / (weeks * 7)
            target_calories = -daily_kcal if goal == "lose" else daily_kcal
            direction       = "deficit" if goal == "lose" else "surplus"
            intake          = int(st.session_state.tdee + target_calories)
            st.markdown(
                f'<div class="callout">'
                f'<div class="callout-big">{int(abs(daily_kcal))} kcal</div>'
                f'<div class="callout-detail">'
                f'Daily <strong>{direction}</strong> needed<br>'
                f'Target daily intake: <strong>{intake} kcal</strong><br>'
                f'Pace: <strong>{rate:.2f} kg/week</strong>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    divider()
    if nav("goal", "Enter weight log", disabled=not ok):
        if goal != "maintain":
            st.session_state.target_change   = change
            st.session_state.target_weeks    = weeks
            st.session_state.target_calories = target_calories
        else:
            st.session_state.target_change   = 0.0
            st.session_state.target_weeks    = 1
            st.session_state.target_calories = 0.0
        go("baseline")
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — BASELINE (weight log entry)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "baseline":
    brand()
    progress_bar("baseline")
    title("Enter your<br><em>weight log.</em>",
          "One reading per day, in kg. Skipped days are fine — mark them, never guess them.")
    demo_banner()

    tab_text, tab_csv, tab_demo = st.tabs(["Type or paste", "Upload CSV", "Simulated demo data"])

    with tab_text:
        slabel("Daily readings, oldest first")
        if "log_text_area" not in st.session_state:
            st.session_state.log_text_area = log_to_text(st.session_state.weight_log)
        log_text = st.text_area(
            "Weights in kg separated by commas or new lines. Use '-' for a day with no reading.",
            height=130, key="log_text_area",
            placeholder="e.g. 78.4, 78.1, -, 78.3, 77.9")
        parsed_log, parse_errors = parse_weight_text(log_text)

    with tab_csv:
        slabel("CSV with a 'weight' column and optionally a 'date' column")
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
        if up is not None:
            csv_log, csv_errors = parse_weight_csv(up.getvalue())
            for e in csv_errors:
                alert(e, "danger")
            if csv_log and not csv_errors:
                st.button("Use this file", key="use_csv",
                          on_click=load_weight_log, args=(csv_log, False))

    with tab_demo:
        alert("This loads a <strong>computer-simulated</strong> 6-week log so you can see how the "
              "analysis works. It is not anyone's real data, and it replaces whatever is entered "
              "in the first tab.", "warn")
        st.button("Load simulated demo data", key="load_demo", on_click=load_demo_log)

    check = validate_weight_log(parsed_log) if parsed_log else None
    for e in parse_errors:
        alert(e, "danger")
    if check is not None:
        for e in check.errors:
            alert(e, "danger")
        for w_msg in check.warnings:
            alert(w_msg, "warn")
        if check.ok and not parse_errors:
            alert(f"<strong>{check.n_obs}</strong> reading(s) covering <strong>{check.n_calendar_days}</strong> "
                  f"day(s); {check.n_missing_days} day(s) without a reading.", "info")
            if check.n_obs < MIN_OBS_FOR_ESTIMATE or check.n_calendar_days < MIN_CALENDAR_DAYS_FOR_ESTIMATE:
                alert(f"An energy-balance estimate needs at least {MIN_OBS_FOR_ESTIMATE} readings covering "
                      f"{MIN_CALENDAR_DAYS_FOR_ESTIMATE} days. You can continue and add readings day by day.", "neutral")
    else:
        alert("No readings entered yet.", "neutral")

    usable = check is not None and check.ok and not parse_errors
    divider()
    if nav("target", "Start daily tracker", disabled=not usable):
        if parsed_log != st.session_state.weight_log:
            st.session_state.is_demo = False
        st.session_state.weight_log = parsed_log
        go("tracker")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — TRACKER (energy-balance trend dashboard)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "tracker":
    brand()
    progress_bar("tracker")
    title("Energy-balance<br><em>dashboard.</em>",
          "What your weight trend says about calorie surplus or deficit — and how sure we can be.")
    demo_banner()

    log    = st.session_state.weight_log
    target = st.session_state.target_calories
    goal   = st.session_state.goal
    latest = latest_reading(log)

    slabel("Log today's weight")
    col_i1, col_i2, col_i3 = st.columns([3, 1.3, 1.7])
    with col_i1:
        new_w = st.number_input("Today's scale reading (kg)", min_value=MIN_PLAUSIBLE_KG,
                                max_value=MAX_PLAUSIBLE_KG,
                                value=float(latest if latest is not None else st.session_state.weight),
                                step=0.1, format="%.1f")
    with col_i2:
        st.write("")
        if st.button("Add reading", **BTN_WIDE):
            st.session_state.weight_log.append(round(new_w, 1))
            st.rerun()
    with col_i3:
        st.write("")
        if st.button("No reading today", **BTN_WIDE):
            st.session_state.weight_log.append(None)
            st.rerun()

    window = window_selector()
    res, rel = analyse(log, window)
    target_weekly = (target * 7) / KCAL_PER_KG

    if res["status"] == "invalid":
        for e in res["errors"]:
            alert(e, "danger")
    else:
        render_verdict(res, rel)
        render_reliability(rel)
        if res["status"] == "ok":
            render_plan_comparison(res, target_weekly, goal)
        render_trend_chart(log, res)
        if res["status"] == "ok":
            render_rolling_chart(log, window, target)
        render_assumptions()

        slabel("Ask your coach")
        question = st.text_input("Ask anything about your progress...",
                                 placeholder="Why is my weight fluctuating?")
        if st.button("Ask coach", **BTN_WIDE):
            if question.strip():
                response = get_coach_response(
                    {"diseases": st.session_state.diseases}, log, question, target_weekly
                )
                render_coach_reply(response)

    divider()
    if nav("baseline", "View full report", disabled=res["status"] == "invalid"):
        go("report")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 7 — REPORT
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "report":
    brand()
    progress_bar("report")
    log      = st.session_state.weight_log
    target   = st.session_state.target_calories
    goal     = st.session_state.goal
    h        = st.session_state.height
    diseases = st.session_state.diseases
    w_kg     = st.session_state.weight
    tdee     = st.session_state.tdee
    checked  = validate_weight_log(log)

    title("Your progress<br><em>report.</em>",
          f"Generated {datetime.today().strftime('%d %B %Y')} · "
          f"{checked.n_obs} readings over {checked.n_calendar_days} days")
    demo_banner()

    window = window_selector()
    res, rel = analyse(log, window)
    res_all = estimate_energy_balance(log, window_days=None)
    target_weekly = (target * 7) / KCAL_PER_KG

    if res["status"] == "invalid":
        for e in res["errors"]:
            alert(e, "danger")
        st.stop()

    first_w, last_w = float(checked.weights[0]), float(checked.weights[-1])
    slabel("Summary")
    if res_all["status"] == "ok":
        trend_change = res_all["trend_line"][-1] - res_all["trend_line"][0]
        change_html = (f'<div class="c-val">{trend_change:+.1f}<span class="c-unit">kg</span></div>'
                       f'<div class="c-sub">Trend change over {checked.n_calendar_days} days</div>')
    else:
        change_html = ('<div class="c-val">—</div>'
                       '<div class="c-sub">Not enough data for a trend</div>')
    st.markdown(f"""
    <div class="cards">
      <div class="card"><div class="c-lbl">First reading</div>
        <div class="c-val">{first_w:.1f}<span class="c-unit">kg</span></div>
        <div class="c-sub">Day 1</div></div>
      <div class="card"><div class="c-lbl">Latest reading</div>
        <div class="c-val">{last_w:.1f}<span class="c-unit">kg</span></div>
        <div class="c-sub">Single readings include water weight</div></div>
      <div class="card"><div class="c-lbl">Trend change</div>{change_html}</div>
    </div>
    """, unsafe_allow_html=True)

    slabel(f"Current estimate — last {window} days" if window else "Estimate — all data")
    render_verdict(res, rel)
    render_reliability(rel)
    if window and res["status"] == "ok" and res_all["status"] == "ok":
        lo_a, hi_a = res_all["kcal_ci"]
        alert(f"Whole-period average ({checked.n_calendar_days} days): about "
              f"<strong>{round10(res_all['kcal_per_day']):+d} kcal/day</strong> "
              f"(approximate 95% range {round10(lo_a):+d} to {round10(hi_a):+d}). "
              f"If your habits changed during the period, the recent-window figure above is the "
              f"relevant one.", "neutral")
    if res["status"] == "ok":
        render_plan_comparison(res, target_weekly, goal)
    render_trend_chart(log, res)
    if res["status"] == "ok":
        render_rolling_chart(log, window, target)

    # ── Secondary feature: 7-day forecast ───────────────────────
    slabel("7-day weight forecast (secondary)")
    fc = forecast_detail(log)
    if fc is None:
        alert("A forecast needs at least 3 readings in the last 7 days.", "neutral")
    else:
        chart_wrap("Recent readings · expected level of the next 7 readings · 80% prediction band")
        recent = checked.days > checked.days[-1] - 28
        fig_f, ax_f = make_fig(7, 3.0)
        ax_f.plot(checked.days[recent] + 1, checked.weights[recent], linestyle="none", marker="o",
                  markersize=4.5, color="#1A1A1A", label="Readings")
        ax_f.plot(fc["days"] + 1, fc["mean"], color="#2D6A4F", linewidth=2.0, label="Forecast")
        if fc["lo"] is not None:
            ax_f.fill_between(fc["days"] + 1, fc["lo"], fc["hi"], color="#2D6A4F", alpha=0.15,
                              label="80% prediction band")
        ax_f.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax_f.set_xlabel("Day", fontsize=8.5)
        ax_f.set_ylabel("Weight (kg)", fontsize=8.5)
        ax_f.legend(loc="best", fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig_f, **PLOT_WIDE)
        plt.close()
        chart_end()
        st.caption("The forecast is the average of your last 3 readings carried forward; over one week, "
                   "day-to-day water weight is larger than the trend, so this simple rule was the most "
                   "accurate of those tested on simulated data. It reads slightly high during steady loss. "
                   + ("The band is a prediction interval: where a single future scale reading is expected "
                      "to fall, sized from how this rule would have performed on your own last six weeks. "
                      "It is much wider than the uncertainty of the trend."
                      if fc["lo"] is not None else
                      "No prediction band is shown yet: it needs about three weeks of readings."))

    # ── Goal projection (only when the trend direction is established) ──
    if goal != "maintain":
        slabel("Goal projection")
        tc    = st.session_state.target_change
        t_end = w_kg + (-tc if goal == "lose" else tc)
        wanted = "deficit" if goal == "lose" else "surplus"
        if res["status"] != "ok" or res["label"] != wanted:
            alert(f"No projection is shown: your recent trend is "
                  f"<strong>{LABEL_TEXT.get(res['label'], 'not yet measurable')}</strong>, so there is no "
                  f"established movement towards your goal of {t_end:.1f} kg to extrapolate.", "neutral")
        else:
            now_w = float(res["trend_line"][-1])
            remaining = t_end - now_w
            lo_s, hi_s = res["kg_per_day_ci"]
            if remaining * res["kg_per_day"] <= 0:
                alert(f"Your trend weight ({now_w:.1f} kg) has already reached your goal of {t_end:.1f} kg.", "success")
            else:
                d_mid = remaining / res["kg_per_day"]
                d_fast, d_slow = sorted(remaining / s_ for s_ in (lo_s, hi_s))
                alert(f"At the current trend ({res['weekly_kg_change']:+.2f} kg/week) your goal of "
                      f"<strong>{t_end:.1f} kg</strong> is about <strong>{d_mid / 7:.0f} weeks</strong> away "
                      f"(range {d_fast / 7:.0f}–{d_slow / 7:.0f} weeks given the uncertainty in the trend). "
                      f"This assumes the trend stays constant, which usually does not hold over long periods.",
                      "info")

    # ── Coach ───────────────────────────────────────────────────
    slabel("Ask your AI coach")
    question_rep = st.text_input("Ask anything about your progress or timeline...",
                                 placeholder="How much time will it take to achieve my goal?")
    if st.button("Ask AI coach", **BTN_WIDE):
        if question_rep.strip():
            response = get_coach_response({"diseases": diseases}, log, question_rep, target_weekly)
            render_coach_reply(response)

    # ── BMI ─────────────────────────────────────────────────────
    bmi_w   = float(res_all["trend_line"][-1]) if res_all["status"] == "ok" else last_w
    bmi_cur = bmi_w / (h/100)**2
    bmi_lbl, bmi_pill, card_col = bmi_cat(bmi_cur)
    slabel("Body-mass index")
    st.markdown(f"""
    <div class="cards">
      <div class="card {card_col}">
        <div class="c-lbl">Current BMI</div>
        <div class="c-val">{bmi_cur:.1f}</div>
        <span class="pill {bmi_pill}">{bmi_lbl}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("BMI categories are population screening bands (WHO). BMI does not distinguish fat "
               "from muscle and is not a diagnosis.")

    # Meal plan (illustrative, unchanged from the internship version)
    render_meal_plan(goal, tdee, target)
    st.caption("Illustrative example menu only; portion energies are rough typical values. "
               "Not a prescription.")

    # General tips
    slabel("General tips")
    recs = [
        ("Weigh yourself at the same time each day, for example in the morning before eating. Consistent conditions reduce scatter and make the trend estimate more reliable.", False),
        ("Judge progress by the trend over several weeks, not by any single reading.", False),
        ("Public-health guidance for adults commonly recommends at least 150 minutes of moderate aerobic activity per week.", False),
    ]
    if diseases:
        recs.append(("You have listed a medical condition. Set calorie, protein and fluid targets with "
                     "your clinician or dietitian rather than from this app.", True))
        for d in diseases:
            matches = rag_engine.search_guidelines("dietary guidelines", condition=d, top_k=1)
            if matches:
                recs.append((f"<strong>Reference note ({d}):</strong> {matches[0]['recommendation']} "
                             f"<em>(Attributed source: {matches[0]['source']}; confirm with your clinician.)</em>", True))

    recs_html = ""
    for i, (text, flagged) in enumerate(recs, 1):
        cls = "rec flagged" if flagged else "rec"
        recs_html += (f'<div class="{cls}"><span class="rec-n">{i:02d}</span>'
                      f'<span>{text}</span></div>')
    st.markdown(recs_html, unsafe_allow_html=True)

    render_assumptions()

    # Validation expander
    slabel("Method validation (simulation only)")
    with st.expander("How these methods were tested"):
        st.markdown(
            "All testing used **computer-simulated** weight logs with known ground truth. "
            "No real-world or clinical validation has been done. See `EXPERIMENT_RESULTS.md` "
            "in the project for the full results, including the cases where the method fails."
        )
        summary_path = os.path.join(ROOT_DIR, "results", "experiments", "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summ = json.load(f)
            hl = summ.get("app_headline", {})
            if hl:
                st.table(pd.DataFrame(hl).T)
                st.caption(summ.get("app_headline_caption", ""))
        if st.button("Run demographic realism check (NHANES)", key="ks_test_report_orig"):
            ks_res = run_nhanes_calibration_test(n_users=500)
            st.write(f"• **Start-weight distribution:** KS {ks_res['weight_ks_before']} (naive parameters) → "
                     f"{ks_res['weight_ks_after']} (NHANES-fitted)")
            st.write(f"• **Age distribution:** KS {ks_res['age_ks_before']} (naive parameters) → "
                     f"{ks_res['age_ks_after']} (NHANES-fitted)")
            st.caption(f"Compares only the simulated cohort's age and start-weight distributions with "
                       f"{ks_res['n_nhanes']:,} adults from CDC NHANES 2017-2018. It says nothing about "
                       f"whether simulated weight dynamics resemble real ones.")
        st.markdown("**7-day forecast error on the internship synthetic cohort** (historical saved file; "
                    "see `results/forecast_reevaluation.json` for the re-evaluation)")
        try:
            with open(os.path.join(ROOT_DIR, "results", "model_evaluation.json")) as f:
                ev = json.load(f)
            st.table(pd.DataFrame({
                "Method": ["Linear extrapolation", "Persistence", "LSTM (as originally evaluated)"],
                "MAE (kg)": [round(ev["mae_kg"]["linear"], 3),
                              round(ev["mae_kg"]["persistence"], 3),
                              round(ev["mae_kg"]["lstm"], 3)],
            }).set_index("Method"))
        except FileNotFoundError:
            st.caption("results/model_evaluation.json not found.")

    divider()
    _, col_r = st.columns([1, 1])
    with col_r:
        if st.button("Start over", **BTN_WIDE):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown(
        '<div class="footer">SIMPLY-FIT &nbsp;·&nbsp; M.Tech Project &nbsp;·&nbsp; '
        'Approximate estimates from weight trends &nbsp;·&nbsp; '
        'Not a substitute for professional medical advice</div>',
        unsafe_allow_html=True,
    )
