import numpy as np
import pytest

from src.agent_coach import agent_coach
from src.data_generator import generate_dataset
from src.ml_engine import run_pipeline


@pytest.fixture(scope="module")
def sample_context():
    df = generate_dataset(n_users=3, seed=11)
    weights = df[df.user_id == 0]["weight"].tolist()
    profile = {"age": 40, "gender": "Male", "target_weight": 78.0,
               "tdee": 2400, "disease": "Hypertension"}
    ml_res = run_pipeline(weights)
    return profile, weights, ml_res


INTENT_CASES = [
    ("How long until I reach my goal weight?", "time_to_goal"),
    ("How many calories should I eat every day?", "calorie_nutrition"),
    ("Why does my weight fluctuate so much day to day?", "water_weight"),
    ("I think I have hit a plateau, nothing moves", "plateau"),
    ("What should my doctor know about my hypertension diet?", "medical_guideline"),
    ("How am I doing overall?", "general_progress"),
]


@pytest.mark.parametrize("question,expected", INTENT_CASES)
def test_intent_routing(sample_context, question, expected):
    profile, weights, ml_res = sample_context
    info = agent_coach.analyze_query_intent(
        question, profile, 80.0, ml_res, target_weekly_change=-0.5
    )
    assert info["intent"] == expected


def test_run_agent_without_api_key(sample_context):
    """The deterministic path must produce a grounded answer with no key."""
    profile, weights, _ = sample_context
    res = agent_coach.run_agent(profile, weights,
                                 "How long until I reach my goal weight?",
                                 target_weekly_change=-0.5)
    assert res["intent"]["intent"] == "time_to_goal"
    assert res["response"].strip()
    assert "kg" in res["response"]
    # execution log: intent step + 3 tool observations
    assert len(res["trace"]) >= 4
    assert res["trace"][0]["action"] == "classify_intent"
    # no fabricated 'thought' keys in the trace
    assert all("thought" not in step for step in res["trace"])


def test_time_to_goal_math(sample_context):
    profile, weights, _ = sample_context
    res = agent_coach.run_agent(profile, weights,
                                "How many weeks to reach my goal?",
                                target_weekly_change=-0.5)
    info = res["intent"]
    assert info["days_needed"] == pytest.approx(info["weeks_needed"] * 7, abs=1)
    assert info["remaining_kg"] >= 0


# ───────────────────────── M.Tech additions ─────────────────────────
UNSUPPORTED_TERMS = ["leptin", "T3", "fat mass", "Fat Weight", "metabolic slowdown",
                     "Isolation Forest", "restore"]


@pytest.mark.parametrize("question,_", INTENT_CASES)
def test_templates_make_no_unsupported_physiological_claims(sample_context, question, _):
    profile, weights, _ml = sample_context
    text = agent_coach.run_agent(profile, weights, question, target_weekly_change=-0.5)["response"]
    for term in UNSUPPORTED_TERMS:
        assert term not in text, f"unsupported claim '{term}' in: {text}"


def test_balance_statement_carries_interval_and_caveat(sample_context):
    profile, weights, _ = sample_context
    res = agent_coach.run_agent(profile, weights, "How am I doing overall?", target_weekly_change=-0.5)
    text = res["response"]
    assert "approximate 95% range" in text
    assert "not a measurement" in text
    assert f"{res['ml_output']['kcal_per_day']:+.0f} kcal/day" in text     # same number as the backend


def test_short_history_gets_an_honest_answer_not_a_number():
    profile = {"target_weight": 75.0, "tdee": 2300, "disease": "none"}
    res = agent_coach.run_agent(profile, [80.0, 79.9, 80.2, 79.8], "How am I doing overall?", -0.5)
    assert res["ml_output"]["status"] == "insufficient"
    assert "can't estimate" in res["response"]
    assert "kcal/day**" not in res["response"]


def test_time_to_goal_uses_goal_weight_not_planned_change():
    """v1 passed the planned kg change (e.g. 5) as the goal weight, giving ~70 kg 'remaining'."""
    weights = list(80 - 0.07 * np.arange(30))
    profile = {"target_weight": 75.0, "tdee": 2300, "disease": "none"}
    info = agent_coach.run_agent(profile, weights, "How many weeks to reach my goal?", -0.5)["intent"]
    assert info["target_weight"] == 75.0
    assert 2.0 < info["remaining_kg"] < 4.0
    assert info["weeks_needed"] == pytest.approx(info["remaining_kg"] / 0.5, abs=0.11)


def test_no_single_protein_target_when_a_condition_is_listed():
    """v1 told every user to eat 1.8 g/kg protein, contradicting its own CKD guideline (0.6-0.8 g/kg)."""
    weights = list(80 - 0.07 * np.arange(30))
    profile = {"target_weight": 75.0, "tdee": 2300, "disease": "CKD"}
    text = agent_coach.run_agent(profile, weights, "How many calories and protein should I eat?", -0.5)["response"]
    assert "1.8" not in text and "g / day" not in text
    assert "clinician" in text


def test_plateau_requires_statistical_support():
    flat = [80.0 + 0.1 * ((i * 7) % 3 - 1) for i in range(42)]
    falling = list(80 - 0.08 * np.arange(42))
    assert agent_coach._tool_check_plateau_status(flat)["is_plateau"] is True
    assert agent_coach._tool_check_plateau_status(falling)["is_plateau"] is False
    assert agent_coach._tool_check_plateau_status([80.0, 80.1])["is_plateau"] is None


def test_coach_accepts_logs_with_missed_days(sample_context):
    profile, weights, _ = sample_context
    gappy = list(weights)
    for i in (3, 10, 11, 40):
        gappy[i] = None
    res = agent_coach.run_agent(profile, gappy, "How am I doing overall?", -0.5)
    assert res["ml_output"]["status"] == "ok"
