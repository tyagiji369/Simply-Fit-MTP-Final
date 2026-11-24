"""Gemini handles free-form questions. Local templates provide clearly labelled offline support."""

import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.energy_balance import DEFAULT_WINDOW_DAYS
from src.ml_engine import KCAL_PER_KG, run_pipeline as run_ml_pipeline
from src.lstm_forecaster import forecast as forecast_lstm
from src.rag_pipeline import rag_engine
from src import grounded_coach as grounded

load_dotenv()


class SimplyFitAgent:
    def __init__(self):
        self.tools = {
            "infer_calorie_balance": self._tool_infer_calorie_balance,
            "forecast_trajectory": self._tool_forecast_trajectory,
            "retrieve_clinical_guidelines": self._tool_retrieve_clinical_guidelines,
            "check_plateau_status": self._tool_check_plateau_status
        }

    # ── Tool Definitions ──────────────────────────────────────
    def _tool_infer_calorie_balance(self, weight_log, target_weekly_change, window_days=DEFAULT_WINDOW_DAYS):
        return run_ml_pipeline(weight_log, target_weekly_change=target_weekly_change, window_days=window_days)

    def _tool_forecast_trajectory(self, weight_log):
        """Recent-level forecast (the app's default). None if too few recent readings."""
        pred = forecast_lstm(weight_log)
        if pred is None:
            return None
        return [round(float(x), 1) for x in pred]

    def _tool_retrieve_clinical_guidelines(self, query, disease):
        return rag_engine.search_guidelines(query, condition=disease, top_k=2)

    def _tool_check_plateau_status(self, weight_log, ml_res=None):
        """
        A plateau is reported only when the trend estimate itself supports it:
        the estimate is within the maintenance tolerance AND its interval is
        narrow enough to rule out a meaningful deficit or surplus
        (label == "maintenance"). Two noisy readings 14 days apart, which the
        internship version compared, cannot establish that.
        """
        ml_res = ml_res or run_ml_pipeline(weight_log)
        if ml_res["status"] != "ok":
            return {"is_plateau": None, "label": None,
                    "explanation": "Not enough readings to tell whether weight has plateaued."}
        return {
            "is_plateau": ml_res["label"] == "maintenance",
            "label": ml_res["label"],
            "weekly_kg_change": ml_res["weekly_kg_change"],
            "explanation": {
                "maintenance": "The trend over the analysis window is flat within its uncertainty.",
                "inconclusive": "The data cannot yet distinguish a flat trend from a slow change.",
            }.get(ml_res["label"], "The trend is still moving; no plateau is indicated."),
        }

    # ── Query Intent & Semantics Analyzer ──────────────────────
    # Keyword groups per intent. The intent with the highest keyword score
    # wins (ties broken by group order). Disease-specific terms count
    # double: a disease name is an unambiguous medical signal, while
    # generic words like "diet" or "eat" routinely appear in medical
    # questions too.
    INTENT_KEYWORDS = {
        "time_to_goal":     ["goal", "target weight", "planned pace", "reach my target"],
        "calorie_nutrition": ["calorie", "eat", "intake", "deficit", "macro",
                              "food", "diet", "tdee", "protein", "meal"],
        "water_weight":     ["fluctuat", "water weight", "water retention", "spike", "salt", "sodium",
                             "scale", "anomaly", "retention", "bounce",
                             "up and down", "vary"],
        "plateau":          ["plateau", "stuck", "slow", "stopped",
                             "not moving", "stagnant", "stagnate"],
        "medical_guideline": ["disease", "pressure", "sugar", "condition",
                              "doctor", "health"],
    }
    # Weight-2 terms for the medical intent.
    MEDICAL_DISEASE_TERMS = ("hypertension", "diabetes", "pcos", "ckd",
                             "hypothyroidism", "nafld", "kidney", "thyroid")

    def analyze_query_intent(self, question, profile, current_weight, ml_res, target_weekly_change):
        q_lower = question.lower()
        target_weight = profile.get("target_weight", current_weight - 5.0)

        water = re.search(r"\b(water|fluids?|hydrat(?:e|ed|ion))\b", q_lower)
        scale = re.search(r"\b(scale|weigh(?:t|ing)?|retention|retain(?:ing)?|fluctuat\w*|bloated)\b", q_lower)
        intake = re.search(r"\b(drink\w*|intake|cups?|glasses|lit(?:er|re)s?|hydration|hydrate|hydrated)\b", q_lower)
        if water and intake and (not scale or re.search(r"how much (?:water|fluid)|how many (?:cups|glasses|liters|litres)", q_lower)):
            conditions = f"{q_lower} {profile.get('disease', '')} {profile.get('diseases', '')}".lower()
            restricted = re.search(r"kidney|renal|dialysis|heart failure|fluid (?:limit|restrict)|ckd", conditions)
            return {"intent": "hydration", "fluid_restriction": bool(restricted)}
        if water and not scale and not intake:
            return {"intent": "clarify_water"}

        scores = {
            intent: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw) + (r"\w*" if kw in ("fluctuat", "calorie", "anomaly", "meal", "macro") else r"\b"), q_lower))
            for intent, kws in self.INTENT_KEYWORDS.items()
        }
        scores["medical_guideline"] += 2 * sum(
            1 for kw in self.MEDICAL_DISEASE_TERMS if re.search(r"\b" + re.escape(kw) + (r"\w*" if kw in ("fluctuat", "calorie", "anomaly", "meal", "macro") else r"\b"), q_lower)
        )
        best_intent = max(scores, key=lambda k: (scores[k], -list(scores).index(k)))
        if scores[best_intent] == 0:
            if re.search(r"\b(progress|trend|doing|overall|on track|summary|my weight)\b", q_lower):
                best_intent = "general_progress"
            else:
                return {"intent": "unsupported"}

        # 1. Time to goal / duration
        if best_intent == "time_to_goal":
            remaining_kg = abs(current_weight - target_weight)
            planned_rate = abs(target_weekly_change)
            if planned_rate < 1e-9:
                return {"intent": "time_to_goal", "remaining_kg": round(remaining_kg, 1),
                        "target_weight": target_weight, "weeks_needed": None, "days_needed": None}
            weeks_needed = remaining_kg / planned_rate
            return {
                "intent": "time_to_goal",
                "remaining_kg": round(remaining_kg, 1),
                "target_weight": target_weight,
                "weeks_needed": round(weeks_needed, 1),
                "days_needed": int(round(weeks_needed * 7)),
            }

        # 2. Calorie / Food / Nutrition / Protein
        if best_intent == "calorie_nutrition":
            tdee = profile.get("tdee", 2200)
            target_intake = tdee + (target_weekly_change * KCAL_PER_KG) / 7.0
            return {
                "intent": "calorie_nutrition",
                "tdee": int(tdee),
                "balance_kcal": ml_res["kcal_per_day"],
                "balance_ci": ml_res["kcal_per_day_ci95"],
                "target_intake": int(round(target_intake)),
            }

        # 3. Fluctuation / Scale noise / Water weight / Salt
        if best_intent == "water_weight":
            return {
                "intent": "water_weight",
                "anomalies": ml_res["anomalies_detected"]
            }

        # 4. Plateau / Slowdown
        if best_intent == "plateau":
            return {
                "intent": "plateau",
                "anomalies": ml_res["anomalies_detected"]
            }

        # 5. Disease / Medical guideline
        if best_intent == "medical_guideline":
            return {
                "intent": "medical_guideline",
                "disease": profile.get("disease", "none")
            }

        # 6. General progress
        return {"intent": "general_progress"}

    # ── Tool execution pipeline ───────────────────────────────
    def run_agent(self, profile, weight_log, question, target_weekly_change=-0.5, gemini_api_key=None, window_days=DEFAULT_WINDOW_DAYS):
        conditions = list(dict.fromkeys(c for c in [profile.get("disease"), *(profile.get("diseases") or [])]
                                        if c and str(c).lower() != "none"))
        ml = self._tool_infer_calorie_balance(weight_log, target_weekly_change, window_days)
        forecast = self._tool_forecast_trajectory(weight_log)
        result = dict(ml_output=ml, forecast=forecast, sources=[], trace=[])
        # An explicitly empty key disables the online coach, including when an environment key exists.
        key = (gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        notice = "No Gemini API key is configured. Showing general recommendations from the offline coach."
        if grounded.urgent_question(question):
            return result | dict(mode="urgent", notice="Urgent care guidance", response=grounded.URGENT,
                                 intent={"intent":"urgent"}, trace=[{"action":"urgent_guidance"}])
        if key:
            metrics = {k:ml[k] for k in ("status", "message", "label", "reliability", "current_trend_weight",
                                          "kcal_per_day", "kcal_per_day_ci95", "weekly_kg_change")}
            metrics.update(simulated_demo=bool(profile.get("is_demo")), goal_weight_kg=profile.get("target_weight"), planned_kg_per_week=target_weekly_change,
                           forecast_kg=forecast, window_days=window_days,
                           definition="Energy equivalent of dated Huber weight trend; 7700 kcal/kg. "
                           "Simulation-calibrated uncertainty. Not measured calorie intake or expenditure.")
            try:
                answer, ids = grounded.online(key, question, conditions, metrics)
            except Exception as exc:
                notice = ("Gemini could not complete this request. " + grounded.failure_notice(exc)
                          + " Showing general recommendations from the offline coach.")
            else:
                if grounded.restricted_request(question) and answer != grounded.URGENT:
                    answer, ids = grounded.LIMIT, []
                return result | dict(mode="gemini", notice="Gemini coach", response=answer,
                                     intent={"intent":"gemini"}, sources=[grounded.evidence()[i] for i in ids],
                                     trace=[{"action":"gemini_first"}])

        current = ml["current_trend_weight"]
        info = self.analyze_query_intent(question, profile, current or 0, ml, target_weekly_change)
        intent = info["intent"]
        plateau = self._tool_check_plateau_status(weight_log, ml)
        ids = []
        if grounded.restricted_request(question) or re.search(r"\b(dose|dosage|insulin units|medication|medications|prescription|tablets?)\b", question, re.I):
            answer = grounded.LIMIT
        elif intent == "hydration":
            if conditions or info["fluid_restriction"]:
                answer = "Ask your clinician whether your condition or treatment changes your fluid target. Follow any prescribed fluid limit."
                if re.search("kidney|ckd|dialysis", " ".join(conditions + [question]), re.I):
                    ids = ["kidney"]
            else:
                ids = ["hydration"]
                answer = grounded.evidence()["hydration"]["note"]
        elif intent in ("unsupported", "clarify_water"):
            answer = self._topic_reply(info)
        elif (intent == "medical_guideline"
              or ((conditions or re.search(r"diabet|dialysis|kidney|hypertens|pregnan|heart failure|thyroid", question, re.I))
                  and intent == "calorie_nutrition")):
            answer = grounded.LIMIT
        elif current is None:
            answer = "Enter a weight log first so I can answer questions about your trend."
        else:
            answer = self._smart_intent_synthesis(info, question, round(current,1), ml,
                                                  target_weekly_change, profile, plateau, [])
        return result | dict(mode="offline", notice=notice, response=answer, intent=info,
                             sources=[grounded.evidence()[i] for i in ids],
                             trace=[{"action":"classify_intent","intent":intent},
                                    {"action":"infer_calorie_balance"}, {"action":"forecast_trajectory"},
                                    {"action":"check_plateau_status"}])

    @staticmethod
    def _topic_reply(info):
        if info["intent"] == "clarify_water":
            return "Are you asking how much water to drink, or why water affects your scale weight?"
        if info["intent"] == "unsupported":
            return ("I don’t have a specific answer for that question. I can help with daily hydration, "
                    "weight trends, scale fluctuations, calorie estimates, plateaus and time to your goal.")
        raise ValueError("Unexpected clarification intent")

    @staticmethod
    def _balance_sentence(ml_res):
        """One honest sentence about the estimate, shared by every template."""
        if ml_res["status"] != "ok":
            return f"I can't estimate your energy balance yet. {ml_res['message']}"
        lo, hi = ml_res["kcal_per_day_ci95"]
        words = {
            "deficit": "which points to a calorie deficit",
            "surplus": "which points to a calorie surplus",
            "maintenance": "which is consistent with roughly maintaining",
            "inconclusive": "but the range is too wide to say whether you are in a deficit, "
                            "a surplus or at maintenance",
        }[ml_res["label"]]
        return (f"Your weight trend corresponds to roughly **{ml_res['kcal_per_day']:+.0f} kcal/day** "
                f"(approximate 95% range {lo:+.0f} to {hi:+.0f}), {words}. "
                f"Data reliability: **{ml_res['reliability']}**. This is an approximation from the "
                f"weight trend using 7,700 kcal per kg; it is not a measurement of what you ate or burned.")

    def _smart_intent_synthesis(self, intent_info, question, current_weight, ml_res, target_weekly_change, profile, plateau_res, rag_res):
        intent = intent_info["intent"]
        balance = self._balance_sentence(ml_res)

        # Intent 1: Time to Goal
        if intent == "time_to_goal":
            rem_kg = intent_info["remaining_kg"]
            target_w = intent_info["target_weight"]
            if intent_info["weeks_needed"] is None:
                return (f"Your goal weight is **{target_w} kg** and your current trend weight is "
                        f"**{current_weight} kg** ({rem_kg} kg apart), but no planned rate is set, "
                        f"so I can't work out a timeline.")
            weeks, days = intent_info["weeks_needed"], intent_info["days_needed"]
            target_date = (datetime.now() + timedelta(days=days)).strftime("%d %b %Y")
            msg = (
                f"**Timeline at your planned pace**\n\n"
                f"From your current trend weight of **{current_weight} kg** to your goal of "
                f"**{target_w} kg** is {rem_kg} kg. At the planned **{abs(target_weekly_change):.2f} kg/week** "
                f"that is about **{weeks} weeks** (~{days} days, around {target_date}). "
                f"This assumes the planned pace is held the whole way, which is rarely exact.\n\n"
            )
            if ml_res["status"] == "ok" and ml_res["label"] in ("deficit", "surplus"):
                msg += f"Your measured trend is currently {ml_res['weekly_kg_change']:+.2f} kg/week."
            else:
                msg += "Your recent readings do not yet show a clear trend to compare with the plan."
            return msg

        # Intent 2: Calorie & Nutrition
        if intent == "calorie_nutrition":
            msg = (
                f"**Calories**\n\n"
                f"• Formula-based maintenance estimate (TDEE): about {intent_info['tdee']} kcal/day. "
                f"This comes from a population equation and can be off by a few hundred kcal for an individual.\n"
                f"• Intake that matches your planned pace ({target_weekly_change:+.2f} kg/week): "
                f"about **{intent_info['target_intake']} kcal/day**, on the same assumptions.\n"
                f"• {balance}\n"
            )
            if str(profile.get("disease", "none")).lower() not in ("none", ""):
                msg += ("\nBecause you have listed a medical condition, please set protein and other "
                        "nutrient targets with your clinician or dietitian rather than from this app.")
            return msg

        # Intent 3: Fluctuation / scale noise
        if intent == "water_weight":
            n_unusual = intent_info["anomalies"]
            return (
                f"**Why daily scale weight moves around**\n\n"
                f"Day-to-day readings commonly differ by several hundred grams or more because of "
                f"body water, food and drink still in the gut, and when you weigh yourself. Those "
                f"swings are much larger than one day's change in body tissue, so a single reading "
                f"says little.\n\n"
                f"• Your current trend weight is **{current_weight} kg**.\n"
                f"• {n_unusual} reading(s) in the analysis window sit unusually far from the trend. "
                f"They are kept and given less weight; nothing is deleted or replaced.\n"
                f"• {balance}"
            )

        # Intent 4: Plateau
        if intent == "plateau":
            if plateau_res.get("is_plateau") is None:
                return f"**Plateau check**\n\n{plateau_res['explanation']}"
            if plateau_res["is_plateau"]:
                return (
                    f"**Plateau check**\n\n"
                    f"{plateau_res['explanation']} {balance}\n\n"
                    f"A flat trend means intake and expenditure are currently about equal. Weight "
                    f"alone cannot tell which of the two changed. If this is not what you intended, "
                    f"review your plan; for anything medical, ask a clinician."
                )
            return f"**Plateau check**\n\n{plateau_res['explanation']} {balance}"

        # Intent 5: Guideline notes
        if intent == "medical_guideline":
            disease = profile.get("disease", "none")
            msg = f"**Guideline note for {disease}**\n\n"
            if rag_res and disease != "none":
                top = rag_res[0]
                msg += (f"• {top['recommendation']}\n• Attributed source: {top['source']}\n\n"
                        f"This is a short reference note from the app's small built-in list, not "
                        f"medical advice. Please confirm it with your clinician.")
            else:
                msg += "No medical condition is recorded in your profile, so there is no specific note to show."
            return msg

        # Intent 6: General progress
        msg = f"**Your weight trend**\n\n• Current trend weight: {current_weight} kg\n• {balance}\n"
        if ml_res["status"] == "ok" and ml_res["label"] in ("deficit", "surplus", "maintenance"):
            gap = ml_res["weekly_kg_change"] - target_weekly_change
            msg += (f"• Measured trend {ml_res['weekly_kg_change']:+.2f} kg/week against a plan of "
                    f"{target_weekly_change:+.2f} kg/week (difference {gap:+.2f} kg/week).")
        else:
            msg += "• There is not yet a clear enough trend to compare with your plan."
        return msg


# Global Agent instance
agent_coach = SimplyFitAgent()
