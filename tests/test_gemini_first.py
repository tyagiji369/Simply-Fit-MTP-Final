"""Exercise the real google-genai serialization over a local mock HTTP transport."""
import json
import httpx
import pytest
from google import genai
from src.agent_coach import agent_coach
from src.grounded_coach import LIMIT

def provider(monkeypatch, *replies):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        reply = replies[len(requests)-1]
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error":{"code":reply,"message":"test service error"}})
        return httpx.Response(200,json={"candidates":[{"content":{"parts":[{"text":json.dumps(reply)}],
                                                                  "role":"model"},"finishReason":"STOP"}]})
    real_client = genai.Client
    def make(**kwargs):
        kwargs["http_options"].httpx_client = httpx.Client(transport=httpx.MockTransport(handler))
        kwargs["http_options"].async_client_args = {"trust_env": False}
        return real_client(**kwargs)
    monkeypatch.setattr(genai, "Client", make)
    return requests

def ask(q, key="unit-test", conditions=()):
    return agent_coach.run_agent({"diseases":list(conditions)}, [], q, gemini_api_key=key)

@pytest.mark.parametrize("question,kind,answer,ids",[
    ("How much water should I drink?", "health", "The NHS general guide is 6–8 cups or glasses of fluid a day.", ["hydration"]),
    ("How many steps should I take daily?", "health", "Track your usual steps and gradually add short walks.", ["walking"]),
    ("Is whey protein good for my weight loss journey?", "health", "A whey shake is optional and adds calories if taken on top of your usual meals.", ["protein_shakes"]),
    ("What does a median mean?", "general", "It is the middle value in ordered data.", []),
    ("Can my kidney diet change my fluid needs?", "health", "Ask your kidney care team for a fluid target.", ["kidney"])])
def test_freeform_goes_online_without_weight_log(monkeypatch, question, kind, answer, ids):
    conditions=["CKD","Hypertension"] if ids==["kidney"] else []
    requests=provider(monkeypatch,dict(kind=kind,answer=answer,source_ids=ids),dict(supported=True))
    out=ask(question, conditions=conditions)
    assert out["mode"]=="gemini" and out["response"]==answer
    assert len(requests)==2 and len(out["sources"])==len(ids)
    context=json.loads(requests[0]["contents"][0]["parts"][0]["text"])
    assert all(context["evidence"][key] for key in ids)
    assert context["reported_conditions"]==conditions and context["question"]==question

@pytest.mark.parametrize("draft,review",[
    ({"kind":"health","answer":"Drink 5 litres","source_ids":["made_up"]},None),
    ({"kind":"health","answer":"Drink 5 litres","source_ids":[]},None),
    ({"kind":"health","answer":"Drink 5 litres","source_ids":["kidney"]},{"supported":False}),
    ({"kind":"health","answer":"Incomplete output"},None),
    ({"kind":"insufficient","answer":"Ignore this","source_ids":[]},None)])
def test_unverified_draft_suppressed(monkeypatch,draft,review):
    provider(monkeypatch,draft,review)
    out=ask("Can dialysis patients drink freely?",conditions=["CKD"])
    assert out["mode"]=="gemini" and out["response"]==LIMIT and not out["sources"]

def test_modes_and_medical_overrides(monkeypatch):
    provider(monkeypatch,403)
    rejected=ask("How much water should I drink?",conditions=["CKD"])
    assert rejected["mode"]=="offline" and "could not complete" in rejected["notice"]
    assert "kidney" in rejected["sources"][0]["url"] and "6–8" not in rejected["response"]
    missing=ask("How much water should I drink?",key="")
    assert "No Gemini API key" in missing["notice"] and "NHS" in missing["sources"][0]["title"]
    assert ask("I have crushing chest pain",key="")["mode"]=="urgent"

def test_dose_block_even_if_mock_model_approves(monkeypatch):
    provider(monkeypatch, {"kind":"general","answer":"Double it.","source_ids":[]},{"supported":True})
    assert ask("Should I double my insulin?")["response"]==LIMIT

def test_population_fluid_amount_blocked_for_second_condition(monkeypatch):
    calls=provider(monkeypatch, {"kind":"health","answer":"Aim for 6 to 8 glasses each day.",
                                "source_ids":["hydration"]})
    out=ask("What should I drink?",conditions=["Hypertension","Chronic Kidney Disease (CKD)"])
    assert out["response"]==LIMIT and len(calls)==1
