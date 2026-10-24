"""OpenAI layer: turns free text into model inputs, and explains results in plain English.

Design choice: the LLM never makes the prediction. A validated ML model does that.
The LLM is the "receptionist and translator". Everything works without an API key
(the app falls back to a form and template explanations).
"""

from __future__ import annotations

import json
import os

from .config import SCREENING_FEATURES
from .guardrails import make_safe, scrub_pii

FALLBACK_MODEL = "gpt-4o-mini"


def _model(model: str | None = None) -> str:
    """Model name, read at call time so Streamlit secrets/env changes are picked up."""
    return model or os.getenv("OPENAI_MODEL") or FALLBACK_MODEL


def get_client(api_key: str | None = None, base_url: str | None = None):
    """Return an OpenAI-compatible client, or None if no key / package is available.

    Works with any OpenAI-compatible provider by setting base_url, e.g.
      Groq:   https://api.groq.com/openai/v1
      Gemini: https://generativelanguage.googleapis.com/v1beta/openai/
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI(api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL") or None)


# ---------------------------------------------------------------------------
# 1. Extraction: conversation -> structured answers
# ---------------------------------------------------------------------------
def extraction_schema() -> dict:
    props = {}
    for name, spec in SCREENING_FEATURES.items():
        if name == "bmi":
            continue
        json_type = "boolean" if spec["kind"] == "binary" else "number"
        props[name] = {"type": [json_type, "null"], "description": spec["label"]}
    props["bmi"] = {"type": ["number", "null"],
                    "description": "BMI, only if the user states the number directly"}
    props["height_cm"] = {"type": ["number", "null"],
                          "description": "Height in centimetres (convert from feet/inches)"}
    props["weight_kg"] = {"type": ["number", "null"],
                          "description": "Weight in kilograms (convert from stones/pounds)"}
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


EXTRACT_SYSTEM = """You extract answers for a PCOS screening questionnaire from a chat.
Rules:
- Fill a field only if the user clearly stated it. Otherwise use null. Never guess or infer from vague hints.
- cycle_irregular: true if periods are irregular, often missed, or usually more than ~35 days apart;
  false if they are regular (roughly every 21-35 days).
- period_length: number of days the bleeding usually lasts.
- Yes/no fields: true for yes, false for no, null for "not sure" or no answer.
- Convert units: feet/inches -> cm, stones/pounds -> kg.
- If the user is replying to the assistant's last question, map the reply to that field.
"""


def extract_answers(client, conversation: list[dict], model: str | None = None) -> dict:
    """conversation: [{"role": "user"|"assistant", "content": str}, ...] -> non-null answers."""
    schema = extraction_schema()
    chat = [{"role": m["role"], "content": scrub_pii(m["content"])} for m in conversation[-12:]]
    try:
        # Best option: strict JSON schema (OpenAI, Gemini and some Groq models support it)
        response = client.chat.completions.create(
            model=_model(model),
            messages=[{"role": "system", "content": EXTRACT_SYSTEM}] + chat,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "pcos_answers", "schema": schema, "strict": True},
            },
        )
    except Exception:
        # Fallback for providers/models without strict schemas: plain JSON mode
        keys = ", ".join(schema["properties"])
        response = client.chat.completions.create(
            model=_model(model),
            messages=[{"role": "system", "content": EXTRACT_SYSTEM + (
                f"\nReply with ONLY a JSON object with exactly these keys: {keys}. "
                "Use true/false for yes-no fields, numbers for numeric fields, null if unknown.")}]
            + chat,
            temperature=0,
            response_format={"type": "json_object"},
        )
    content = response.choices[0].message.content or "{}"
    start, end = content.find("{"), content.rfind("}")
    data = json.loads(content[start:end + 1]) if start != -1 else {}
    answers = {}
    for key, value in data.items():
        if key not in schema["properties"] or value is None:
            continue
        if isinstance(value, str):  # tolerate "yes"/"no"/"12" from looser models
            low = value.strip().lower()
            if low in {"true", "yes"}:
                value = True
            elif low in {"false", "no"}:
                value = False
            else:
                try:
                    value = float(low)
                except ValueError:
                    continue
        answers[key] = int(value) if isinstance(value, bool) else value
    return answers


# ---------------------------------------------------------------------------
# 2. Explanation: model output -> plain English + GP questions
# ---------------------------------------------------------------------------
EXPLAIN_SYSTEM = """You are a warm, careful women's-health information assistant writing in UK English.
You receive the output of a PCOS screening model. It is NOT a diagnosis.
Hard rules:
- Never say the person has, or does not have, PCOS. Say their answers "share patterns" with, or
  "match fewer patterns" seen in, people diagnosed with PCOS.
- In 2-3 sentences, explain what the result means and name the 2-3 answers that influenced it most.
- If flag_for_gp is true, gently encourage booking a GP appointment. Otherwise say it is still worth
  seeing a GP if symptoms bother them, because a screening tool can miss things.
- Mention that GPs usually check for PCOS with blood tests (hormone levels) and sometimes an ultrasound.
- Then give 3-4 short questions they could ask their GP, as a bulleted list.
- Under 170 words. No scary language. No medication or diet prescriptions.
- Never present weight or BMI as the cause, and never blame the person. If BMI is a factor,
  mention it neutrally as one of several patterns.
"""

GP_QUESTIONS = [
    "Could my symptoms be linked to PCOS or another hormonal condition?",
    "Would blood tests to check my hormone levels be useful?",
    "Do I need a pelvic ultrasound?",
    "What can I do now to manage my symptoms while we find out more?",
]


def fallback_explanation(result: dict) -> str:
    raising = [f["label"].lower() for f in result["top_factors"] if f["effect"] > 0][:3]
    parts = [result["band_text"]]
    if raising:
        parts.append("The answers that influenced this most were: " + ", ".join(raising) + ".")
    if result["flag_for_gp"]:
        parts.append("It would be a good idea to book a GP appointment to talk this through.")
    else:
        parts.append("If your symptoms bother you, it's still worth seeing a GP: "
                     "a screening tool can miss things.")
    parts.append("GPs usually check for PCOS with blood tests (hormone levels) and sometimes an "
                 "ultrasound.")
    text = " ".join(parts)
    text += "\n\n**Questions you could ask your GP:**\n" + "\n".join(f"- {q}" for q in GP_QUESTIONS)
    return make_safe(text)


def explain(client, result: dict, model: str | None = None) -> str:
    if client is None:
        return fallback_explanation(result)
    payload = {
        "screening_score_0_to_100": round(result["risk"] * 100),
        "band": result["band"],
        "flag_for_gp": result["flag_for_gp"],
        "top_factors": [{"answer": f["label"], "pushes_score": "up" if f["effect"] > 0 else "down"}
                        for f in result["top_factors"]],
        "answers_not_given": result["assumed_typical"],
    }
    try:
        response = client.chat.completions.create(
            model=_model(model),
            messages=[{"role": "system", "content": EXPLAIN_SYSTEM},
                      {"role": "user", "content": json.dumps(payload)}],
            temperature=0.3,
        )
        return make_safe(response.choices[0].message.content)
    except Exception as exc:  # network/API problems must never break the demo
        return fallback_explanation(result) + f"\n\n(AI explanation unavailable: {type(exc).__name__})"
