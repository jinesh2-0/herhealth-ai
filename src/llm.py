"""OpenAI layer: turns free text into model inputs, and explains results in plain English.

Design choice: the LLM never makes the prediction. A validated ML model does that.
The LLM is the "receptionist and translator". Everything works without an API key
(the app falls back to a form and template explanations).
"""

from __future__ import annotations

import json
import os

from .config import EXTRA_KEYS, SCREENING_FEATURES
from .guardrails import make_safe, scrub_pii

FALLBACK_MODEL = "gpt-4o-mini"


def _model(model: str | None = None) -> str:
    """Model name, read at call time so Streamlit secrets/env changes are picked up."""
    return model or os.getenv("OPENAI_MODEL") or FALLBACK_MODEL


def _chat(client, **kwargs):
    """chat.completions.create with low reasoning effort for reasoning models (faster replies).
    Falls back to a plain call if the provider doesn't accept the parameter."""
    effort = os.getenv("OPENAI_REASONING_EFFORT") or (
        "low" if "gpt-oss" in kwargs.get("model", "") else None)
    if effort:
        try:
            return client.chat.completions.create(reasoning_effort=effort, **kwargs)
        except Exception:
            pass
    return client.chat.completions.create(**kwargs)


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
    for key, desc in EXTRA_KEYS.items():          # extra NHS/NICE check questions
        kind = "number" if key in ("impact", "period_pain") else "boolean"
        props[key] = {"type": [kind, "null"], "description": desc}
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


EXTRACT_SYSTEM = """You extract answers for a PMOS (formerly PCOS) screening questionnaire from a chat.
Rules:
- Fill a field only if the user clearly stated it. Otherwise use null. Never guess or infer from vague hints.
- cycle_irregular: true if periods are irregular, often missed, or usually more than ~35 days apart;
  false if they are regular (roughly every 21-35 days).
- period_length: number of days the bleeding usually lasts.
- Yes/no fields: true for yes, false for no, null for "not sure" or no answer.
- Convert units: feet/inches -> cm, stones/pounds -> kg.
- If the user is replying to the assistant's last question, map the reply to that field.
- impact and period_pain are 0-10 ratings. For the bleeding and pain checklist, set the items the
  user says they have to true; if they say "none", set every checklist item to false.
- postmeno_bleeding: false if the user says it doesn't apply (their periods haven't stopped).
"""


def extract_answers(client, conversation: list[dict], model: str | None = None) -> dict:
    """conversation: [{"role": "user"|"assistant", "content": str}, ...] -> non-null answers."""
    schema = extraction_schema()
    chat = [{"role": m["role"], "content": scrub_pii(m["content"])} for m in conversation[-12:]]
    try:
        # Best option: strict JSON schema (OpenAI, Gemini and some Groq models support it)
        response = _chat(
            client,
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
        response = _chat(
            client,
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
You receive the output of a screening model for PMOS (polyendocrine metabolic ovarian syndrome),
which the NHS now uses as the name for what used to be called PCOS. It is NOT a diagnosis.
On first mention write "PMOS (formerly PCOS)", then "PMOS".
Hard rules:
- Never say the person has, or does not have, PMOS. Say their answers "share patterns" with, or
  "match fewer patterns" seen in, people diagnosed with PMOS.
- In 2-3 sentences, explain what the result means and name the 2-3 answers that influenced it most.
- If flag_for_gp is true, gently encourage booking a GP appointment. Otherwise say it is still worth
  seeing a GP if symptoms bother them, because a screening tool can miss things.
- Mention that GPs usually check for PMOS with blood tests (hormone levels and insulin resistance) and sometimes an ultrasound.
- Then give 3-4 short questions they could ask their GP, as a bulleted list.
- Under 170 words. No scary language. No medication or diet prescriptions.
- Never present weight or BMI as the cause, and never blame the person. If BMI is a factor,
  mention it neutrally as one of several patterns.
"""

GP_QUESTIONS = [
    "Could my symptoms be linked to PMOS (formerly PCOS) or another hormonal condition?",
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
    parts.append("GPs usually check for PMOS with blood tests (hormone levels) and sometimes an "
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
        response = _chat(
            client,
            model=_model(model),
            messages=[{"role": "system", "content": EXPLAIN_SYSTEM},
                      {"role": "user", "content": json.dumps(payload)}],
            temperature=0.3,
        )
        return make_safe(response.choices[0].message.content)
    except Exception as exc:  # network/API problems must never break the demo
        return fallback_explanation(result) + f"\n\n(AI explanation unavailable: {type(exc).__name__})"


# ---------------------------------------------------------------------------
# 3. Optional: personalised questions for the GP summary
#    Privacy: only the NAMES of the warnings are sent (no dates, numbers or free text).
# ---------------------------------------------------------------------------
SUGGEST_SYSTEM = """You help someone in the UK prepare for a GP appointment about women's health.
You are given only the names of warnings an app showed them. Write up to 2 short, specific
questions the PERSON should answer about their own symptoms before the appointment, so the GP
gets useful detail (timing, triggers, what has been tried, impact). Each under 20 words, plain UK
English. No diagnosis, no medical advice, no questions already in the list provided.
Reply with only JSON: {"questions": ["...", "..."]}"""


def suggest_questions(client, warning_titles: list[str], existing: list[str],
                      model: str | None = None) -> list[str]:
    if client is None or not warning_titles:
        return []
    user = ("Warnings shown: " + "; ".join(warning_titles)
            + "\nQuestions already asked: " + "; ".join(existing))
    response = _chat(client, model=_model(model), temperature=0.3,
                     messages=[{"role": "system", "content": SUGGEST_SYSTEM},
                               {"role": "user", "content": user}],
                     response_format={"type": "json_object"})
    content = response.choices[0].message.content or "{}"
    start, end = content.find("{"), content.rfind("}")
    try:
        items = json.loads(content[start:end + 1]).get("questions", []) if start != -1 else []
    except ValueError:
        items = []
    out = []
    for q in items:
        q = make_safe(str(q)).split("\n")[0].strip() if isinstance(q, str) else ""
        q = q.replace("Not a diagnosis", "").strip()
        if 8 < len(q) < 160 and q not in existing and q not in out:
            out.append(q)
    return out[:2]
