"""Safety layer: red-flag detection, privacy scrubbing and safe wording of outputs."""

from __future__ import annotations

import re

DISCLAIMER = (
    "This tool gives a screening estimate based on patterns in a public research dataset. "
    "It is not a diagnosis and cannot rule PCOS in or out. Please speak to a GP or "
    "healthcare professional about any symptoms that worry you."
)

# (pattern, message). Checked on everything the user types, before anything else happens.
RED_FLAGS = [
    (r"\b(severe|unbearable|extreme|worst)\b.{0,30}\b(pain|cramp)",
     "Severe pain needs prompt medical attention. Please contact NHS 111 or your GP today, "
     "or call 999 / go to A&E if it is sudden or very bad."),
    (r"\b(soak(ing|ed)?|flood(ing)?)\b|bleeding (a lot|heavily)|heavy bleeding",
     "Very heavy bleeding (for example soaking a pad or tampon every hour) needs medical advice "
     "today. Contact NHS 111 or your GP, or 999 if you feel faint or unwell."),
    (r"\b(faint(ed|ing)?|pass(ed)? out|collapsed?)\b",
     "Fainting or collapsing needs urgent assessment. Call 999 or go to A&E if it happens again "
     "or you feel very unwell."),
    (r"\bpregnan\w*\b.{0,40}\b(bleed\w*|pain)\b|\b(bleed\w*|pain)\b.{0,40}\bpregnan\w*",
     "Bleeding or pain during pregnancy should be checked straight away. Contact your midwife, "
     "maternity unit or NHS 111 now."),
    (r"\b(after|post)[- ]?menopaus\w*\b.{0,30}\bbleed\w*|\bbleed\w*.{0,30}\b(after|post)[- ]?menopaus",
     "Any bleeding after the menopause should be checked by a GP soon, even if it is light."),
    (r"\bchest pain\b",
     "Chest pain can be serious. Call 999 if it is severe, spreading, or with breathlessness."),
    (r"\b(suicid\w*|kill myself|end my life|self[- ]?harm)\b",
     "It sounds like you're going through a lot. Please reach out for support now: call 999 if "
     "you are in immediate danger, or contact NHS 111 (option 2 for mental health) or Samaritans "
     "on 116 123 (free, 24/7)."),
]

# Phrases the LLM must never use: it never diagnoses.
_DIAGNOSTIC_PHRASES = [
    (re.compile(r"\byou (definitely |probably |likely )?have pcos\b", re.I),
     "your answers share some patterns linked with PCOS"),
    (re.compile(r"\byou (are|have been) diagnosed\b", re.I), "you may want to ask about"),
    (re.compile(r"\byou (do not|don't) have pcos\b", re.I),
     "your answers match fewer PCOS-linked patterns, but this cannot rule it out"),
]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(\+?\d[\d\s-]{8,}\d)")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
_NHS_NUMBER = re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b")


def red_flags(text: str) -> list[str]:
    """Return urgent-care messages triggered by the user's text (empty list if none)."""
    found = []
    for pattern, message in RED_FLAGS:
        if re.search(pattern, text or "", re.I) and message not in found:
            found.append(message)
    return found


def scrub_pii(text: str) -> str:
    """Remove obvious identifiers before anything is sent to an external API."""
    text = _EMAIL.sub("[email removed]", text or "")
    text = _NHS_NUMBER.sub("[number removed]", text)
    # Only treat it as a phone number if it has 10+ digits (so "21 165 60" survives)
    text = _PHONE.sub(
        lambda m: "[number removed]" if sum(ch.isdigit() for ch in m.group()) >= 10 else m.group(),
        text,
    )
    text = _POSTCODE.sub("[postcode removed]", text)
    return text


def make_safe(text: str) -> str:
    """Rewrite any diagnostic phrasing and make sure the disclaimer is present."""
    for pattern, replacement in _DIAGNOSTIC_PHRASES:
        text = pattern.sub(replacement, text or "")
    if "not a diagnosis" not in text.lower():
        text = text.rstrip() + "\n\n_" + DISCLAIMER + "_"
    return text
