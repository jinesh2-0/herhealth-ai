"""HerHealth: a women's health tracker and PMOS (formerly PCOS) screening assistant.

Concept prototype for a women's health feature in the NHS App. Not affiliated with the NHS.
Run locally:   streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
from datetime import date, timedelta
from statistics import median
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import apple_health, gp_pdf, llm, tracker
from src.config import EXTRA_FEATURES, SCREENING_FEATURES
from src.guardrails import DISCLAIMER, red_flags
from src.predict import Screener, compute_bmi

load_dotenv()
st.set_page_config(page_title="HerHealth · women's health tracker (concept)", page_icon="🩺",
                   layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "pcos_screening.joblib"
BAND_TITLES = {
    "higher": "Worth talking to a GP",
    "moderate": "Some patterns found",
    "lower": "Fewer patterns found",
}

# NHS-style look (NHS Design System colours; no NHS logo or name used as branding)
st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown, .stText { font-family: Arial, Helvetica, sans-serif; }
.hh-header { background:#005EB8; color:#fff; padding:18px 22px; border-radius:4px; margin-bottom:8px; }
.hh-header h1 { color:#fff !important; font-size:1.9rem; margin:0; padding:0; font-weight:700; }
.hh-header p { color:#fff; margin:6px 0 0; font-size:1.02rem; }
.hh-concept { background:#fff; border:1px solid #d8dde0; padding:8px 12px; font-size:0.85rem;
              color:#4c6272; border-radius:4px; margin-bottom:12px; }
.hh-card { background:#fff; border:1px solid #d8dde0; border-radius:4px; margin:10px 0; }
.hh-card .hd { padding:10px 16px; font-weight:700; color:#fff; font-size:1.02rem; }
.hh-card .bd { padding:12px 16px; color:#212b32; line-height:1.5; }
.hh-card.gp .hd { background:#005EB8; }
.hh-card.urgent .hd { background:#d5281b; }
.hh-card.info .hd { background:#4c6272; }
.hh-card a { color:#005EB8; font-weight:600; }
.hh-card details { margin-top:8px; border-top:1px solid #d8dde0; padding-top:6px; }
.hh-card summary { cursor:pointer; color:#005EB8; font-weight:600; }
.hh-card .ev { color:#4c6272; font-size:0.93rem; margin-top:6px; }
.hh-card .tests { background:#f0f4f5; border-left:4px solid #005EB8; padding:6px 10px;
                  margin:8px 0 4px; font-size:0.95rem; }
.hh-card .tests ul { margin:4px 0 0 0; padding-left:18px; }
.hh-acc { background:#fff; border-left:6px solid #005EB8; padding:12px 16px; margin:8px 0;
          border-radius:4px; color:#212b32; line-height:1.5; }
.hh-grid { display:grid; grid-template-columns:repeat(20, 11px); gap:3px; margin:6px 0 4px; }
.hh-grid span { width:11px; height:11px; border-radius:2px; background:#d8dde0; display:block; }
.hh-grid span.on { background:#005EB8; }
.hh-grid span.fa { background:#d5281b; }
[data-testid="stFormSubmitButton"] button, .stButton button[kind="primary"],
.stDownloadButton button[kind="primary"] { background:#007f3b !important; border-color:#007f3b !important;
  color:#fff !important; font-weight:600; }
</style>
""", unsafe_allow_html=True)


def care_card(level: str, title: str, body: str, link: str | None = None,
              source: str | None = None, evidence: str | None = None,
              tests: list[str] | None = None) -> None:
    """NHS-style care card: blue = see a GP, red = urgent, grey = good to know.
    `evidence` is shown in a collapsible 'How reliable is this sign?' section."""
    label = {"urgent": "Urgent advice", "gp": "See a GP", "info": "Good to know"}[level]
    more = (f'<br><a href="{html.escape(link)}" target="_blank">Read the NHS guidance</a>'
            if link else "")
    ev = ""
    if evidence:
        ev = (f'<details><summary>How reliable is this sign?</summary>'
              f'<div class="ev">{html.escape(evidence)}'
              + (f'<br><b>Source:</b> {html.escape(source)}' if source else "")
              + '</div></details>')
    tst = ""
    if tests:
        tst = ('<div class="tests"><b>Tests to ask your GP about:</b><ul>'
               + "".join(f"<li>{html.escape(t)}</li>" for t in tests) + "</ul></div>")
    st.markdown(f'<div class="hh-card {level}"><div class="hd">{label}: {html.escape(title)}</div>'
                f'<div class="bd">{html.escape(body)}{tst}{more}{ev}</div></div>',
                unsafe_allow_html=True)


def show_cards(alert_list: list[dict]) -> None:
    order = {"urgent": 0, "gp": 1, "info": 2}
    for a in sorted(alert_list, key=lambda a: order[a["level"]]):
        care_card(a["level"], a["title"], a["detail"], a.get("link"), a.get("source"),
                  a.get("evidence"), a.get("tests"))
    if any(a["level"] in ("gp", "urgent") for a in alert_list):
        st.link_button("How to book a GP appointment (NHS)", tracker.NHS_LINKS["gp_booking"])
        st.caption("Blood tests on the NHS are arranged by your GP, so take these suggestions to "
                   "your appointment. In the NHS App, you'd book straight from here.")


def symptom_sliders(prefix: str, skip_groups: tuple = (), bloating_label: str | None = None):
    """One None/Mild/Moderate/Severe slider per symptom, in two columns.
    Returns (flags, severity, bloating_persistent)."""
    st.caption(tracker.SEVERITY_HELP)
    flags, severity, bloat = {}, {}, False
    groups = [g for g in tracker.SYMPTOM_GROUPS if g not in skip_groups]
    cols = st.columns(2)
    for i, group in enumerate(groups):
        with cols[i % 2]:
            st.markdown(f"**{group}**")
            for k, label in tracker.SYMPTOM_GROUPS[group].items():
                if k == "bloating" and bloating_label:
                    bloat = st.checkbox(bloating_label, key=f"{prefix}_bloat")
                    continue
                level = st.select_slider(label, options=tracker.SEVERITY_OPTIONS, value="None",
                                         key=f"{prefix}_{k}")
                if level != "None":
                    flags[k] = True
                    severity[k] = tracker.SEVERITY_OPTIONS.index(level)
    return flags, severity, bloat


def impact_slider(key: str, label: str = "How much did your symptoms affect your day?"):
    """0-10 'effect on daily life' slider; 'Skip' means not rated (we never assume 0)."""
    value = st.select_slider(label, options=["Skip"] + list(range(11)), value="Skip", key=key)
    st.caption(tracker.IMPACT_HELP)
    return None if value == "Skip" else int(value)


def pain_slider(key: str, label: str = "Worst period pain (0 = none, 10 = worst imaginable)"):
    value = st.select_slider(label, options=["Skip"] + list(range(11)), value="Skip", key=key)
    return None if value == "Skip" else int(value)


def impact_note(impact: int | None, result: dict | None) -> list[str]:
    """Make sure a low score never hides symptoms that are disrupting someone's life."""
    if impact is not None and impact >= tracker.IMPACT_GP and result and result["band"] != "higher":
        return [f"Your PMOS score isn't high, but you rated the effect on your life as "
                f"{impact}/10 ({tracker.impact_label(impact)}). That is a reason to see a GP on "
                "its own, whatever the score says."]
    return []


def accuracy_numbers(bundle: dict) -> dict | None:
    m = bundle.get("cv_metrics_at_threshold", {})
    tp, fn, tn, fp = (m.get(k, 0) for k in ("tp", "fn", "tn", "fp"))
    if not (tp + fn and tn + fp):
        return None
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prev = REAL_WORLD_PREVALENCE
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    return {"n": bundle.get("meta", {}).get("n", tp + fn + tn + fp), "caught": round(sens * 100),
            "false_alarm": round((1 - spec) * 100), "ppv_in_10": round(ppv * 10)}


def notify(message: str) -> None:
    """Pop-up confirmation that's visible wherever the user has scrolled to."""
    long_msg = len(message) > 120
    try:
        st.toast(message, icon="✅", duration="long" if long_msg else "short")
    except TypeError:  # older Streamlit without `duration`
        st.toast(message, icon="✅")
    if long_msg:       # import summaries also stay on screen to read
        st.success(message)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score interval)."""
    if not n:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def dot_grid(hits: int, cls: str) -> str:
    """100 squares, `hits` of them coloured: a picture of '86 out of 100'."""
    return '<div class="hh-grid">' + "".join(
        f'<span class="{cls if i < hits else ""}"></span>' for i in range(100)) + "</div>"


REAL_WORLD_PREVALENCE = 0.13   # WHO: PMOS affects up to 13% of women of reproductive age


def accuracy_panel(bundle: dict) -> None:
    """Honest, plain-English accuracy of the PMOS pattern check."""
    m = bundle.get("cv_metrics_at_threshold", {})
    meta = bundle.get("meta", {})
    tp, fn, tn, fp = (m.get(k, 0) for k in ("tp", "fn", "tn", "fp"))
    if not (tp + fn and tn + fp):
        return
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    s_lo, s_hi = wilson(tp, tp + fn)
    f_lo, f_hi = wilson(fp, tn + fp)
    prev_data = (tp + fn) / (tp + fn + tn + fp)
    prev = REAL_WORLD_PREVALENCE
    ppv_real = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    ppv_data = sens * prev_data / (sens * prev_data + (1 - spec) * (1 - prev_data))
    auc = bundle.get("cv_auc_by_model", {}).get(bundle.get("model_name"))
    caught, false_alarm = round(sens * 100), round((1 - spec) * 100)
    st.markdown(
        f'<div class="hh-acc"><b>Out of 100 people with PMOS</b>, it flags about <b>{caught}</b> '
        f'(95% range {s_lo:.0%} to {s_hi:.0%}) and misses about {100 - caught}.'
        f'{dot_grid(caught, "on")}'
        f'<b>Out of 100 people without PMOS</b>, about <b>{false_alarm}</b> get flagged anyway '
        f'(95% range {f_lo:.0%} to {f_hi:.0%}).{dot_grid(false_alarm, "fa")}'
        f'<b>What a flag means in real life:</b> in the training data {prev_data:.0%} of people '
        f'had PMOS, so about {ppv_data:.0%} of flags were right. In the general population it is '
        f'up to {prev:.0%} (WHO), so only about <b>{round(ppv_real * 10)} in 10</b> people flagged '
        f'would actually have PMOS. A flag means "worth checking with a GP", never "you have it". '
        f'A low score does not rule it out.</div>', unsafe_allow_html=True)
    st.caption(
        f"Tested on {meta.get('n', '?')} people from a public dataset using 5-fold "
        f"cross-validation (each person was scored by a model that had not seen them). "
        f"AUC {auc:.2f} (0.5 is a coin toss, 1.0 is perfect). Limits: one region of India, ages "
        "20 to 48, and more false alarms for people with a BMI of 30 or more. "
        "See RESPONSIBLE_DATA.md." if auc else "")


def reliability_table() -> None:
    st.markdown("| Check | Based on | How it's backed | What the evidence says |\n"
                "|---|---|---|---|\n"
                + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in tracker.CHECKS))
    st.caption("Only the PMOS pattern check is a model we trained and tested ourselves. The other "
               "checks are rules taken from NHS and NICE guidance, so we quote published "
               "studies instead of claiming our own accuracy.")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@st.cache_resource
def load_screener():
    return Screener(MODEL_PATH) if MODEL_PATH.exists() else None


def setting(name: str) -> str | None:
    """Read from Streamlit secrets (deployed) or environment/.env (local)."""
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file locally
        pass
    return value or os.getenv(name)


screener = load_screener()
if setting("OPENAI_MODEL"):
    os.environ["OPENAI_MODEL"] = setting("OPENAI_MODEL")
client = llm.get_client(setting("OPENAI_API_KEY"), setting("OPENAI_BASE_URL"))

st.markdown('<div class="hh-header"><h1>HerHealth</h1><p>Track your periods and symptoms, '
            'spot warning signs early, and know when to see a GP.</p></div>',
            unsafe_allow_html=True)
st.markdown('<div class="hh-concept"><b>Concept prototype</b> for a women\'s health feature in the '
            'NHS App ("My Health"). Not affiliated with or endorsed by the NHS. Uses public '
            'research data and made-up demo data only. Not a diagnosis.</div>',
            unsafe_allow_html=True)

if screener is None:
    st.error("Model file not found. Train it first:\n\n"
             "`python -m src.train --data data/raw/PCOS_data_without_infertility.xlsx`")
    st.stop()

if screener.bundle.get("meta", {}).get("synthetic"):
    st.warning("Demo mode: this model was trained on synthetic test data, not real patients.")

# ---------------------------------------------------------------------------
# Sidebar: transparency about the model and data
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Log** your periods and symptoms over time (or chat, or fill in a quick form).\n"
        "2. **Alerts** follow NHS and NICE guidance across women's health: cycles, heavy "
        "periods, endometriosis symptoms, perimenopause, bleeding that should always be "
        "checked, persistent bloating and cervical screening.\n"
        "3. A **PMOS pattern check** (formerly PCOS) compares your log with a public research "
        "dataset, and shows how accurate it is.\n"
        "4. Download a **summary for your GP**."
    )
    meta = screener.bundle.get("meta", {})
    m = screener.bundle.get("cv_metrics_at_threshold", {})
    auc = screener.bundle.get("cv_auc_by_model", {}).get(screener.bundle.get("model_name"))
    st.header("How accurate is the PMOS check?")
    st.caption("Tested with 5-fold cross-validation on the training data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensitivity", f"{m.get('sensitivity', 0):.0%}")
    c2.metric("Specificity", f"{m.get('specificity', 0):.0%}")
    c3.metric("AUC", f"{auc:.2f}" if auc else "n/a")
    st.caption(f"Of 100 people with PMOS it flags about {round(m.get('sensitivity', 0) * 100)}; "
               f"of 100 without, about {round((1 - m.get('specificity', 1)) * 100)} are flagged "
               f"too. Trained on {meta.get('n', '?')} people and tuned to catch most cases. "
               "The other checks follow NHS and NICE guidance; see 'How reliable are these "
               "checks?' in My health.")
    st.header("Your data")
    st.markdown("- **This prototype stores nothing.** Your log lives only in this browser tab. "
                "Download it to keep it, delete it any time.\n"
                "- Names, emails, phone numbers and postcodes are removed before any text is "
                "sent to the AI.\n"
                "- **My health** and **Quick check** work without sending your log to any AI.\n"
                "- In the NHS App, this would sit behind NHS login under NHS data rules.")
    st.caption("AI chat: " + ("on" if client else "off (no API key)"))


# ---------------------------------------------------------------------------
# Shared result display
# ---------------------------------------------------------------------------
def show_result(result: dict, cache_key: str, heading: str = "Your screening result",
                use_llm: bool = True, extra_notes: list[str] | None = None) -> None:
    score = round(result["risk"] * 100)
    st.subheader(heading)
    c1, c2 = st.columns([1, 2])
    c1.metric("Screening score", f"{score} / 100")
    c2.markdown(f"**{BAND_TITLES[result['band']]}**  \n{result['band_text']}")
    st.progress(min(score, 100) / 100)
    for note in result.get("reliability_notes", []) + (extra_notes or []):
        st.warning(note)

    with st.expander("What influenced this", expanded=True):
        for f in result["top_factors"]:
            direction = "⬆️ raised" if f["effect"] > 0 else "⬇️ lowered"
            st.write(f"{direction} the score: **{f['label']}**")
        if result["assumed_typical"]:
            st.caption("Not answered (treated as typical): " + ", ".join(result["assumed_typical"]))

    with st.expander("How accurate is this check?"):
        accuracy_panel(screener.bundle)

    # Cache the explanation so Streamlit reruns don't call the API again
    store = st.session_state.setdefault("explanations", {})
    if cache_key not in store:
        with st.spinner("Writing your summary..."):
            store[cache_key] = llm.explain(client if use_llm else None, result)
    st.markdown(store[cache_key])


def answered(feature: str, answers: dict) -> bool:
    if feature in EXTRA_FEATURES and EXTRA_FEATURES[feature]["kind"] == "group":
        return any(answers.get(k) is not None for k in EXTRA_FEATURES[feature]["keys"])
    if feature == "bmi":
        return answers.get("bmi") is not None or (
            answers.get("height_cm") is not None and answers.get("weight_kg") is not None)
    return answers.get(feature) is not None


YES = {"yes", "y", "yeah", "yep", "yup", "ya", "definitely", "i do", "i have", "correct", "true"}
NO = {"no", "n", "nope", "nah", "never", "not really", "i dont", "i do not", "i havent", "none",
      "false"}
SKIP = {"not sure", "unsure", "dont know", "i dont know", "idk", "no idea", "skip",
        "prefer not to say", "rather not say", "maybe"}
NONE_OF_THEM = {"none", "none of them", "none of those", "none of these", "nothing", "no none",
                "no to all", "no none of them"}
NOT_APPLICABLE = {"doesnt apply", "does not apply", "not applicable", "na", "n a",
                  "my periods havent stopped", "they havent stopped", "not relevant"}
ALL_QUESTIONS = {**SCREENING_FEATURES, **EXTRA_FEATURES}


def chat_snapshot(answers: dict) -> dict:
    """Chat answers -> the one-off snapshot used by the NHS/NICE checks."""
    yes = (lambda k: bool(answers.get(k)))  # noqa: E731
    flags = {"acne": yes("pimples"), "hair_growth": yes("hair_growth"),
             "hair_loss": yes("hair_loss"), "skin_darkening": yes("skin_darkening"),
             "weight_gain": yes("weight_gain"), "pain_sex": yes("pain_sex"),
             "pain_poo": yes("pain_poo"), "pain_pee": yes("pain_pee"),
             "bleeding_between": yes("bleeding_between"),
             "bleeding_after_sex": yes("bleeding_after_sex"), "hot_flushes": yes("hot_flushes"),
             "postmeno_bleeding": yes("postmeno_bleeding")}
    pain = answers.get("period_pain")
    impact = answers.get("impact")
    return {"age": answers.get("age"), "cycle_irregular": answers.get("cycle_irregular"),
            "period_length": answers.get("period_length"),
            "flow": tracker.VERY_HEAVY if yes("heavy_bleeding") else None,
            "pain": int(pain) if pain is not None else None,
            "impact": int(impact) if impact is not None else None,
            "flags": flags, "bloating_persistent": yes("bloating_persistent")}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def parse_height_weight(text: str, answers: dict) -> dict | None:
    """'165 cm', '60kg', '165 60', '165cm and 58 kg' -> {'height_cm':..., 'weight_kg':...}.
    Feet/inches/stones/pounds are left to the LLM."""
    low = text.lower()
    if re.search(r"\b(ft|feet|foot|inch|inches|stone|stones|st|lb|lbs|pound|pounds)\b|'|\"", low):
        return None
    found = {}
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(cm|kg)?", low):
        v = float(value)
        has_height = "height_cm" in found or answers.get("height_cm") is not None
        if unit == "cm" or (not unit and 120 <= v <= 220 and not has_height):
            found["height_cm"] = v
        elif unit == "kg" or (not unit and 30 <= v <= 200):
            found["weight_kg"] = v
    return found or None


def quick_parse(text: str, feature: str | None, answers: dict):
    """Answer simple replies (yes/no/a number) instantly without calling the LLM.
    Returns a value (or a dict for height/weight), "SKIP" for not sure,
    or None if the LLM should interpret it."""
    if feature is None:
        return None
    t = _norm(text)
    if t in SKIP:
        return "SKIP"
    if feature == "bmi":
        return parse_height_weight(text, answers)
    if feature in EXTRA_FEATURES:
        spec = EXTRA_FEATURES[feature]
        if spec["kind"] == "scale":
            m = re.search(r"\d+(\.\d+)?", t)
            return float(m.group()) if m and 0 <= float(m.group()) <= 10 else None
        if spec["kind"] == "group":
            if t in NO or t in NONE_OF_THEM:
                return {k: 0 for k in spec["keys"]}
            words = t.replace(",", " ").split()
            letters = dict(zip("abcdef", spec["keys"]))
            if words and all(w in letters or w in {"and", "or", "also", "both", "only"}
                             for w in words) and any(w in letters for w in words):
                return {k: int(letter in words) for letter, k in letters.items()}
            return None
        if t in NOT_APPLICABLE:
            return 0
        return 1 if t in YES else 0 if t in NO else None
    spec = SCREENING_FEATURES[feature]
    if feature == "cycle_irregular":
        if "irregular" in t or "missed" in t:
            return 1
        if t in {"regular", "regular ones", "theyre regular", "they are regular"}:
            return 0
        return None  # "yes"/"no" is ambiguous for this question, so let the LLM handle it
    if spec["kind"] == "binary":
        return 1 if t in YES else 0 if t in NO else None
    if feature in ("age", "period_length"):
        m = re.search(r"\d+(\.\d+)?", t)
        if m and spec["min"] <= float(m.group()) <= spec["max"]:
            return float(m.group())
    return None


def handle_turn(ss, new_user: list[dict]) -> None:
    for m in new_user:
        for warning in red_flags(m["content"]):
            ss.messages.append({"role": "assistant", "content": f"⚠️ **{warning}**"})
    reply = new_user[-1]["content"]
    pending = ss.asked[-1] if ss.asked else None

    quick = quick_parse(reply, pending, ss.answers)
    if quick is not None:
        if isinstance(quick, dict):
            ss.answers.update(quick)
        elif quick != "SKIP":
            ss.answers[pending] = quick
    else:
        try:
            ss.answers.update(llm.extract_answers(client, ss.messages))
        except Exception as exc:
            ss.messages.append({"role": "assistant", "content":
                                f"Sorry, I couldn't reach the AI just now ({type(exc).__name__}). "
                                "Please try again, or use the **Quick check** tab."})
            return
    if ss.answers.get("bmi") is None:
        ss.answers["bmi"] = compute_bmi(ss.answers.get("height_cm"), ss.answers.get("weight_kg"))
    if pending == "checklist" and answered("checklist", ss.answers):
        for k in EXTRA_FEATURES["checklist"]["keys"]:
            if ss.answers.get(k) is None:
                ss.answers[k] = 0

    # Height/weight given only half: ask for the missing half (once each)
    if pending == "bmi" and quick != "SKIP" and ss.answers.get("bmi") is None:
        h, w = ss.answers.get("height_cm"), ss.answers.get("weight_kg")
        if h is not None and w is None and "bmi_weight" not in ss.retried:
            ss.retried.append("bmi_weight")
            ss.messages.append({"role": "assistant",
                                "content": "Thanks! And what's your weight in kg?"})
            return
        if w is not None and h is None and "bmi_height" not in ss.retried:
            ss.retried.append("bmi_height")
            ss.messages.append({"role": "assistant",
                                "content": "Thanks! And what's your height in cm?"})
            return

    # Unclear answer to the question just asked: ask again once, with a hint
    if (pending and quick != "SKIP" and not answered(pending, ss.answers)
            and pending not in ss.retried):
        ss.retried.append(pending)
        spec = ALL_QUESTIONS[pending]
        hint = {"binary": "You can answer yes, no or not sure.",
                "scale": "A number from 0 to 10 is fine, or say 'not sure'.",
                "group": "You can list the letters, for example 'a and d', or say none."
                }.get(spec["kind"], "A number is fine, or say 'not sure'.")
        ss.messages.append({"role": "assistant", "content":
                            f"Sorry, I didn't quite catch that. {spec['question']} {hint}"})
        return

    age = ss.answers.get("age")
    next_q = next((f for f in SCREENING_FEATURES
                   if not answered(f, ss.answers) and f not in ss.asked), None)
    if next_q is None:
        next_q = next((f for f, spec in EXTRA_FEATURES.items()
                       if not answered(f, ss.answers) and f not in ss.asked
                       and (not spec.get("min_age") or (age is not None
                                                        and age >= spec["min_age"]))), None)
        if next_q and "extra_intro" not in ss.retried:
            ss.retried.append("extra_intro")
            ss.messages.append({"role": "assistant", "content":
                                "Thanks! A few more quick questions so I can check your answers "
                                "against NHS and NICE guidance for other conditions too."})
    if next_q:
        ss.asked.append(next_q)
        ss.messages.append({"role": "assistant", "content": ALL_QUESTIONS[next_q]["question"]})
    else:
        ss.chat_alerts = tracker.snapshot_alerts(chat_snapshot(ss.answers), date.today())
        ss.result = screener.predict(ss.answers)
        ss.messages.append({"role": "assistant",
                            "content": "Thank you, that's everything I need. Here is your result:"})


# ---------------------------------------------------------------------------
# My cycle tracker
# ---------------------------------------------------------------------------
def _apple_summary(summ: dict) -> str:
    parts = [f"{summ['periods']} periods", f"{summ['symptom_days']} days of symptoms",
             f"{summ['weights']} weights"]
    if summ["deviations"]:
        parts.append(f"{summ['deviations']} Apple cycle alert(s)")
    msg = "Imported from Apple Health: " + ", ".join(parts) + "."
    if summ["periods_skipped"]:
        msg += f" Kept {summ['periods_skipped']} period(s) you'd already logged yourself."
    if summ["filled"]:
        msg += f" Filled in your {' and '.join(summ['filled'])}."
    msg += (f" Read {summ['records_used']:,} relevant records and skipped "
            f"{summ['records_ignored']:,} others. Nothing was stored or sent to any AI.")
    return msg


def render_apple_import(log: dict, today: date) -> None:
    """Import periods, symptoms and Apple Watch data from the Health app's export file."""
    ss = st.session_state
    with st.expander("Import from Apple Health (iPhone and Apple Watch)"):
        st.markdown(
            "Apple Health has no web connection, so you export your data yourself:\n"
            "1. On your iPhone, open the **Health** app and tap your **profile picture** "
            "(top right).\n"
            "2. Tap **Export All Health Data**, then **Export**. It can take a few minutes.\n"
            "3. Save **export.zip** (for example to Files), then upload it here.\n\n"
            "We read only what this app needs from the **last 12 months**: periods, spotting, "
            "cycle symptoms, Apple's cycle alerts, weight, height, sleeping wrist temperature "
            "and resting heart rate. Everything else in the file (workouts, routes, heart "
            "rhythm, medical records) is skipped. Your date of birth is only used to work out "
            "your age. Nothing is stored or sent to any AI. Periods you've already logged "
            "yourself are kept as they are.")
        up = st.file_uploader("Upload export.zip (or export.xml)", type=["zip", "xml"],
                              key="apple_upload")
        if up is not None and ss.get("apple_loaded") != (up.name, up.size):
            try:
                with st.spinner("Reading your Apple Health export..."):
                    parsed = apple_health.parse_export(up.getvalue(), today)
                summ = apple_health.merge_into_log(log, parsed, today)
                ss.apple_loaded = (up.name, up.size)
                ss.flash = _apple_summary(summ)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.caption("No iPhone? Try it with a made-up sample export ('Olivia', 26, not a real "
                   "person), or download the sample file to try the upload yourself.")
        b1, b2 = st.columns(2)
        sample = apple_health.make_sample_export(today)
        if b1.button("Load the made-up sample", width="stretch",
                     help="Replaces what's on screen with the sample's data"):
            ss.log = tracker.empty_log()
            summ = apple_health.merge_into_log(
                ss.log, apple_health.parse_export(sample, today), today)
            ss.flash = _apple_summary(summ).replace("Apple Health:", "the made-up sample:")
            st.rerun()
        b2.download_button("Download the sample export.zip", sample,
                           file_name="sample-apple-health-export.zip", mime="application/zip",
                           width="stretch")


def render_wearable(log: dict, today: date) -> None:
    """Apple's own cycle alerts plus sensor trends (context only, no alerts)."""
    wear = log["wearable"]
    st.markdown("### From your Apple Watch")
    st.caption(f"Imported from {wear.get('source', 'Apple Health')} on "
               f"{tracker._d(wear['imported_on']).strftime('%d %b %Y')}.")
    for a in apple_health.deviation_cards(log, today):
        care_card(a["level"], a["title"], a["detail"], None, a.get("source"), a.get("evidence"))

    start = today - timedelta(days=tracker.RECENT_DAYS)
    temps = [(tracker._d(d), v) for d, v in wear.get("wrist_temp", []) if tracker._d(d) >= start]
    hrs = [(tracker._d(d), v) for d, v in wear.get("resting_hr", []) if tracker._d(d) >= start]
    if temps:
        base = median(v for _, v in temps)
        df = pd.DataFrame({"Night": [d for d, _ in temps],
                           "Difference from your usual (°C)": [round(v - base, 2) for _, v in temps]})
        line = alt.Chart(df).mark_line(color="#005EB8", point=alt.OverlayMarkDef(size=12)).encode(
            x=alt.X("Night:T", title=None),
            y=alt.Y("Difference from your usual (°C):Q"),
            tooltip=["Night:T", "Difference from your usual (°C):Q"])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#4c6272").encode(y="y:Q")
        st.altair_chart((zero + line).properties(height=200, title="Sleeping wrist temperature"),
                        width="stretch")
    if hrs:
        df = pd.DataFrame({"Day": [d for d, _ in hrs], "Resting heart rate (bpm)": [v for _, v in hrs]})
        st.altair_chart(alt.Chart(df).mark_line(color="#d5281b").encode(
            x=alt.X("Day:T", title=None), y=alt.Y("Resting heart rate (bpm):Q",
                                                   scale=alt.Scale(zero=False)),
            tooltip=["Day:T", "Resting heart rate (bpm):Q"]).properties(
            height=180, title="Resting heart rate"), width="stretch")
    if temps or hrs:
        st.caption("For context only. Body temperature normally rises slightly after ovulation, "
                   "which is how Apple estimates it. There are no NHS or NICE thresholds for "
                   "these readings, so HerHealth doesn't raise any alerts from them.")
    if wear.get("unmapped"):
        st.caption("Also in your export but not used (no matching HerHealth symptom): "
                   + ", ".join(f"{k.lower()} ({n})" for k, n in wear["unmapped"].items()) + ".")


def render_gp_summary(log: dict, today: date, alert_list: list[dict], result: dict | None) -> None:
    """Extra questions for the GP, then a printable NHS-style PDF."""
    ss = st.session_state
    st.markdown("### Summary for my GP")
    flagged = [a for a in alert_list if a["level"] in ("urgent", "gp")]
    questions = tracker.follow_up_questions(alert_list)
    with st.expander("Add details for your GP (recommended)", expanded=bool(flagged)):
        concern = st.text_area("What would you most like help with at the appointment?",
                               key="gp_concern", placeholder="e.g. My periods are so painful I "
                               "miss lectures every month and I'd like to find out why.")
        meds = st.text_area("Medicines you take now (including the pill, HRT, supplements)",
                            key="gp_meds", placeholder="e.g. Ibuprofen 400 mg when needed")
        tried = st.multiselect("Things you've already tried for these symptoms",
                               tracker.TREATMENTS, key="gp_tried")
        notes = st.text_input("What helped, and what didn't?", key="gp_tried_notes")
        ai_qs = ss.get("gp_ai_qs", [])
        answers = []
        if questions or ai_qs:
            st.markdown("**A few questions about your symptoms** (your GP will find these useful)")
            for i, q in enumerate(questions + ai_qs):
                answers.append((q, st.text_input(q, key=f"gp_q_{i}_{abs(hash(q)) % 10**6}")))
        if client and flagged and not ai_qs:
            if st.button("Suggest 2 more personal questions with AI"):
                try:
                    with st.spinner("Thinking of useful questions..."):
                        ss.gp_ai_qs = llm.suggest_questions(
                            client, [a["title"] for a in flagged], questions)
                except Exception:
                    ss.gp_ai_qs = []
                    st.warning("The AI couldn't suggest questions just now.")
                st.rerun()
            st.caption("Only the names of your warnings are sent to the AI: no dates, numbers or "
                       "anything you've typed.")
    extras = {"concern": concern, "medicines": meds, "tried": tried, "tried_notes": notes,
              "answers": answers}
    if st.button("Create my GP summary (PDF)", type="primary"):
        with st.spinner("Creating your PDF..."):
            ss.gp_pdf = gp_pdf.build_pdf(log, today, alert_list, result, extras,
                                         accuracy_numbers(screener.bundle))
    if ss.get("gp_pdf"):
        st.download_button("Download my GP summary (PDF)", ss.gp_pdf,
                           file_name=f"my-gp-summary-{today.isoformat()}.pdf",
                           mime="application/pdf", type="primary")
        st.caption("If you change your log or answers, click Create again to update the PDF.")
    with st.expander("Preview the summary text"):
        st.markdown(tracker.gp_summary(log, today, alert_list, result, extras))


def render_tracker() -> None:
    ss = st.session_state
    if "log" not in ss:
        ss.log = tracker.empty_log()
    log = ss.log
    today = date.today()
    has_data = bool(log["periods"] or log["symptoms"])

    if ss.get("flash"):
        notify(ss.pop("flash"))

    # --- Data controls -------------------------------------------------------
    d1, d2 = st.columns([3, 1], vertical_alignment="bottom")
    demo_name = d1.selectbox("Made-up demo profiles (not real people)", list(tracker.DEMOS))
    if d2.button("Load demo profile", width="stretch"):
        ss.log = tracker.DEMOS[demo_name](today)
        ss.flash = f"Loaded a made-up demo profile ({demo_name.split(':')[0]}). Not a real person."
        st.rerun()
    c2, c3 = st.columns(2)
    c2.download_button("Download my data", tracker.to_json(log), file_name="herhealth-my-data.json",
                       mime="application/json", disabled=not has_data, width="stretch")
    if c3.button("Delete all my data", disabled=not has_data, width="stretch"):
        ss.log = tracker.empty_log()
        ss.flash = "All your data has been deleted from this session."
        st.rerun()
    upload = st.file_uploader("Load a HerHealth file you saved earlier", type=["json"])
    if upload is not None and ss.get("loaded_file") != (upload.name, upload.size):
        try:
            ss.log = tracker.from_json(upload.getvalue().decode("utf-8"))
            ss.loaded_file = (upload.name, upload.size)
            ss.flash = "Your saved data has been loaded."
            st.rerun()
        except Exception:
            st.error("That file couldn't be read. Is it a file downloaded from HerHealth?")

    render_apple_import(log, today)

    with st.expander("About me", expanded=bool(tracker.missing_profile(log))):
        prof = log["profile"]
        st.caption("The PMOS pattern check needs your age, height and weight. We ask rather "
                   "than guess, because they change the result.")
        p1, p2, p3 = st.columns(3)
        num = (lambda v, cast: cast(v) if v is not None else None)  # noqa: E731
        age = p1.number_input("Age", 12, 90, value=num(prof.get("age"), int), step=1,
                              placeholder="e.g. 24")
        height = p2.number_input("Height (cm)", 120.0, 210.0,
                                 value=num(prof.get("height_cm"), float), step=1.0,
                                 format="%.0f", placeholder="e.g. 165")
        weight = p3.number_input("Weight (kg)", 30.0, 250.0,
                                 value=num(prof.get("weight_kg"), float), step=0.1,
                                 format="%.1f", placeholder="e.g. 62.5")
        screening = st.selectbox(
            "When was your last cervical screening (smear test)?", tracker.SCREENING_OPTIONS,
            index=tracker.SCREENING_OPTIONS.index(prof.get("last_screening")
                                                  or tracker.SCREENING_OPTIONS[0]))
        meno = st.checkbox("My periods stopped 12 or more months ago (menopause)",
                           value=bool(prof.get("menopause")))
        if st.button("Save my details"):
            log["profile"] = {"age": int(age) if age is not None else None,
                              "height_cm": float(height) if height is not None else None,
                              "weight_kg": float(weight) if weight is not None else None,
                              "last_screening": screening, "menopause": meno}
            ss.flash = "Your details have been saved for this session."
            st.rerun()

    # --- Dashboard -------------------------------------------------------------
    stt = tracker.stats(log, today)
    result = None
    alert_list = tracker.alerts(log, today) if has_data else []
    if not has_data:
        st.info("Start by logging your last period below, or click **Load demo profile** "
                "to see how it works.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Last period started",
                  f"{stt['days_since_last']} days ago" if stt["days_since_last"] is not None else "-")
        m2.metric("Typical cycle",
                  f"{stt['typical_cycle']:.0f} days" if stt["typical_cycle"] else "-")
        nxt = stt["next_expected"]
        m3.metric("Next period expected",
                  "-" if not nxt else nxt.strftime("%d %b") if nxt >= today
                  else f"Was due {nxt.strftime('%d %b')}")

        if stt["cycles"]:
            df = pd.DataFrame([{
                "Cycle starting": s.strftime("%d %b %y"), "Days": c,
                "Status": "Usual range" if 21 <= c <= 35 else "Outside usual range"}
                for s, c in stt["cycles"]])
            band = alt.Chart(pd.DataFrame({"lo": [21], "hi": [35]})).mark_rect(
                opacity=0.15, color="#007f3b").encode(y="lo:Q", y2="hi:Q")
            bars = alt.Chart(df).mark_bar(size=28).encode(
                x=alt.X("Cycle starting:N", sort=None, title=None),
                y=alt.Y("Days:Q", title="Days between periods"),
                color=alt.Color("Status:N", legend=alt.Legend(orient="bottom", title=None),
                                scale=alt.Scale(domain=["Usual range", "Outside usual range"],
                                                range=["#005EB8", "#d5281b"])),
                tooltip=["Cycle starting", "Days"])
            st.altair_chart((band + bars).properties(
                height=230, title="Cycle length (green band = usual 21 to 35 days)"),
                width="stretch")

        st.markdown("### What we noticed")
        show_cards(alert_list)
        if not any(a["level"] in ("gp", "urgent") for a in alert_list):
            st.success("Nothing in your log needs a GP visit right now, based on NHS guidance. "
                       "Keep logging, and see a GP any time something worries you.")
        with st.expander("How reliable are these checks?"):
            reliability_table()

        if log.get("wearable"):
            render_wearable(log, today)

        answers = tracker.screening_answers(log, today)
        missing = tracker.missing_profile(log)
        if answers and missing and not log["profile"].get("menopause"):
            st.subheader("Pattern check: PMOS (formerly PCOS)")
            st.info(f"To run the pattern check, add your {tracker._join(missing)} in **About me** "
                    "above. We don't guess these, because they change the result.")
        elif answers and not log["profile"].get("menopause"):
            result = screener.predict(answers)
            show_result(result, cache_key="track-" + str(sorted(answers.items())),
                        heading="Pattern check: PMOS (formerly PCOS)", use_llm=False,
                        extra_notes=tracker.pattern_check_notes(log, today) + impact_note(
                            tracker.recent_impact(log, today), result))
            st.caption("Worked out automatically from your log: no questions needed, and your log "
                       "is not sent to any AI.")

        render_gp_summary(log, today, alert_list, result)

    # --- Logging forms ----------------------------------------------------------
    st.markdown("### Add to my log")
    with st.form("log_period", clear_on_submit=True):
        st.markdown("**Log a period**")
        p1, p2 = st.columns(2)
        start = p1.date_input("First day", value=today, max_value=today, format="DD/MM/YYYY")
        end = p2.date_input("Last day", value=today, max_value=today, format="DD/MM/YYYY")
        ongoing = st.checkbox("I'm still on my period")
        p3, p4 = st.columns(2)
        flow = p3.selectbox("Heaviest flow", tracker.FLOWS, index=1)
        pain = p4.slider("Worst pain (0 = none, 10 = worst imaginable)", 0, 10, 3)
        if st.form_submit_button("Save period", width="stretch"):
            try:
                replaced = tracker.add_period(log, start, None if ongoing else end, flow, pain)
                ss.flash = "Period saved."
                if replaced:
                    old = ", ".join(
                        tracker._d(p["start"]).strftime("%d %b") + " to "
                        + (tracker._d(p["end"]).strftime("%d %b") if p.get("end") else "ongoing")
                        for p in replaced)
                    ss.flash = (f"Period saved. It overlapped the period you'd logged ({old}), "
                                "so that entry was replaced with these dates.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with st.form("log_symptoms", clear_on_submit=True):
        st.markdown("**Log symptoms** (slide each one you noticed on this day: None to Severe)")
        s1, s2 = st.columns(2)
        day = s1.date_input("Date", value=today, max_value=today, format="DD/MM/YYYY")
        weight = s2.number_input("Weight in kg (optional)", 0.0, 250.0, value=0.0, step=0.1)
        flags, severity, _ = symptom_sliders("sym")
        impact = impact_slider("sym_impact")
        if st.form_submit_button("Save symptoms", width="stretch"):
            tracker.add_symptoms(log, day, flags, weight or None, impact, severity)
            ss.flash = "Symptoms saved."
            st.rerun()

    if has_data:
        with st.expander("My log"):
            if log["periods"]:
                st.dataframe(pd.DataFrame([{
                    "Start": p["start"], "End": p["end"] or "ongoing", "Flow": p["flow"],
                    "Pain": p["pain"]} for p in log["periods"]]), hide_index=True,
                    width="stretch")
            if log["symptoms"]:
                st.dataframe(pd.DataFrame([{
                    "Date": s["date"],
                    "Symptoms": tracker.describe_symptoms(s),
                    "Effect on day (0-10)": "" if s.get("impact") is None else str(s["impact"]),
                    "Weight (kg)": "" if not s.get("weight_kg") else str(s["weight_kg"])}
                    for s in log["symptoms"]]), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
track_tab, chat_tab, form_tab = st.tabs(["My health", "Ask about my symptoms", "Quick check"])

with track_tab:
    render_tracker()

with chat_tab:
    if client is None:
        st.info("The AI chat needs an LLM API key (see README). "
                "Use **Quick check**: it uses exactly the same model, without AI.")
    else:
        ss = st.session_state
        if "messages" not in ss:
            ss.messages = [{"role": "assistant", "content":
                            "Hi! Tell me in your own words about your periods and any symptoms "
                            "you've noticed (for example pain, heavy bleeding, acne, hair or "
                            "weight changes, hot flushes) and how much they affect your day. "
                            "Please don't share your name or contact details."}]
            ss.answers, ss.asked, ss.retried, ss.result = {}, [], [], None
            ss.processed = len(ss.messages)  # messages already handled

        for msg in ss.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if ss.result is None:
            text = st.chat_input("Type your answer...")
            if text:
                ss.messages.append({"role": "user", "content": text})
                with st.chat_message("user"):
                    st.markdown(text)
            # Handle every user message not processed yet. This survives Streamlit
            # interrupting a run when someone types again before the reply arrives.
            new_user = [m for m in ss.messages[ss.processed:] if m["role"] == "user"]
            if new_user:
                with st.spinner("Thinking..."):
                    handle_turn(ss, new_user)
                ss.processed = len(ss.messages)
                st.rerun()

        if ss.result is not None:
            st.markdown("### What we noticed")
            if ss.get("chat_alerts"):
                show_cards(ss.chat_alerts)
            else:
                st.success("Nothing you told me matches NHS or NICE 'see a GP' guidance. See a GP "
                           "any time something worries you.")
            impact = ss.answers.get("impact")
            show_result(ss.result, cache_key="chat-" + str(sorted(ss.answers.items())),
                        extra_notes=impact_note(int(impact) if impact is not None else None,
                                                ss.result))

        if st.button("Start over", key="reset_chat"):
            for k in ["messages", "answers", "asked", "retried", "result", "processed",
                      "chat_alerts"]:
                ss.pop(k, None)
            st.rerun()

with form_tab:
    YES_NO = {"Yes": 1, "No": 0, "Not sure": None}
    ss = st.session_state
    st.caption("This tab uses no AI: nothing you enter leaves the app.")
    with st.form("quick_check"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=12, max_value=90, value=None, step=1,
                              placeholder="e.g. 24")
        height = c2.number_input("Height (cm)", min_value=120.0, max_value=210.0, value=None,
                                 step=1.0, format="%.0f", placeholder="e.g. 165")
        weight = c3.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=None,
                                 step=0.1, format="%.1f", placeholder="e.g. 62.5")
        cycle = st.radio("Your periods are usually...",
                         ["Regular (every 21-35 days)", "Irregular or often missed", "Not sure"],
                         horizontal=True, index=2)
        period_length = st.select_slider("How many days does your period usually last?",
                                         options=["Not sure"] + list(range(1, 13)),
                                         value="Not sure")
        binary_answers = {}
        for name, spec in SCREENING_FEATURES.items():
            if spec["kind"] == "binary" and name != "cycle_irregular":
                choice = st.radio(spec["question"], list(YES_NO), horizontal=True,
                                  index=2, key=f"q_{name}")
                binary_answers[name] = YES_NO[choice]

        st.markdown("#### Other symptoms")
        st.caption("Checked against NHS and NICE guidance, so we can flag things beyond PMOS, "
                   "such as heavy periods, endometriosis symptoms or perimenopause.")
        f1, f2 = st.columns(2)
        flow = f1.selectbox("Heaviest flow in a usual period", ["Not sure"] + tracker.FLOWS)
        with f2:
            pain = pain_slider("qc_pain")
        impact = impact_slider("qc_impact", "How much do your symptoms affect your day?")
        flags, severity, bloat = symptom_sliders(
            "qc", skip_groups=("Skin, hair and weight",),
            bloating_label="Bloating most days for 3 weeks or more")
        m1, m2 = st.columns(2)
        meno = m1.checkbox("My periods stopped 12 or more months ago", key="qc_meno")
        screening = m2.selectbox("Last cervical screening (smear test)", tracker.SCREENING_OPTIONS)
        submitted = st.form_submit_button("Check my answers", type="primary")

    if submitted:
        missing = [n for n, v in (("age", age), ("height", height), ("weight", weight))
                   if v is None]
        if missing:
            st.error(f"Please enter your {tracker._join(missing)}. We don't guess these, "
                     "because they change the result.")
        else:
            answers = {
                "age": age,
                "bmi": compute_bmi(height, weight),
                "cycle_irregular": {"Regular (every 21-35 days)": 0,
                                    "Irregular or often missed": 1}.get(cycle),
                "period_length": None if period_length == "Not sure" else period_length,
                **binary_answers,
            }
            flags.update({"acne": binary_answers.get("pimples") == 1,
                          **{k: binary_answers.get(k) == 1 for k in
                             ("hair_growth", "hair_loss", "skin_darkening", "weight_gain")}})
            snap = {"age": age, "menopause": meno, "cycle_irregular": answers["cycle_irregular"],
                    "period_length": answers["period_length"],
                    "flow": None if flow == "Not sure" else flow, "pain": pain,
                    "impact": impact, "flags": flags, "severity": severity,
                    "bloating_persistent": bloat,
                    "last_screening": screening}
            ss.form_answers = answers
            ss.form_result = screener.predict(answers)
            ss.form_alerts = tracker.snapshot_alerts(snap, date.today())
            ss.form_impact, ss.form_meno = impact, meno

    if ss.get("form_result"):
        st.markdown("### What we noticed")
        if ss.get("form_alerts"):
            show_cards(ss.form_alerts)
        else:
            st.success("Nothing you entered matches NHS or NICE 'see a GP' guidance. See a GP any "
                       "time something worries you.")
        notes = impact_note(ss.get("form_impact"), ss.form_result)
        if ss.get("form_meno"):
            notes.append("You said your periods have stopped, so this PMOS check doesn't apply "
                         "well to you.")
        show_result(ss.form_result, extra_notes=notes,
                    cache_key="form-" + str(sorted(ss.form_answers.items())))
