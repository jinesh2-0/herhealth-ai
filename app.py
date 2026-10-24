"""HerHealth: a women's health tracker and PMOS (formerly PCOS) screening assistant.

Concept prototype for a women's health feature in the NHS App. Not affiliated with the NHS.
Run locally:   streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import llm, tracker
from src.config import SCREENING_FEATURES
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
              source: str | None = None, evidence: str | None = None) -> None:
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
    st.markdown(f'<div class="hh-card {level}"><div class="hd">{label}: {html.escape(title)}</div>'
                f'<div class="bd">{html.escape(body)}{more}{ev}</div></div>',
                unsafe_allow_html=True)


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
    if feature == "bmi":
        return answers.get("bmi") is not None or (
            answers.get("height_cm") is not None and answers.get("weight_kg") is not None)
    return answers.get(feature) is not None


YES = {"yes", "y", "yeah", "yep", "yup", "ya", "definitely", "i do", "i have", "correct", "true"}
NO = {"no", "n", "nope", "nah", "never", "not really", "i dont", "i do not", "i havent", "none",
      "false"}
SKIP = {"not sure", "unsure", "dont know", "i dont know", "idk", "no idea", "skip",
        "prefer not to say", "rather not say", "maybe"}


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
        spec = SCREENING_FEATURES[pending]
        hint = ("You can answer yes, no or not sure." if spec["kind"] == "binary"
                else "A number is fine, or say 'not sure'.")
        ss.messages.append({"role": "assistant", "content":
                            f"Sorry, I didn't quite catch that. {spec['question']} {hint}"})
        return

    next_q = next((f for f in SCREENING_FEATURES
                   if not answered(f, ss.answers) and f not in ss.asked), None)
    if next_q:
        ss.asked.append(next_q)
        ss.messages.append({"role": "assistant", "content": SCREENING_FEATURES[next_q]["question"]})
    else:
        ss.result = screener.predict(ss.answers)
        ss.messages.append({"role": "assistant",
                            "content": "Thank you, that's everything I need. Here is your result:"})


# ---------------------------------------------------------------------------
# My cycle tracker
# ---------------------------------------------------------------------------
def render_tracker() -> None:
    ss = st.session_state
    if "log" not in ss:
        ss.log = tracker.empty_log()
    log = ss.log
    today = date.today()
    has_data = bool(log["periods"] or log["symptoms"])

    if ss.get("flash"):
        st.success(ss.pop("flash"))

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

    with st.expander("About me", expanded=log["profile"].get("age") is None):
        prof = log["profile"]
        p1, p2 = st.columns(2)
        age = p1.number_input("Age", 12, 90, value=int(prof.get("age") or 25))
        height = p2.number_input("Height (cm)", 120, 210, value=int(prof.get("height_cm") or 162))
        screening = st.selectbox(
            "When was your last cervical screening (smear test)?", tracker.SCREENING_OPTIONS,
            index=tracker.SCREENING_OPTIONS.index(prof.get("last_screening")
                                                  or tracker.SCREENING_OPTIONS[0]))
        meno = st.checkbox("My periods stopped 12 or more months ago (menopause)",
                           value=bool(prof.get("menopause")))
        if st.button("Save my details"):
            log["profile"] = {"age": int(age), "height_cm": float(height),
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
        order = {"urgent": 0, "gp": 1, "info": 2}
        for a in sorted(alert_list, key=lambda a: order[a["level"]]):
            care_card(a["level"], a["title"], a["detail"], a["link"],
                      a.get("source"), a.get("evidence"))
        if not any(a["level"] in ("gp", "urgent") for a in alert_list):
            st.success("Nothing in your log needs a GP visit right now, based on NHS guidance. "
                       "Keep logging, and see a GP any time something worries you.")
        with st.expander("How reliable are these checks?"):
            reliability_table()

        answers = tracker.screening_answers(log, today)
        if answers and not log["profile"].get("menopause"):
            result = screener.predict(answers)
            show_result(result, cache_key="track-" + str(sorted(answers.items())),
                        heading="Pattern check: PMOS (formerly PCOS)", use_llm=False,
                        extra_notes=tracker.pattern_check_notes(log, today))
            st.caption("Worked out automatically from your log: no questions needed, and your log "
                       "is not sent to any AI.")

        summary = tracker.gp_summary(log, today, alert_list, result)
        st.download_button("Download a summary for my GP", summary,
                           file_name="my-cycle-summary-for-gp.md", mime="text/markdown",
                           type="primary")
        with st.expander("Preview the GP summary"):
            st.markdown(summary)

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
                tracker.add_period(log, start, None if ongoing else end, flow, pain)
                ss.flash = "Period saved."
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with st.form("log_symptoms", clear_on_submit=True):
        st.markdown("**Log symptoms** (tick anything you noticed on this day)")
        s1, s2 = st.columns(2)
        day = s1.date_input("Date", value=today, max_value=today, format="DD/MM/YYYY")
        weight = s2.number_input("Weight in kg (optional)", 0.0, 250.0, value=0.0, step=0.1)
        flags = {}
        cols = st.columns(2)
        for i, (group, items) in enumerate(tracker.SYMPTOM_GROUPS.items()):
            with cols[i % 2]:
                st.markdown(f"**{group}**")
                for k, label in items.items():
                    flags[k] = st.checkbox(label, key=f"sym_{k}")
        if st.form_submit_button("Save symptoms", width="stretch"):
            tracker.add_symptoms(log, day, flags, weight or None)
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
                    "Symptoms": ", ".join(tracker.SYMPTOMS[k] for k in tracker.SYMPTOMS if s.get(k))
                    or "None", "Weight (kg)": s.get("weight_kg") or ""}
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
                            "you've noticed (for example acne, hair changes or weight changes). "
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
            show_result(ss.result, cache_key="chat-" + str(sorted(ss.answers.items())))

        if st.button("Start over", key="reset_chat"):
            for k in ["messages", "answers", "asked", "retried", "result", "processed"]:
                ss.pop(k, None)
            st.rerun()

with form_tab:
    YES_NO = {"Yes": 1, "No": 0, "Not sure": None}
    with st.form("quick_check"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=12, max_value=60, value=25)
        height = c2.number_input("Height (cm)", min_value=120, max_value=210, value=162)
        weight = c3.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
        cycle = st.radio("Your periods are usually...",
                         ["Regular (every 21-35 days)", "Irregular or often missed", "Not sure"],
                         horizontal=True)
        period_length = st.slider("How many days does your period usually last?", 0, 12, 5)
        binary_answers = {}
        for name, spec in SCREENING_FEATURES.items():
            if spec["kind"] == "binary" and name != "cycle_irregular":
                choice = st.radio(spec["question"], list(YES_NO), horizontal=True,
                                  index=2, key=f"q_{name}")
                binary_answers[name] = YES_NO[choice]
        submitted = st.form_submit_button("Check my answers", type="primary")

    if submitted:
        answers = {
            "age": age,
            "bmi": compute_bmi(height, weight),
            "cycle_irregular": {"Regular (every 21-35 days)": 0,
                                "Irregular or often missed": 1}.get(cycle),
            "period_length": period_length,
            **binary_answers,
        }
        st.session_state.form_answers = answers
        st.session_state.form_result = screener.predict(answers)

    if st.session_state.get("form_result"):
        show_result(st.session_state.form_result,
                    cache_key="form-" + str(sorted(st.session_state.form_answers.items())))
