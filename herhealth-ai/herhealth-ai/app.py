"""HerHealth AI: a PCOS screening assistant.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import llm
from src.config import SCREENING_FEATURES
from src.guardrails import DISCLAIMER, red_flags
from src.predict import Screener, compute_bmi

load_dotenv()
st.set_page_config(page_title="HerHealth AI · PCOS screening", page_icon="🌸", layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "pcos_screening.joblib"
BAND_TITLES = {
    "higher": "Worth talking to a GP",
    "moderate": "Some patterns found",
    "lower": "Fewer patterns found",
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@st.cache_resource
def load_screener():
    return Screener(MODEL_PATH) if MODEL_PATH.exists() else None


def api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets file locally
        pass
    return key or os.getenv("OPENAI_API_KEY")


screener = load_screener()
client = llm.get_client(api_key())

st.title("🌸 HerHealth AI")
st.markdown("**Could your symptoms be linked to PCOS?** A free, private screening check "
            "that uses only things you already know about yourself. No blood test needed.")
st.info(DISCLAIMER, icon="ℹ️")

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
        "1. You describe your symptoms (chat) or fill in a short form.\n"
        "2. A machine-learning model trained on a **public, anonymised research dataset** "
        "compares your answers with patterns from people with and without PCOS.\n"
        "3. You get a plain-English summary and questions to take to your GP."
    )
    meta = screener.bundle.get("meta", {})
    m = screener.bundle.get("cv_metrics_at_threshold", {})
    auc = screener.bundle.get("cv_auc_by_model", {}).get(screener.bundle.get("model_name"))
    st.header("Model performance")
    st.caption("5-fold cross-validation on the training data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensitivity", f"{m.get('sensitivity', 0):.0%}")
    c2.metric("Specificity", f"{m.get('specificity', 0):.0%}")
    c3.metric("AUC", f"{auc:.2f}" if auc else "n/a")
    st.caption(f"Trained on {meta.get('n', '?')} people. Tuned to catch most cases "
               "(high sensitivity), accepting some false alarms.")
    st.header("Your privacy")
    st.markdown("- Nothing you enter is stored.\n"
                "- Names, emails, phone numbers and postcodes are removed before any text "
                "is sent to the AI.\n"
                "- The **Quick check** tab works with no AI at all.")
    st.caption("AI chat: " + ("on" if client else "off (no API key)"))


# ---------------------------------------------------------------------------
# Shared result display
# ---------------------------------------------------------------------------
def show_result(result: dict, cache_key: str) -> None:
    score = round(result["risk"] * 100)
    st.subheader("Your screening result")
    c1, c2 = st.columns([1, 2])
    c1.metric("Screening score", f"{score} / 100")
    c2.markdown(f"**{BAND_TITLES[result['band']]}**  \n{result['band_text']}")
    st.progress(min(score, 100) / 100)
    for note in result.get("reliability_notes", []):
        st.warning(note)

    with st.expander("What influenced this", expanded=True):
        for f in result["top_factors"]:
            direction = "⬆️ raised" if f["effect"] > 0 else "⬇️ lowered"
            st.write(f"{direction} the score: **{f['label']}**")
        if result["assumed_typical"]:
            st.caption("Not answered (treated as typical): " + ", ".join(result["assumed_typical"]))

    # Cache the explanation so Streamlit reruns don't call the API again
    store = st.session_state.setdefault("explanations", {})
    if cache_key not in store:
        with st.spinner("Writing your summary..."):
            store[cache_key] = llm.explain(client, result)
    st.markdown(store[cache_key])


def answered(feature: str, answers: dict) -> bool:
    if feature == "bmi":
        return answers.get("bmi") is not None or (
            answers.get("height_cm") is not None and answers.get("weight_kg") is not None)
    return answers.get(feature) is not None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
chat_tab, form_tab = st.tabs(["💬 Chat", "📝 Quick check"])

with chat_tab:
    if client is None:
        st.info("The AI chat needs an OpenAI API key (see README). "
                "Use **Quick check**: it uses exactly the same model, without AI.")
    else:
        ss = st.session_state
        if "messages" not in ss:
            ss.messages = [{"role": "assistant", "content":
                            "Hi! Tell me in your own words about your periods and any symptoms "
                            "you've noticed (for example acne, hair changes or weight changes). "
                            "Please don't share your name or contact details."}]
            ss.answers, ss.asked, ss.result = {}, [], None

        for msg in ss.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if ss.result is None and (text := st.chat_input("Type your answer...")):
            ss.messages.append({"role": "user", "content": text})
            for warning in red_flags(text):
                ss.messages.append({"role": "assistant", "content": f"⚠️ **{warning}**"})
            try:
                ss.answers.update(llm.extract_answers(client, ss.messages))
            except Exception as exc:
                st.warning(f"Couldn't reach the AI ({type(exc).__name__}). "
                           "Try again, or use the Quick check tab.")
            if ss.answers.get("bmi") is None:
                ss.answers["bmi"] = compute_bmi(ss.answers.get("height_cm"),
                                                ss.answers.get("weight_kg"))

            next_q = next((f for f in SCREENING_FEATURES
                           if not answered(f, ss.answers) and f not in ss.asked), None)
            if next_q:
                ss.asked.append(next_q)
                ss.messages.append({"role": "assistant",
                                    "content": SCREENING_FEATURES[next_q]["question"]})
            else:
                ss.result = screener.predict(ss.answers)
                ss.messages.append({"role": "assistant",
                                    "content": "Thank you, that's everything I need. "
                                               "Here is your result:"})
            st.rerun()

        if ss.result is not None:
            show_result(ss.result, cache_key="chat-" + str(sorted(ss.answers.items())))

        if st.button("Start over", key="reset_chat"):
            for k in ["messages", "answers", "asked", "result"]:
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
