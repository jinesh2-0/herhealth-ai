"""End-to-end tests on synthetic data. Run: pytest -q"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import make  # noqa: E402
from src import llm  # noqa: E402
from src.guardrails import make_safe, red_flags, scrub_pii  # noqa: E402
from src.predict import Screener, compute_bmi  # noqa: E402
from src.train import main as train_main  # noqa: E402


def test_train_and_predict(tmp_path):
    data = tmp_path / "pcos_synthetic.csv"
    make(n=300, seed=1).to_csv(data, index=False)
    summary = train_main(["--data", str(data), "--models-dir", str(tmp_path / "models"),
                          "--reports-dir", str(tmp_path / "reports"), "--fast"])
    assert summary["meta"]["synthetic"] is True
    assert (tmp_path / "reports" / "model_card.md").exists()
    assert (tmp_path / "reports" / "figures" / "roc_curves.png").exists()

    screener = Screener(tmp_path / "models" / "pcos_screening.joblib")
    high = screener.predict({"age": 24, "height_cm": 160, "weight_kg": 75, "cycle_irregular": 1,
                             "hair_growth": 1, "skin_darkening": 1, "weight_gain": 1,
                             "pimples": 1, "hair_loss": 1})
    low = screener.predict({"age": 33, "bmi": 21, "cycle_irregular": 0, "hair_growth": 0,
                            "skin_darkening": 0, "weight_gain": 0, "pimples": 0, "hair_loss": 0})
    for r in (high, low):
        assert 0 <= r["risk"] <= 1
        assert r["band"] in {"higher", "moderate", "lower"}
    assert high["risk"] > low["risk"]
    assert high["top_factors"], "expected at least one explaining factor"


def test_bmi():
    assert compute_bmi(160, 64) == 25.0
    assert compute_bmi(None, 64) is None


def test_red_flags():
    assert red_flags("I have severe pain in my pelvis")
    assert red_flags("I'm soaking a pad every hour")
    assert not red_flags("my periods are irregular and I have acne")


def test_scrub_pii():
    text = scrub_pii("email me at jo@example.com or call 07700 900123, I'm 21 165 60")
    assert "example.com" not in text and "900123" not in text
    assert "21 165 60" in text


def test_make_safe_never_diagnoses():
    out = make_safe("Based on this, you have PCOS.")
    assert "you have pcos" not in out.lower()
    assert "not a diagnosis" in out.lower()


def test_extraction_schema_is_strict():
    schema = llm.extraction_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_tracker_alerts_and_roundtrip():
    from datetime import date, timedelta
    from src import tracker as T
    today = date(2026, 10, 24)

    demo = T.demo_log(today)
    titles = {a["title"] for a in T.alerts(demo, today)}
    assert "Your periods look irregular" in titles
    assert "Irregular periods with other symptoms" in titles
    assert T.from_json(T.to_json(demo))["periods"] == demo["periods"]

    regular = T.empty_log()
    for i in range(4):
        s = today - timedelta(days=28 * (3 - i) + 5)
        T.add_period(regular, s, s + timedelta(days=4), "Medium", 3)
    assert not [a for a in T.alerts(regular, today) if a["level"] in ("gp", "urgent")]

    missed = T.empty_log()
    T.add_period(missed, today - timedelta(days=120), today - timedelta(days=115), "Medium", 3)
    assert any("No period logged" in a["title"] for a in T.alerts(missed, today))

    painful = T.empty_log()
    T.add_period(painful, today - timedelta(days=3), None, T.VERY_HEAVY, 9)
    levels = {a["level"] for a in T.alerts(painful, today)}
    assert "urgent" in levels


def test_new_guideline_checks():
    from datetime import date, timedelta
    from src import tracker as T
    today = date(2026, 10, 24)
    keys = lambda log: {a["key"]: a["level"] for a in T.alerts(log, today)}  # noqa: E731

    # Every demo profile loads, and every alert says where it comes from and how reliable it is
    for make in T.DEMOS.values():
        log = make(today)
        for a in T.alerts(log, today):
            assert a["source"] and a["evidence"] and a["level"] in {"info", "gp", "urgent"}
        assert T.from_json(T.to_json(log))["profile"] == log["profile"]
    assert "endometriosis" in keys(T.demo_emma(today)) and "heavy" in keys(T.demo_emma(today))
    assert "perimenopause" in keys(T.demo_claire(today))

    # Endometriosis (NICE NG73): one listed symptom is enough to flag
    log = T.empty_log()
    T.add_symptoms(log, today, {"pain_sex": True})
    assert keys(log).get("endometriosis") == "gp"

    # Heavy periods + anaemia symptoms -> blood test advice (NICE NG88)
    log = T.empty_log()
    T.add_symptoms(log, today, {"clots": True, "tiredness": True})
    heavy = [a for a in T.alerts(log, today) if a["key"] == "heavy"][0]
    assert "full blood count" in heavy["detail"]

    # Bleeding after the menopause is always urgent
    log = T.empty_log()
    log["profile"]["menopause"] = True
    T.add_period(log, today - timedelta(days=2), today, "Light", 0)
    assert keys(log).get("postmeno_bleeding") == "urgent"

    # Bloating on 12+ days in a month (NICE NG12), but not on fewer
    log = T.empty_log()
    for i in range(11):
        T.add_symptoms(log, today - timedelta(days=i), {"bloating": True})
    assert "bloating" not in keys(log)
    T.add_symptoms(log, today - timedelta(days=20), {"bloating": True})
    assert keys(log).get("bloating") == "gp"

    # Perimenopause advice differs under 45
    log = T.empty_log()
    log["profile"]["age"] = 38
    T.add_symptoms(log, today, {"hot_flushes": True})
    peri = [a for a in T.alerts(log, today) if a["key"] == "perimenopause"][0]
    assert "under 45" in peri["detail"]

    # Cervical screening: overdue only for ages 25-64
    log = T.empty_log()
    log["profile"].update({"age": 40, "last_screening": T.SCREENING_OPTIONS[2]})
    assert keys(log).get("screening") == "gp"
    log["profile"]["age"] = 70
    assert "screening" not in keys(log)


def test_fallback_explanation_without_api():
    result = {"band_text": "Some patterns.", "flag_for_gp": True, "assumed_typical": [],
              "top_factors": [{"label": "Irregular periods", "effect": 0.2}]}
    text = llm.explain(None, result)
    assert "GP" in text and "not a diagnosis" in text.lower()
