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


def test_apple_health_import():
    import io
    import zipfile
    from datetime import date, timedelta
    import pytest
    from src import apple_health as A, tracker as T
    today = date(2026, 10, 24)

    # The made-up sample parses into periods, symptoms, profile and Apple's own alerts
    sample = A.make_sample_export(today)
    parsed = A.parse_export(sample, today)
    assert len(parsed["periods"]) == 7 and parsed["age"] == 26 and parsed["height_cm"] == 165.0
    assert {d["kind"] for d in parsed["deviations"]} == {"irregular", "spotting"}
    assert parsed["records_ignored"] > 0            # step counts are skipped, not kept

    # Merging never overwrites a period the user logged by hand
    log = T.empty_log()
    first = parsed["periods"][0]["start"]
    T.add_period(log, first + timedelta(days=1), first + timedelta(days=4), "Light", 6)
    summary = A.merge_into_log(log, parsed, today)
    assert summary["periods_skipped"] == 1
    mine = [p for p in log["periods"] if p.get("source") is None]
    assert len(mine) == 1 and mine[0]["pain"] == 6
    assert any(p.get("source") == "Apple Health" and p["pain"] is None for p in log["periods"])
    keys = {a["key"] for a in T.alerts(log, today)}
    assert "irregular" in keys and "bleeding_between" in keys
    assert len(A.deviation_cards(log, today)) == 2
    again = T.from_json(T.to_json(log))              # survives download + re-upload
    assert again["wearable"]["deviations"] == log["wearable"]["deviations"]
    assert "not recorded" in T.gp_summary(again, today, T.alerts(again, today), None)

    # A bare export.xml works too; "not present" symptoms and pounds are handled
    xml = (b'<?xml version="1.0"?><HealthData locale="en_GB">'
           b'<Record type="HKCategoryTypeIdentifierAcne" startDate="2026-10-20 08:00:00 +0100" '
           b'value="HKCategoryValueSeverityNotPresent"/>'
           b'<Record type="HKCategoryTypeIdentifierHotFlashes" startDate="2026-10-21 08:00:00 +0100" '
           b'value="HKCategoryValueSeverityMild"/>'
           b'<Record type="HKQuantityTypeIdentifierBodyMass" unit="lb" '
           b'startDate="2026-10-21 08:00:00 +0100" value="150"/></HealthData>')
    p2 = A.parse_export(xml, today)
    assert p2["symptoms"] == {date(2026, 10, 21): {"hot_flushes"}}
    assert p2["weights"][date(2026, 10, 21)] == 68.0

    # Files that aren't Apple exports, or that try XML entity tricks, are rejected safely
    with pytest.raises(ValueError):
        A.parse_export(b"not xml at all", today)
    bomb = (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
            b'<HealthData>&b;</HealthData>')
    with pytest.raises(ValueError):
        A.parse_export(bomb, today)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other.txt", "hello")
    with pytest.raises(ValueError):
        A.parse_export(buf.getvalue(), today)


def test_overlapping_periods_and_required_details():
    from datetime import date, timedelta
    from src import tracker as T
    today = date(2026, 10, 24)
    log = T.empty_log()
    T.add_period(log, today - timedelta(days=40), today - timedelta(days=35), "Medium", 3)
    first = today - timedelta(days=10)
    T.add_period(log, first, first + timedelta(days=4), "Medium", 3)
    # Overlapping entry replaces the old one, and the replaced period is reported back
    replaced = T.add_period(log, first + timedelta(days=2), first + timedelta(days=6), "Heavy", 5)
    assert [T._d(p["start"]) for p in replaced] == [first]
    assert len(log["periods"]) == 2 and T._d(log["periods"][-1]["start"]) == first + timedelta(days=2)
    # Same first day still updates in place; a separate period is simply added
    assert T.add_period(log, first + timedelta(days=2), None, "Heavy", 6)
    assert not T.add_period(log, today - timedelta(days=70), today - timedelta(days=66), "Light", 2)
    assert len(log["periods"]) == 3

    # The pattern check asks for age, height and weight instead of assuming them
    assert T.missing_profile(log) == ["age", "height", "weight"]
    log["profile"].update({"age": 24, "height_cm": 165.0})
    assert T.missing_profile(log) == ["weight"]
    log["profile"]["weight_kg"] = 60.0
    assert T.missing_profile(log) == []
    assert T.screening_answers(log, today)["bmi"] == 22.0
    for make in T.DEMOS.values():                      # demo profiles are complete
        assert T.missing_profile(make(today)) == []


def test_impact_tests_snapshot_and_pdf():
    from datetime import date, timedelta
    from src import gp_pdf, tracker as T
    today = date(2026, 10, 24)

    # Effect on daily life raises a warning on its own (7+), stronger wording at 9+
    log = T.demo_sophie(today)
    T.add_symptoms(log, today - timedelta(days=2), {}, None, 9)
    impact = [a for a in T.alerts(log, today) if a["key"] == "impact"][0]
    assert impact["level"] == "gp" and "stopping you" in impact["title"]
    assert T.impact_label(0) == "no effect on my day" and "bed" in T.impact_label(10)

    # Every warning carries tests; endometriosis says no blood test can diagnose it
    assert all("tests" in a for a in T.alerts(log, today))
    assert any("No blood test" in t for t in T.tests_for("endometriosis"))
    assert any("no blood test" in t.lower() for t in T.tests_for("perimenopause", 47))
    assert any("FSH" in t for t in T.tests_for("perimenopause", 35))

    # One-off answers (Quick check / chat) raise the same NHS/NICE warnings
    snap = {"age": 29, "cycle_irregular": 1, "period_length": 9, "flow": T.VERY_HEAVY,
            "pain": 9, "impact": 8, "bloating_persistent": True,
            "flags": {"pain_sex": True, "acne": True, "bleeding_after_sex": True}}
    keys = [a["key"] for a in T.snapshot_alerts(snap, today)]
    for k in ("severe_pain", "heavy", "long_period", "endometriosis", "bloating", "impact",
              "irregular", "pmos", "bleeding_after_sex"):
        assert k in keys, k
    assert keys[0] == "severe_pain"                       # urgent first
    assert T.snapshot_alerts({"age": 30}, today) == []    # nothing reported, nothing flagged

    # Follow-up questions and a real PDF
    al = T.alerts(log, today)
    assert T.follow_up_questions(al)
    extras = {"concern": "Irregular periods and acne", "medicines": "None",
              "tried": [T.TREATMENTS[0]], "answers": [("When did it start?", "2025")]}
    md = T.gp_summary(log, today, al, None, extras)
    assert "Tests I'd like to ask about" in md and "When did it start?" in md
    pdf = gp_pdf.build_pdf(log, today, al, None, extras,
                           {"n": 541, "caught": 86, "false_alarm": 24, "ppv_in_10": 3})
    assert pdf[:4] == b"%PDF" and len(pdf) > 20000


def test_symptom_severity():
    from datetime import date, timedelta
    from src import apple_health as A, tracker as T
    today = date(2026, 10, 24)
    log = T.empty_log()
    T.add_symptoms(log, today - timedelta(days=1), {}, None, None,
                   {"pelvic_pain": 3, "bloating": 3, "acne": 1})
    keys = {a["key"]: a["level"] for a in T.alerts(log, today)}
    assert keys["severe_pelvic_pain"] == "urgent" and keys["severe_symptoms"] == "gp"
    assert "Bloating or a swollen tummy (severe)" in T.describe_symptoms(log["symptoms"][0])
    T.add_symptoms(log, today, {}, None, None, {"breathless": 3, "low_mood": 3,
                                               "pain_stops_activities": 3})
    keys = {a["key"]: a["level"] for a in T.alerts(log, today)}
    assert keys["severe_breathless"] == "urgent" and keys["severe_low_mood"] == "gp"
    assert keys["severe_pain"] == "urgent"
    mild = T.empty_log()
    T.add_symptoms(mild, today, {}, None, None, {"bloating": 1, "acne": 2})
    assert not [a for a in T.alerts(mild, today) if a["key"].startswith("severe")]
    assert T.from_json(T.to_json(log))["symptoms"][-1]["severity"]["breathless"] == 3
    snap = T.snapshot_alerts({"age": 30, "flags": {}, "severity": {"pelvic_pain": 3}}, today)
    assert snap[0]["key"] == "severe_pelvic_pain"
    # Apple Health severity comes through too
    xml = (b'<HealthData><Record type="HKCategoryTypeIdentifierBloating" '
           b'startDate="2026-10-20 08:00:00 +0100" value="HKCategoryValueSeveritySevere"/>'
           b'</HealthData>')
    parsed = A.parse_export(xml, today)
    merged = T.empty_log()
    A.merge_into_log(merged, parsed, today)
    assert merged["symptoms"][0]["severity"] == {"bloating": 3}
