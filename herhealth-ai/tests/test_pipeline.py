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


def test_fallback_explanation_without_api():
    result = {"band_text": "Some patterns.", "flag_for_gp": True, "assumed_typical": [],
              "top_factors": [{"label": "Irregular periods", "effect": 0.2}]}
    text = llm.explain(None, result)
    assert "GP" in text and "not a diagnosis" in text.lower()
