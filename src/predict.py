"""Load a trained bundle and score one person's answers, with per-person explanations."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FEATURE_LABELS, SCREENING_FEATURES


def compute_bmi(height_cm: float | None, weight_kg: float | None) -> float | None:
    try:
        if height_cm and weight_kg and height_cm > 0:
            return round(float(weight_kg) / (float(height_cm) / 100) ** 2, 1)
    except (TypeError, ValueError):
        pass
    return None


class Screener:
    """Wraps a saved model bundle (see src/train.py)."""

    def __init__(self, bundle_path: str | Path):
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle["model"]
        self.features: list[str] = self.bundle["features"]
        self.threshold: float = self.bundle["threshold"]
        self.medians: dict = self.bundle["medians"]

    # ------------------------------------------------------------------
    def required_missing(self, answers: dict) -> list[str]:
        """Required screening questions that are still unanswered."""
        return [f for f in self.features
                if SCREENING_FEATURES.get(f, {}).get("required") and answers.get(f) is None]

    def _row(self, answers: dict) -> pd.DataFrame:
        values = {}
        for f in self.features:
            v = answers.get(f)
            if isinstance(v, bool):
                v = float(v)
            values[f] = np.nan if v is None else float(v)
        return pd.DataFrame([values], columns=self.features)

    def _proba(self, row: pd.DataFrame) -> float:
        return float(self.model.predict_proba(row)[0, 1])

    # ------------------------------------------------------------------
    def predict(self, answers: dict, top_k: int = 5) -> dict:
        answers = dict(answers)
        if answers.get("bmi") is None:
            answers["bmi"] = compute_bmi(answers.get("height_cm"), answers.get("weight_kg"))

        row = self._row(answers)
        risk = self._proba(row)

        # Per-person explanation: how much does each answer move the score compared
        # with a "typical" value (the training median)? Measured in log-odds so that
        # effects stay visible even when the score is close to 0 or 1.
        def logit(p: float) -> float:
            p = min(max(p, 1e-6), 1 - 1e-6)
            return float(np.log(p / (1 - p)))

        contributions = []
        for f in self.features:
            if pd.isna(row.at[0, f]):
                continue
            alt = row.copy()
            alt.at[0, f] = self.medians.get(f, np.nan)
            delta = logit(risk) - logit(self._proba(alt))
            if abs(delta) >= 0.05:
                contributions.append({
                    "feature": f,
                    "label": FEATURE_LABELS.get(f, f),
                    "value": row.at[0, f],
                    "effect": round(delta, 3),  # + raises the score, - lowers it (log-odds)
                })
        contributions.sort(key=lambda d: abs(d["effect"]), reverse=True)

        if risk >= self.threshold:
            band = "higher"
            band_text = "Several of your answers match patterns seen in people diagnosed with PMOS (formerly PCOS)."
        elif risk >= self.threshold / 2:
            band = "moderate"
            band_text = "Some of your answers match patterns seen in people diagnosed with PMOS (formerly PCOS)."
        else:
            band = "lower"
            band_text = "Your answers match fewer of the patterns seen in people diagnosed with PMOS (formerly PCOS)."

        assumed = [FEATURE_LABELS.get(f, f) for f in self.features if pd.isna(row.at[0, f])]

        # Flag answers outside what the model has seen (results less reliable there)
        notes = []
        for f, (lo, hi) in self.bundle.get("ranges", {}).items():
            v = row.at[0, f] if f in row.columns else np.nan
            if f in ("age", "bmi") and not pd.isna(v) and not (lo <= v <= hi):
                notes.append(f"The model was trained on people with {FEATURE_LABELS.get(f, f)} "
                             f"between {lo:.0f} and {hi:.0f}, so this result is less reliable for you.")
        return {
            "risk": round(risk, 3),
            "band": band,
            "band_text": band_text,
            "threshold": self.threshold,
            "flag_for_gp": risk >= self.threshold,
            "top_factors": contributions[:top_k],
            "assumed_typical": assumed,
            "reliability_notes": notes,
            "model_name": self.bundle.get("model_name"),
            "metrics": self.bundle.get("cv_metrics_at_threshold", {}),
            "synthetic_model": self.bundle.get("meta", {}).get("synthetic", False),
        }
