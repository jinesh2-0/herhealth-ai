"""Load and clean the public Kaggle PCOS dataset (or a synthetic file with the same columns)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMN_MAP, TARGET, CLINICAL_FEATURE_LIST


def _normalise(name: str) -> str:
    """'  Age (yrs)' -> 'ageyrs', 'Height(Cm) ' -> 'heightcm'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle file. Supports the original .xlsx (sheet 'Full_new') or a .csv export."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for how to download it."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet = "Full_new" if "Full_new" in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=sheet)
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, coerce types and derive the features the models use."""
    rename = {}
    for col in raw.columns:
        key = _normalise(col)
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = raw.rename(columns=rename)
    df = df[[c for c in df.columns if c in set(COLUMN_MAP.values())]].copy()

    if TARGET not in df.columns:
        raise ValueError("Could not find the 'PCOS (Y/N)' column in the dataset.")

    # Everything numeric; stray text values (e.g. 'a' in AMH) become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Cycle: 2 = regular, 4 (and occasional 5) = irregular in the source data
    if "cycle_ri" in df.columns:
        df["cycle_irregular"] = np.where(df["cycle_ri"].isna(), np.nan,
                                         (df["cycle_ri"] >= 4).astype(float))
        df = df.drop(columns=["cycle_ri"])

    # Fill missing BMI from height and weight where possible
    if {"weight_kg", "height_cm"} <= set(df.columns):
        computed = df["weight_kg"] / (df["height_cm"] / 100) ** 2
        df["bmi"] = df["bmi"].fillna(computed) if "bmi" in df.columns else computed

    # Remove physically implausible values (data-entry errors) by setting them to NaN
    bounds = {"age": (12, 60), "bmi": (10, 70), "period_length": (0, 15)}
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    keep = [c for c in CLINICAL_FEATURE_LIST if c in df.columns] + [TARGET]
    return df[keep].reset_index(drop=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return clean(read_raw(path))
