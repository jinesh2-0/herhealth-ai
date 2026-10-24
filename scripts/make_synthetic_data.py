"""Create a SYNTHETIC dataset with the same column names as the Kaggle PCOS file.

Purpose: test the pipeline end to end before (or without) the real data.
The numbers are invented. Never report results from this file.

Usage:  python scripts/make_synthetic_data.py  ->  data/synthetic/pcos_synthetic.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 541, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.33, n)

    def yes(p_pos: float, p_neg: float) -> np.ndarray:
        return rng.binomial(1, np.where(y == 1, p_pos, p_neg))

    age = np.clip(rng.normal(np.where(y == 1, 29, 31), 5, n), 18, 48).round()
    height = rng.normal(156, 6, n).round(1)
    bmi = np.clip(rng.normal(np.where(y == 1, 25.5, 23.5), 4, n), 15, 40)
    weight = (bmi * (height / 100) ** 2).round(1)

    df = pd.DataFrame({
        "Sl. No": np.arange(1, n + 1),
        "Patient File No.": np.arange(10001, 10001 + n),
        "PCOS (Y/N)": y,
        " Age (yrs)": age,
        "Weight (Kg)": weight,
        "Height(Cm) ": height,
        "BMI": bmi.round(1),
        "Cycle(R/I)": np.where(yes(0.6, 0.15) == 1, 4, 2),
        "Cycle length(days)": np.clip(rng.normal(5, 1.3, n), 1, 10).round(),
        "Weight gain(Y/N)": yes(0.6, 0.25),
        "hair growth(Y/N)": yes(0.55, 0.15),
        "Skin darkening (Y/N)": yes(0.5, 0.15),
        "Hair loss(Y/N)": yes(0.5, 0.35),
        "Pimples(Y/N)": yes(0.6, 0.35),
        "Fast food (Y/N)": yes(0.65, 0.4),
        "Reg.Exercise(Y/N)": yes(0.2, 0.25),
        "FSH(mIU/mL)": rng.lognormal(np.where(y == 1, 1.5, 1.7), 0.4, n).round(2),
        "LH(mIU/mL)": rng.lognormal(np.where(y == 1, 1.6, 1.2), 0.5, n).round(2),
        "AMH(ng/mL)": rng.lognormal(np.where(y == 1, 1.9, 1.1), 0.5, n).round(2),
        "TSH (mIU/L)": rng.lognormal(0.9, 0.4, n).round(2),
        "PRL(ng/mL)": rng.lognormal(3.0, 0.4, n).round(2),
        "Vit D3 (ng/mL)": rng.lognormal(3.4, 0.5, n).round(1),
        "PRG(ng/mL)": rng.lognormal(-1.2, 0.5, n).round(2),
        "RBS(mg/dl)": rng.normal(100, 15, n).round(),
        "Hb(g/dl)": rng.normal(11.2, 0.9, n).round(1),
        "Waist:Hip Ratio": rng.normal(np.where(y == 1, 0.9, 0.88), 0.04, n).round(2),
        "Follicle No. (L)": rng.poisson(np.where(y == 1, 12, 5)),
        "Follicle No. (R)": rng.poisson(np.where(y == 1, 13, 6)),
        "Avg. F size (L) (mm)": rng.normal(15, 3.5, n).round(1),
        "Avg. F size (R) (mm)": rng.normal(15, 3.5, n).round(1),
        "Endometrium (mm)": rng.normal(8.5, 2, n).round(1),
    })
    df["FSH/LH"] = (df["FSH(mIU/mL)"] / df["LH(mIU/mL)"]).round(2)

    # Imitate the messiness of the real file: a few blanks and a stray text value
    df.loc[rng.choice(n, 5, replace=False), "Fast food (Y/N)"] = np.nan
    df["AMH(ng/mL)"] = df["AMH(ng/mL)"].astype(object)
    df.loc[3, "AMH(ng/mL)"] = "a"
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "pcos_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"Wrote {out} (SYNTHETIC - for testing only)")
