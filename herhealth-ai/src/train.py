"""Train the PCOS screening (self-report) and clinical (with labs/ultrasound) models.

Usage (from the repo root):
    python -m src.train --data data/raw/PCOS_data_without_infertility.xlsx

Outputs:
    models/pcos_screening.joblib, models/pcos_clinical.joblib
    reports/metrics.json, reports/model_card.md, reports/figures/*.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import (  # noqa: E402
    CLINICAL_FEATURE_LIST,
    DEFAULT_TARGET_SENSITIVITY,
    FEATURE_LABELS,
    RANDOM_STATE,
    SCREENING_FEATURE_LIST,
    TARGET,
)
from .data import load_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def candidate_models(fast: bool = False) -> dict[str, Pipeline]:
    trees = 100 if fast else 400
    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=trees, min_samples_leaf=3, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "sensitivity": _safe_div(tp, tp + fn),   # of real cases, how many we flag
        "specificity": _safe_div(tn, tn + fp),   # of non-cases, how many we clear
        "ppv": _safe_div(tp, tp + fp),           # when we flag, how often it's real
        "npv": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(tp + tn, len(y)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def pick_threshold(y: np.ndarray, p: np.ndarray, target_sensitivity: float) -> float:
    """Highest probability threshold that still catches >= target_sensitivity of cases."""
    _, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target_sensitivity)[0]
    if len(ok) == 0 or not np.isfinite(thr[ok[0]]):
        return 0.5
    return float(np.clip(thr[ok[0]], 0.0, 1.0))


def subgroup_report(X: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float) -> list[dict]:
    """Bias check: does performance hold across age and BMI groups?"""
    rows = []
    groupings = {
        "age": ([0, 25, 30, 35, 200], ["<25", "25-29", "30-34", "35+"]),
        "bmi": ([0, 18.5, 25, 30, 200], ["<18.5", "18.5-24.9", "25-29.9", "30+"]),
    }
    for col, (bins, labels) in groupings.items():
        if col not in X.columns:
            continue
        cats = pd.cut(X[col], bins=bins, labels=labels, right=False)
        for label in labels:
            mask = (cats == label).to_numpy()
            n, pos = int(mask.sum()), int(y[mask].sum())
            row = {"variable": col, "group": label, "n": n, "positives": pos,
                   "auc": None, "sensitivity": None, "specificity": None}
            if n >= 15 and 0 < pos < n:
                row["auc"] = float(roc_auc_score(y[mask], p[mask]))
                m = metrics_at(y[mask], p[mask], threshold)
                row["sensitivity"], row["specificity"] = m["sensitivity"], m["specificity"]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Training one feature set
# ---------------------------------------------------------------------------
def train_feature_set(name: str, df: pd.DataFrame, wanted: list[str],
                      target_sensitivity: float, fast: bool) -> dict:
    features = [f for f in wanted if f in df.columns]
    dropped = [f for f in wanted if f not in df.columns]
    X, y = df[features], df[TARGET].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {}
    for model_name, model in candidate_models(fast).items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        candidates[model_name] = {"auc": float(roc_auc_score(y, proba)), "proba": proba}

    best_name = max(candidates, key=lambda k: candidates[k]["auc"])
    best_proba = candidates[best_name]["proba"]
    threshold = pick_threshold(y, best_proba, target_sensitivity)

    final_model = candidate_models(fast)[best_name].fit(X, y)

    imp = permutation_importance(final_model, X, y, scoring="roc_auc",
                                 n_repeats=5 if fast else 15, random_state=RANDOM_STATE)
    importance = sorted(
        ({"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": float(m)}
         for f, m in zip(features, imp.importances_mean)),
        key=lambda d: d["importance"], reverse=True,
    )

    return {
        "name": name,
        "model": final_model,
        "model_name": best_name,
        "features": features,
        "dropped_features": dropped,
        "threshold": threshold,
        "medians": X.median(numeric_only=True).to_dict(),
        "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in features},
        "cv_auc_by_model": {k: v["auc"] for k, v in candidates.items()},
        "cv_metrics_at_threshold": metrics_at(y, best_proba, threshold),
        "cv_metrics_at_0_5": metrics_at(y, best_proba, 0.5),
        "subgroups": subgroup_report(X, y, best_proba, threshold),
        "importance": importance,
        "_candidates": candidates,  # used for plots only, not saved
        "_y": y,
    }


# ---------------------------------------------------------------------------
# Figures and reports
# ---------------------------------------------------------------------------
def save_figures(results: dict[str, dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC curves for every candidate model in both feature sets
    plt.figure(figsize=(6, 5))
    for set_name, res in results.items():
        for model_name, c in res["_candidates"].items():
            fpr, tpr, _ = roc_curve(res["_y"], c["proba"])
            plt.plot(fpr, tpr, label=f"{set_name} / {model_name} (AUC {c['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("Cross-validated ROC curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curves.png", dpi=150)
    plt.close()

    # Confusion matrix + importance for the screening model
    res = results["screening"]
    m = res["cv_metrics_at_threshold"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Purples")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Predicted no", "Predicted yes"])
    plt.yticks([0, 1], ["Actual no", "Actual yes"])
    plt.title(f"Screening model @ threshold {m['threshold']:.2f}")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_screening.png", dpi=150)
    plt.close()

    top = res["importance"][:10][::-1]
    plt.figure(figsize=(6, 4))
    plt.barh([d["label"] for d in top], [d["importance"] for d in top], color="#7b4fa0")
    plt.xlabel("Drop in AUC when shuffled (permutation importance)")
    plt.title("What drives the screening model")
    plt.tight_layout()
    plt.savefig(fig_dir / "importance_screening.png", dpi=150)
    plt.close()


def _fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def write_model_card(results: dict[str, dict], meta: dict, path: Path) -> None:
    lines = ["# Model card (auto-generated by `src/train.py`)", ""]
    if meta["synthetic"]:
        lines += ["> **WARNING: trained on SYNTHETIC test data. These numbers mean nothing "
                  "clinically. Re-run on the real Kaggle dataset before reporting results.**", ""]
    lines += [
        f"- Data file: `{meta['data_file']}`",
        f"- Patients: **{meta['n']}**, with PCOS: **{meta['positives']}** "
        f"({meta['positives'] / meta['n']:.0%})",
        f"- Trained: {meta['trained_at']}",
        f"- Evaluation: 5-fold stratified cross-validation; threshold chosen to reach "
        f"sensitivity >= {meta['target_sensitivity']:.0%}",
        "",
    ]
    for set_name, res in results.items():
        m, m5 = res["cv_metrics_at_threshold"], res["cv_metrics_at_0_5"]
        lines += [
            f"## {set_name.title()} model",
            f"- Features ({len(res['features'])}): " + ", ".join(res["features"]),
            "- Cross-validated AUC by model: " + ", ".join(
                f"{k} {v:.3f}" for k, v in res["cv_auc_by_model"].items()),
            f"- Chosen: **{res['model_name']}**",
            "",
            "| Threshold | Sensitivity | Specificity | PPV | NPV | Accuracy |",
            "|---|---|---|---|---|---|",
            f"| {m['threshold']:.2f} (screening) | {_fmt(m['sensitivity'])} | {_fmt(m['specificity'])} "
            f"| {_fmt(m['ppv'])} | {_fmt(m['npv'])} | {_fmt(m['accuracy'])} |",
            f"| 0.50 | {_fmt(m5['sensitivity'])} | {_fmt(m5['specificity'])} | {_fmt(m5['ppv'])} "
            f"| {_fmt(m5['npv'])} | {_fmt(m5['accuracy'])} |",
            "",
            "Top factors: " + ", ".join(d["label"] for d in res["importance"][:6]),
            "",
            "**Subgroup check (bias):**",
            "",
            "| Variable | Group | n | PCOS cases | AUC | Sensitivity | Specificity |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in res["subgroups"]:
            lines.append(f"| {r['variable']} | {r['group']} | {r['n']} | {r['positives']} | "
                         f"{_fmt(r['auc'])} | {_fmt(r['sensitivity'])} | {_fmt(r['specificity'])} |")
        lines.append("")
    lines += [
        "## Limitations",
        "- Small dataset from a single region (hospitals in Kerala, India, per the dataset "
        "description): results may not transfer to other populations or ethnicities.",
        "- Self-reported symptoms are subjective; the model estimates similarity to diagnosed "
        "cases, it does not diagnose.",
        "- Subgroups with fewer than 15 people are not scored (too small to be meaningful).",
        "- The threshold is chosen on the same cross-validated predictions it is evaluated on, so "
        "the reported sensitivity/specificity are slightly optimistic (nested CV or a held-out "
        "test set would fix this).",
        "- The training data has a much higher PCOS rate than the general population, so scores "
        "are not calibrated probabilities and PPV in the real world would be lower.",
        "- Ages and BMIs outside the training range are flagged in the app as less reliable.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Path to the Kaggle .xlsx/.csv file")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--target-sensitivity", type=float, default=DEFAULT_TARGET_SENSITIVITY)
    ap.add_argument("--fast", action="store_true", help="Fewer trees/repeats (for tests)")
    args = ap.parse_args(argv)

    df = load_dataset(args.data)
    synthetic = "synthetic" in Path(args.data).name.lower()
    meta = {
        "data_file": Path(args.data).name,
        "n": int(len(df)),
        "positives": int(df[TARGET].sum()),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "target_sensitivity": args.target_sensitivity,
        "synthetic": synthetic,
    }

    results = {
        "screening": train_feature_set("screening", df, SCREENING_FEATURE_LIST,
                                       args.target_sensitivity, args.fast),
        "clinical": train_feature_set("clinical", df, CLINICAL_FEATURE_LIST,
                                      args.target_sensitivity, args.fast),
    }

    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = {"meta": meta}
    for set_name, res in results.items():
        bundle = {k: v for k, v in res.items() if not k.startswith("_")}
        bundle["meta"] = meta
        joblib.dump(bundle, models_dir / f"pcos_{set_name}.joblib")
        summary[set_name] = {k: v for k, v in bundle.items() if k not in {"model", "meta"}}

    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float),
                                              encoding="utf-8")
    save_figures(results, reports_dir / "figures")
    write_model_card(results, meta, reports_dir / "model_card.md")

    for set_name, res in results.items():
        m = res["cv_metrics_at_threshold"]
        print(f"[{set_name}] best={res['model_name']} "
              f"AUC={res['cv_auc_by_model'][res['model_name']]:.3f} "
              f"sens={m['sensitivity']:.2f} spec={m['specificity']:.2f} @t={m['threshold']:.2f}")
    if synthetic:
        print("WARNING: trained on synthetic data - for testing the pipeline only.")
    return summary


if __name__ == "__main__":
    main()
