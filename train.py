"""Train and calibrate the credit default model.

Reproduces the final model from the raw Kaggle CSVs in one command:

    python train.py --data-dir data --out-dir artifacts

Writes the fitted calibrated model, a metrics file and a reliability diagram.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import build_features

RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIBRATION_SIZE = 0.25  # fraction of the remaining training data

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

log = logging.getLogger("train")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix and cast object columns to category dtype."""
    X, y = build_features(str(data_dir))
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("category")
    log.info("features: %d rows x %d columns, base rate %.4f", *X.shape, y.mean())
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Three-way split: train, calibration, test.

    The calibration set is held out from training so the calibration mapping is
    fitted on scores the model has never seen.
    """
    X_fit, X_test, y_fit, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_fit, y_fit, test_size=CALIBRATION_SIZE, stratify=y_fit,
        random_state=RANDOM_STATE,
    )
    log.info("split: train=%d calibration=%d test=%d",
             len(X_train), len(X_cal), len(X_test))
    return X_train, y_train, X_cal, y_cal, X_test, y_test


def fit_model(X_train, y_train, X_cal, y_cal):
    """Fit LightGBM, then fit an isotonic calibration map on the held-out set."""
    log.info("fitting LightGBM on %d rows", len(X_train))
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    log.info("fitting isotonic calibration on %d rows", len(X_cal))
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_cal, y_cal)
    return model, calibrated


def evaluate(model, calibrated, X_test, y_test) -> dict:
    """Score both the raw and the calibrated model on the test set."""
    raw = model.predict_proba(X_test)[:, 1]
    cal = calibrated.predict_proba(X_test)[:, 1]
    base_rate = float(y_test.mean())

    metrics = {
        "n_test": int(len(y_test)),
        "base_rate": base_rate,
        "auc_roc": float(roc_auc_score(y_test, cal)),
        "auc_pr": float(average_precision_score(y_test, cal)),
        "brier_uncalibrated": float(brier_score_loss(y_test, raw)),
        "brier_calibrated": float(brier_score_loss(y_test, cal)),
        "brier_constant_baseline": float(
            brier_score_loss(y_test, np.full_like(cal, base_rate))
        ),
    }
    return metrics, raw, cal


def plot_reliability(y_test, raw, cal, path: Path) -> None:
    """Reliability diagram: predicted probability against observed frequency."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for scores, label in ((raw, "uncalibrated"), (cal, "isotonic")):
        prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=20)
        ax.plot(prob_pred, prob_true, marker="o", ms=4, label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="directory holding the unzipped Kaggle CSVs")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"),
                        help="where to write the model, metrics and figure")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data_dir)
    X_train, y_train, X_cal, y_cal, X_test, y_test = split(X, y)
    model, calibrated = fit_model(X_train, y_train, X_cal, y_cal)
    metrics, raw, cal = evaluate(model, calibrated, X_test, y_test)

    joblib.dump(calibrated, args.out_dir / "model.joblib")
    log.info("wrote %s", args.out_dir / "model.joblib")

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote %s", args.out_dir / "metrics.json")

    plot_reliability(y_test, raw, cal, args.out_dir / "calibration.png")

    print()
    print(f"  AUC-ROC              {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR               {metrics['auc_pr']:.4f}"
          f"   (base rate {metrics['base_rate']:.4f})")
    print(f"  Brier uncalibrated   {metrics['brier_uncalibrated']:.4f}")
    print(f"  Brier calibrated     {metrics['brier_calibrated']:.4f}")
    print(f"  Brier constant       {metrics['brier_constant_baseline']:.4f}")
    print()


if __name__ == "__main__":
    main()