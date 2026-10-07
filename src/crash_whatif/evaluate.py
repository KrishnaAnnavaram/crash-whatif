"""Metrics on the real test rows only. No test row is resampled or synthetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

from .models import FittedModel, build_pipeline
from .schema import FEATURES, SEVERITY_LEVELS


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Mean |observed rate - mean probability| over equal-width bins, weighted by bin size."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def bootstrap_auc(y, p, n_boot: int = 300, seed: int = 42) -> tuple[float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        ix = rng.integers(0, y.size, y.size)
        if len(np.unique(y[ix])) == 2:
            stats.append(roc_auc_score(y[ix], p[ix]))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def binary_metrics(model: FittedModel, X: pd.DataFrame, y, n_boot: int = 300, seed: int = 42) -> dict:
    y = np.asarray(y)
    p = model.proba(X)
    pred = (p >= model.threshold).astype(int)
    has_both = len(np.unique(y)) == 2
    return {
        "model": model.name,
        "threshold": model.threshold,
        "roc_auc": float(roc_auc_score(y, p)) if has_both else float("nan"),
        "roc_auc_ci95": bootstrap_auc(y, p, n_boot=n_boot, seed=seed) if has_both else (float("nan"),) * 2,
        "pr_auc_severe": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "recall_severe": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_not_severe": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


def three_class_report(train: pd.DataFrame, y_train, test: pd.DataFrame, y_test, seed: int = 42) -> dict:
    """Severity with three classes (minor, moderate, severe) and class weights.

    The minor class is rare. This report shows its recall instead of a merge in silence.
    """
    out = {"class_counts_train": {lvl: int((np.asarray(y_train) == i).sum()) for i, lvl in enumerate(SEVERITY_LEVELS)}}
    for name in ("logreg", "random_forest"):
        pipe = build_pipeline(name, resample="class_weight", seed=seed).fit(train[FEATURES], y_train)
        pred = pipe.predict(test[FEATURES])
        recalls = recall_score(y_test, pred, labels=[0, 1, 2], average=None, zero_division=0)
        out[name] = {
            "f1_macro": float(f1_score(y_test, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "recall": {lvl: float(r) for lvl, r in zip(SEVERITY_LEVELS, recalls)},
        }
    return out
