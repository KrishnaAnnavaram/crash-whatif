"""Model pipelines. Imputation, encoding, scaling and resampling are steps of one
pipeline, so ``fit`` with the training rows fits all of them on those rows only.

Nominal columns are one-hot encoded. Resampling (if any) runs on imputed columns
before the encoding, and ``SMOTENC`` treats the nominal columns as categories, so a
synthetic row never gets a fractional category.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from .schema import FEATURES, FLAGS, NOMINAL, NUMERIC

MODEL_NAMES = ("majority", "make_only", "logreg", "linear_svm", "random_forest", "gradient_boosting")


def _split_columns(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    return (
        [c for c in NUMERIC if c in columns],
        [c for c in FLAGS if c in columns],
        [c for c in NOMINAL if c in columns],
    )


def build_imputer(columns: list[str]) -> ColumnTransformer:
    numeric, flags, nominal = _split_columns(columns)
    parts = []
    if numeric:
        parts.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if flags:
        parts.append(("flags", SimpleImputer(strategy="most_frequent"), flags))
    if nominal:
        parts.append(("nominal", SimpleImputer(strategy="constant", fill_value="missing"), nominal))
    return ColumnTransformer(parts, verbose_feature_names_out=False).set_output(transform="pandas")


def build_encoder(columns: list[str]) -> ColumnTransformer:
    numeric, flags, nominal = _split_columns(columns)
    parts = []
    if numeric:
        parts.append(("numeric", StandardScaler(), numeric))
    if flags:
        parts.append(("flags", "passthrough", flags))
    if nominal:
        parts.append(("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build_classifier(name: str, balanced: bool, seed: int):
    weight = "balanced" if balanced else None
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name in {"logreg", "make_only"}:
        return LogisticRegression(max_iter=2000, C=1.0, class_weight=weight)
    if name == "linear_svm":
        svm = LinearSVC(C=0.5, class_weight=weight, dual="auto", max_iter=5000, random_state=seed)
        return CalibratedClassifierCV(svm, cv=3, method="sigmoid")  # gives probabilities
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample" if balanced else None,
            random_state=seed, n_jobs=1,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight=weight, random_state=seed)
    raise ValueError(f"unknown model {name!r}, use one of {MODEL_NAMES}")


def build_pipeline(name: str, resample: str = "class_weight", seed: int = 42, columns: list[str] | None = None):
    """Make the full pipeline of one model. ``columns`` restricts the input columns."""
    if columns is None:
        columns = ["make"] if name == "make_only" else list(FEATURES)
    clf = build_classifier(name, balanced=(resample == "class_weight"), seed=seed)
    steps = [("impute", build_imputer(columns)), ("encode", build_encoder(columns)), ("model", clf)]
    if resample in {"none", "class_weight"} or name == "majority":
        return Pipeline(steps)
    imblearn_pipeline = importlib.import_module("imblearn.pipeline")  # optional extra "imbalance"
    if resample == "oversample":
        sampler = importlib.import_module("imblearn.over_sampling").RandomOverSampler(random_state=seed)
    elif resample == "smotenc":
        nominal = [c for c in NOMINAL if c in columns]
        over = importlib.import_module("imblearn.over_sampling")
        sampler = over.SMOTENC(categorical_features=nominal, random_state=seed) if nominal and len(nominal) < len(columns) else over.RandomOverSampler(random_state=seed)
    else:
        raise ValueError(f"unknown resample method {resample!r}")
    return imblearn_pipeline.Pipeline([steps[0], ("resample", sampler), steps[1], steps[2]])


@dataclass
class FittedModel:
    """A fitted pipeline and its decision threshold, chosen on the validation rows."""

    name: str
    pipeline: object
    threshold: float = 0.5
    columns: tuple[str, ...] = tuple(FEATURES)

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[list(self.columns)])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.proba(X) >= self.threshold).astype(int)


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold with the best balanced accuracy on the validation rows (grid 0.05 .. 0.95)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [balanced_accuracy_score(y_val, (p_val >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def fit_model(name: str, train: pd.DataFrame, y_train, val: pd.DataFrame, y_val, resample: str, seed: int) -> FittedModel:
    columns = ["make"] if name == "make_only" else list(FEATURES)
    pipe = build_pipeline(name, resample=resample, seed=seed, columns=columns)
    pipe.fit(train[columns], y_train)
    model = FittedModel(name, pipe, 0.5, tuple(columns))
    if name != "majority" and len(np.unique(y_val)) == 2:
        model.threshold = choose_threshold(np.asarray(y_val), model.proba(val))
    return model
