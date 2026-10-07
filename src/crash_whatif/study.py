"""The full study: validate, split first, fit, evaluate, explain, audit and benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cf_metrics import Plausibility, query_metrics, summarize
from .config import Settings
from .counterfactuals import FeatureSpace, RandomSearchCF
from .evaluate import binary_metrics, three_class_report
from .importance import auc_drop
from .models import MODEL_NAMES, FittedModel, fit_model
from .schema import FEATURES, ValidationReport, make_target, validate
from .split import Split, stratified_split

CF_MODELS = ("logreg", "linear_svm", "random_forest", "gradient_boosting")


@dataclass
class Study:
    settings: Settings
    validation: ValidationReport
    split: Split
    y: dict[str, np.ndarray]
    models: dict[str, FittedModel] = field(default_factory=dict)


def prepare(raw: pd.DataFrame, settings: Settings) -> Study:
    clean, report = validate(raw)
    y_all = make_target(clean, "binary")
    split = stratified_split(clean, y_all, seed=settings.seed)
    y = {part: make_target(getattr(split, part), "binary") for part in ("train", "val", "test")}
    return Study(settings, report, split, y)


def fit_all(study: Study, names=MODEL_NAMES) -> Study:
    s = study.split
    for name in names:
        study.models[name] = fit_model(
            name, s.train, study.y["train"], s.val, study.y["val"], resample=study.settings.resample, seed=study.settings.seed
        )
    return study


def metrics_table(study: Study, n_boot: int = 300) -> list[dict]:
    return [
        binary_metrics(m, study.split.test, study.y["test"], n_boot=n_boot, seed=study.settings.seed)
        for m in study.models.values()
    ]


def importance_tables(study: Study, names=("logreg", "random_forest"), n_repeats: int = 10) -> dict[str, list[dict]]:
    out = {}
    for name in names:
        if name in study.models:
            table = auc_drop(study.models[name], study.split.test, study.y["test"], n_repeats=n_repeats, seed=study.settings.seed)
            out[name] = table.to_dict(orient="records")
    return out


def signal_audit(study: Study) -> dict:
    """How much does ``make`` alone explain, and how much do the other columns add?"""
    s, y, seed = study.split, study.y, study.settings.seed
    from .models import build_pipeline

    def auc_of(columns: list[str]) -> float:
        from sklearn.metrics import roc_auc_score

        pipe = build_pipeline("logreg", resample="class_weight", seed=seed, columns=columns).fit(s.train[columns], y["train"])
        return float(roc_auc_score(y["test"], pipe.predict_proba(s.test[columns])[:, 1]))

    make_only = auc_of(["make"])
    without_make = auc_of([c for c in FEATURES if c != "make"])
    all_columns = auc_of(list(FEATURES))
    crosstab = (
        pd.crosstab(s.train["make"], s.train["severity"], normalize="index").round(3).to_dict(orient="index")
    )
    return {
        "auc_make_only": make_only,
        "auc_without_make": without_make,
        "auc_all_columns": all_columns,
        "make_dominates": bool(make_only > 0.9 and without_make < 0.6),
        "severity_share_by_make_train": crosstab,
    }


def counterfactual_benchmark(study: Study, names=CF_MODELS, n_queries: int | None = None, n_cfs: int = 3) -> dict:
    """Counterfactuals for N real test rows that each model predicts as severe."""
    s, settings = study.split, study.settings
    n_queries = n_queries or settings.n_queries
    space = FeatureSpace.from_train(s.train, settings.actionable)
    plaus = Plausibility(s.train)
    out = {"actionable": list(space.actionable), "n_cfs": n_cfs}
    for name in names:
        model = study.models[name]
        severe_rows = s.test[model.predict(s.test) == 1]
        queries = severe_rows.sample(n=min(n_queries, len(severe_rows)), random_state=settings.seed)
        generator = RandomSearchCF(model, space, seed=settings.seed)
        per_query, counts = [], {}
        for i, (_, row) in enumerate(queries.iterrows()):
            result = generator.generate(row, n_cfs=n_cfs, desired=0, seed=settings.seed + i)
            per_query.append(query_metrics(result, model, space, plaus))
            for _, cf in result.counterfactuals.iterrows():
                for col in space.changed(row, cf):
                    counts[col] = counts.get(col, 0) + 1
        out[name] = summarize(per_query, seed=settings.seed)
        out[name]["changed_columns"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return out


def three_class(study: Study) -> dict:
    s = study.split
    return three_class_report(s.train, make_target(s.train, "three"), s.test, make_target(s.test, "three"), seed=study.settings.seed)
