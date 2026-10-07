"""Global explanation: permutation importance with repeats, a seed and a clear sign.

The value of a column is the DROP of ROC AUC when the column is shuffled. A positive
value means that the model uses the column. Each input column is shuffled as one unit,
so all one-hot columns of a nominal column move together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .models import FittedModel


class _Scorer:
    """Adapter so that ``permutation_importance`` scores the full pipeline on raw columns."""

    def __init__(self, model: FittedModel) -> None:
        self.model = model

    def fit(self, X, y):  # pragma: no cover - never called
        return self

    def predict_proba(self, X):
        p = self.model.proba(X)
        return np.column_stack([1 - p, p])


def auc_drop(model: FittedModel, X: pd.DataFrame, y, n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    columns = list(model.columns)

    def scorer(est, X_, y_):
        return roc_auc_score(y_, est.predict_proba(X_)[:, 1])

    result = permutation_importance(
        _Scorer(model), X[columns], np.asarray(y), scoring=scorer, n_repeats=n_repeats, random_state=seed
    )
    table = pd.DataFrame(
        {"column": columns, "auc_drop": result.importances_mean, "std": result.importances_std}
    )
    return table.sort_values("auc_drop", ascending=False, ignore_index=True)
