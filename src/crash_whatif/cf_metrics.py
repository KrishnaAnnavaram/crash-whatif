"""Counterfactual quality over MANY real test rows, with bootstrap intervals.

* validity: share of counterfactuals that the model predicts as the desired class
* proximity: mean MAD-scaled distance from the query (lower is better)
* sparsity: mean number of changed columns (lower is better)
* diversity: mean pairwise distance between the counterfactuals of one query
* plausibility: distance to the nearest real training row, divided by the median
  nearest-neighbour distance between training rows (near 1 or lower: like real data)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from .counterfactuals import CFResult, FeatureSpace
from .models import FittedModel, build_encoder, build_imputer
from .schema import FEATURES


class Plausibility:
    """kNN distance to the training rows in the scaled, one-hot space (fit on train only)."""

    def __init__(self, train: pd.DataFrame) -> None:
        self.encoder = Pipeline([("impute", build_imputer(FEATURES)), ("encode", build_encoder(FEATURES))])
        Z = self.encoder.fit_transform(train[FEATURES])
        self.nn = NearestNeighbors(n_neighbors=2).fit(Z)
        d, _ = self.nn.kneighbors(Z)
        self.reference = float(np.median(d[:, 1])) or 1.0

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        d, _ = self.nn.kneighbors(self.encoder.transform(rows[FEATURES]), n_neighbors=1)
        return d[:, 0] / self.reference


def query_metrics(result: CFResult, model: FittedModel, space: FeatureSpace, plaus: Plausibility) -> dict:
    cfs = result.counterfactuals
    if cfs.empty:
        return {"found": 0.0, "validity": 0.0}
    rows = [r for _, r in cfs.iterrows()]
    pairs = list(combinations(rows, 2))
    return {
        "found": 1.0,
        "validity": float(np.mean(model.predict(cfs) == result.desired)),
        "proximity": float(np.mean([space.distance(result.query, r) for r in rows])),
        "sparsity": float(np.mean([len(space.changed(result.query, r)) for r in rows])),
        "diversity": float(np.mean([space.distance(a, b) for a, b in pairs])) if pairs else 0.0,
        "plausibility": float(np.mean(plaus.score(cfs))),
    }


def summarize(per_query: list[dict], n_boot: int = 500, seed: int = 42) -> dict:
    """Mean and 95 % bootstrap interval of each metric over the queries."""
    frame = pd.DataFrame(per_query)
    rng = np.random.default_rng(seed)
    out = {"queries": int(len(frame))}
    for col in ["found", "validity", "proximity", "sparsity", "diversity", "plausibility"]:
        if col not in frame:
            continue
        values = frame[col].dropna().to_numpy(float)
        if values.size == 0:
            continue
        boots = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
        out[col] = {
            "mean": float(values.mean()),
            "ci95": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        }
    return out
