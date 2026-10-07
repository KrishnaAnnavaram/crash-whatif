"""Local explanation: counterfactuals ("what change makes the model predict not severe?").

``RandomSearchCF`` is a model-agnostic, seeded search. It changes only actionable
columns, keeps each value inside the permitted range of the training rows, prefers few
changes and small MAD-scaled distances, and removes each change that is not necessary.
``DiceCF`` in ``dice_adapter.py`` gives the same interface with the DiCE library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import FittedModel
from .schema import FLAGS, INTEGER, NOMINAL, NUMERIC


def to_frame(rows) -> pd.DataFrame:
    """Rows (Series) to a DataFrame with float numeric and flag columns."""
    df = pd.DataFrame([r for r in rows]).reset_index(drop=True)
    for col in NUMERIC + FLAGS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


@dataclass
class FeatureSpace:
    """Permitted values of each actionable column and the scale of each numeric column."""

    actionable: list[str]
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_train(cls, train: pd.DataFrame, actionable, permitted: dict | None = None) -> "FeatureSpace":
        actionable = list(actionable)
        unknown = [c for c in actionable if c not in NUMERIC + FLAGS + NOMINAL]
        if unknown:
            raise ValueError(f"actionable columns are not model features: {unknown}")
        space = cls(actionable=actionable)
        for col in NUMERIC + FLAGS:
            values = train[col].dropna().to_numpy(float)
            mad = float(np.median(np.abs(values - np.median(values)))) if values.size else 0.0
            space.scale[col] = mad if mad > 0 else (float(values.std()) if values.size and values.std() > 0 else 1.0)
            if col in actionable:
                space.ranges[col] = (float(values.min()), float(values.max()))
        for col in NOMINAL:
            if col in actionable:
                space.categories[col] = sorted(train[col].dropna().unique().tolist())
        for col, allowed in (permitted or {}).items():
            if col in space.ranges:
                space.ranges[col] = (float(allowed[0]), float(allowed[1]))
            elif col in space.categories:
                space.categories[col] = list(allowed)
        return space

    def distance(self, query: pd.Series, cf: pd.Series) -> float:
        """MAD-scaled L1 distance over numeric columns plus 1 for each changed nominal column."""
        total = 0.0
        for col in self.scale:
            a, b = query.get(col), cf.get(col)
            if pd.isna(a) and pd.isna(b):
                continue
            total += 1.0 if pd.isna(a) or pd.isna(b) else abs(float(a) - float(b)) / self.scale[col]
        for col in NOMINAL:
            a, b = query.get(col), cf.get(col)
            if not (pd.isna(a) and pd.isna(b)) and a != b:
                total += 1.0
        return total

    def changed(self, query: pd.Series, cf: pd.Series) -> list[str]:
        out = []
        for col in self.actionable:
            a, b = query.get(col), cf.get(col)
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b) or a != b:
                out.append(col)
        return out

    def sample(self, col: str, rng: np.random.Generator, size: int) -> np.ndarray:
        if col in self.categories:
            return rng.choice(np.array(self.categories[col], dtype=object), size=size)
        low, high = self.ranges[col]
        if col in INTEGER:
            return rng.integers(int(round(low)), int(round(high)) + 1, size=size).astype(float)
        return rng.uniform(low, high, size=size)


@dataclass
class CFResult:
    query: pd.Series
    counterfactuals: pd.DataFrame  # one row for each counterfactual
    desired: int
    found: bool


class RandomSearchCF:
    def __init__(self, model: FittedModel, space: FeatureSpace, n_samples: int = 300, max_changes: int = 3, seed: int = 42):
        self.model = model
        self.space = space
        self.n_samples = n_samples
        self.max_changes = max(1, min(max_changes, len(space.actionable)))
        self.seed = seed

    def _sparsify(self, query: pd.Series, cf: pd.Series, desired: int) -> pd.Series:
        """Put back the query value of each changed column if the prediction stays the desired class."""
        current = cf.copy()
        for col in self.space.changed(query, cf):
            trial = current.copy()
            trial[col] = query[col]
            if self.model.predict(to_frame([trial]))[0] == desired:
                current = trial
        return current

    def generate(self, query: pd.Series, n_cfs: int = 3, desired: int | None = None, seed: int | None = None) -> CFResult:
        frame = to_frame([query])
        if desired is None:
            desired = 1 - int(self.model.predict(frame)[0])
        rng = np.random.default_rng(self.seed if seed is None else seed)
        valid: list[pd.Series] = []
        actionable = self.space.actionable
        for k in range(1, self.max_changes + 1):
            cand = pd.concat([frame] * self.n_samples, ignore_index=True)
            for col in actionable:
                if col in self.space.categories:
                    cand[col] = cand[col].astype(object)
            picks = np.array([rng.choice(len(actionable), size=k, replace=False) for _ in range(self.n_samples)])
            for j, col in enumerate(actionable):
                rows = np.flatnonzero((picks == j).any(axis=1))
                if rows.size:
                    cand.loc[rows, col] = self.space.sample(col, rng, rows.size)
            hits = cand[self.model.predict(cand) == desired]
            valid.extend(row for _, row in hits.iterrows())
            if len(valid) >= n_cfs:
                break
        if not valid:
            return CFResult(query, frame.iloc[0:0], desired, False)
        valid.sort(key=lambda cf: self.space.distance(query, cf))
        # a single change cannot get smaller: only counterfactuals with 2 or more changes are sparsified
        cleaned = [
            self._sparsify(query, cf, desired) if len(self.space.changed(query, cf)) > 1 else cf
            for cf in valid[: 5 * n_cfs]
        ]
        scored = sorted(cleaned, key=lambda cf: (len(self.space.changed(query, cf)), self.space.distance(query, cf)))
        chosen, seen = [], set()
        for cf in scored:
            key = tuple((c, str(cf[c])) for c in self.space.changed(query, cf))
            if key and key not in seen:
                seen.add(key)
                chosen.append(cf)
            if len(chosen) == n_cfs:
                break
        return CFResult(query, to_frame(chosen), desired, bool(chosen))
