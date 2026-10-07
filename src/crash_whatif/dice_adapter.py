"""DiCE adapter (extra ``dice``). It gives the same ``generate`` interface and result as
``RandomSearchCF``, so the benchmark can compare the two generators.

``dice_ml`` is imported only when the class is made, so the core package works without it.
"""
from __future__ import annotations

import importlib

import pandas as pd

from .counterfactuals import CFResult, FeatureSpace, to_frame
from .models import FittedModel
from .schema import FLAGS, NUMERIC

_OUTCOME = "_severe"


class DiceCF:
    def __init__(self, model: FittedModel, space: FeatureSpace, train: pd.DataFrame, y_train, seed: int = 42):
        dice_ml = importlib.import_module("dice_ml")
        columns = list(model.columns)
        frame = to_frame(r for _, r in train[columns].iterrows())
        frame[_OUTCOME] = list(y_train)
        continuous = [c for c in NUMERIC + FLAGS if c in columns]
        # DiCE refuses missing values: queries get the training median or mode
        self.fill = {c: (frame[c].median() if c in continuous else frame[c].mode().iloc[0]) for c in columns}
        data = dice_ml.Data(dataframe=frame.dropna(), continuous_features=continuous, outcome_name=_OUTCOME)
        wrapped = dice_ml.Model(model=model.pipeline, backend="sklearn")
        self.explainer = dice_ml.Dice(data, wrapped, method="random")
        self.model = model
        self.space = space
        self.seed = seed

    def generate(self, query: pd.Series, n_cfs: int = 3, desired: int | None = None, seed: int | None = None) -> CFResult:
        frame = to_frame([query])[list(self.model.columns)].fillna(self.fill)
        if desired is None:
            desired = 1 - int(self.model.predict(frame)[0])
        permitted = {**{c: list(r) for c, r in self.space.ranges.items()}, **self.space.categories}
        try:
            result = self.explainer.generate_counterfactuals(
                frame,
                total_CFs=n_cfs,
                desired_class=int(desired),
                features_to_vary=list(self.space.actionable),
                permitted_range=permitted,
                random_seed=self.seed if seed is None else seed,
            )
        except Exception as exc:  # DiCE raises when it finds no counterfactual
            if type(exc).__name__ == "UserConfigValidationException":
                return CFResult(query, frame.iloc[0:0], desired, False)
            raise
        cfs = result.cf_examples_list[0].final_cfs_df
        if cfs is None or cfs.empty:
            return CFResult(query, frame.iloc[0:0], desired, False)
        cfs = to_frame(r for _, r in cfs.drop(columns=[_OUTCOME], errors="ignore").iterrows())
        return CFResult(query, cfs, desired, True)
