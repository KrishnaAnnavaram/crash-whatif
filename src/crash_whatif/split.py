"""Stratified, seeded train / validation / test split. It runs FIRST, before any fit step."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def stratified_split(
    df: pd.DataFrame, y: np.ndarray, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 42
) -> Split:
    """Split rows into train, validation and test parts with the class shares of ``y``."""
    if not 0 < val_frac < 0.5 or not 0 < test_frac < 0.5:
        raise ValueError("val_frac and test_frac must be in (0, 0.5)")
    idx = np.arange(len(df))
    rest, test_idx = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    rel_val = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(rest, test_size=rel_val, stratify=y[rest], random_state=seed)
    pick = lambda ix: df.iloc[np.sort(ix)].reset_index(drop=True)  # noqa: E731
    return Split(pick(train_idx), pick(val_idx), pick(test_idx))
