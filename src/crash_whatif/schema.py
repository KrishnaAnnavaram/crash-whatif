"""Crash table schema: column names, types, ranges and the clean-up of each column.

Each safety flag (``abs``, ``esc``, ``tcs``, ``tpms``) is read from its own source
column with one explicit mapping. No flag is copied from another flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "severity"
SEVERITY_LEVELS = ("minor", "moderate", "severe")

# source column (public Kaggle file) -> project column
SOURCE_COLUMNS = {
    "Vehicle_make": "make",
    "Vehicle_type": "vehicle_type",
    "Vehicle_year": "vehicle_year",
    "Engine_type": "engine_type",
    "Engine_displacement": "engine_cc",
    "Transmission_type": "transmission",
    "Number_of_cylinders": "cylinders",
    "Vehicle_weight": "weight_kg",
    "Vehicle_length": "length_mm",
    "Vehicle_width": "width_mm",
    "Vehicle_height": "height_mm",
    "Safety_rating": "safety_rating",
    "Number_of_airbags": "airbags",
    "ABS_presence": "abs",
    "ESC_presence": "esc",
    "TCS_presence": "tcs",
    "TPMS_presence": "tpms",
    "Crash_location": "location",
    "Weather_conditions": "weather",
    "Road_surface_conditions": "road_surface",
    "Time_of_day": "time_of_day",
    "Day_of_week": "day_of_week",
    "Driver_age": "driver_age",
    "Driver_gender": "driver_gender",
    "Crash_severity": "severity",
}

NOMINAL = [
    "make", "vehicle_type", "engine_type", "transmission", "location",
    "weather", "road_surface", "time_of_day", "day_of_week", "driver_gender",
]
NUMERIC = [
    "vehicle_year", "engine_cc", "cylinders", "weight_kg", "length_mm",
    "width_mm", "height_mm", "safety_rating", "airbags", "driver_age",
]
FLAGS = ["abs", "esc", "tcs", "tpms"]
FEATURES = NUMERIC + FLAGS + NOMINAL
INTEGER = set(NUMERIC) | set(FLAGS)

RANGES = {
    "vehicle_year": (1980, 2030),
    "engine_cc": (0, 8000),
    "cylinders": (0, 16),
    "weight_kg": (400, 6000),
    "length_mm": (2500, 7000),
    "width_mm": (1200, 2600),
    "height_mm": (1100, 2500),
    "safety_rating": (0, 5),
    "airbags": (0, 12),
    "driver_age": (16, 100),
}
REQUIRED = ["make", "vehicle_type", TARGET]

_TRUE = {"1", "1.0", "true", "yes", "y", "t"}
_FALSE = {"0", "0.0", "false", "no", "n", "f"}
_MISSING = {"", "nan", "none", "null", "na", "n/a"}


class SchemaError(ValueError):
    """The table cannot be used."""


def to_flag(values: pd.Series) -> pd.Series:
    """Map ``True/1/1.0/yes`` to 1.0, ``False/0/0.0/no`` to 0.0 and other values to NaN."""
    text = values.map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
    return text.map(lambda v: 1.0 if v in _TRUE else (0.0 if v in _FALSE else np.nan)).astype(float)


def clean_category(values: pd.Series) -> pd.Series:
    """Strip, lower-case and join words with ``_``. The text ``'nan'`` and blanks become NaN."""
    text = values.map(lambda v: "" if pd.isna(v) else str(v).strip().lower())
    return text.map(lambda v: np.nan if v in _MISSING else re.sub(r"\s+", "_", v)).astype(object)


def rename_source(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the public source columns to project columns. Other columns stay."""
    return df.rename(columns={k: v for k, v in SOURCE_COLUMNS.items() if k in df.columns})


@dataclass
class ValidationReport:
    rows_in: int
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(count)

    def summary(self) -> str:
        lines = [f"crashes: {self.rows_in} rows in, {self.rows_out} rows out, classes {self.class_counts}"]
        lines += [f"  dropped {n:>6}  {reason}" for reason, n in sorted(self.dropped.items())]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all known columns to their types. Absent feature columns become NaN."""
    out = rename_source(df).copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float) if col in out else np.nan
    for col in FLAGS:
        out[col] = to_flag(out[col]) if col in out else np.nan
    for col in NOMINAL:
        out[col] = clean_category(out[col]) if col in out else np.nan
    if TARGET in out:
        out[TARGET] = clean_category(out[TARGET])
    return out


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport(rows_in=len(df))
    renamed = rename_source(df)
    absent = [c for c in REQUIRED if c not in renamed.columns]
    if absent:
        raise SchemaError(f"required columns are absent: {absent}")
    out = coerce(renamed)
    for col in REQUIRED:
        miss = out[col].isna()
        report.drop(f"missing {col}", miss.sum())
        out = out[~miss]
    bad_label = ~out[TARGET].isin(SEVERITY_LEVELS)
    report.drop("unknown severity value", bad_label.sum())
    out = out[~bad_label]
    for col, (low, high) in RANGES.items():
        bad = out[col].notna() & ((out[col] < low) | (out[col] > high))
        if bad.any():
            report.warnings.append(f"{int(bad.sum())} values of {col} outside [{low}, {high}] set to NaN")
            out.loc[bad, col] = np.nan
    for col in FEATURES:
        share = out[col].isna().mean()
        if share > 0.2:
            report.warnings.append(f"{col} is missing in {share:.0%} of rows")
    if out.empty:
        raise SchemaError("no valid rows left after validation")
    report.rows_out = len(out)
    report.class_counts = {k: int(v) for k, v in out[TARGET].value_counts().sort_index().items()}
    return out.reset_index(drop=True), report


def make_target(df: pd.DataFrame, mode: str = "binary") -> np.ndarray:
    """``binary``: 1 for ``severe``, 0 for ``moderate`` and ``minor``. ``three``: 0, 1, 2 in severity order."""
    if mode == "binary":
        return (df[TARGET] == "severe").to_numpy(int)
    if mode == "three":
        return df[TARGET].map({level: i for i, level in enumerate(SEVERITY_LEVELS)}).to_numpy(int)
    raise ValueError(f"unknown target mode {mode!r}")
