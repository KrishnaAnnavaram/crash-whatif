"""Seeded synthetic crash tables with a KNOWN cause structure.

Two modes:

* ``signal`` (default): the log-odds of a severe crash is a documented sum of effects
  (``LOGIT_EFFECTS``). Some columns have no effect (``NOISE_COLUMNS``), driver gender
  among them. Thus a test can check that an explanation finds the true causes.
* ``make_rule``: the label copies a rule on ``make`` with 2 % flips. This mode copies the
  defect of the public synthetic file, where the make alone decides the label.

The output uses the column names and the messy values of the public file
(``'True'``, ``'1.0'``, blanks and the text ``'nan'``), so it also tests the clean-up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAKES = ["Maruti Suzuki", "Honda", "Hyundai", "Mahindra", "Tata Motors"]
MAKE_EFFECT = {"Maruti Suzuki": -0.5, "Honda": -0.2, "Hyundai": 0.0, "Mahindra": 0.3, "Tata Motors": 0.2}
TYPES = ["sedan", "hatchback", "pickup", "SUV"]
ENGINES = ["petrol", "diesel", "CNG", "electric"]
WEATHER = ["clear", "rain", "fog"]
ROAD = ["dry", "wet", "muddy"]
TIMES = ["morning", "afternoon", "evening", "night"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# The documented log-odds model of the "signal" mode (project column names).
LOGIT_INTERCEPT = 0.9
LOGIT_EFFECTS = {
    "intercept": LOGIT_INTERCEPT,
    "make": "MAKE_EFFECT by make",
    "vehicle_type": "+0.5 for pickup",
    "safety_rating": "-0.45 for each step above 2.5",
    "airbags": "-0.18 for each airbag above 4",
    "esc": "-0.8 if present",
    "abs": "-0.4 if present",
    "weather": "+0.3 rain, +0.7 fog",
    "road_surface": "+0.5 wet, +0.8 muddy",
    "time_of_day": "+0.6 at night",
    "driver_age": "+0.03 for each year away from 45",
    "location": "+0.4 rural",
}
SIGNAL_COLUMNS = (
    "make", "vehicle_type", "safety_rating", "airbags", "esc", "abs",
    "weather", "road_surface", "time_of_day", "driver_age", "location",
)
NOISE_COLUMNS = (
    "vehicle_year", "engine_type", "engine_cc", "transmission", "cylinders", "weight_kg",
    "length_mm", "width_mm", "height_mm", "tcs", "tpms", "day_of_week", "driver_gender",
)


def _messy_flag(rng: np.random.Generator, present: np.ndarray) -> np.ndarray:
    """Write 0/1 flags the way the public file does: '1.0', 'True', '0.0', 'False' or blank."""
    out = np.where(present, "1.0", "0.0").astype(object)
    as_word = rng.random(present.size) < 0.25
    out[as_word & present] = "True"
    out[as_word & ~present] = "False"
    out[rng.random(present.size) < 0.07] = None
    return out


def true_logit(df: pd.DataFrame) -> np.ndarray:
    """Log-odds of a severe crash for clean rows (project names, clean values)."""
    make = df["make"].map({k.lower().replace(" ", "_"): v for k, v in MAKE_EFFECT.items()}).fillna(0.0)
    return (
        LOGIT_INTERCEPT
        + make.to_numpy(float)
        + 0.5 * (df["vehicle_type"] == "pickup").to_numpy(float)
        - 0.45 * (df["safety_rating"].to_numpy(float) - 2.5)
        - 0.18 * (df["airbags"].to_numpy(float) - 4)
        - 0.8 * df["esc"].fillna(0).to_numpy(float)
        - 0.4 * df["abs"].fillna(0).to_numpy(float)
        + df["weather"].map({"rain": 0.3, "fog": 0.7}).fillna(0.0).to_numpy(float)
        + df["road_surface"].map({"wet": 0.5, "muddy": 0.8}).fillna(0.0).to_numpy(float)
        + 0.6 * (df["time_of_day"] == "night").to_numpy(float)
        + 0.03 * np.abs(df["driver_age"].to_numpy(float) - 45)
        + 0.4 * (df["location"] == "rural").to_numpy(float)
    )



def make_crashes(n: int = 6000, seed: int = 42, mode: str = "signal") -> pd.DataFrame:
    """Make ``n`` crash rows with the public column names and a ``Crash_severity`` label."""
    if mode not in {"signal", "make_rule"}:
        raise ValueError("mode must be 'signal' or 'make_rule'")
    rng = np.random.default_rng(seed)
    make = rng.choice(MAKES, size=n)
    vtype = rng.choice(TYPES, size=n, p=[0.3, 0.3, 0.2, 0.2])
    rating = rng.integers(1, 5, size=n)
    airbags = rng.integers(1, 8, size=n)
    flags = {name: rng.random(n) < p for name, p in (("abs", 0.7), ("esc", 0.6), ("tcs", 0.65), ("tpms", 0.6))}
    clean = pd.DataFrame(
        {
            "make": pd.Series(make).str.lower().str.replace(" ", "_"),
            "vehicle_type": pd.Series(vtype).str.lower(),
            "safety_rating": rating,
            "airbags": airbags,
            "esc": flags["esc"].astype(float),
            "abs": flags["abs"].astype(float),
            "weather": rng.choice(WEATHER, size=n, p=[0.6, 0.25, 0.15]),
            "road_surface": rng.choice(ROAD, size=n, p=[0.6, 0.25, 0.15]),
            "time_of_day": rng.choice(TIMES, size=n, p=[0.2, 0.3, 0.3, 0.2]),
            "driver_age": rng.integers(18, 85, size=n),
            "location": rng.choice(["urban", "rural"], size=n, p=[0.65, 0.35]),
        }
    )
    if mode == "signal":
        p_severe = 1.0 / (1.0 + np.exp(-true_logit(clean)))
        severe = rng.random(n) < p_severe
    else:
        severe = (make != "Maruti Suzuki") ^ (rng.random(n) < 0.02)
    minor = ~severe & (rng.random(n) < 0.01)
    severity = np.where(severe, "severe", np.where(minor, "minor", "moderate"))

    df = pd.DataFrame(
        {
            "Vehicle_make": make,
            "Vehicle_type": vtype,
            "Vehicle_year": rng.integers(2000, 2023, size=n),
            "Engine_type": rng.choice(ENGINES, size=n, p=[0.4, 0.2, 0.2, 0.2]),
            "Engine_displacement": rng.integers(800, 3000, size=n),
            "Transmission_type": rng.choice(["manual", "automatic"], size=n),
            "Number_of_cylinders": rng.choice([3, 4, 6], size=n),
            "Vehicle_weight": rng.integers(800, 2500, size=n),
            "Vehicle_length": rng.integers(3500, 5000, size=n),
            "Vehicle_width": rng.integers(1400, 2000, size=n),
            "Vehicle_height": rng.integers(1400, 1800, size=n),
            "Safety_rating": rating,
            "Number_of_airbags": airbags,
            "ABS_presence": _messy_flag(rng, flags["abs"]),
            "ESC_presence": _messy_flag(rng, flags["esc"]),
            "TCS_presence": _messy_flag(rng, flags["tcs"]),
            "TPMS_presence": _messy_flag(rng, flags["tpms"]),
            "Crash_location": clean["location"],
            "Weather_conditions": clean["weather"],
            "Road_surface_conditions": clean["road_surface"],
            "Time_of_day": clean["time_of_day"],
            "Day_of_week": rng.choice(DAYS, size=n),
            "Driver_age": clean["driver_age"],
            "Driver_gender": rng.choice(["Male", "Female"], size=n),
            "Crash_severity": severity,
        }
    )
    # blanks and the text 'nan', as in the public file (the label stays complete)
    for col, share in (("Crash_location", 0.08), ("Weather_conditions", 0.08), ("Day_of_week", 0.1)):
        df.loc[rng.random(n) < share, col] = "nan"
    df.loc[rng.random(n) < 0.03, "Driver_gender"] = None
    return df
