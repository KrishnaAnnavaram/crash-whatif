"""Schema clean-up (reference problems 3 and 9), synthetic data (problem 2) and the split."""
import numpy as np
import pandas as pd
import pytest

from crash_whatif.schema import SchemaError, clean_category, coerce, make_target, to_flag, validate
from crash_whatif.split import stratified_split
from crash_whatif.synthetic import NOISE_COLUMNS, SIGNAL_COLUMNS, make_crashes


def test_flag_mapping_is_explicit():
    out = to_flag(pd.Series(["1.0", "True", "0.0", "False", "", None, "yes", "maybe", 1, 0.0]))
    expected = [1.0, 1.0, 0.0, 0.0, np.nan, np.nan, 1.0, np.nan, 1.0, 0.0]
    np.testing.assert_array_equal(out.to_numpy(), np.array(expected))


def test_each_flag_comes_from_its_own_column():
    """Reference problem 3: TCS must not be a copy of ABS."""
    raw = pd.DataFrame(
        {"Vehicle_make": ["Honda"] * 3, "Vehicle_type": ["sedan"] * 3, "Crash_severity": ["severe"] * 3,
         "ABS_presence": ["1.0", "1.0", "1.0"], "TCS_presence": ["0.0", "False", None]}
    )
    clean, _ = validate(raw)
    assert clean["abs"].tolist() == [1.0, 1.0, 1.0]
    assert clean["tcs"].tolist()[:2] == [0.0, 0.0] and np.isnan(clean["tcs"].iloc[2])


def test_text_nan_and_blanks_become_missing():
    out = clean_category(pd.Series(["nan", " ", "Urban", "SUV", None]))
    assert pd.isna(out[0]) and pd.isna(out[1]) and pd.isna(out[4])
    assert out[2] == "urban" and out[3] == "suv"


def test_validation_drops_bad_labels_and_warns_on_ranges():
    raw = make_crashes(200, seed=1)
    raw.loc[0, "Crash_severity"] = "fatal?"
    raw.loc[1, "Driver_age"] = 250
    clean, report = validate(raw)
    assert report.dropped["unknown severity value"] == 1
    assert any("driver_age" in w for w in report.warnings)
    assert clean["driver_age"].max() <= 100
    assert sum(report.class_counts.values()) == len(clean) == 199


def test_missing_required_column_raises():
    with pytest.raises(SchemaError):
        validate(pd.DataFrame({"Vehicle_make": ["Honda"]}))


def test_targets_keep_the_minor_class_visible():
    """Reference problem 9: the binary target is documented and the three-class target exists."""
    df = coerce(pd.DataFrame({"Crash_severity": ["minor", "moderate", "severe"], "Vehicle_make": ["a"] * 3}))
    assert make_target(df, "binary").tolist() == [0, 0, 1]
    assert make_target(df, "three").tolist() == [0, 1, 2]


def test_synthetic_is_seeded_and_messy():
    a, b = make_crashes(500, seed=4), make_crashes(500, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert {"True", "1.0", "0.0"} <= set(a["ABS_presence"].dropna())
    assert (a["Crash_location"] == "nan").any()


def test_signal_and_noise_columns_are_disjoint():
    assert not set(SIGNAL_COLUMNS) & set(NOISE_COLUMNS)
    assert "driver_gender" in NOISE_COLUMNS


def test_make_rule_mode_copies_the_public_file_defect():
    clean, _ = validate(make_crashes(2000, seed=5, mode="make_rule"))
    share = clean.groupby("make")["severity"].apply(lambda s: (s == "severe").mean())
    assert share["maruti_suzuki"] < 0.05
    assert share.drop("maruti_suzuki").min() > 0.95


def test_split_is_stratified_disjoint_and_seeded():
    clean, _ = validate(make_crashes(2000, seed=6))
    y = make_target(clean)
    a = stratified_split(clean, y, seed=1)
    b = stratified_split(clean, y, seed=1)
    pd.testing.assert_frame_equal(a.test, b.test)
    assert a.sizes() == {"train": 1200, "val": 400, "test": 400}
    shares = [make_target(p).mean() for p in (a.train, a.val, a.test)]
    assert max(shares) - min(shares) < 0.02
    key = lambda d: set(map(tuple, d[["make", "driver_age", "weight_kg", "length_mm"]].astype(str).to_numpy()))  # noqa: E731
    assert len(key(a.train) & key(a.test)) < 5  # (near) no row is in two parts
