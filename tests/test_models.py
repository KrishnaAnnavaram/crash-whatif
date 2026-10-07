"""Pipelines fit on train only (problems 1, 4, 6, 7), baselines and metrics."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from crash_whatif import study as st
from crash_whatif.evaluate import binary_metrics, expected_calibration_error
from crash_whatif.models import build_pipeline, choose_threshold
from crash_whatif.schema import FEATURES, NOMINAL, NUMERIC


def test_imputer_and_scaler_use_train_rows_only(study):
    pipe = study.models["logreg"].pipeline
    imputer = pipe.named_steps["impute"].named_transformers_["numeric"]
    assert imputer.statistics_[NUMERIC.index("driver_age")] == np.nanmedian(study.split.train["driver_age"])
    scaler = pipe.named_steps["encode"].named_transformers_["numeric"]
    assert scaler.mean_[NUMERIC.index("weight_kg")] == pytest.approx(study.split.train["weight_kg"].mean())


def test_nominal_columns_are_one_hot_not_codes(study):
    """Problem 4: a nominal column becomes 0/1 columns, never a number such as make = 2.37."""
    pipe = study.models["logreg"].pipeline
    Z = pipe[:-1].transform(study.split.test[FEATURES])
    encoder = pipe.named_steps["encode"]
    names = list(encoder.get_feature_names_out())
    make_cols = [i for i, n in enumerate(names) if n.startswith("make_")]
    assert len(make_cols) == 5
    assert set(np.unique(Z[:, make_cols])) <= {0.0, 1.0}


def test_unknown_category_in_new_rows_is_ignored(study):
    rows = study.split.test.head(5).copy()
    rows["make"] = "unknown_brand"
    assert np.isfinite(study.models["logreg"].proba(rows)).all()


def test_logistic_regression_inputs_are_scaled(study):
    """Problem 6: numeric columns have mean 0 and SD 1 on the training rows."""
    pipe = study.models["logreg"].pipeline
    Z = pipe[:-1].transform(study.split.train[FEATURES])
    numeric = Z[:, : len(NUMERIC)]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-2)


def test_models_beat_the_baselines_on_real_test_rows(study):
    test_auc = {
        name: roc_auc_score(study.y["test"], m.proba(study.split.test)) for name, m in study.models.items()
    }
    assert test_auc["majority"] == 0.5
    assert test_auc["logreg"] > test_auc["make_only"] + 0.05
    assert test_auc["random_forest"] > 0.62


def test_test_rows_are_not_resampled(study):
    """Problem 1: the test part keeps its natural class shares."""
    assert study.split.test.shape[0] == study.y["test"].size == 800
    assert 0.6 < study.y["test"].mean() < 0.9


def test_threshold_is_chosen_on_validation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = choose_threshold(y, p)
    assert 0.3 < t <= 0.7


def test_metrics_row_has_all_fields(study):
    row = binary_metrics(study.models["logreg"], study.split.test, study.y["test"], n_boot=50)
    assert row["roc_auc_ci95"][0] <= row["roc_auc"] <= row["roc_auc_ci95"][1]
    for key in ("pr_auc_severe", "balanced_accuracy", "f1_macro", "brier", "log_loss", "ece"):
        assert 0 <= row[key] <= 1 or key == "log_loss"


def test_expected_calibration_error_known_values():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0
    assert expected_calibration_error([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_three_class_report_shows_minor_recall(study):
    out = st.three_class(study)
    assert set(out["logreg"]["recall"]) == {"minor", "moderate", "severe"}
    assert out["class_counts_train"]["minor"] >= 1


def test_audit_finds_the_make_rule():
    """Problem 2: on make-rule data the audit says that make alone decides the label."""
    from crash_whatif.config import Settings
    from crash_whatif.synthetic import make_crashes

    s = st.prepare(make_crashes(2000, seed=8, mode="make_rule"), Settings())
    audit = st.signal_audit(s)
    assert audit["make_dominates"] is True
    assert audit["auc_make_only"] > 0.95


def test_audit_on_signal_data(study):
    audit = st.signal_audit(study)
    assert audit["make_dominates"] is False
    assert audit["auc_without_make"] > audit["auc_make_only"]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_pipeline("xgboost")


@pytest.mark.parametrize("method", ["smotenc", "oversample"])
def test_resampling_runs_inside_the_pipeline(study, method):
    pytest.importorskip("imblearn")
    pipe = build_pipeline("logreg", resample=method, seed=1)
    pipe.fit(study.split.train[FEATURES], study.y["train"])
    sampler = pipe.named_steps["resample"]
    imputed = pipe.named_steps["impute"].transform(study.split.train[FEATURES])
    X_res, y_res = sampler.fit_resample(imputed, study.y["train"])
    assert y_res.mean() == pytest.approx(0.5)
    for col in NOMINAL:  # no synthetic row has a category that does not exist
        assert set(X_res[col]) <= set(imputed[col])
    p = pipe.predict_proba(study.split.test[FEATURES])[:, 1]
    assert p.shape[0] == len(study.split.test)
