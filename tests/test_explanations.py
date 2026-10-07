"""Permutation importance (problem 5) and counterfactuals over many rows (problems 8 and 10)."""
import numpy as np
import pandas as pd
import pytest

from crash_whatif import study as st
from crash_whatif.cf_metrics import Plausibility, query_metrics, summarize
from crash_whatif.counterfactuals import FeatureSpace, RandomSearchCF, to_frame
from crash_whatif.importance import auc_drop
from crash_whatif.schema import FEATURES


@pytest.fixture(scope="module")
def importance(study):
    return auc_drop(study.models["logreg"], study.split.test, study.y["test"], n_repeats=5, seed=1)


@pytest.fixture(scope="module")
def space(study, settings):
    return FeatureSpace.from_train(study.split.train, settings.actionable)


def test_importance_sign_and_ranking_match_the_generator(importance):
    top = set(importance.head(5)["column"])
    assert {"safety_rating", "esc", "airbags"} <= top
    noise = importance.set_index("column").loc[["driver_gender", "day_of_week", "tcs", "tpms", "length_mm"], "auc_drop"]
    assert (noise.abs() < 0.01).all()
    assert importance["auc_drop"].iloc[0] > 0.02  # a positive value means: the model uses the column


def test_importance_is_seeded(study, importance):
    again = auc_drop(study.models["logreg"], study.split.test, study.y["test"], n_repeats=5, seed=1)
    pd.testing.assert_frame_equal(importance, again)
    assert (importance["std"] >= 0).all()


def test_feature_space_uses_train_ranges(study, space):
    assert space.ranges["airbags"] == (study.split.train["airbags"].min(), study.split.train["airbags"].max())
    assert "driver_age" not in space.ranges  # not actionable
    with pytest.raises(ValueError):
        FeatureSpace.from_train(study.split.train, ["not_a_column"])


def _severe_query(study):
    model = study.models["random_forest"]
    rows = study.split.test[model.predict(study.split.test) == 1]
    return model, rows.iloc[0]


def test_counterfactuals_are_valid_actionable_and_in_range(study, space):
    model, query = _severe_query(study)
    result = RandomSearchCF(model, space, seed=3).generate(query, n_cfs=3, desired=0)
    assert result.found
    assert (model.predict(result.counterfactuals) == 0).all()
    for _, cf in result.counterfactuals.iterrows():
        changed = space.changed(query, cf)
        assert changed and set(changed) <= set(space.actionable)
        for col in FEATURES:
            if col not in space.actionable:
                assert (pd.isna(cf[col]) and pd.isna(query[col])) or cf[col] == query[col]
        for col in changed:
            if col in space.ranges:
                low, high = space.ranges[col]
                assert low <= cf[col] <= high
            else:
                assert cf[col] in space.categories[col]


def test_counterfactuals_have_no_unnecessary_change(study, space):
    model, query = _severe_query(study)
    result = RandomSearchCF(model, space, seed=3).generate(query, n_cfs=3, desired=0)
    for _, cf in result.counterfactuals.iterrows():
        for col in space.changed(query, cf):
            reverted = cf.copy()
            reverted[col] = query[col]
            assert model.predict(to_frame([reverted]))[0] == 1  # each change is necessary


def test_counterfactuals_are_seeded(study, space):
    model, query = _severe_query(study)
    a = RandomSearchCF(model, space, seed=5).generate(query, desired=0)
    b = RandomSearchCF(model, space, seed=5).generate(query, desired=0)
    pd.testing.assert_frame_equal(a.counterfactuals, b.counterfactuals)


def test_plausibility_of_real_rows_is_near_one(study):
    plaus = Plausibility(study.split.train)
    scores = plaus.score(study.split.test.head(100))
    assert 0.6 < np.median(scores) < 1.6


def test_benchmark_covers_many_queries(study):
    bench = st.counterfactual_benchmark(study, names=("logreg",), n_queries=8)
    b = bench["logreg"]
    assert b["queries"] == 8
    assert b["validity"]["mean"] > 0.7
    low, high = b["sparsity"]["ci95"]
    assert 1 <= low <= b["sparsity"]["mean"] <= high <= 3
    assert set(b["changed_columns"]) <= set(bench["actionable"])


def test_summarize_and_empty_result(study, space):
    model, query = _severe_query(study)
    empty = RandomSearchCF(model, FeatureSpace.from_train(study.split.train, ["transmission"]), seed=1)
    result = empty.generate(query, desired=0)
    metrics = query_metrics(result, model, space, Plausibility(study.split.train))
    if not result.found:
        assert metrics == {"found": 0.0, "validity": 0.0}
    out = summarize([{"found": 1.0, "validity": 1.0, "sparsity": 1.0}, metrics])
    assert out["queries"] == 2


def test_dice_adapter(study, space):
    pytest.importorskip("dice_ml")
    from crash_whatif.dice_adapter import DiceCF

    model, query = _severe_query(study)
    generator = DiceCF(model, space, study.split.train, study.y["train"], seed=1)
    result = generator.generate(query, n_cfs=2, desired=0)
    if result.found:
        assert set(result.counterfactuals.columns) >= set(FEATURES)
