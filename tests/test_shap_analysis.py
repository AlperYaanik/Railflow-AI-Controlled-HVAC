"""Tests for M9's SHAP feature attribution (src/shap_analysis.py).

The core computation is fast (~0.03s for the full 2558-row test split, since
TreeExplainer is exact rather than sampled) so these run the real thing on a
small slice, not a mock -- unlike the model benchmark or tau_act sweep,
there's no expensive-multi-minute-run reason to fake it.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import numpy as np
import pytest

from src.features import FEATURE_COLUMNS
from src.shap_analysis import compute, importance_table, held_out_test_features
from src.train import MODEL_PATH


@pytest.fixture(scope="module")
def require_model(require_weather):
    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")


def test_shap_handles_this_projects_actual_categorical_columns(require_model):
    """Regression guard for a real, documented compatibility class: LightGBM's
    native categorical_feature support has broken SHAP's TreeExplainer on
    some version combinations ("could not convert string to float",
    shap/shap#170). This project's model uses categorical_feature for
    city/direction/pattern (src/train.py) -- confirmed working on the
    versions pinned here before src/shap_analysis.py was written, and
    locked in so an upgrade that reintroduces the incompatibility fails a
    test instead of silently producing wrong or crashing output.
    """
    X = held_out_test_features().iloc[:20]
    assert X["city"].dtype.name == "category"  # sanity check this IS exercising the risky path
    shap_values, X_out = compute(X=X)
    assert shap_values.shape == (20, len(FEATURE_COLUMNS))


def test_shap_values_reconstruct_the_models_actual_predictions(require_model):
    """SHAP's own additivity property: sum(shap_values) + base_value must
    equal the model's raw prediction, exactly (to floating-point precision).
    This is the difference between "ran without crashing" and "computed the
    right thing" -- checked directly before trusting any importance ranking
    built on top of it.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    X = held_out_test_features().iloc[:100]
    shap_values, X_out = compute(booster=booster, X=X)

    import shap
    explainer = shap.TreeExplainer(booster)
    reconstructed = shap_values.sum(axis=1) + explainer.expected_value
    raw_pred = booster.predict(X_out)
    assert reconstructed == pytest.approx(raw_pred, abs=1e-6)


def test_importance_table_covers_every_feature_exactly_once_and_is_sorted(require_model):
    X = held_out_test_features().iloc[:50]
    shap_values, X_out = compute(X=X)
    table = importance_table(shap_values, X_out)

    assert sorted(table["feature"]) == sorted(FEATURE_COLUMNS)
    assert (table["mean_abs_shap_c"].diff().dropna() <= 1e-12).all(), "must be sorted descending"
    assert (table["mean_abs_shap_c"] >= 0.0).all()


def test_forecast_features_rank_above_current_state_alone(require_model):
    """Not pinning exact SHAP values (a real empirical result, not a
    constant to freeze) -- but the ordering claim behind this project's
    whole anticipatory-control premise (the model leans on FORECAST
    information, not just current state) is worth a real regression guard,
    not just an eyeballed plot. t_out_fcst_h (the forecasted disturbance)
    should matter more than t_air_c/error_c (current state alone) on
    average across the test split.
    """
    shap_values, X = compute()  # full test split -- this is the one test
    # worth spending the (still sub-second) cost of the whole split on,
    # since a small sample could accidentally under/over-represent the tail
    # scenarios where t_mass_c/t_out_fcst_h swing hardest.
    table = importance_table(shap_values, X).set_index("feature")["mean_abs_shap_c"]
    assert table["t_out_fcst_h"] > table["t_air_c"]
    assert table["t_out_fcst_h"] > table["error_c"]
