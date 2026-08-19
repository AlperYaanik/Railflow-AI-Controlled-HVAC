"""Smoke tests for M9's model benchmark (src/benchmark_models.py).

Deliberately NOT a test of the full run() -- training six candidates,
several with a 9-point grid each, takes minutes and its result is a
genuine empirical finding (see ROADMAP.md's M9 section: model family
barely matters here), not a fixed number safe to pin in a fast unit test.
What IS safe and worth guarding is the MECHANISM: that the data prep and
one-hot encoding produce consistent, leakage-free shapes, and that the
scoring helper computes what it claims to.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import numpy as np
import pandas as pd
import pytest

from src.benchmark_models import _one_hot, _prep_data, _score
from src.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from src.train import MODEL_PATH


@pytest.fixture(scope="module")
def require_model(require_weather):
    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")


def test_prep_data_returns_consistent_shapes(require_model):
    feat, X, y, masks = _prep_data()
    assert len(X) == len(y) == len(feat)
    assert set(X.columns) == set(FEATURE_COLUMNS)
    assert sum(mask.sum() for mask in masks.values()) == len(feat), (
        "train/val/test masks must partition every row exactly once"
    )


def test_one_hot_encoding_preserves_rows_and_drops_no_numeric_column(require_model):
    _, X, _, masks = _prep_data()
    Xoh = _one_hot(X, masks)
    assert len(Xoh) == len(X)
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    assert set(numeric_cols).issubset(Xoh.columns)
    for col in CATEGORICAL_COLUMNS:
        assert col not in Xoh.columns, f"{col} should have been replaced by dummy columns"


def test_one_hot_encoding_is_fit_only_on_the_train_split(monkeypatch):
    """A category present only in val/test must not crash the encoder --
    handle_unknown='ignore' means it's encoded as all-zero dummies, not a
    ValueError at transform time. Fitting on the full frame instead of just
    the train split would hide this distinction entirely (every category
    would be "seen"), so this is checked with a synthetic frame, not this
    project's real data, where every category already appears in train.

    Patches src.benchmark_models's OWN bindings of FEATURE_COLUMNS/
    CATEGORICAL_COLUMNS (from its `from src.features import ...`), not
    src.features's -- _one_hot() looks these up as module globals in its own
    namespace, so that's the one patch target that actually takes effect.
    """
    import src.benchmark_models as bm

    monkeypatch.setattr(bm, "FEATURE_COLUMNS", ["city", "direction", "pattern", "numeric_col"])
    monkeypatch.setattr(bm, "CATEGORICAL_COLUMNS", ("city", "direction", "pattern"))

    X = pd.DataFrame({
        "city": ["cairo", "cairo", "aswan"],
        "direction": ["down", "down", "up"],
        "pattern": ["express", "express", "semi_express"],
        "numeric_col": [1.0, 2.0, 3.0],
    })
    masks = {"train": pd.Series([True, True, False]), "val": pd.Series([False, False, True]),
              "test": pd.Series([False, False, False])}

    Xoh = _one_hot(X, masks)

    assert len(Xoh) == 3
    # aswan/up/semi_express (val-only categories) produce all-zero dummies,
    # not a crash and not a spurious new column.
    assert Xoh.loc[2, [c for c in Xoh.columns if c.startswith("city_")]].sum() == 0


def test_score_matches_a_manual_mae_computation():
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 33.0, 35.0])
    mask = pd.Series([True, True, True, True])
    feat = pd.DataFrame({"t_air_c": [10.0, 20.0, 30.0, 40.0], "target_t_air_c": [11.0, 19.0, 31.0, 39.0]})

    result = _score(pred, y, feat, mask)
    assert result["model_mae"] == pytest.approx(np.mean([2.0, 2.0, 3.0, 5.0]))
    assert result["n"] == 4
