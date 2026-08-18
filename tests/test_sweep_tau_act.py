"""Smoke tests for M6's tau_act sweep (src/sweep_tau_act.py).

Deliberately NOT a test of the full 23-scenario x 5-value sweep -- that takes
several minutes and its result is a genuine, scenario-dependent empirical
finding (see ROADMAP.md's M6 section: the sweep falsified its own original
"monotonic increase" hypothesis), not a fixed number safe to pin in a fast
unit test. What IS safe and worth guarding here is the MECHANISM: that
overriding cfg reaches CabinModel's actuator, that it does NOT reach the
feature builder's EWMA halflife (see tests/test_features.py for the direct
version of that check), and that the function runs cleanly end to end.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pytest

from src.compare_controllers import held_out_test_scenarios
from src.config import load_config
from src.sweep_tau_act import sweep
from src.train import MODEL_PATH


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def require_model(require_weather):
    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")


@pytest.fixture(scope="module")
def small_scenarios(require_weather):
    return held_out_test_scenarios().head(2)


def test_sweep_runs_and_has_one_row_per_tau_value(require_model, cfg, small_scenarios):
    results = sweep(tau_act_values_min=(0.0, 20.0), cfg=cfg, scenarios=small_scenarios)
    assert list(results["tau_act_min"]) == [0.0, 20.0]
    assert len(results) == 2


def test_sweep_produces_finite_physically_sane_numbers(require_model, cfg, small_scenarios):
    """Loose bounds, same spirit as test_integration.py's M5 sanity check --
    not asserting a direction (the sweep's own headline finding is that the
    naive expected direction doesn't hold), just that nothing has gone
    unphysical."""
    results = sweep(tau_act_values_min=(0.0, 20.0), cfg=cfg, scenarios=small_scenarios)
    for col in ("onoff_mean_saving_pct", "ff_mean_saving_pct"):
        assert results[col].between(-100.0, 100.0).all()
    for col in ("onoff_total_baseline_dh", "onoff_total_candidate_dh",
                "ff_total_baseline_dh", "ff_total_candidate_dh"):
        assert (results[col] >= 0.0).all()


def test_sweeping_tau_act_actually_changes_the_onoff_baseline(require_model, cfg, small_scenarios):
    """Wiring check: the swept cfg must reach CabinModel's actuator, or the
    sweep would silently report identical numbers at every tau_act_min --
    exactly the kind of "parameter with zero effect" bug found and fixed for
    held_out_test_scenarios() during the M6-readiness audit.
    """
    results = sweep(tau_act_values_min=(0.0, 20.0), cfg=cfg, scenarios=small_scenarios)
    dh_at_0 = results.loc[results["tau_act_min"] == 0.0, "onoff_total_baseline_dh"].iloc[0]
    dh_at_20 = results.loc[results["tau_act_min"] == 20.0, "onoff_total_baseline_dh"].iloc[0]
    assert dh_at_0 != pytest.approx(dh_at_20)
