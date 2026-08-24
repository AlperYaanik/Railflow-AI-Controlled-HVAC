"""Regression test for M5's headline number: AnticipatorySetpointAdvisor vs
ThermostatController across the M4 held-out test split.

Locks in the shape of the result src/compare_controllers.py reports after the
deadband fix (see controllers.py's AnticipatorySetpointAdvisor docstring and
docs/PARAMETERS.md's M5 correction log): a positive mean energy saving and
zero scenarios worse on both energy and comfort. Before the fix, the same
comparison returned a NEGATIVE mean (-4.5%) with 6/23 scenarios strictly
worse on both -- this test exists so that failure mode cannot silently ship
again without a test going red.

M10 -- the three directional tests below (test_anticipatory_beats_thermostat_
on_average, test_no_scenario_is_worse_on_both_energy_and_comfort,
test_most_scenarios_show_some_energy_saving) are marked xfail, NOT deleted
and NOT silently left red: they pin numbers earned by the retired
watts-dispatch law (gain_k-based). AnticipatorySetpointAdvisor's new
setpoint-shift law uses a fresh, untuned max_shift_k placeholder (see its
docstring in src/controllers.py) -- re-earning these directional guarantees
is Phase 2's job (ROADMAP.md's M10 section), via the same VAL-grid/TEST-once
retune discipline used to earn them the first time.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pytest

from src.compare_controllers import (
    compare,
    compare_feedforward_contribution,
    held_out_test_scenarios,
    summarize,
)
from src.config import load_config
from src.train import MODEL_PATH


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def require_model(require_weather):
    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")


@pytest.fixture(scope="module")
def summary(require_model, cfg):
    results = compare(cfg)
    return summarize(results)


def test_test_split_is_the_documented_23_scenarios(require_weather):
    """Wiring check: the split this comparison runs on is the one ROADMAP.md
    and docs/PARAMETERS.md cite (23 scenarios, 2024-08-18 to 2024-08-30) -- if
    data_generator.py's seed or row target ever changes, this is what would
    catch the cited numbers no longer matching what actually runs.
    """
    scenarios = held_out_test_scenarios()
    assert len(scenarios) == 23
    assert str(scenarios["date"].min()) == "2024-08-18"
    assert str(scenarios["date"].max()) == "2024-08-30"


@pytest.mark.xfail(
    reason="pending Phase 2 retune of AnticipatorySetpointAdvisor under the "
           "new setpoint-output law -- see ROADMAP.md's M10 section",
    strict=False,
)
def test_anticipatory_beats_thermostat_on_average(summary):
    """The headline claim. Bounded, not pinned to a decimal -- the result is
    deterministic given the fixed seed/split/model, but a tight pin would
    fail on any harmless retrain. 0% is the bar that actually matters: this
    is exactly what the pre-deadband controller failed.
    """
    assert 0.0 < summary["mean_energy_saving_pct"] < 20.0
    assert 0.0 < summary["median_energy_saving_pct"] < 20.0


@pytest.mark.xfail(
    reason="pending Phase 2 retune of AnticipatorySetpointAdvisor under the "
           "new setpoint-output law -- see ROADMAP.md's M10 section",
    strict=False,
)
def test_no_scenario_is_worse_on_both_energy_and_comfort(summary):
    """The specific failure mode the deadband fixed: 6/23 scenarios were
    strictly dominated (more energy AND more discomfort) before it. The
    actual claim is "never worse on both", not merely "usually better" --
    this is the stronger, more specific guarantee.
    """
    assert summary["n_worse_on_both"] == 0


@pytest.mark.xfail(
    reason="pending Phase 2 retune of AnticipatorySetpointAdvisor under the "
           "new setpoint-output law -- see ROADMAP.md's M10 section",
    strict=False,
)
def test_most_scenarios_show_some_energy_saving(summary):
    assert summary["n_energy_better"] >= summary["n_scenarios"] * 0.5


def test_feedforward_contribution_uses_the_same_scenarios(require_model, cfg):
    """A different question from compare() (see the module's docstring): does
    M4's forecast earn its keep over pure proportional feedback, not whether
    it beats ThermostatController. No direction is asserted here -- unlike
    the on/off comparison, this one was never tuned or validated on any
    split before being run, so a hard bound would be asserting an outcome
    rather than checking one. Just needs to run cleanly on the same 23
    scenarios and produce finite, physically sane numbers.
    """
    results = compare_feedforward_contribution(cfg)
    assert len(results) == 23
    s = summarize(results, baseline="proportional", candidate="anticipatory")
    assert s["n_scenarios"] == 23
    assert -100.0 < s["mean_energy_saving_pct"] < 100.0
    assert s["total_baseline_degree_hours"] >= 0.0
    assert s["total_candidate_degree_hours"] >= 0.0
