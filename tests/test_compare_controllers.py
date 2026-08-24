"""Regression test for M5's headline number: AnticipatorySetpointAdvisor vs
ThermostatController across the M4 held-out test split.

M10 PHASE 2 RESULT (see controllers.py's AnticipatorySetpointAdvisor
docstring and src/tune_advisor.py for the full three-attempt, 80-
configuration search this comes from): once Railflow's output became a
setpoint recommendation fed through a bang-bang PlantResponse rather than a
direct watts command, the pre-M10 "+3.9% energy, comfort improves too"
headline no longer held -- not a bug, a diagnosed structural limit (a
bang-bang receiver has no proportional response, so a forecast-driven
setpoint shift can only move WHEN it switches, never HOW HARD it runs). The
honestly re-tuned result was PARITY, not a win: mean energy delta -0.2%
(median +0.0%), zero scenarios worse on both energy and comfort, total
degree-hours slightly BETTER in aggregate (41.43 vs 42.79 K*h) despite
energy being a wash.

M12 RESULT (PlantResponse replaced the bang-bang plant with a proportional
PI one -- see controllers.py's PlantResponse/AnticipatorySetpointAdvisor
docstrings and src/tune_advisor_m12.py for the full two-attempt search):
re-tuning (ff_weight, max_shift_k, deadband_k) alone against the new plant,
at the OLD every-minute update cadence, came back a REGRESSION, not a win --
comfort-ok collapsed from Phase 2's 21/23 to 2/23. Diagnosed as a cascaded-
control timescale mismatch (the advisor moving its target faster than
PlantResponse's own PI loop can settle -- ~17 min integral time). Fixed by
slowing advisor_update_interval_min instead (SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN
= 50 min, src/controllers.py), which let the two loops stop fighting
entirely (worse-on-both hits 0/31 on VAL from 40 minutes up). Re-confirmed
once on this TEST split: mean +0.04% (median +0.00%), 0/23 worse on both
energy and comfort, total degree-hours 15.46 -> 15.45 K*h -- matching
Phase 2's own old-plant headline almost exactly, this time under a plant
that actually behaves like the black-box "converges and holds" unit M12 set
out to model. Confirmed once on this TEST split each time, matching this
project's standing tuning discipline -- not silently reused from a prior
run, not cherry-picked to look better than it is.
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


def test_anticipatory_is_roughly_at_parity_with_thermostat(summary):
    """M10's honest headline: NOT a beats-the-baseline claim (that was the
    pre-M10, retired watts-dispatch law's result -- see module docstring).
    Bounded around zero, not pinned to a decimal -- the result is
    deterministic given the fixed seed/split/model, but a tight pin would
    fail on any harmless retrain. The actual result (-0.2% mean, +0.0%
    median) sits well inside this band; a regression toward the old failure
    mode (-4.5%, before the original deadband fix) or an unexplained jump to
    a large positive number would both fall outside it and are what this
    guards against.
    """
    assert -2.0 < summary["mean_energy_saving_pct"] < 1.0
    assert -2.0 < summary["median_energy_saving_pct"] < 1.0


def test_no_scenario_is_worse_on_both_energy_and_comfort(summary):
    """The specific failure mode the ORIGINAL (pre-M10) deadband fixed: 6/23
    scenarios were strictly dominated (more energy AND more discomfort)
    before it. Held under M10 Phase 2's re-tuned law (0/23 on TEST) --
    the actual claim is "never worse on both", not merely "usually better",
    the stronger, more specific guarantee. M12's plant swap broke this
    guarantee once (attempt 1: 6/23 worse-on-both, a real regression, not
    assumed to transfer and not silently reused) before re-earning it with a
    diagnosed fix (attempt 2: slowing advisor_update_interval_min to resolve
    a cascaded-control timescale mismatch against PlantResponse's PI loop) --
    see the module docstring and src/tune_advisor_m12.py for the full story.
    """
    assert summary["n_worse_on_both"] == 0


def test_total_comfort_is_no_worse_in_aggregate(summary):
    """A genuine, if modest, positive finding worth locking in on its own:
    TOTAL degree-hours across all 23 TEST scenarios comes out lower for the
    anticipatory advisor than ThermostatController's (15.45 vs 15.46 K*h,
    M12) -- essentially tied, not a large margin, but never worse in
    aggregate. Small tolerance for harmless retrain noise, not an exact pin.
    M12's plant swap broke this once (attempt 1, before the
    advisor_update_interval_min fix) and re-earned it under the new plant,
    not merely carried it over unchecked from Phase 2's bang-bang-tuned law.
    """
    assert summary["total_candidate_degree_hours"] <= summary["total_baseline_degree_hours"] + 1.0


def test_forcing_the_forecast_on_does_not_help(require_model, cfg):
    """A different question from compare() (see the module's docstring):
    does deliberately turning M4's forecast back on (ff_weight=0.3) beat
    what's actually shipped (ff_weight=0.0)? Unlike the pre-M10 version of
    this ablation, a direction now IS asserted: Phase 2's tuning found this
    consistently, repeatably negative (0/23 scenarios showed ANY energy
    improvement from turning the forecast on, TEST-confirmed) -- this test
    exists so a future change that silently makes the forecast start
    helping again (which would be a genuinely interesting result) gets
    caught and looked at, not lost as an unnoticed drive-by improvement.
    """
    results = compare_feedforward_contribution(cfg)
    assert len(results) == 23
    s = summarize(results, baseline="shipped", candidate="with_forecast")
    assert s["n_scenarios"] == 23
    assert s["mean_energy_saving_pct"] <= 0.0
    assert s["total_baseline_degree_hours"] >= 0.0
    assert s["total_candidate_degree_hours"] >= 0.0
