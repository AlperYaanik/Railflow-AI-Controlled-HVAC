"""Regression test for M5's headline number: AnticipatorySetpointAdvisor vs
ThermostatController across the M4 held-out test split.

M10 PHASE 2 RESULT (see controllers.py's AnticipatorySetpointAdvisor
docstring and src/tune_advisor.py for the full three-attempt, 80-
configuration search this comes from): once Railflow's output became a
setpoint recommendation fed through a bang-bang PlantResponse rather than a
direct watts command, the pre-M10 "+3.9% energy, comfort improves too"
headline no longer holds -- not because of a bug, but a diagnosed structural
limit (a bang-bang receiver has no proportional response, so a forecast-
driven setpoint shift can only move WHEN it switches, never HOW HARD it
runs). The honestly re-tuned result is PARITY, not a win: mean energy delta
-0.2% (median +0.0%), zero scenarios worse on both energy and comfort, and
total degree-hours slightly BETTER in aggregate (41.43 vs 42.79 K*h) even
though energy is a wash. Confirmed once on this TEST split, matching this
project's standing tuning discipline -- not silently reused from a
pre-M10 run, not cherry-picked to look better than it is.
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
    before it. Still holds under M10's re-tuned law (confirmed 0/23 on
    TEST) -- the actual claim is "never worse on both", not merely "usually
    better", the stronger, more specific guarantee, and the one property
    Phase 2's re-tune was explicitly ranked to protect first (see
    src/tune_advisor.py's n_both_better-first ranking).
    """
    assert summary["n_worse_on_both"] == 0


def test_total_comfort_is_no_worse_in_aggregate(summary):
    """A genuine, if modest, positive finding worth locking in on its own:
    even though per-scenario energy is roughly a wash (previous test) and
    per-scenario comfort is only equal-or-better on 19/23, TOTAL degree-hours
    across all 23 scenarios comes out lower for the anticipatory advisor
    (41.43 vs ThermostatController's 42.79 K*h) -- the scenarios it helps
    outweigh the ones it doesn't, in aggregate. Small tolerance for harmless
    retrain noise, not an exact pin.
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
