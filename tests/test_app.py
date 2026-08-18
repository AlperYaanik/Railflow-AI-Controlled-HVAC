"""Tests for M7's Streamlit demo (src/app.py), using Streamlit's own headless
AppTest framework rather than a real browser.

WHY THIS FILE EXISTS. app.py deliberately doesn't reimplement any
simulation logic -- it calls run_controller()/score() exactly as
src/compare_controllers.py does (see app.py's module docstring). That
reuse is only worth something if it's actually verified, not assumed: a
copy-paste slip when wiring the scenario picker's city/date/direction/
pattern/load_factor through to run_both() would silently feed the live
demo a DIFFERENT scenario than the one the sidebar claims to show, and
nothing about the page would look wrong. These tests catch that class of
bug by cross-checking the app's displayed numbers against
src/compare_controllers.py's compare() output for the identical scenario,
the same way the app was manually verified in a real browser before this
file was written (see ROADMAP.md's M7 section).

Also guards against the specific bug found when the app was first run:
`streamlit run src/app.py` does not put the project root on sys.path the
way `python -m` does, so `from src...` imports failed with
ModuleNotFoundError until app.py added its own sys.path fix. AppTest runs
the file the same way a real `streamlit run` would, so it would have
caught that bug directly.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pytest
from streamlit.testing.v1 import AppTest

from src.compare_controllers import compare, held_out_test_scenarios
from src.config import load_config
from src.train import MODEL_PATH


@pytest.fixture(scope="module")
def require_model(require_weather):
    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")


def _metric_values(at) -> dict:
    """Streamlit metrics don't expose a stable key, only display order --
    read them positionally, matching app.py's fixed column layout
    (on/off energy, anticipatory energy, on/off comfort, anticipatory comfort)."""
    metrics = at.get("metric")
    return {
        "thermo_energy": metrics[0].value, "antic_energy": metrics[1].value,
        "antic_energy_delta": metrics[1].delta,
        "thermo_dh": metrics[2].value, "antic_dh": metrics[3].value,
    }


def test_app_runs_without_exception(require_model):
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception


def test_default_scenario_matches_the_cli_comparison_tool(require_model):
    """The exact check that caught nothing wrong when this app was first
    verified in a real browser, now automated: the app's displayed numbers
    for its default (first) scenario must equal compare()'s output for
    that same scenario -- not approximately, since it's the identical
    code path, not a reimplementation.
    """
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception

    scenarios = held_out_test_scenarios()
    first = scenarios.iloc[[0]]
    expected = compare(scenarios=first).iloc[0]

    shown = _metric_values(at)
    assert shown["thermo_energy"] == f"{expected.thermo_energy_kwh:.2f} kWh"
    assert shown["antic_energy"] == f"{expected.antic_energy_kwh:.2f} kWh"
    assert shown["thermo_dh"] == f"{expected.thermo_degree_hours:.2f} K·h"
    assert shown["antic_dh"] == f"{expected.antic_degree_hours:.2f} K·h"


def test_switching_scenarios_produces_the_matching_different_numbers(require_model):
    """Regression guard for the specific risk this file's module docstring
    describes: a scenario-wiring slip that shows one scenario's label but
    simulates a different one. Picks a NON-default scenario, switches to
    it via the sidebar selectbox exactly as a real user would, and checks
    the result against compare() for that specific scenario -- not just
    "the numbers changed to something", but "changed to the RIGHT thing".
    """
    scenarios = held_out_test_scenarios()
    target = scenarios.iloc[2]  # third scenario in the list, arbitrary but fixed
    label = (f"#{target.scenario_id} — {target.city}, {target.date}, "
             f"{target.depart_hour:.1f}h, {target.direction}/{target.pattern}, "
             f"load {target.load_factor}")

    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    at.selectbox[0].select(label).run()
    assert not at.exception

    expected = compare(scenarios=scenarios.iloc[[2]]).iloc[0]
    shown = _metric_values(at)
    assert shown["thermo_energy"] == f"{expected.thermo_energy_kwh:.2f} kWh"
    assert shown["antic_energy"] == f"{expected.antic_energy_kwh:.2f} kWh"


def test_no_model_shows_an_actionable_error_not_a_crash(monkeypatch, require_weather):
    """If models/lgbm_t_air_forecaster.txt is missing (a fresh clone before
    `python -m src.train` has run), the app must say so and stop cleanly --
    not throw a raw exception at whoever's driving the demo machine.

    Two non-obvious things had to be found by running this, not guessed:

    1. AppTest.from_file() re-executes src/app.py's own source fresh each
       run rather than reusing an already-imported module object, so
       `monkeypatch.setattr(a_previously_imported_app_module, "MODEL_PATH",
       ...)` silently does nothing -- the fresh exec's own `from src.train
       import MODEL_PATH` re-reads it from src.train (already in
       sys.modules, not re-executed) regardless. The patch has to target
       src.train.MODEL_PATH, the actual source of the name, not app.py's
       copy of it.
    2. get_booster() takes no arguments, so its @st.cache_resource cache is
       one slot shared across every AppTest run in this process. Without
       clearing it, a real model loaded by an earlier test in this module
       would still be returned here even with MODEL_PATH patched away, and
       this test would pass for the wrong reason.

    Both were confirmed by watching this test fail for exactly those
    reasons before being fixed, not assumed from reading Streamlit's docs.
    """
    import streamlit as st

    import src.train as train_module

    st.cache_resource.clear()
    monkeypatch.setattr(train_module, "MODEL_PATH", MODEL_PATH.parent / "does_not_exist.txt")
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception
    assert any("no trained model" in e.value.lower() for e in at.get("error"))
    st.cache_resource.clear()  # don't leak the patched empty result to tests after this one
    st.cache_resource.clear()  # don't leak the patched empty result to tests after this one
