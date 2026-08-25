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
    """Looked up by LABEL, not position -- st.metric() exposes .label
    (confirmed via dataclasses.fields(Metric), not just documented). A
    positional lookup broke once already: M13 added a "Station precool"
    section with its own 6 metrics ABOVE this comparison's 4, silently
    shifting every hardcoded index (metrics[1] started reading precool's
    "Energy -- with precool" instead of this section's "Energy --
    anticipatory"). Label lookup survives any future reordering the same
    way this one didn't.
    """
    by_label = {m.label: m for m in at.get("metric")}
    return {
        "thermo_energy": by_label["Energy — static schedule"].value,
        "antic_energy": by_label["Energy — anticipatory"].value,
        "antic_energy_delta": by_label["Energy — anticipatory"].delta,
        "thermo_dh": by_label["Comfort — static schedule"].value,
        "antic_dh": by_label["Comfort — anticipatory"].value,
    }


def test_app_runs_without_exception(require_model):
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception


def test_default_scenario_matches_the_cli_comparison_tool(require_model):
    """The exact check that caught nothing wrong when this app was first
    verified in a real browser, now automated: the app's displayed numbers
    for its default scenario must equal compare()'s output for that same
    scenario -- not approximately, since it's the identical code path, not
    a reimplementation.

    scenario_id=140 (cairo, 2024-08-25, 8.93h), NOT scenarios.iloc[0] --
    app.py's sidebar picker deliberately defaults there (M12), not to
    whichever scenario_id sorts first. See app.py's own comment on that
    selectbox: under M12's tuned cadence most scenarios are near-
    indistinguishable or bit-identical (10/23), a weak opening view for a
    live demo, so the default opens on the TEST-split scenario with the
    cleanest illustrative story instead (+0.29% energy, zero comfort cost).
    Caught by this exact test going red once already, the first time this
    file's own default was changed without updating what this test expected
    -- a real, if narrow, regression this test earned its keep catching.
    """
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception

    scenarios = held_out_test_scenarios()
    default = scenarios.loc[scenarios["scenario_id"] == 140]
    assert len(default) == 1, "scenario_id=140 must still be in the TEST split for this test to mean anything"
    expected = compare(scenarios=default).iloc[0]

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


def test_missing_scenario_data_shows_an_actionable_error_not_a_crash(
    monkeypatch, require_model, tmp_path,
):
    """If data/scenarios_raw.parquet is missing -- weather fetched and the
    model trained, but `python -m src.data_generator` never run, or its
    output pruned by a partial copy -- the app must say so and stop cleanly,
    the same guarantee the missing-model test above locks in.

    Found by code inspection: unlike the model check (`if booster is None:
    st.error(...); st.stop()`), get_scenarios() had no equivalent guard, so
    this path would crash with a raw FileNotFoundError from deep inside
    pandas.read_parquet -- confirmed by reproducing it directly (patching
    DATA_DIR and calling held_out_test_scenarios() outside Streamlit
    entirely) before this test or the fix existed.

    Patches src.compare_controllers.DATA_DIR, not src.app's or src.config's:
    held_out_test_scenarios() looks up DATA_DIR as a module global in
    src.compare_controllers's own namespace at call time, so that's the one
    binding that actually affects it -- same reasoning the MODEL_PATH patch
    above already relies on for src.train.

    get_scenarios() is @st.cache_data with no arguments, a single slot
    shared across this whole process -- without clearing it, an earlier
    test's successfully-cached real scenario table would still come back
    here regardless of the DATA_DIR patch, and this test would pass for the
    wrong reason (see the analogous get_booster()/cache_resource note
    above).
    """
    import streamlit as st

    import src.compare_controllers as compare_controllers_module

    st.cache_data.clear()
    monkeypatch.setattr(compare_controllers_module, "DATA_DIR", tmp_path)
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception
    assert any("scenarios_raw.parquet" in e.value for e in at.get("error"))
    st.cache_data.clear()  # don't leak the patched empty result to tests after this one


def test_missing_weather_data_shows_an_actionable_error_not_a_crash(
    monkeypatch, require_model, tmp_path,
):
    """If a weather_{city}_summer.csv is missing -- the model trained on an
    older weather cache that was since deleted, or a partial copy that kept
    data/scenarios_raw.parquet but not data/weather_*.csv -- run_both() must
    say so and stop cleanly rather than crash raw, same as the two cases
    above.

    Patches src.evaluate.DATA_DIR, not src.app's: run_controller() (called
    by run_both()) looks up DATA_DIR as a module global in src.evaluate's
    own namespace, independent of the src.compare_controllers.DATA_DIR
    binding the scenarios-file test above patches -- confirmed these two
    are genuinely separate bindings (each module did its own `from
    src.config import DATA_DIR`), so patching one leaves the other, and this
    test leaves get_scenarios() reading the real, present
    scenarios_raw.parquet -- only the weather read fails here.

    run_both() is @st.cache_data keyed on its scenario arguments; the
    default scenario this test exercises is the same one other tests in
    this file already ran successfully, so without clearing the cache here
    too, the earlier real result would mask this patch exactly like the
    get_scenarios() case above.
    """
    import streamlit as st

    import src.evaluate as evaluate_module

    st.cache_data.clear()
    monkeypatch.setattr(evaluate_module, "DATA_DIR", tmp_path)
    at = AppTest.from_file("src/app.py", default_timeout=60).run()
    assert not at.exception
    # Not a hardcoded city -- the default (first) scenario's city depends on
    # scenario_id ordering, not something this test should assume.
    assert any("weather_" in e.value and "_summer.csv" in e.value for e in at.get("error"))
    st.cache_data.clear()  # don't leak the patched empty result to tests after this one
