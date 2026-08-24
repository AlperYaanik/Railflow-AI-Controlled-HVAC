"""Tests for the M5 scoring metrics.

degree_hours_outside_band is the one that matters most: it replaces an
earlier, uncited "degree-minutes" unit this project invented rather than
sourced. Real HVAC comfort literature uses degree-hours ("Exceedance
Degree-Hours", Salimi et al., Indoor Air, 2021). These tests check the
formula against hand-computed cases, not just that it runs.
"""

import numpy as np
import pandas as pd
import pytest

from src.cabin_model import CabinInputs, CabinModel
from src.config import load_config
from src.controllers import ThermostatController
from src.evaluate import degree_hours_outside_band, run_controller, score


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------- degree_hours_outside_band


def test_zero_when_always_inside_the_band():
    t_air = pd.Series([25.0, 25.5, 24.5, 26.0])
    setpoint = pd.Series([25.0] * 4)
    assert degree_hours_outside_band(t_air, setpoint, band_k=2.0) == pytest.approx(0.0)


def test_matches_hand_computed_single_excursion():
    """One minute at 4 K above a 2 K band -> 2 K of exceedance for 1 minute
    -> 2/60 K*h."""
    t_air = pd.Series([25.0, 29.0, 25.0])  # setpoint 25, band 2 -> excess at index 1 is 2 K
    setpoint = pd.Series([25.0] * 3)
    expected = 2.0 * (1.0 / 60.0)
    assert degree_hours_outside_band(t_air, setpoint, band_k=2.0) == pytest.approx(expected)


def test_scales_with_duration_not_just_magnitude():
    """The same 1 K exceedance sustained for twice as long must score twice
    as high -- this is what distinguishes degree-hours from a minute-count,
    which the earlier degree-minutes framing risked being confused with.
    """
    setpoint = pd.Series([25.0] * 4)
    short = pd.Series([25.0, 28.0, 25.0, 25.0])   # 1 minute at +3 (1 K over a 2 K band)
    long = pd.Series([25.0, 28.0, 28.0, 25.0])    # 2 minutes at +3
    dh_short = degree_hours_outside_band(short, setpoint, band_k=2.0)
    dh_long = degree_hours_outside_band(long, setpoint, band_k=2.0)
    assert dh_long == pytest.approx(2.0 * dh_short)


def test_scales_with_magnitude_not_just_duration():
    """Twice as far outside the band for the same duration must score twice
    as high."""
    setpoint = pd.Series([25.0] * 2)
    mild = pd.Series([25.0, 28.0])    # excess over a 2 K band = 1 K
    severe = pd.Series([25.0, 30.0])  # excess over a 2 K band = 3 K
    dh_mild = degree_hours_outside_band(mild, setpoint, band_k=2.0)
    dh_severe = degree_hours_outside_band(severe, setpoint, band_k=2.0)
    assert dh_severe == pytest.approx(3.0 * dh_mild)


def test_symmetric_for_overcooling_and_overheating():
    setpoint = pd.Series([25.0] * 2)
    hot = pd.Series([25.0, 30.0])
    cold = pd.Series([25.0, 20.0])
    assert degree_hours_outside_band(hot, setpoint, band_k=2.0) == pytest.approx(
        degree_hours_outside_band(cold, setpoint, band_k=2.0)
    )


def test_a_wider_band_never_increases_the_score():
    t_air = pd.Series([25.0, 31.0, 27.0, 25.0])
    setpoint = pd.Series([25.0] * 4)
    narrow = degree_hours_outside_band(t_air, setpoint, band_k=1.0)
    wide = degree_hours_outside_band(t_air, setpoint, band_k=3.0)
    assert wide <= narrow


def test_dt_min_scales_the_result_linearly():
    """Native resolution can be anything; the K*h conversion must track it."""
    t_air = pd.Series([25.0, 29.0])
    setpoint = pd.Series([25.0] * 2)
    per_minute = degree_hours_outside_band(t_air, setpoint, band_k=2.0, dt_min=1.0)
    per_five_minutes = degree_hours_outside_band(t_air, setpoint, band_k=2.0, dt_min=5.0)
    assert per_five_minutes == pytest.approx(5.0 * per_minute)


def test_never_negative():
    rng = np.random.default_rng(0)
    t_air = pd.Series(rng.uniform(15, 35, 200))
    setpoint = pd.Series([25.0] * 200)
    assert degree_hours_outside_band(t_air, setpoint, band_k=2.0) >= 0.0


# --------------------------------------------------------------- run_controller


def test_run_controller_matches_the_manually_traced_reference(cfg, require_weather):
    """Locks in the numbers this file's own __main__ block reports, on the
    same fixed scenario used throughout M5's development. A silent
    regression in run_controller or score() should move these.

    Updated to 31.08 kWh / +0.50 K / 5.06 K for M12: PlantResponse became a
    PI-controlled black box (replacing M10-M11's bang-bang default -- see
    src/controllers.py's PlantResponse docstring), a real behavioural
    change, not a bug -- energy is nearly unchanged (30.86->31.08 kWh) but
    the underlying trajectory is now close to flat rather than swinging
    several K around setpoint (see ROADMAP.md's M12 section for the trace).
    worst_excursion_k is UNCHANGED at 5.06 -- the same value under both
    plant models, consistent with it being a hard physical ceiling (how
    fast rated capacity can arrest a sudden disturbance) rather than an
    artefact of which control law is running. The pre-M12 30.86/0.47
    reference is in git history if ever needed for comparison.
    """
    traj = run_controller(lambda model: ThermostatController(model), cfg)
    result = score(traj, cfg)
    assert result.energy_kwh == pytest.approx(31.08, abs=0.1)
    assert result.mean_err_k == pytest.approx(0.50, abs=0.05)
    assert result.worst_excursion_k == pytest.approx(5.06, abs=0.1)
    assert result.minutes == len(traj)


def test_run_controller_reports_advised_setpoint_distinct_from_scoring_setpoint(cfg, require_weather):
    """M10 wiring check: advised_setpoint_c (what THIS advisor recommended)
    and setpoint_c (the fixed EN13129 reference every arm is SCORED against)
    are different columns answering different questions -- easy to conflate
    since they're often equal. For ThermostatController, whose whole job is
    "always recommend the static schedule", they must be equal every minute:
    if they aren't, PlantResponse and CabinModel.setpoint() have drifted
    apart. For an advisor that actually adjusts its recommendation, they
    must differ at least sometimes, or the "recommend a setpoint" reframing
    isn't actually doing anything.
    """
    from src.controllers import AnticipatorySetpointAdvisor
    from src.features import LiveFeatureBuilder
    from src.train import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    thermo = run_controller(lambda model: ThermostatController(model), cfg)
    assert (thermo["advised_setpoint_c"] == thermo["setpoint_c"]).all()

    def antic_factory(model):
        builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                      pattern="semi_express", load_factor=1.0, depart_hour=8.0)
        return AnticipatorySetpointAdvisor(model, booster, builder)

    antic = run_controller(antic_factory, cfg)
    assert (antic["advised_setpoint_c"] != antic["setpoint_c"]).any()


def test_run_controller_reports_real_weather_driven_humidity(cfg, require_weather):
    """M9's humidity state was built and tested inside CabinModel but never
    reached run_controller()'s output -- found during a whole-project review
    (not visible in the demo, the dataset, anywhere). This locks in the fix:
    w_air_g_kg/rh_air_pct must be present and, more importantly, must vary
    with the SCENARIO's real weather rather than sitting on CabinInputs'
    rh_out_pct=50.0 default the whole run -- confirmed by comparing two
    different cities' weather, not assumed from the code alone.
    """
    cairo = run_controller(lambda model: ThermostatController(model), cfg, city="cairo")
    aswan = run_controller(lambda model: ThermostatController(model), cfg, city="aswan")

    for traj in (cairo, aswan):
        assert "w_air_g_kg" in traj.columns and "rh_air_pct" in traj.columns
        assert traj["w_air_g_kg"].between(0.0, 40.0).all()
        assert traj["rh_air_pct"].between(0.0, 100.0).all()
        # Real weather varies minute to minute -- a constant 50% default
        # would make every row identical.
        assert traj["rh_air_pct"].nunique() > 1

    assert not cairo["rh_air_pct"].equals(aswan["rh_air_pct"]), (
        "Cairo and Aswan have different real outdoor humidity -- identical "
        "output here would mean rh_out_pct isn't actually being read from "
        "the weather file"
    )


def test_moisture_generation_and_exchange_have_no_direct_q_hvac_term(cfg):
    """The precise, structural claim: the moisture balance's SOURCE terms
    (passenger generation, ventilation/door exchange) have no q_hvac
    dependence at all. Checked directly on two otherwise-identical runs that
    only differ in commanded power, in a deliberately dry/hot regime (25%
    outdoor RH, 38 C) chosen so the saturation cap below never binds --
    isolating the claim this test is actually about from the separate,
    known effect the next test covers.
    """
    model_a, model_b = CabinModel(cfg), CabinModel(cfg)
    state_a, state_b = model_a.initial_state(30.0), model_b.initial_state(30.0)
    w_a, w_b = [], []
    for t in range(60):
        cmd = -model_a.cooling_capacity_w if t % 20 < 10 else 0.0
        w_a.append(model_a.step(state_a, CabinInputs(
            t_out_c=38.0, rh_out_pct=25.0, n_pax=60, q_hvac_cmd_w=cmd), dt_s=60.0).w_air_g_kg)
        w_b.append(model_b.step(state_b, CabinInputs(
            t_out_c=38.0, rh_out_pct=25.0, n_pax=60, q_hvac_cmd_w=0.0), dt_s=60.0).w_air_g_kg)
    assert w_a == w_b


def test_saturation_cap_can_make_two_controllers_moisture_diverge(cfg, require_weather):
    """A real, second-order effect found by a failing test, not designed in
    advance: the previous test shows the moisture SOURCE terms don't depend
    on q_hvac, but the saturation cap added to step() (see its comment --
    w_air_g_kg can't exceed what the air holds at 100% RH for its CURRENT
    t_air_c) does depend on temperature, which DOES depend on q_hvac. Since
    w_air_g_kg is a persistent state, even one substep where the cap binds
    differently for two controllers forks their moisture trajectories
    forward from that point -- so on a real (humid enough) scenario, two
    controllers' absolute moisture can end up close but NOT bit-identical.
    This test locks in "close" (a small bound) rather than the stronger,
    now-known-false "identical" claim an earlier version of this test made.
    """
    from src.controllers import AnticipatorySetpointAdvisor
    from src.features import LiveFeatureBuilder
    from src.train import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    def antic_factory(model):
        builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                      pattern="semi_express", load_factor=1.0, depart_hour=8.0)
        return AnticipatorySetpointAdvisor(model, booster, builder)

    thermo = run_controller(lambda model: ThermostatController(model), cfg)
    antic = run_controller(antic_factory, cfg)

    assert not thermo["t_air_c"].equals(antic["t_air_c"])  # sanity check: they do differ
    max_diff = (thermo["w_air_g_kg"] - antic["w_air_g_kg"]).abs().max()
    assert max_diff < 2.0, (
        f"moisture trajectories diverged by {max_diff:.2f} g/kg -- more than the "
        "saturation-cap effect alone should plausibly cause"
    )
    # %RH depends on t_air_c too, so it can differ even more than w_air_g_kg does.
    assert thermo["rh_air_pct"].between(0.0, 100.0).all()
    assert antic["rh_air_pct"].between(0.0, 100.0).all()


def test_score_energy_matches_trajectory_sum(cfg, require_weather):
    traj = run_controller(lambda model: ThermostatController(model), cfg)
    result = score(traj, cfg)
    assert result.energy_kwh == pytest.approx(traj["electrical_w"].sum() / 60000.0)


def test_score_uses_the_configured_band(cfg, require_weather):
    """A wider configured band must never increase degree-hours."""
    import copy

    narrow_cfg = copy.deepcopy(cfg)
    narrow_cfg["comfort"]["band_k"] = 0.5
    wide_cfg = copy.deepcopy(cfg)
    wide_cfg["comfort"]["band_k"] = 5.0

    traj = run_controller(lambda model: ThermostatController(model), cfg)
    narrow_result = score(traj, narrow_cfg)
    wide_result = score(traj, wide_cfg)
    assert wide_result.degree_hours <= narrow_result.degree_hours
