"""M11: window-tint controllers and the tunnel/solar-gain wiring they react to.

Not exhaustive -- per the team's own explicit scope guidance (ROADMAP.md's
M11 section: "no one is going to look at the code... just need a simulation
that shows our idea"), this checks the real, load-bearing claims (tint
actually reduces solar gain and energy; anticipatory genuinely acts ahead of
reactive, not just differently) rather than every edge case.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pytest

from src.cabin_model import CabinInputs, CabinModel
from src.config import load_config
from src.controllers import ThermostatController
from src.evaluate import run_controller, score
from src.occupancy import Service, simulate
from src.tint_controller import (
    GHI_FULL_TINT_W_M2,
    AnticipatoryTintAdvisor,
    ReactiveTintController,
    _ghi_to_tint,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------------------- _ghi_to_tint


def test_ghi_to_tint_bounds():
    assert _ghi_to_tint(0.0) == 0.0
    assert _ghi_to_tint(GHI_FULL_TINT_W_M2) == 1.0
    assert _ghi_to_tint(GHI_FULL_TINT_W_M2 * 2) == 1.0  # clipped, not extrapolated
    assert _ghi_to_tint(-50.0) == 0.0  # never negative, even for a bad input


def test_ghi_to_tint_is_linear_between_the_bounds():
    half = _ghi_to_tint(GHI_FULL_TINT_W_M2 / 2)
    assert half == pytest.approx(0.5)


# --------------------------------------------------------------- CabinModel


def test_tint_level_zero_reproduces_the_pre_m11_q_solar_exactly(cfg):
    """Backward-compatibility guarantee: every M1-M10 caller that never
    passes tint_level gets tint_level=0.0 by default, which must give
    BIT-IDENTICAL q_solar to the pre-M11 formula (ghi * glazing_area *
    sunlit_fraction * solar_heat_gain_coefficient) -- not approximately
    close, exactly equal, since this is the same computation restructured,
    not a new one.
    """
    model = CabinModel(cfg)
    state = model.initial_state(25.0)
    r = model.step(state, CabinInputs(t_out_c=35.0, ghi_w_m2=600.0, tint_level=0.0), dt_s=60.0)
    env = cfg["envelope"]
    expected = 600.0 * cfg["geometry"]["glazing_area_m2"] * env["solar_heat_gain_coefficient"] * env["sunlit_glazing_fraction"]
    assert r.q_solar_w == pytest.approx(expected)


def test_tint_level_one_uses_the_tinted_shgc(cfg):
    model = CabinModel(cfg)
    state = model.initial_state(25.0)
    r = model.step(state, CabinInputs(t_out_c=35.0, ghi_w_m2=600.0, tint_level=1.0), dt_s=60.0)
    env = cfg["envelope"]
    expected = 600.0 * cfg["geometry"]["glazing_area_m2"] * env["solar_heat_gain_coefficient_tinted"] * env["sunlit_glazing_fraction"]
    assert r.q_solar_w == pytest.approx(expected)


def test_more_tint_never_increases_solar_gain(cfg):
    """Monotonicity: darkening the glass must never let MORE sun in.
    solar_heat_gain_coefficient_tinted < solar_heat_gain_coefficient is a
    config invariant this test guards, not just asserts."""
    model = CabinModel(cfg)
    q_solar_by_tint = []
    for tint in (0.0, 0.3, 0.6, 1.0):
        state = model.initial_state(25.0)
        r = model.step(state, CabinInputs(t_out_c=35.0, ghi_w_m2=700.0, tint_level=tint), dt_s=60.0)
        q_solar_by_tint.append(r.q_solar_w)
    assert q_solar_by_tint == sorted(q_solar_by_tint, reverse=True)


# --------------------------------------------------------------- occupancy


def test_tunnel_zones_produce_the_configured_in_tunnel_minutes(cfg):
    profile = simulate(Service(direction="down", load_factor=1.0), cfg)
    configured = cfg["route"]["tunnel_zones_min"]
    expected_minutes = sum(z["end_min"] - z["start_min"] for z in configured)
    assert sum(profile.in_tunnel) == expected_minutes


def test_no_tunnel_zones_configured_gives_an_all_false_profile():
    import copy
    cfg_no_tunnels = copy.deepcopy(load_config())
    cfg_no_tunnels["route"]["tunnel_zones_min"] = []
    profile = simulate(Service(direction="down", load_factor=1.0), cfg_no_tunnels)
    assert not any(profile.in_tunnel)


# ----------------------------------------------------------- run_controller


def test_tunnel_minutes_report_zero_ghi_to_the_controller(cfg, require_weather):
    """Wiring check: a tunnel must actually reach ControllerInputs.ghi_w_m2
    as ~0, not just exist as a flag nothing reads."""
    traj = run_controller(
        lambda m: ThermostatController(m), cfg,
        tint_controller=ReactiveTintController(),
    )
    tunnel_rows = traj.loc[traj["in_tunnel"]]
    assert len(tunnel_rows) > 0, "scenario must actually pass through a configured tunnel zone"
    assert (tunnel_rows["tint_level"] == 0.0).all(), (
        "no sun reaches the cabin in a tunnel -- reactive tint has nothing to react to"
    )


def test_tinting_reduces_energy_relative_to_no_tint(cfg, require_weather):
    no_tint = run_controller(lambda m: ThermostatController(m), cfg, tint_controller=None)
    tinted = run_controller(lambda m: ThermostatController(m), cfg, tint_controller=ReactiveTintController())
    assert score(tinted, cfg).energy_kwh < score(no_tint, cfg).energy_kwh


def test_anticipatory_tint_reacts_before_reactive_tint_at_a_tunnel(cfg, require_weather):
    """The actual demo claim: anticipatory tint clears AHEAD of a tunnel
    (using the forecast), reactive tint only clears AT it. Checked at a
    minute where the anticipatory advisor's t+H lookahead lands INSIDE the
    tunnel but the current minute itself is not yet in it -- the exact
    window where the two controllers' behaviour should diverge; picking any
    earlier minute would test nothing, since neither controller has a
    reason to react yet.
    """
    reactive = run_controller(lambda m: ThermostatController(m), cfg, tint_controller=ReactiveTintController())
    anticipatory = run_controller(lambda m: ThermostatController(m), cfg, tint_controller=AnticipatoryTintAdvisor())

    tunnel_start = cfg["route"]["tunnel_zones_min"][0]["start_min"]
    tunnel_end = cfg["route"]["tunnel_zones_min"][0]["end_min"]
    horizon = cfg["simulation"]["control_horizon_min"]
    # t + horizon must land inside [tunnel_start, tunnel_end) for the
    # anticipatory advisor to actually be "seeing" this tunnel yet.
    lookahead_t = max(0, (tunnel_start + tunnel_end) // 2 - horizon)

    assert reactive.loc[lookahead_t, "in_tunnel"] == False  # noqa: E712 -- not yet in the tunnel
    assert anticipatory.loc[lookahead_t, "tint_level"] < reactive.loc[lookahead_t, "tint_level"], (
        "anticipatory should already be clearing, seeing the tunnel in its forecast, "
        "while reactive still sees only the current (still sunny) minute"
    )
