"""Physics sanity tests for the grey-box cabin model.

These are the foundation of trust in every number downstream. If the simulator
disagrees with the time constants derived from the config, either the model or
the config is wrong — and catching that here costs minutes rather than
invalidating the whole evaluation later.
"""

import math

import pytest

from src.cabin_model import CabinInputs, CabinModel
from src.config import (
    air_time_constant,
    envelope_ua,
    load_config,
    thermal_time_constant,
    ventilation_ua,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def model(cfg):
    return CabinModel(cfg)


def run(model, minutes, inputs, state=None, t0=24.0, dt_s=60.0):
    state = state if state is not None else model.initial_state(t0)
    last = None
    for _ in range(int(minutes * 60 / dt_s)):
        last = model.step(state, inputs, dt_s=dt_s)
    return state, last


# --------------------------------------------------------------------- basics


def test_free_float_approaches_outdoor_temperature(model):
    """With HVAC off and nobody aboard, the cabin must tend to T_out."""
    t_out = 40.0
    state, _ = run(model, minutes=600, inputs=CabinInputs(t_out_c=t_out), t0=20.0)
    assert state.t_air_c == pytest.approx(t_out, abs=0.5)
    assert state.t_mass_c == pytest.approx(t_out, abs=0.5)


def test_free_float_converges_from_above_too(model):
    """Symmetry check: a hot cabin must cool to T_out, not just warm to it."""
    t_out = 15.0
    state, _ = run(model, minutes=600, inputs=CabinInputs(t_out_c=t_out), t0=45.0)
    assert state.t_air_c == pytest.approx(t_out, abs=0.5)


def test_no_gain_no_change_at_equilibrium(model):
    """Starting at T_out with no loads, nothing should move."""
    state, _ = run(model, minutes=120, inputs=CabinInputs(t_out_c=30.0), t0=30.0)
    assert state.t_air_c == pytest.approx(30.0, abs=1e-6)
    assert state.t_mass_c == pytest.approx(30.0, abs=1e-6)


# ------------------------------------------------------------------ linearity


def test_passenger_heat_load_scales_linearly(model):
    """Doubling passenger count must double the sensible passenger heat."""
    _, r40 = run(model, minutes=1, inputs=CabinInputs(t_out_c=30.0, n_pax=40))
    _, r80 = run(model, minutes=1, inputs=CabinInputs(t_out_c=30.0, n_pax=80))
    assert r80.q_passengers_w == pytest.approx(2.0 * r40.q_passengers_w)


def test_solar_gain_scales_linearly(model):
    """Solar gain is linear in irradiance."""
    _, r1 = run(model, minutes=1, inputs=CabinInputs(t_out_c=30.0, ghi_w_m2=400))
    _, r2 = run(model, minutes=1, inputs=CabinInputs(t_out_c=30.0, ghi_w_m2=800))
    assert r2.q_solar_w == pytest.approx(2.0 * r1.q_solar_w)


def test_solar_uses_only_the_sunlit_glazing_fraction(model, cfg):
    """Both sides of the coach are glazed but only one faces the sun."""
    _, r = run(model, minutes=1, inputs=CabinInputs(t_out_c=30.0, ghi_w_m2=1000))
    naive = 1000 * cfg["geometry"]["glazing_area_m2"] * cfg["envelope"]["solar_heat_gain_coefficient"]
    assert r.q_solar_w == pytest.approx(naive * cfg["envelope"]["sunlit_glazing_fraction"])
    assert r.q_solar_w < naive  # guards against the double-counting bug


# ------------------------------------------------------- derived time constants


def test_slow_mode_matches_derived_time_constant(model, cfg):
    """63% of a step response should land near the derived tau_slow.

    This is the test that ties the simulator to the config. tau is never
    written down — it is C_eff / (UA_env + UA_vent), and the simulation must
    reproduce it.
    """
    t0, t_out = 20.0, 40.0
    tau_min = thermal_time_constant(cfg, n_pax=0) / 60.0
    target = t0 + 0.632 * (t_out - t0)

    state = model.initial_state(t0)
    crossing = None
    for minute in range(1, 400):
        r = model.step(state, CabinInputs(t_out_c=t_out), dt_s=60.0)
        if crossing is None and r.t_mass_c >= target:
            crossing = minute
            break

    assert crossing is not None, "cabin never reached 63% of the step"
    assert crossing == pytest.approx(tau_min, rel=0.15), (
        f"simulated 63% time {crossing} min vs derived tau_slow {tau_min:.1f} min"
    )


def test_air_node_responds_much_faster_than_the_cabin(model, cfg):
    """The whole point of the 2-node model: a fast air mode must exist."""
    tau_fast = air_time_constant(cfg) / 60.0
    tau_slow = thermal_time_constant(cfg) / 60.0
    assert tau_fast < tau_slow / 5.0, "air node is not meaningfully faster than the mass"


def test_boarding_spike_is_visible_in_air_temperature(model):
    """A large boarding event must move air temperature sharply within a dwell.

    If this fails, the model has smeared out the disturbance that anticipatory
    control is supposed to catch, and the whole premise is untestable.
    """
    state = model.initial_state(24.0)
    _, r = run(model, minutes=8, inputs=CabinInputs(t_out_c=40.0, n_pax=68), state=state)
    rise = r.t_air_c - 24.0
    assert rise > 5.0, f"boarding spike only {rise:.2f} K — disturbance is too muted"
    assert r.t_air_c > r.t_mass_c, "air must lead the interior mass during a spike"


# ----------------------------------------------------------------- actuator


def test_capacity_limit_is_enforced(model):
    """Commanding more cooling than the unit has must clamp to its capacity."""
    huge = -10 * model.cooling_capacity_w
    state = model.initial_state(30.0)
    _, r = run(model, minutes=120, inputs=CabinInputs(t_out_c=40.0, q_hvac_cmd_w=huge), state=state)
    assert r.q_hvac_actual_w >= -model.cooling_capacity_w - 1e-6


def test_dead_time_delays_the_response(model):
    """Nothing should be delivered before the dead time has elapsed."""
    dead_min = model.dead_time_s / 60.0
    assert dead_min > 0, "this test assumes a non-zero dead time in config"

    state = model.initial_state(30.0)
    delivered = []
    for _ in range(int(dead_min * 2) + 4):
        r = model.step(state, CabinInputs(t_out_c=40.0, q_hvac_cmd_w=-20000.0), dt_s=60.0)
        delivered.append(abs(r.q_hvac_actual_w))

    assert delivered[0] == pytest.approx(0.0, abs=1.0), "power delivered before dead time elapsed"
    assert delivered[-1] > 1000.0, "power never arrived after the dead time"


def test_actuator_lag_is_first_order(model):
    """After the dead time, delivered power approaches the command exponentially."""
    cmd = -20000.0
    state = model.initial_state(30.0)
    inputs = CabinInputs(t_out_c=40.0, q_hvac_cmd_w=cmd)

    for _ in range(200):
        r = model.step(state, inputs, dt_s=60.0)
    assert r.q_hvac_actual_w == pytest.approx(cmd, rel=0.02), "never converged to the command"


def test_zero_tau_act_removes_the_lag(cfg):
    """tau_act = 0 must degenerate to instant delivery after the dead time.

    M6 sweeps tau_act down to zero, and the controller comparison relies on
    that endpoint behaving as a true no-lag reference.
    """
    import copy

    c = copy.deepcopy(cfg)
    c["hvac"]["tau_act_min"] = 0.0
    c["hvac"]["dead_time_min"] = 0.0
    model = CabinModel(c)

    state = model.initial_state(30.0)
    r = model.step(state, CabinInputs(t_out_c=40.0, q_hvac_cmd_w=-15000.0), dt_s=60.0)
    assert r.q_hvac_actual_w == pytest.approx(-15000.0)


# ------------------------------------------------------------------ energetics


def test_cop_degrades_with_outdoor_temperature(model):
    assert model.cop(45.0) < model.cop(30.0)


def test_cop_never_falls_below_its_floor(model, cfg):
    assert model.cop(200.0) == pytest.approx(cfg["hvac"]["cop_min"])


def test_electrical_power_reflects_cop(model):
    """Electrical draw is coil load divided by COP, so it exceeds nothing physical."""
    state = model.initial_state(26.0)
    _, r = run(model, minutes=60, inputs=CabinInputs(t_out_c=40.0, n_pax=60, q_hvac_cmd_w=-20000.0), state=state)
    expected = (abs(r.q_hvac_actual_w) + r.q_latent_w) / r.cop
    assert r.electrical_w == pytest.approx(expected)


def test_no_electrical_draw_when_hvac_is_off(model):
    _, r = run(model, minutes=10, inputs=CabinInputs(t_out_c=40.0, n_pax=60))
    assert r.electrical_w == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ numerical health


def test_substepping_does_not_change_the_answer(model):
    """A 60 s call and six 10 s calls must agree — proves the substepping works."""
    inputs = CabinInputs(t_out_c=45.0, n_pax=70, ghi_w_m2=900, q_hvac_cmd_w=-25000.0)

    a = model.initial_state(24.0)
    for _ in range(60):
        model.step(a, inputs, dt_s=60.0)

    b = model.initial_state(24.0)
    for _ in range(360):
        model.step(b, inputs, dt_s=10.0)

    assert a.t_air_c == pytest.approx(b.t_air_c, abs=0.05)


def test_state_stays_finite_under_extreme_forcing(model):
    """Guards against a stability blow-up that would silently poison the dataset."""
    state = model.initial_state(24.0)
    for _ in range(2000):
        r = model.step(
            state,
            CabinInputs(t_out_c=48.0, n_pax=80, ghi_w_m2=1000, door_open=True, q_hvac_cmd_w=-40000.0),
            dt_s=60.0,
        )
    assert math.isfinite(state.t_air_c) and math.isfinite(state.t_mass_c)
    assert -50.0 < state.t_air_c < 100.0


def test_door_opening_increases_heat_ingress(model):
    """An open door on a hot day must add heat, not remove it."""
    _, closed = run(model, minutes=1, inputs=CabinInputs(t_out_c=45.0, n_pax=70))
    _, opened = run(model, minutes=1, inputs=CabinInputs(t_out_c=45.0, n_pax=70, door_open=True))
    assert opened.q_door_w > 0.0
    assert closed.q_door_w == pytest.approx(0.0)
