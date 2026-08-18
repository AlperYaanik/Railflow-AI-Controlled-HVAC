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
    state, _ = run(model, minutes=600, inputs=CabinInputs(t_out_c=t_out, fan_on=False), t0=20.0)
    assert state.t_air_c == pytest.approx(t_out, abs=0.5)
    assert state.t_mass_c == pytest.approx(t_out, abs=0.5)


def test_free_float_converges_from_above_too(model):
    """Symmetry check: a hot cabin must cool to T_out, not just warm to it."""
    t_out = 15.0
    state, _ = run(model, minutes=600, inputs=CabinInputs(t_out_c=t_out, fan_on=False), t0=45.0)
    assert state.t_air_c == pytest.approx(t_out, abs=0.5)


def test_no_gain_no_change_at_equilibrium(model):
    """Starting at T_out with no loads, nothing should move."""
    state, _ = run(model, minutes=120, inputs=CabinInputs(t_out_c=30.0, fan_on=False), t0=30.0)
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
        r = model.step(state, CabinInputs(t_out_c=t_out, fan_on=False), dt_s=60.0)
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


def test_boarding_spike_is_visible_in_air_temperature(model, cfg):
    """A large boarding event must move air temperature sharply within a dwell.

    Deliberately ISOTHERMAL: outdoor temperature equals the starting cabin
    temperature, so passengers are the only forcing. An earlier version of this
    test ran T_out = 40 C against a 24 C cabin, where the outdoor gradient
    contributed 3.3x more heat than the passengers did — it looked like a
    boarding test but was mostly measuring envelope gain.

    The comfort band half-width is the yardstick: if a boarding event does not
    move the cabin by at least that much, anticipatory control has nothing to
    win and the project's premise is untestable.
    """
    t_amb = 30.0
    band = cfg["comfort"]["band_k"]

    state = model.initial_state(t_amb)
    _, r = run(model, minutes=8, inputs=CabinInputs(t_out_c=t_amb, n_pax=68, fan_on=False),
                state=state)

    rise = r.t_air_c - t_amb
    assert rise > band, (
        f"boarding raises air only {rise:.2f} K over 8 min, inside the +/-{band} K "
        "comfort band — the disturbance is too muted to control against"
    )
    assert r.t_air_c > r.t_mass_c, "air must lead the interior mass during a spike"


def test_single_node_would_understate_the_boarding_spike(model, cfg):
    """Justifies the 2R2C choice, and guards it against being 'simplified' later.

    Same isothermal boarding event, integrated as a single lumped node. If a
    future change collapses the two nodes into one, this test fails and says
    why.
    """
    from src.config import effective_heat_capacity, envelope_ua, ventilation_ua

    t_amb, n_pax, minutes = 30.0, 68, 8
    ua = envelope_ua(cfg) + ventilation_ua(cfg, n_pax)
    q_pax = n_pax * cfg["occupancy"]["sensible_heat_w_per_pax"]
    c_eff = effective_heat_capacity(cfg)

    t_lumped = t_amb
    for _ in range(minutes * 60):
        t_lumped += (q_pax + ua * (t_amb - t_lumped)) / c_eff

    state = model.initial_state(t_amb)
    _, r = run(model, minutes=minutes,
                inputs=CabinInputs(t_out_c=t_amb, n_pax=n_pax, fan_on=False), state=state)

    two_node_rise = r.t_air_c - t_amb
    one_node_rise = t_lumped - t_amb
    assert two_node_rise > 1.5 * one_node_rise, (
        f"two-node rise {two_node_rise:.2f} K vs single-node {one_node_rise:.2f} K — "
        "the fast air mode has been lost, which is the whole reason for 2R2C"
    )


def test_results_are_insensitive_to_interior_mass(cfg):
    """C_mass is our least defensible parameter, so prove it barely matters.

    It is an [ASSUMPTION] built from a mass budget. Varying it across the full
    plausible range must not move the quantities the controller actually sees.
    """
    import copy

    fast, spikes = [], []
    for c_mass_mj in (1.6, 4.34):
        c = copy.deepcopy(cfg)
        c["thermal_mass"]["interior_mass_capacity_j_k"] = c_mass_mj * 1e6
        m = CabinModel(c)
        fast.append(air_time_constant(c))

        state = m.initial_state(30.0)
        _, r = run(m, minutes=8, inputs=CabinInputs(t_out_c=30.0, n_pax=68, fan_on=False),
                     state=state)
        spikes.append(r.t_air_c - 30.0)

    assert abs(fast[1] - fast[0]) / fast[0] < 0.05, "tau_fast should not depend on C_mass"
    assert abs(spikes[1] - spikes[0]) / spikes[0] < 0.15, "boarding spike should not depend on C_mass"


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
    """Delivered power converges to the command, when the air stream can carry it.

    The command is kept modest on purpose: a large one would be clipped by the
    supply-air limit as the cabin cooled, and this test is about the lag, not
    about the clamp.
    """
    cmd = -8000.0
    state = model.initial_state(30.0)
    inputs = CabinInputs(t_out_c=40.0, q_hvac_cmd_w=cmd)

    for _ in range(200):
        r = model.step(state, inputs, dt_s=60.0)

    assert abs(cmd) < abs(model.deliverable_cooling_w(state.t_air_c)), (
        "test setup invalid: the supply-air limit is binding, so this is not "
        "measuring the lag"
    )
    assert r.q_hvac_actual_w == pytest.approx(cmd, rel=0.02), "never converged to the command"


# ---------------------------------------------------------------- supply air


def test_cooling_is_limited_by_the_supply_air_stream(model):
    """Rated capacity is not deliverable unless the air stream can carry it."""
    hot = model.deliverable_cooling_w(35.0)
    mild = model.deliverable_cooling_w(24.0)
    assert abs(mild) < abs(hot), "cooling authority must shrink as the cabin cools"
    assert abs(hot) <= model.cooling_capacity_w + 1e-6


def test_cooling_authority_vanishes_at_supply_temperature(model):
    """At supply temperature the stream can do nothing more."""
    assert model.deliverable_cooling_w(model.supply_min_c) == pytest.approx(0.0)
    assert model.deliverable_cooling_w(model.supply_min_c - 5.0) == pytest.approx(0.0)


def test_cabin_cannot_be_cooled_below_supply_temperature(model):
    """The physical floor. Without it the model cools without bound.

    Commanding maximum cooling forever previously drove the air node far below
    anything a coil could produce, which is what let a bang-bang baseline
    oscillate for a numerical rather than a physical reason.
    """
    state = model.initial_state(40.0)
    for _ in range(600):
        model.step(state, CabinInputs(t_out_c=45.0, q_hvac_cmd_w=-1e6), dt_s=60.0)
    assert state.t_air_c >= model.supply_min_c - 0.5


def test_delivered_power_is_clamped_as_the_cabin_cools(model):
    """The lag state must not keep delivering power the stream can no longer carry.

    Bounded against the temperature at the START of each step. While cooling,
    temperature falls monotonically through the step and the limit falls with
    it, so the start-of-step limit is the largest value any substep could
    legitimately have used.
    """
    state = model.initial_state(38.0)
    inputs = CabinInputs(t_out_c=42.0, q_hvac_cmd_w=-model.cooling_capacity_w)

    for _ in range(400):
        limit_at_start = abs(model.deliverable_cooling_w(state.t_air_c))
        r = model.step(state, inputs, dt_s=60.0)
        assert abs(r.q_hvac_actual_w) <= limit_at_start + 1.0


def test_pulldown_is_faster_from_a_hotter_cabin(model):
    """A hot cabin has more cooling authority, so early pull-down is quicker.

    This is the behaviour that makes the limit worth modelling: authority is
    state-dependent, so a controller cannot assume full capacity is always on
    tap near setpoint.
    """
    def drop_over(minutes, t_start):
        state = model.initial_state(t_start)
        for _ in range(minutes):
            model.step(state, CabinInputs(t_out_c=45.0, q_hvac_cmd_w=-1e6), dt_s=60.0)
        return t_start - state.t_air_c

    assert drop_over(5, 40.0) > drop_over(5, 27.0)


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


def test_cop_cooling_degrades_with_outdoor_temperature(model):
    assert model.cop_cooling(45.0) < model.cop_cooling(30.0)


def test_cop_cooling_never_falls_below_its_floor(model, cfg):
    assert model.cop_cooling(200.0) == pytest.approx(cfg["hvac"]["cop_min"])


def test_electrical_power_reflects_cop(model):
    """Compressor draw is coil load over COP; total adds the supply fan."""
    state = model.initial_state(26.0)
    _, r = run(model, minutes=60,
               inputs=CabinInputs(t_out_c=40.0, n_pax=60, q_hvac_cmd_w=-20000.0), state=state)

    expected_compressor = (abs(r.q_hvac_actual_w) + r.q_latent_w) / r.cop
    assert r.compressor_w == pytest.approx(expected_compressor)
    assert r.electrical_w == pytest.approx(r.compressor_w + r.fan_w)


def test_only_the_fan_draws_power_when_cooling_is_off(model):
    """With no cooling called for, the fan is still running and still costs.

    This is the whole reason fan power matters: it is a floor on energy that
    no controller can reduce, so omitting it inflates any percentage saving.
    """
    _, r = run(model, minutes=10, inputs=CabinInputs(t_out_c=40.0, n_pax=60))
    assert r.compressor_w == pytest.approx(0.0, abs=1e-9)
    assert r.electrical_w == pytest.approx(model.supply_fan_w)


def test_no_draw_at_all_when_the_fan_is_off(model):
    _, r = run(model, minutes=10, inputs=CabinInputs(t_out_c=40.0, n_pax=60, fan_on=False))
    assert r.electrical_w == pytest.approx(0.0, abs=1e-9)


def test_fan_heat_enters_the_cabin(model):
    """The motor sits in the air stream, so its power lands as sensible heat."""
    warm = run(model, minutes=45, inputs=CabinInputs(t_out_c=25.0), t0=25.0)[0]
    cool = run(model, minutes=45, inputs=CabinInputs(t_out_c=25.0, fan_on=False), t0=25.0)[0]
    assert warm.t_air_c > cool.t_air_c + 0.5, "fan heat is not reaching the cabin"


def test_fan_power_matches_specific_fan_power(model, cfg):
    """Derived from SFP and airflow, not written down as a wattage."""
    hv = cfg["hvac"]
    expected = hv["supply_fan_sfp_kw_per_m3s"] * 1000.0 * hv["supply_air_m3_h"] / 3600.0
    assert model.supply_fan_w == pytest.approx(expected)


# -------------------------------------------------------------------- heating
#
# Mirrors the cooling actuator/energetics tests above. This section did not
# exist before cop() was split into cop_cooling()/cop_heating() — every
# actuator test up to that point commanded cooling only, which is how a
# heating COP formula that read backwards (COP 3.8 at 5 C, for what is
# actually a resistive heater) went unnoticed.


def test_cop_heating_is_fixed_regardless_of_outdoor_temperature(model):
    """Resistive heating is a physical identity, not a fitted curve."""
    assert model.cop_heating(-10.0) == model.cop_heating(5.0) == model.cop_heating(30.0)


def test_cop_heating_matches_the_configured_value(model, cfg):
    assert model.cop_heating(5.0) == pytest.approx(cfg["hvac"]["heating_cop"])


def test_heating_and_cooling_disagree_at_the_same_temperature(model):
    """Regression guard for the actual bug: they must never be reunified.

    Before the split, both modes shared one formula built for cooling. At
    5 C that produced COP 3.8 for what is really a resistive heater — a
    heat pump reading applied to equipment that isn't a heat pump.
    """
    t_out = 5.0
    assert model.cop_heating(t_out) != pytest.approx(model.cop_cooling(t_out))
    assert model.cop_heating(t_out) == pytest.approx(1.0)
    assert model.cop_cooling(t_out) > 3.0, "cooling COP formula has changed unexpectedly"


def test_cop_dispatch_follows_the_sign_of_delivered_power(model):
    """step() must select cop_heating when q_hvac ends up positive, not by t_out_c."""
    cold_state = model.initial_state(5.0)
    _, heating_result = run(
        model, minutes=30, inputs=CabinInputs(t_out_c=5.0, q_hvac_cmd_w=15000.0),
        state=cold_state,
    )
    assert heating_result.q_hvac_actual_w > 0.0, "test setup did not actually heat"
    assert heating_result.cop == pytest.approx(model.cop_heating(5.0))

    hot_state = model.initial_state(35.0)
    _, cooling_result = run(
        model, minutes=30, inputs=CabinInputs(t_out_c=40.0, q_hvac_cmd_w=-15000.0),
        state=hot_state,
    )
    assert cooling_result.q_hvac_actual_w < 0.0, "test setup did not actually cool"
    assert cooling_result.cop == pytest.approx(model.cop_cooling(40.0))


def test_heating_dead_time_delays_the_response(model):
    """Mirrors test_dead_time_delays_the_response for the heating direction."""
    dead_min = model.dead_time_s / 60.0
    assert dead_min > 0, "this test assumes a non-zero dead time in config"

    state = model.initial_state(5.0)
    delivered = []
    for _ in range(int(dead_min * 2) + 4):
        r = model.step(state, CabinInputs(t_out_c=5.0, q_hvac_cmd_w=20000.0), dt_s=60.0)
        delivered.append(r.q_hvac_actual_w)

    assert delivered[0] == pytest.approx(0.0, abs=1.0), "power delivered before dead time elapsed"
    assert delivered[-1] > 1000.0, "power never arrived after the dead time"


def test_heating_lag_is_first_order(model):
    """Mirrors test_actuator_lag_is_first_order for the heating direction."""
    cmd = 8000.0
    state = model.initial_state(5.0)
    inputs = CabinInputs(t_out_c=5.0, q_hvac_cmd_w=cmd)

    for _ in range(200):
        r = model.step(state, inputs, dt_s=60.0)

    assert cmd < model.deliverable_heating_w(state.t_air_c), (
        "test setup invalid: the heating limit is binding, so this is not "
        "measuring the lag"
    )
    assert r.q_hvac_actual_w == pytest.approx(cmd, rel=0.02), "never converged to the command"


def test_heating_capacity_limit_is_enforced(model):
    """Mirrors test_capacity_limit_is_enforced for the heating direction."""
    huge = 10 * model.heating_capacity_w
    state = model.initial_state(5.0)
    _, r = run(model, minutes=120, inputs=CabinInputs(t_out_c=5.0, q_hvac_cmd_w=huge), state=state)
    assert r.q_hvac_actual_w <= model.heating_capacity_w + 1e-6


def test_heating_electrical_draw_equals_thermal_output(model):
    """With COP = 1.0 exactly, electrical input must equal heat delivered.

    The cleanest possible check on the fix: before it, this ratio was 3.8,
    not 1.0, because the cooling formula was degrading COP in the wrong
    direction for a resistive heater.
    """
    state = model.initial_state(5.0)
    _, r = run(model, minutes=60, inputs=CabinInputs(t_out_c=5.0, q_hvac_cmd_w=15000.0), state=state)
    assert r.compressor_w == pytest.approx(abs(r.q_hvac_actual_w))


def test_no_latent_load_is_charged_during_heating(model):
    """Only cooling dehumidifies; heating must never carry a latent term."""
    state = model.initial_state(5.0)
    _, r = run(model, minutes=30,
               inputs=CabinInputs(t_out_c=5.0, n_pax=60, q_hvac_cmd_w=15000.0), state=state)
    assert r.q_hvac_actual_w > 0.0, "test setup did not actually heat"
    assert r.compressor_w == pytest.approx(abs(r.q_hvac_actual_w))


def test_zero_tau_act_removes_the_lag_when_heating(cfg):
    """Heating counterpart of test_zero_tau_act_removes_the_lag.

    M6's sweep runs both directions of the actuator dead-time/lag test; the
    endpoint must degenerate correctly regardless of which way the command
    points.
    """
    import copy

    c = copy.deepcopy(cfg)
    c["hvac"]["tau_act_min"] = 0.0
    c["hvac"]["dead_time_min"] = 0.0
    model = CabinModel(c)

    state = model.initial_state(5.0)
    r = model.step(state, CabinInputs(t_out_c=5.0, q_hvac_cmd_w=15000.0), dt_s=60.0)
    assert r.q_hvac_actual_w == pytest.approx(15000.0)


# ------------------------------------------------------------ numerical health


def test_outer_step_size_does_not_change_the_answer(model):
    """Bookkeeping check: calling in 60 s chunks must equal calling in 10 s chunks.

    Note what this does NOT test. Both paths integrate internally at
    SUBSTEP_S, so this cannot detect discretisation error — it only proves the
    outer/inner loop and the actuator pipeline stay consistent across call
    granularity. Convergence is tested separately below.
    """
    inputs = CabinInputs(t_out_c=45.0, n_pax=70, ghi_w_m2=900, q_hvac_cmd_w=-25000.0)

    a = model.initial_state(24.0)
    for _ in range(60):
        model.step(a, inputs, dt_s=60.0)

    b = model.initial_state(24.0)
    for _ in range(360):
        model.step(b, inputs, dt_s=10.0)

    assert a.t_air_c == pytest.approx(b.t_air_c, abs=0.05)


def test_integration_has_converged_at_the_chosen_substep(cfg):
    """The real convergence test: is SUBSTEP_S small enough to be accurate?

    Explicit Euler against a 0.25 s reference over a full hour of hard forcing.
    The fast mode is ~1.1 min, so this is the step size that matters.
    """
    import src.cabin_model as cm

    inputs = CabinInputs(t_out_c=45.0, n_pax=70, ghi_w_m2=900, q_hvac_cmd_w=-25000.0)

    def run_at(substep_s):
        original = cm.SUBSTEP_S
        cm.SUBSTEP_S = substep_s
        try:
            m = cm.CabinModel(cfg)
            state = m.initial_state(24.0)
            for _ in range(60):
                m.step(state, inputs, dt_s=60.0)
            return state.t_air_c
        finally:
            cm.SUBSTEP_S = original

    reference = run_at(0.25)
    chosen = run_at(cm.SUBSTEP_S)
    coarse = run_at(60.0)

    assert abs(chosen - reference) < 0.05, (
        f"SUBSTEP_S={cm.SUBSTEP_S}s gives {chosen - reference:+.4f} K error vs a 0.25 s reference"
    )
    # Sanity: the test can actually detect error, i.e. a bad step size fails it.
    assert abs(coarse - reference) > abs(chosen - reference), (
        "a 60 s step should be measurably worse — otherwise this test proves nothing"
    )


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


def test_door_physics_is_independent_of_the_timetable(cfg):
    """Editing the route must not change how fast air crosses an open door.

    An earlier version expressed door infiltration as a total per stop divided
    by the route's mean dwell. Lengthening one station's dwell by 12 minutes
    then moved door_ua by 36% at *every* station — the timetable was silently
    rewriting the physics.
    """
    import copy

    baseline = CabinModel(cfg).door_ua

    edited = copy.deepcopy(cfg)
    edited["route"]["stations"][0]["dwell_min"] = 20
    assert CabinModel(edited).door_ua == pytest.approx(baseline)

    fewer = copy.deepcopy(cfg)
    fewer["route"]["stations"] = fewer["route"]["stations"][:3]
    assert CabinModel(fewer).door_ua == pytest.approx(baseline)


def test_door_exchange_accumulates_with_dwell_length(cfg):
    """A longer stop must let in proportionally more heat."""
    model = CabinModel(cfg)
    inputs = CabinInputs(t_out_c=45.0, n_pax=70, door_open=True)

    short_state = model.initial_state(25.0)
    short = sum(model.step(short_state, inputs, dt_s=60.0).q_door_w for _ in range(2))

    long_state = model.initial_state(25.0)
    long = sum(model.step(long_state, inputs, dt_s=60.0).q_door_w for _ in range(6))

    assert long > 2.0 * short, "tripling the dwell should let in substantially more heat"
