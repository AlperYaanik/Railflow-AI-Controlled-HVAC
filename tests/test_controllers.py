"""Tests for M5's setpoint advisors and the shared PlantResponse.

Two real bugs were found here, both invisible from reading the code and both
found only by running a real simulation and tracing it minute by minute, not
by checking an aggregate summary:

1. A sign bug in AnticipatorySetpointAdvisor (then AnticipatoryController)
   -- the exact same class of mistake already made once in
   data_generator.py's StochasticController: error computed backwards, so
   the dispatch sent cooling-direction fractions to heating. Fixing only the
   error sign produced NO visible change in the aggregate test, because a
   SECOND, independent sign bug in the dispatch line was silently cancelling
   it out. Two bugs stacked to look like "the fix did nothing" -- worth
   remembering as a debugging lesson as much as a code lesson. UNDER M10:
   that second bug's entire class is now structurally impossible -- the
   dispatch tail that used to branch on a capacity sign no longer exists
   (see AnticipatorySetpointAdvisor's docstring in src/controllers.py). The
   test that specifically covered it,
   test_anticipatory_command_sign_matches_frac_sign, was retired for that
   reason rather than ported; its remaining sign-direction coverage lives on
   in test_anticipatory_setpoint_shift_sign_matches_error_sign below.

2. ThermostatController, run dual-mode (heat below the lower threshold, cool
   above the upper one), oscillated between full heat and full cool 18 times
   in a 166-minute journey -- an 8+ K limit cycle, not the intended +/-1 K
   band. This cabin's air node is fast enough (tau_fast ~1.1-1.3 min) that a
   full-power command overshoots the opposite switching threshold before the
   actuator's own dead-time/lag can arrest it. Fixed by making the baseline
   cooling-only, matching every other baseline already in this project and
   the documented cooling-dominated climate. This hysteresis mechanism now
   lives on PlantResponse (M10), shared verbatim by every advisor -- see its
   own tests near the bottom of this file.

Both are why every test below runs the REAL simulator (weather, occupancy,
cabin model) rather than only checking the control law in isolation --
that's what actually exposed them.
"""

import lightgbm as lgb
import numpy as np
import pytest

from src.cabin_model import CabinInputs, CabinModel
from src.config import load_config
from src.controllers import (
    AnticipatorySetpointAdvisor,
    ControllerInputs,
    PlantResponse,
    ThermostatController,
)
from src.evaluate import run_controller
from src.features import LiveFeatureBuilder
from src.train import MODEL_PATH


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def require_model(require_weather):
    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")


def _anticipatory_factory(cfg, booster, **kwargs):
    def factory(model):
        builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                      pattern="semi_express", load_factor=1.0, depart_hour=8.0)
        return AnticipatorySetpointAdvisor(model, booster, builder, **kwargs)
    return factory


# --------------------------------------------------------- ThermostatController


def test_thermostat_never_oscillates_between_extremes(cfg):
    """Regression guard for the exact bug found: dual-mode bang-bang on this
    fast plant oscillated 21.8-30 C, 18 switches in 166 minutes. Cooling-only
    must not reproduce anything close to that.
    """
    df = run_controller(lambda model: ThermostatController(model), cfg)
    assert df["t_air_c"].max() - df["t_air_c"].min() < 12.0, (
        "T_air swings more than a full comfort-band multiple -- possible "
        "regression toward the dual-mode oscillation bug"
    )
    assert df["err_c"].min() > -5.0, "overcooling far below setpoint -- check for overshoot"


def test_thermostat_is_cooling_only(cfg):
    df = run_controller(lambda model: ThermostatController(model), cfg)
    assert (df["cmd_w"] <= 0.0).all(), "ThermostatController must never command heating"


def test_thermostat_switches_at_a_sane_rate(cfg):
    """Some cycling is expected and correct for bang-bang control; the bug
    produced 18 switches from HEAT<->COOL flip-flopping specifically. This
    bounds total switch count generously -- not asserting a tight number,
    since normal cycling rate depends on the day -- as a blow-up guard.
    """
    df = run_controller(lambda model: ThermostatController(model), cfg)
    on = df["cmd_w"] < 0.0
    switches = int((on != on.shift(1)).sum())
    assert switches < 60, f"{switches} on/off switches in {len(df)} minutes is excessive cycling"


def test_thermostat_respects_the_hysteresis_band_config(cfg):
    """Changing the config value must actually change behaviour -- a wiring
    check, not just a physics check."""
    import copy

    narrow = copy.deepcopy(cfg)
    narrow["hvac"]["thermostat_hysteresis_k"] = 0.5
    wide = copy.deepcopy(cfg)
    wide["hvac"]["thermostat_hysteresis_k"] = 4.0

    df_narrow = run_controller(lambda model: ThermostatController(model), narrow)
    df_wide = run_controller(lambda model: ThermostatController(model), wide)

    def switch_count(df):
        on = df["cmd_w"] < 0.0
        return int((on != on.shift(1)).sum())

    assert switch_count(df_narrow) >= switch_count(df_wide), (
        "a narrower hysteresis band should cycle at least as often as a wider one"
    )


def test_thermostat_actuator_limits_are_never_exceeded(cfg):
    """Loose start/end-of-row bound -- same methodology as M2/M4's clamp
    tests, needed because each 60 s row is 6x10s substeps and T_air moves
    within it."""
    model_probe = CabinModel(cfg)
    df = run_controller(lambda model: ThermostatController(model), cfg)
    df["t_air_start"] = df["t_air_c"].shift(1).fillna(df["t_air_c"])

    limit = np.minimum(
        df["t_air_c"].apply(model_probe.deliverable_cooling_w),
        df["t_air_start"].apply(model_probe.deliverable_cooling_w),
    )
    cooling = df["q_actual_w"] < -1.0
    violation = cooling & (df["q_actual_w"] < limit - 5.0)
    assert violation.sum() == 0


# --------------------------------------------------- AnticipatorySetpointAdvisor


def test_anticipatory_cools_when_hot_not_heats(require_model, cfg):
    """Regression guard for the exact bug found: current_error/predicted_error
    computed with the wrong sign, then a second, independently wrong sign in
    the dispatch line that cancelled the first fix and looked like no change
    had happened. A hot day (T_out to 36 C) must end near setpoint, not at
    43 C from actively fighting itself with heat.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = run_controller(_anticipatory_factory(cfg, booster), cfg)
    assert df["t_air_c"].iloc[-1] < 30.0, (
        f"final T_air {df['t_air_c'].iloc[-1]:.1f} C -- controller may be "
        "heating instead of cooling on a hot day"
    )
    assert df["err_c"].mean() < 2.0


def test_anticipatory_setpoint_shift_sign_matches_error_sign(require_model, cfg):
    """Direct check on the control law itself, isolated from the full
    simulation: hot -> negative shift (call for more cooling), cold ->
    positive shift (tolerate a warmer cabin / call for heating headroom).

    Formerly test_anticipatory_frac_sign_matches_error_sign, checking
    ctrl.last_frac; renamed and repointed at ctrl.last_setpoint_shift_c under
    M10 (see module docstring). test_anticipatory_command_sign_matches_frac_sign,
    which used to separately check that the watts dispatch agreed in sign
    with frac, has NO replacement here: that second dispatch step doesn't
    exist anymore (recommend_setpoint() returns the shift-adjusted setpoint
    directly, with no further sign-dependent branch downstream in this
    class), so there is no second sign decision left to test.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model = CabinModel(cfg)
    builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    ctrl = AnticipatorySetpointAdvisor(model, booster, builder)

    # Warm up past the LiveFeatureBuilder history requirement first.
    warm_inputs = ControllerInputs(
        t_air_c=26.0, t_mass_c=26.0, t_out_c=30.0, ghi_w_m2=400.0, n_pax=30,
        door_open=False, q_hvac_actual_w=0.0, time_to_next_station_min=20,
        expected_boarding=5, t_out_fcst_h=30.0, ghi_fcst_h=400.0,
    )
    for _ in range(25):
        ctrl.recommend_setpoint(warm_inputs)
    assert ctrl.last_setpoint_shift_c is not None  # warmup path exercised

    hot = ControllerInputs(t_air_c=40.0, t_mass_c=38.0, t_out_c=35.0, ghi_w_m2=600.0,
                            n_pax=60, door_open=False, q_hvac_actual_w=0.0,
                            time_to_next_station_min=20, expected_boarding=10,
                            t_out_fcst_h=35.0, ghi_fcst_h=600.0)
    ctrl.recommend_setpoint(hot)
    assert ctrl.last_setpoint_shift_c < 0.0, (
        "cabin far above setpoint must produce a cooling-direction (negative) shift"
    )

    cold = ControllerInputs(t_air_c=10.0, t_mass_c=12.0, t_out_c=15.0, ghi_w_m2=0.0,
                             n_pax=0, door_open=False, q_hvac_actual_w=0.0,
                             time_to_next_station_min=20, expected_boarding=0,
                             t_out_fcst_h=15.0, ghi_fcst_h=0.0)
    ctrl.recommend_setpoint(cold)
    assert ctrl.last_setpoint_shift_c > 0.0, (
        "cabin far below setpoint must produce a heating-direction (positive) shift"
    )


def test_anticipatory_actuator_limits_are_never_exceeded(require_model, cfg):
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model_probe = CabinModel(cfg)
    df = run_controller(_anticipatory_factory(cfg, booster), cfg)
    df["t_air_start"] = df["t_air_c"].shift(1).fillna(df["t_air_c"])

    cool_limit = np.minimum(
        df["t_air_c"].apply(model_probe.deliverable_cooling_w),
        df["t_air_start"].apply(model_probe.deliverable_cooling_w),
    )
    heat_limit = np.maximum(
        df["t_air_c"].apply(model_probe.deliverable_heating_w),
        df["t_air_start"].apply(model_probe.deliverable_heating_w),
    )
    cooling = df["q_actual_w"] < -1.0
    heating = df["q_actual_w"] > 1.0
    assert (cooling & (df["q_actual_w"] < cool_limit - 5.0)).sum() == 0
    assert (heating & (df["q_actual_w"] > heat_limit + 5.0)).sum() == 0


def test_anticipatory_degenerates_to_pure_feedback_at_ff_weight_zero(require_model, cfg):
    """ff_weight=0 must ignore the forecast entirely and match a plain
    proportional-feedback advisor. Correctness check on the blend formula,
    not just a plausibility check. max_shift_k=3.0 is this test's own
    arbitrary bound, chosen to keep the manual hand-computation below simple
    -- unrelated to AnticipatorySetpointAdvisor's production default (4.0,
    still an unrelated [ASSUMPTION], see its docstring).
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = run_controller(_anticipatory_factory(cfg, booster, ff_weight=0.0, max_shift_k=3.0), cfg)

    model = CabinModel(cfg)
    setpoint = model.setpoint(35.0)
    half_band = cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
    # With ff_weight=0, raw = (setpoint - t_air), deadbanded (see the class
    # docstring), then clipped to +/-max_shift_k -- verify on one manual
    # point rather than the whole trajectory.
    ctrl = AnticipatorySetpointAdvisor(
        model, booster,
        LiveFeatureBuilder(cfg, "cairo", "down", "semi_express", 1.0, 8.0),
        ff_weight=0.0, max_shift_k=3.0,
    )
    inputs = ControllerInputs(t_air_c=30.0, t_mass_c=29.0, t_out_c=35.0, ghi_w_m2=500.0,
                               n_pax=40, door_open=False, q_hvac_actual_w=-5000.0,
                               time_to_next_station_min=20, expected_boarding=10,
                               t_out_fcst_h=36.0, ghi_fcst_h=520.0)
    ctrl.recommend_setpoint(inputs)
    raw = setpoint - 30.0
    excess = abs(raw) - half_band
    raw_db = np.sign(raw) * excess if excess > 0.0 else 0.0
    expected_shift = np.clip(raw_db, -3.0, 3.0)
    assert ctrl.last_setpoint_shift_c == pytest.approx(expected_shift)


def test_anticipatory_deadband_zeroes_a_small_error(require_model, cfg):
    """Regression guard for the exact mechanism the deadband fixes: an error
    smaller than half the hysteresis band must produce a shift EXACTLY 0 (the
    recommendation exactly equals setpoint_now), not a small nonzero nudge.
    Before this was added, the advisor was almost never truly recommending
    "no change" (0.6% of minutes on the canonical scenario, vs
    ThermostatController's 32.5% fully-off), which under the pre-M10 watts
    dispatch meant continuous low-power modulation -- why it used MORE
    energy than the baseline on most of the M4 test split. See
    docs/PARAMETERS.md's M5 correction log.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model = CabinModel(cfg)
    half_band = cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
    ctrl = AnticipatorySetpointAdvisor(
        model, booster,
        LiveFeatureBuilder(cfg, "cairo", "down", "semi_express", 1.0, 8.0),
        ff_weight=0.0, max_shift_k=3.0,
    )
    setpoint = model.setpoint(35.0)
    small_error_t_air = setpoint - (half_band * 0.5)  # inside the deadband
    inputs = ControllerInputs(t_air_c=small_error_t_air, t_mass_c=small_error_t_air, t_out_c=35.0,
                               ghi_w_m2=500.0, n_pax=40, door_open=False, q_hvac_actual_w=0.0,
                               time_to_next_station_min=20, expected_boarding=10,
                               t_out_fcst_h=36.0, ghi_fcst_h=520.0)
    recommended = ctrl.recommend_setpoint(inputs)
    assert ctrl.last_setpoint_shift_c == 0.0
    assert recommended == pytest.approx(setpoint)


@pytest.mark.parametrize("ff_weight", [0.0, 0.3, 0.6, 0.9, 1.0])
def test_anticipatory_sensitivity_to_feedforward_weight(require_model, cfg, ff_weight):
    """Sweeps ff_weight, the way every other under-sourced parameter in this
    project is characterised rather than asserted correct (see
    docs/PARAMETERS.md's calibration priority list). Records behaviour;
    the only hard requirement is that nothing goes unphysical at any setting.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = run_controller(_anticipatory_factory(cfg, booster, ff_weight=ff_weight), cfg)
    assert np.isfinite(df["t_air_c"]).all()
    assert df["t_air_c"].between(-10, 60).all()


def test_anticipatory_warmup_fallback_is_pure_feedback(cfg):
    """Before LiveFeatureBuilder has enough history, the advisor must still
    recommend something sensible (pure feedback) rather than doing nothing
    or crashing.
    """
    model = CabinModel(cfg)
    booster = lgb.Booster(model_file=str(MODEL_PATH)) if MODEL_PATH.exists() else None
    if booster is None:
        pytest.skip("no saved model. Run:  python -m src.train")
    builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    ctrl = AnticipatorySetpointAdvisor(model, booster, builder)

    hot_inputs = ControllerInputs(t_air_c=35.0, t_mass_c=33.0, t_out_c=32.0, ghi_w_m2=500.0,
                                   n_pax=40, door_open=False, q_hvac_actual_w=0.0,
                                   time_to_next_station_min=20, expected_boarding=10,
                                   t_out_fcst_h=33.0, ghi_fcst_h=520.0)
    recommended = ctrl.recommend_setpoint(hot_inputs)  # minute 0: still in warmup
    setpoint_now = model.setpoint(hot_inputs.t_out_c)
    assert recommended < setpoint_now, (
        "warmup fallback should still call for more cooling when clearly too hot"
    )


# ----------------------------------------------------------------- PlantResponse


def test_plant_response_reacts_to_a_discontinuous_setpoint_jump(cfg):
    """Under M10, PlantResponse can be fed a DIFFERENT setpoint every call --
    e.g. ThermostatController's slowly-varying schedule one minute and
    AnticipatorySetpointAdvisor's shifted recommendation the next, if a
    caller ever swapped advisors mid-run. Confirms it reacts to whichever
    setpoint it's given THIS call, not one remembered from a previous call --
    it should hold no state beyond the on/off flag itself. Previously
    impossible to exercise: ThermostatController's own internally-derived
    setpoint only ever varied slowly with t_out_c.
    """
    model = CabinModel(cfg)
    plant = PlantResponse(model)

    # Air well above a low setpoint -- must turn cooling on.
    cmd1 = plant.respond(t_air_c=30.0, setpoint_c=22.0)
    assert cmd1 < 0.0

    # Same air temperature, but a setpoint jump puts it back inside the band
    # around a much higher target -- hysteresis HOLDS the on state (by
    # design, see the class docstring), so still cooling.
    cmd2 = plant.respond(t_air_c=30.0, setpoint_c=29.0)
    assert cmd2 < 0.0

    # Air now clearly below the new, higher setpoint's lower threshold --
    # must turn off.
    cmd3 = plant.respond(t_air_c=27.0, setpoint_c=29.0)
    assert cmd3 == 0.0
