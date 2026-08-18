"""Tests for M5's two controllers.

Two real bugs were found here, both invisible from reading the code and both
found only by running a real simulation and tracing it minute by minute, not
by checking an aggregate summary:

1. A sign bug in AnticipatoryController -- the exact same class of mistake
   already made once in data_generator.py's StochasticController: error
   computed backwards, so the dispatch sent cooling-direction fractions to
   heating. Fixing only the error sign produced NO visible change in the
   aggregate test, because a SECOND, independent sign bug in the dispatch
   line was silently cancelling it out. Two bugs stacked to look like "the
   fix did nothing" -- worth remembering as a debugging lesson as much as a
   code lesson.

2. ThermostatController, run dual-mode (heat below the lower threshold, cool
   above the upper one), oscillated between full heat and full cool 18 times
   in a 166-minute journey -- an 8+ K limit cycle, not the intended +/-1 K
   band. This cabin's air node is fast enough (tau_fast ~1.1-1.3 min) that a
   full-power command overshoots the opposite switching threshold before the
   actuator's own dead-time/lag can arrest it. Fixed by making the baseline
   cooling-only, matching every other baseline already in this project and
   the documented cooling-dominated climate.

Both are why every test below runs the REAL simulator (weather, occupancy,
cabin model) rather than only checking the control law in isolation --
that's what actually exposed them.
"""

import lightgbm as lgb
import numpy as np
import pytest

from src.cabin_model import CabinInputs, CabinModel
from src.config import load_config
from src.controllers import AnticipatoryController, ControllerInputs, ThermostatController
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
        return AnticipatoryController(model, booster, builder, **kwargs)
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


# ------------------------------------------------------- AnticipatoryController


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


def test_anticipatory_frac_sign_matches_error_sign(require_model, cfg):
    """Direct check on the control law itself, isolated from the full
    simulation: hot -> negative frac (cool), cold -> positive frac (heat).
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model = CabinModel(cfg)
    builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    ctrl = AnticipatoryController(model, booster, builder)

    # Warm up past the LiveFeatureBuilder history requirement first.
    warm_inputs = ControllerInputs(
        t_air_c=26.0, t_mass_c=26.0, t_out_c=30.0, ghi_w_m2=400.0, n_pax=30,
        door_open=False, q_hvac_actual_w=0.0, time_to_next_station_min=20,
        expected_boarding=5, t_out_fcst_h=30.0, ghi_fcst_h=400.0,
    )
    for _ in range(25):
        ctrl.command(warm_inputs)
    assert ctrl.last_frac is not None  # warmup path exercised

    hot = ControllerInputs(t_air_c=40.0, t_mass_c=38.0, t_out_c=35.0, ghi_w_m2=600.0,
                            n_pax=60, door_open=False, q_hvac_actual_w=0.0,
                            time_to_next_station_min=20, expected_boarding=10,
                            t_out_fcst_h=35.0, ghi_fcst_h=600.0)
    ctrl.command(hot)
    assert ctrl.last_frac < 0.0, "cabin far above setpoint must produce a cooling (negative) frac"

    cold = ControllerInputs(t_air_c=10.0, t_mass_c=12.0, t_out_c=15.0, ghi_w_m2=0.0,
                             n_pax=0, door_open=False, q_hvac_actual_w=0.0,
                             time_to_next_station_min=20, expected_boarding=0,
                             t_out_fcst_h=15.0, ghi_fcst_h=0.0)
    ctrl.command(cold)
    assert ctrl.last_frac > 0.0, "cabin far below setpoint must produce a heating (positive) frac"


def test_anticipatory_command_sign_matches_frac_sign(require_model, cfg):
    """The dispatch bug specifically: frac and the returned command must
    always agree on sign. cmd < 0 (cooling) iff frac < 0.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = run_controller(_anticipatory_factory(cfg, booster), cfg)
    # Re-run and inspect frac alongside cmd directly, rather than inferring.
    model = CabinModel(cfg)
    builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    ctrl = AnticipatoryController(model, booster, builder)
    state = model.initial_state(26.0)
    for t in range(60):
        inputs = ControllerInputs(
            t_air_c=state.t_air_c, t_mass_c=state.t_mass_c, t_out_c=35.0, ghi_w_m2=500.0,
            n_pax=50, door_open=(t % 20 == 0), q_hvac_actual_w=state.q_hvac_actual_w,
            time_to_next_station_min=15, expected_boarding=10,
            t_out_fcst_h=36.0, ghi_fcst_h=520.0,
        )
        cmd = ctrl.command(inputs)
        if ctrl.last_frac < 0:
            assert cmd < 0, f"t={t}: frac={ctrl.last_frac:.3f} but cmd={cmd:.0f}"
        elif ctrl.last_frac > 0:
            assert cmd > 0, f"t={t}: frac={ctrl.last_frac:.3f} but cmd={cmd:.0f}"
        else:
            assert cmd == 0.0
        model.step(state, CabinInputs(t_out_c=inputs.t_out_c, ghi_w_m2=inputs.ghi_w_m2,
                   n_pax=inputs.n_pax, door_open=inputs.door_open, q_hvac_cmd_w=cmd), dt_s=60.0)


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
    proportional-feedback controller. Correctness check on the blend
    formula, not just a plausibility check.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = run_controller(_anticipatory_factory(cfg, booster, ff_weight=0.0, gain_k=3.0), cfg)

    model = CabinModel(cfg)
    setpoint = model.setpoint(35.0)
    half_band = cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
    # With ff_weight=0, raw = (setpoint - t_air), deadbanded (see the class
    # docstring), then /gain_k, clipped -- verify on one manual point rather
    # than the whole trajectory.
    ctrl = AnticipatoryController(model, booster,
                                   LiveFeatureBuilder(cfg, "cairo", "down", "semi_express", 1.0, 8.0),
                                   ff_weight=0.0, gain_k=3.0)
    inputs = ControllerInputs(t_air_c=30.0, t_mass_c=29.0, t_out_c=35.0, ghi_w_m2=500.0,
                               n_pax=40, door_open=False, q_hvac_actual_w=-5000.0,
                               time_to_next_station_min=20, expected_boarding=10,
                               t_out_fcst_h=36.0, ghi_fcst_h=520.0)
    ctrl.command(inputs)
    raw = setpoint - 30.0
    excess = abs(raw) - half_band
    raw_db = np.sign(raw) * excess if excess > 0.0 else 0.0
    expected_frac = np.clip(raw_db / 3.0, -1.0, 1.0)
    assert ctrl.last_frac == pytest.approx(expected_frac)


def test_anticipatory_deadband_zeroes_a_small_error(require_model, cfg):
    """Regression guard for the exact mechanism the deadband fixes: an error
    smaller than half the hysteresis band must produce frac EXACTLY 0, not a
    small nonzero nudge. Before this was added, the controller was almost
    never truly off (0.6% of minutes on the canonical scenario, vs
    ThermostatController's 32.5%) -- continuous low-power modulation, which
    is why it used MORE energy than the baseline on most of the M4 test
    split. See docs/PARAMETERS.md's M5 correction log.
    """
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    model = CabinModel(cfg)
    half_band = cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
    ctrl = AnticipatoryController(model, booster,
                                   LiveFeatureBuilder(cfg, "cairo", "down", "semi_express", 1.0, 8.0),
                                   ff_weight=0.0, gain_k=3.0)
    setpoint = model.setpoint(35.0)
    small_error_t_air = setpoint - (half_band * 0.5)  # inside the deadband
    inputs = ControllerInputs(t_air_c=small_error_t_air, t_mass_c=small_error_t_air, t_out_c=35.0,
                               ghi_w_m2=500.0, n_pax=40, door_open=False, q_hvac_actual_w=0.0,
                               time_to_next_station_min=20, expected_boarding=10,
                               t_out_fcst_h=36.0, ghi_fcst_h=520.0)
    cmd = ctrl.command(inputs)
    assert ctrl.last_frac == 0.0
    assert cmd == 0.0


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
    """Before LiveFeatureBuilder has enough history, the controller must
    still act sensibly (pure feedback) rather than doing nothing or crashing.
    """
    model = CabinModel(cfg)
    booster = lgb.Booster(model_file=str(MODEL_PATH)) if MODEL_PATH.exists() else None
    if booster is None:
        pytest.skip("no saved model. Run:  python -m src.train")
    builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    ctrl = AnticipatoryController(model, booster, builder)

    hot_inputs = ControllerInputs(t_air_c=35.0, t_mass_c=33.0, t_out_c=32.0, ghi_w_m2=500.0,
                                   n_pax=40, door_open=False, q_hvac_actual_w=0.0,
                                   time_to_next_station_min=20, expected_boarding=10,
                                   t_out_fcst_h=33.0, ghi_fcst_h=520.0)
    cmd = ctrl.command(hot_inputs)  # minute 0: still in warmup
    assert cmd < 0.0, "warmup fallback should still cool when clearly too hot"
