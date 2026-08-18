"""M5's two controllers: the on/off baseline and the anticipatory predictive
controller, driven identically by src/evaluate.py's simulation loop.

TERMINOLOGY NOTE -- this is deliberately NOT called a Smith predictor.

A true Smith predictor is a specific single-loop architecture: an internal
delay-free model plus a separate pure-delay model, with the mismatch between
predicted and actual plant output fed back to correct the primary controller,
built around an analytic (transfer-function) plant model. What's built here
is a data-driven forward model (M4's LightGBM forecaster) feeding a
feedforward+feedback blend -- the same SPIRIT as Smith-predictor and MPC
dead-time compensation (both exist to counteract a control loop that acts too
late because the plant doesn't respond until well after the command), but
not that structure. Calling it a Smith predictor would be a claim a
control-theory-literate reviewer could correctly dispute. It's an
"anticipatory controller using a learned forward model" -- accurate, still
substantive, and doesn't invite a fight over vocabulary it wouldn't win.

WHY FEEDFORWARD ALONE ISN'T USED. Published rail HVAC predictive control
doesn't trust the forecast alone either: a real train MPC study describes
"feedforward and feedback control with dynamic correction... adds real-time
indoor temperature feedback", reporting 13.44% daily energy saving over pure
feedback control. M4's own audit (docs/PARAMETERS.md, "OPEN RISK FOR M5")
found the forecaster's accuracy depends on which policy generated its input
trajectory -- a pure-feedforward controller would have nothing to catch that
when the forecast drifts. So AnticipatoryController blends a feedforward term
(the M4 forecast) with a feedback term (current error), matching the
published design's shape rather than a simpler but riskier "trust the model"
version.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from src.cabin_model import CabinModel
from src.features import FEATURE_COLUMNS, LiveFeatureBuilder


@dataclass
class ControllerInputs:
    """One minute's worth of state, passed identically to both controllers so
    src/evaluate.py can drive either through the same loop.

    `q_hvac_actual_w` is the actuator's CURRENTLY delivered power -- read
    directly from CabinState.q_hvac_actual_w BEFORE calling step() for this
    minute, not something a controller should track or guess itself. The
    actuator's dead-time/lag state already lives on CabinState between calls;
    reading it from there (rather than reinventing it in the controller) is
    what keeps LiveFeatureBuilder's lag/EWMA features consistent with what
    the physics actually delivered.
    """

    t_air_c: float
    t_mass_c: float
    t_out_c: float
    ghi_w_m2: float
    n_pax: float
    door_open: bool
    q_hvac_actual_w: float
    time_to_next_station_min: float
    expected_boarding: float
    t_out_fcst_h: float
    ghi_fcst_h: float


class ThermostatController:
    """On/off, COOLING-ONLY baseline with hysteresis.

    Real rail HVAC controls this way -- on/off compressor cycling via a
    clutch, switching at a calibrated hysteresis band, is documented in
    vehicle A/C patents -- so this is an honest baseline, not a straw man.
    `hvac.thermostat_hysteresis_k` is sourced in config/cabin_params.yaml
    (AIRAH's standard energy-saving target, corroborated by a real vehicle
    A/C patent's 1.5 K figure) rather than an arbitrarily narrow band, which
    would make the baseline cycle unrealistically often and bias the M5
    comparison in the anticipatory controller's favour.

    COOLING-ONLY BY DESIGN, not by omission -- and found the hard way. A
    first version also called for full heating below the lower threshold,
    matching what a generic dual-mode thermostat would do. Tracing it minute
    by minute (not just checking the aggregate result) showed it oscillating
    between full heat and full cool 18 times in a 166-minute journey, T_air
    swinging 21.8-30 C -- an 8+ K limit cycle, not the intended +/-1 K band.
    Mechanism: this cabin's air node is fast (tau_fast ~1.1-1.3 min, the same
    property M2 built the two-node model specifically to capture) relative to
    the rated capacities, so a full-power command overshoots the opposite
    threshold before the actuator's own dead-time/lag can arrest it, and the
    controller flips again. A "textbook" hysteresis band sized for a slower
    plant doesn't hold here. Every other baseline in this project (M2/M3's
    modulating and on/off analysis, data_generator's exploration policy) was
    already cooling-only for exactly this reason -- Egypt is cooling-dominated
    and heating is documented as rarely engaging (docs/PARAMETERS.md) -- this
    class just hadn't matched that precedent yet. Heating capability still
    exists in CabinModel and in AnticipatoryController; it's specifically the
    simple bang-bang baseline that can't run dual-mode stably on this plant.
    """

    def __init__(self, model: CabinModel):
        self.model = model
        self.half_band = model.cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
        self._on = False

    def command(self, inputs: ControllerInputs) -> float:
        setpoint = self.model.setpoint(inputs.t_out_c)
        upper, lower = setpoint + self.half_band, setpoint - self.half_band

        # Standard two-threshold hysteresis: switch at the outer edges, HOLD
        # the previous on/off state anywhere inside the deadband.
        if inputs.t_air_c > upper:
            self._on = True
        elif inputs.t_air_c < lower:
            self._on = False

        return -self.model.cooling_capacity_w if self._on else 0.0


@dataclass
class AnticipatoryController:
    """Feedforward (M4's t+H forecast) blended with feedback (current error).

    raw  = ff_weight * predicted_error + (1-ff_weight) * current_error
    raw  = deadband(raw, thermostat_hysteresis_k / 2)   -- see below
    frac = clip( raw / gain_k, -1, 1 )
    cmd  = -cooling_capacity * frac   if frac < 0  (predicted/current too hot)
         =  heating_capacity * frac   if frac > 0

    predicted_error compares the FORECAST t+H state against the setpoint AT
    t+H (from the same weather-forecast lookahead the feature already uses,
    not today's setpoint) -- predicted state against predicted target, not
    predicted state against today's target.

    THE DEADBAND -- added after the M5 test-split comparison came back
    NEGATIVE (this controller using MORE energy than ThermostatController on
    most of the 23 held-out scenarios). Diagnosis, not guesswork: on the
    canonical scenario, ThermostatController is fully off 32.5% of the time;
    this controller's raw proportional law, with no floor, was fully off
    only 0.6% of the time -- it never stops nudging, so it pays continuous
    low-power compressor cost the baseline's real "off" periods avoid
    entirely. Reusing `hvac.thermostat_hysteresis_k` (already sourced for
    ThermostatController) as this controller's own deadband gives both
    controllers the same real-world switching tolerance instead of inventing
    a second unsourced number. Applied as a continuous shrink-to-zero
    (`excess = |raw| - half_band`, zero below it, `sign(raw) * excess`
    above), not a hard on/off jump -- a discontinuous version risks the exact
    chattering ThermostatController hit the first time dual-mode hysteresis
    was tried on this fast plant (tau_fast ~1.1-1.3 min). Full before/after
    numbers in docs/PARAMETERS.md's M5 correction log.

    ff_weight and gain_k are [ASSUMPTION]: no literature source gives an
    exact blend ratio for this combination of forecaster and plant. Swept in
    tests/test_controllers.py to characterise sensitivity rather than
    asserting the default is uniquely correct -- the same treatment given to
    every other under-sourced parameter in this project (see
    docs/PARAMETERS.md's calibration priority list).
    """

    model: CabinModel
    booster: lgb.Booster
    builder: LiveFeatureBuilder
    ff_weight: float = 0.6
    gain_k: float = 3.0

    last_frac: float = field(default=0.0, init=False, repr=False)
    """Exposed for tests/diagnostics -- the most recent command as a fraction
    of rated capacity, positive heating / negative cooling."""

    def command(self, inputs: ControllerInputs) -> float:
        setpoint_now = self.model.setpoint(inputs.t_out_c)

        features = self.builder.step(
            t_air_c=inputs.t_air_c, t_mass_c=inputs.t_mass_c, t_out_c=inputs.t_out_c,
            ghi_w_m2=inputs.ghi_w_m2, n_pax=inputs.n_pax, door_open=inputs.door_open,
            setpoint_c=setpoint_now, q_hvac_actual_w=inputs.q_hvac_actual_w,
            time_to_next_station_min=inputs.time_to_next_station_min,
            expected_boarding=inputs.expected_boarding,
            t_out_fcst_h=inputs.t_out_fcst_h, ghi_fcst_h=inputs.ghi_fcst_h,
        )

        # Sign convention matches the dispatch below: NEGATIVE frac cools,
        # POSITIVE heats. So the error has to be (setpoint - state), not
        # (state - setpoint) -- too hot must come out negative. Got this
        # backwards on the first pass here; it is the exact same class of bug
        # already found and fixed once in data_generator.py's
        # StochasticController, caught the same way: by running it and
        # noticing the controller was fighting the wrong direction (a 46 C
        # day ending at 43 C instead of near setpoint).
        current_error = setpoint_now - inputs.t_air_c

        if features is None:
            # Not enough history yet (start of journey) -- fall back to pure
            # feedback, the same reactive law ThermostatController's hysteresis
            # midpoint implies, so the controller does something sane rather
            # than nothing during warmup.
            raw = current_error
        else:
            pred_t_air_h = float(self.booster.predict(features[list(FEATURE_COLUMNS)])[0])
            setpoint_h = self.model.setpoint(inputs.t_out_fcst_h)
            predicted_error = setpoint_h - pred_t_air_h
            raw = self.ff_weight * predicted_error + (1.0 - self.ff_weight) * current_error

        # Deadband -- see the class docstring for why this exists. Continuous
        # (shrinks to 0 smoothly, no jump at the edge) rather than a hard
        # on/off threshold, so it can't reintroduce chattering.
        half_band = self.model.cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
        excess = abs(raw) - half_band
        raw = float(np.sign(raw) * excess) if excess > 0.0 else 0.0

        frac = float(np.clip(raw / self.gain_k, -1.0, 1.0))
        self.last_frac = frac
        hv = self.model.cfg["hvac"]
        # frac is ALREADY signed (negative=cool, positive=heat) and capacities
        # are stored as positive magnitudes, so this is frac * capacity, not
        # -frac * capacity -- a second, previously-hidden sign bug stacked on
        # top of the current_error/predicted_error one above: fixing only the
        # error sign left frac correctly negative-when-hot, but this line's
        # extra leading '-' flipped it positive again right before dispatch,
        # which is why the first fix alone produced no visible change when
        # tested. Caught by tracing frac and cmd minute-by-minute rather than
        # trusting that a correct-looking frac implied a correct command.
        return frac * hv["cooling_capacity_w"] if frac < 0 else frac * hv["heating_capacity_w"]
