"""M5's setpoint advisors (ThermostatController, AnticipatorySetpointAdvisor)
and the shared PlantResponse they're evaluated through -- driven identically
by src/evaluate.py's simulation loop.

M10 REFRAMING NOTE -- Railflow's own compute cannot touch a real HVAC unit's
control electronics (would void the manufacturer's warranty), so nothing in
this module outputs a power command anymore. Each advisor's job is now
`.recommend_setpoint(inputs) -> float` (degrees C) -- the same kind of value
a human or the unit's own thermostat interface already accepts, with zero
hardware modification. PlantResponse is the one place a setpoint becomes
watts: a stand-in for the real, third-party Control Unit this project will
never redesign, used IDENTICALLY for every compared arm so M5/M6 isolate the
advisor's actual contribution rather than conflating it with a difference in
dispatch mechanism. See ROADMAP.md's M10 section for the full rationale;
M5/M6's pre-reframing numbers are kept there, marked superseded, not deleted.

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
when the forecast drifts. So AnticipatorySetpointAdvisor blends a feedforward term
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


class PlantResponse:
    """Simulates the RECEIVING unit's own on/off reaction to a setpoint -- a
    stand-in for the real, third-party Control Unit this project will never
    redesign (see the M10 reframing note at the top of this module). Used
    IDENTICALLY for every compared arm in M5/M6: the only thing that ever
    differs between "today's static schedule" and "the AI's recommendation"
    is the setpoint fed to THIS class, never the mechanism -- see
    src/evaluate.py's run_controller() for why that symmetry matters to the
    comparison's validity.

    COOLING-ONLY BY DESIGN, not by omission -- and found the hard way, back
    when this logic still lived directly on ThermostatController. A first
    version also called for full heating below the lower threshold, matching
    what a generic dual-mode thermostat would do. Tracing it minute by minute
    (not just checking the aggregate result) showed it oscillating between
    full heat and full cool 18 times in a 166-minute journey, T_air swinging
    21.8-30 C -- an 8+ K limit cycle, not the intended +/-1 K band.
    Mechanism: this cabin's air node is fast (tau_fast ~1.1-1.3 min, the same
    property M2 built the two-node model specifically to capture) relative to
    the rated capacities, so a full-power command overshoots the opposite
    threshold before the actuator's own dead-time/lag can arrest it, and the
    reaction flips again. A "textbook" hysteresis band sized for a slower
    plant doesn't hold here. Every other baseline in this project (M2/M3's
    modulating and on/off analysis, data_generator's exploration policy) was
    already cooling-only for exactly this reason -- Egypt is cooling-dominated
    and heating is documented as rarely engaging (docs/PARAMETERS.md).
    Heating capability still exists in CabinModel; it's specifically this
    simple bang-bang reaction that can't run dual-mode stably on this plant.

    `hvac.thermostat_hysteresis_k` is sourced in config/cabin_params.yaml
    (AIRAH's standard energy-saving target, corroborated by a real vehicle
    A/C patent's 1.5 K figure) rather than an arbitrarily narrow band, which
    would make this reaction cycle unrealistically often and bias the M5
    comparison in the anticipatory advisor's favour.
    """

    def __init__(self, model: CabinModel):
        self.model = model
        self.half_band = model.cfg["hvac"]["thermostat_hysteresis_k"] / 2.0
        self._on = False

    def respond(self, t_air_c: float, setpoint_c: float) -> float:
        upper, lower = setpoint_c + self.half_band, setpoint_c - self.half_band

        # Standard two-threshold hysteresis: switch at the outer edges, HOLD
        # the previous on/off state anywhere inside the deadband.
        if t_air_c > upper:
            self._on = True
        elif t_air_c < lower:
            self._on = False

        return -self.model.cooling_capacity_w if self._on else 0.0


class ThermostatController:
    """M5's baseline setpoint advisor: always recommends today's static
    EN13129 sliding-schedule setpoint (CabinModel.setpoint()) -- no forecast,
    no adjustment. The trivial case of the setpoint-advisor interface every
    class in this module now implements. Deliberately holds NO hysteresis
    state itself -- that lives in PlantResponse, owned once per run by
    src/evaluate.py's run_controller() and shared identically across every
    compared arm.
    """

    def __init__(self, model: CabinModel):
        self.model = model

    def recommend_setpoint(self, inputs: ControllerInputs) -> float:
        return self.model.setpoint(inputs.t_out_c)


@dataclass
class AnticipatorySetpointAdvisor:
    """Feedforward (M4's t+H forecast) blended with feedback (current error),
    recommending a cabin SETPOINT (deg C) -- Railflow's actual deliverable
    under the M10 reframing, not a watts command. Formerly
    AnticipatoryController; renamed when its dispatch tail changed from
    "capacity fraction -> watts" to "setpoint shift -> recommended setpoint"
    (see ROADMAP.md's M10 section for why).

    raw   = ff_weight * predicted_error + (1-ff_weight) * current_error
    raw   = deadband(raw, thermostat_hysteresis_k / 2)   -- see below
    shift = clip( raw, -max_shift_k, max_shift_k )
    recommendation = clip( setpoint_now + shift, sliding_setpoint_low_c, sliding_setpoint_high_c )

    predicted_error compares the FORECAST t+H state against the setpoint AT
    t+H (from the same weather-forecast lookahead the feature already uses,
    not today's setpoint) -- predicted state against predicted target, not
    predicted state against today's target.

    THE DEADBAND -- added after the M5 test-split comparison came back
    NEGATIVE (this advisor's law using MORE energy than ThermostatController
    on most of the 23 held-out scenarios, back when both fed watts directly).
    Diagnosis, not guesswork: on the canonical scenario, ThermostatController
    is fully off 32.5% of the time; this advisor's raw proportional law, with
    no floor, was fully off only 0.6% of the time -- it never stopped
    nudging, so it paid continuous low-power compressor cost the baseline's
    real "off" periods avoided entirely. Reusing `hvac.thermostat_hysteresis_k`
    (already sourced for PlantResponse) as this advisor's own deadband gives
    both advisors the same real-world switching tolerance instead of
    inventing a second unsourced number. Applied as a continuous
    shrink-to-zero (`excess = |raw| - half_band`, zero below it,
    `sign(raw) * excess` above), not a hard on/off jump -- a discontinuous
    version risks the exact chattering PlantResponse's mechanism hit the
    first time dual-mode hysteresis was tried on this fast plant (tau_fast
    ~1.1-1.3 min). Full before/after numbers in docs/PARAMETERS.md's M5
    correction log.

    ff_weight and max_shift_k are [ASSUMPTION]: no literature source gives an
    exact blend ratio or setpoint-authority bound for this combination of
    forecaster and plant. Swept in tests/test_controllers.py to characterise
    sensitivity rather than asserting the default is uniquely correct -- the
    same treatment given to every other under-sourced parameter in this
    project (see docs/PARAMETERS.md's calibration priority list).

    ff_weight=0.45 carries over from the pre-M10 (gain_k, ff_weight) tune
    (M9): a 25-point grid searched on the VAL split, ranked by n_both_better
    (scenarios strictly better on both energy AND comfort) rather than mean
    energy saving alone, confirmed ONCE on the TEST split -- mean energy
    saving +1.2%->+3.9%, both-better 13/23->19/23, worse-on-both stayed
    0/23, comfort-ok unchanged at 21/23. Full numbers in ROADMAP.md's M9
    section. THAT RESULT DESCRIBES THE RETIRED WATTS-DISPATCH LAW -- it is
    kept here only as the origin of ff_weight's current value, marked
    superseded/pending regeneration in ROADMAP.md's M10 section. `max_shift_k`
    is a fresh [ASSUMPTION] placeholder (see its own field docstring below),
    not carried over from `gain_k` by any principled conversion -- Phase 2
    re-tunes (ff_weight, max_shift_k) jointly, same VAL-grid/TEST-once
    discipline, from scratch.
    """

    model: CabinModel
    booster: lgb.Booster
    builder: LiveFeatureBuilder
    ff_weight: float = 0.45
    max_shift_k: float = 4.0
    """[ASSUMPTION] Phase-1 structural placeholder, carried over from the
    retired gain_k's rough scale but NOT re-derived under the new law (see
    class docstring). Bounds how many degrees C the recommendation may move
    away from setpoint_now in either direction. Phase 2 must re-tune this
    jointly with ff_weight via the same VAL-grid-then-TEST-once methodology
    used for the old (gain_k, ff_weight) pair -- see ROADMAP.md's M10
    section."""

    last_setpoint_shift_c: float = field(default=0.0, init=False, repr=False)
    """Exposed for tests/diagnostics/serial telemetry -- how far the most
    recent recommendation moved from setpoint_now, signed (negative = call
    for more cooling, positive = tolerate a warmer cabin)."""

    def recommend_setpoint(self, inputs: ControllerInputs) -> float:
        setpoint_now = self.model.setpoint(inputs.t_out_c)

        features = self.builder.step(
            t_air_c=inputs.t_air_c, t_mass_c=inputs.t_mass_c, t_out_c=inputs.t_out_c,
            ghi_w_m2=inputs.ghi_w_m2, n_pax=inputs.n_pax, door_open=inputs.door_open,
            setpoint_c=setpoint_now, q_hvac_actual_w=inputs.q_hvac_actual_w,
            time_to_next_station_min=inputs.time_to_next_station_min,
            expected_boarding=inputs.expected_boarding,
            t_out_fcst_h=inputs.t_out_fcst_h, ghi_fcst_h=inputs.ghi_fcst_h,
        )

        # current_error is (setpoint - state): too hot must come out
        # negative, matching "negative shift = call for more cooling" below.
        # Got this backwards on the first pass when this law was first built
        # (ROADMAP.md's M5 section); it is the exact same class of bug
        # already found and fixed once in data_generator.py's
        # StochasticController, caught the same way: by running it and
        # noticing the advisor was fighting the wrong direction (a 46 C day
        # ending at 43 C instead of near setpoint).
        current_error = setpoint_now - inputs.t_air_c

        if features is None:
            # Not enough history yet (start of journey) -- fall back to pure
            # feedback, the same reactive law PlantResponse's hysteresis
            # midpoint implies, so the advisor recommends something sane
            # rather than nothing during warmup.
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

        # NEW tail (M10) -- replaces frac = clip(raw/gain_k, -1, 1);
        # cmd = frac*capacity. raw is ALREADY in degrees C (an error term),
        # so no unit conversion happens here, only an authority bound. This
        # is structurally simpler than the retired dispatch: no
        # cooling/heating capacity-sign branch survives, so the double
        # sign-bug class documented in docs/PARAMETERS.md's M5 correction
        # log has nothing left to hide in.
        shift = float(np.clip(raw, -self.max_shift_k, self.max_shift_k))
        self.last_setpoint_shift_c = shift

        # Clamp the RESULT to the same sliding-setpoint envelope
        # CabinModel.setpoint() itself respects -- keeps every recommendation
        # inside docs/serial_protocol.md's existing, unchanged
        # SETPOINT_RANGE_C (22.0-26.0) with zero protocol changes. Phase 2
        # may find this too tight and choose to widen both together, as one
        # considered joint decision -- see ROADMAP.md's M10 section.
        c = self.model.cfg["comfort"]
        lo, hi = c["sliding_setpoint_low_c"], c["sliding_setpoint_high_c"]
        return float(np.clip(setpoint_now + shift, lo, hi))
