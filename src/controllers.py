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
    raw   = deadband(raw, deadband_k / 2)   -- see deadband_k's own docstring
    shift = clip( raw, -max_shift_k, max_shift_k )
    recommendation = clip( setpoint_now + shift, sliding_setpoint_low_c, sliding_setpoint_high_c )

    predicted_error compares the FORECAST t+H state against the setpoint AT
    t+H (from the same weather-forecast lookahead the feature already uses,
    not today's setpoint) -- predicted state against predicted target, not
    predicted state against today's target.

    M10 PHASE 2 RESULT, STATED PLAINLY: under this architecture, the
    forecast does not earn its keep. ff_weight=0.0 (below) means this
    "anticipatory" advisor currently ignores M4's forecast entirely -- the
    VAL-grid/TEST-once search this project used to tune every other
    under-sourced parameter (see deadband_k's own docstring for the full
    three-attempt, 80-configuration result) found that ANY nonzero
    ff_weight makes energy and comfort outcomes worse, monotonically, not
    better. This is a diagnosed, structural consequence of PlantResponse
    having no proportional response (fully on or fully off, nothing
    between) -- not a forecaster quality problem: M4 itself still beats
    persistence by +21% at the raw t+H prediction level (ROADMAP.md's M4
    section), unaffected by any of this. A forecast-driven setpoint shift
    can only move WHEN PlantResponse's switch fires, never HOW HARD it
    runs, and firing early on a forecast buys no proportional benefit to
    offset its added runtime. See `src/compare_controllers.py`'s
    `compare_feedforward_contribution()` -- deliberately re-pointed in
    Phase 2 to demonstrate this finding directly, not just assert it.

    THE DEADBAND (`deadband_k`) exists for the same reason it always has:
    added after an early M5 test-split comparison came back negative
    because a no-floor proportional law never stopped nudging, paying
    continuous low-power cost real "off" periods avoid entirely. Applied as
    a continuous shrink-to-zero (`excess = |raw| - half_band`, zero below
    it, `sign(raw) * excess` above), not a hard on/off jump, so it can't
    reintroduce chattering. What changed in M10 Phase 2: it no longer
    silently reuses PlantResponse's own switching band (see deadband_k's
    docstring for why that specific reuse was actively harmful, not merely
    redundant).

    ff_weight, max_shift_k, and deadband_k are all [ASSUMPTION]: no
    literature source gives an exact blend ratio, setpoint-authority bound,
    or deadband width for this combination of forecaster, plant, and
    (post-M10) bang-bang receiving unit. Swept in tests/test_controllers.py
    and `src/tune_advisor.py` to characterise sensitivity rather than
    asserting the shipped point is uniquely correct -- the same treatment
    given to every other under-sourced parameter in this project (see
    docs/PARAMETERS.md's calibration priority list).
    """

    model: CabinModel
    booster: lgb.Booster
    builder: LiveFeatureBuilder
    ff_weight: float = 0.0
    max_shift_k: float = 4.0
    """[ASSUMPTION], carried forward unchanged from Phase 1 (was already
    known not to matter much once >= ~1-2 C -- see deadband_k's docstring
    for what Phase 2 actually found needed changing)."""
    deadband_k: float | None = 1.5
    """M10 Phase 2 addition. This advisor's OWN deadband width, separate
    from PlantResponse's switching band. Both this field and `ff_weight`
    above are Phase 2's tuned result -- NOT the Phase-1 placeholders they
    started as -- via the same VAL-grid/TEST-once discipline used for the
    old (gain_k, ff_weight) pair. Full result, including why it landed on
    "the forecast helps not at all," is in ROADMAP.md's M10 section and
    `src/tune_advisor.py`'s module docstring; summary below.

    Phase 1 shipped `deadband_k=None` (reusing `hvac.thermostat_hysteresis_k`
    outright) and `ff_weight=0.45` carried over from the retired law. The
    resulting VAL-split grid search came back negative or flat at EVERY
    point tested -- not a near-miss. Diagnosed, not guessed: at ff_weight=0,
    traced a real scenario minute by minute and found advised_setpoint_c
    differs from the static schedule on 125/166 minutes (by up to ~2.9 C)
    yet cmd_w and t_air_c came out BIT-IDENTICAL to ThermostatController
    throughout -- because the advisor's deadband and PlantResponse's
    switching half-band were the SAME number, a shift only ever became
    nonzero at the exact moment PlantResponse's own switch was already
    firing under the static setpoint, never before it.

    Three grid-search attempts, 80 total (ff_weight, deadband_k)
    configurations, all measured on the honest VAL split: not one ever beat
    the static baseline. The best ANY of them reached was parity (0.00%
    energy delta) from below, as deadband_k widens past
    `hvac.thermostat_hysteresis_k`; any nonzero ff_weight made things
    monotonically worse (ff_weight=1.0: 25/31 VAL scenarios strictly worse
    on both energy and comfort). Mechanism: PlantResponse has no
    proportional response -- it is either fully on or fully off -- so a
    forecast-driven setpoint shift can only move WHEN the switch fires, not
    HOW HARD it runs. Shifting based on CURRENT error just re-derives a
    noisier copy of a decision the switch's own hysteresis already makes;
    shifting based on the FORECAST fires the switch too early relative to
    actual need, paying full-capacity runtime with nothing proportional to
    show for it.

    SHIPPED: `ff_weight=0.0` (the forecast is not used -- turning it on
    never helped, at any deadband tested), `deadband_k=1.5` (the VAL
    grid's own top-ranked point by this project's standing
    n_both_better-first discipline: 10/31 VAL scenarios both-better, only
    1/31 worse-on-both -- confirmed on TEST at 4/23 both-better, 0/23
    worse-on-both, mean energy delta -0.2%, essentially a wash rather than
    a win). This IS effectively a disclosed null result for the forecast's
    contribution under this architecture, not a tuned improvement -- stated
    plainly rather than dressed up, matching M9's own precedent (SHAP-
    diagnosed remedies tested, neither adopted, disclosed as a limitation)."""

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

        # Deadband -- see the class docstring for why this exists, and
        # deadband_k's own docstring for why it's no longer hardcoded to
        # PlantResponse's own switching band. Continuous (shrinks to 0
        # smoothly, no jump at the edge) rather than a hard on/off
        # threshold, so it can't reintroduce chattering.
        deadband_k = (self.deadband_k if self.deadband_k is not None
                      else self.model.cfg["hvac"]["thermostat_hysteresis_k"])
        half_band = deadband_k / 2.0
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
