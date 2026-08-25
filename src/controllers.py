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
    time_to_last_station_min: float
    """M13 fix. Counts down ONCE toward the journey's FINAL station -- see
    OccupancyProfile.time_to_last_station_min's own docstring in
    src/occupancy.py for why StationPrecoolAdvisor needs this alongside
    time_to_next_station_min. Required, not defaulted, matching every other
    field here -- both real construction sites (src/evaluate.py,
    tests/test_controllers.py's _controller_inputs() helper) are updated
    alongside this field, not left to silently fall back to anything."""


class BangBangPlantResponse:
    """M10-M11's on/off reaction to a setpoint -- a stand-in for the real,
    third-party Control Unit this project will never redesign (see the M10
    reframing note at the top of this module). SUPERSEDED as the default by
    PlantResponse (below) in M12, kept here, not deleted, as the honest,
    maximally-conservative baseline model: "assume the least capable real
    unit this project could defensibly claim compatibility with." Still
    useful for exactly that comparison -- see PlantResponse's own docstring
    for why M12 needed a second, more capable model in the first place.

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


class PlantResponse:
    """M12: models the receiving unit as a black box that, given a
    commanded setpoint, converges to it and HOLDS there -- not a claim
    about its literal internal mechanism (a real PID loop, a staged or
    variable-speed compressor -- unknown, and irrelevant to this project),
    but a claim about its aggregate BEHAVIOR: it responds proportionally to
    how far off it currently is, and it doesn't leave a permanent offset
    once settled. Implemented as PI (proportional + integral), deliberately
    not full PID: a derivative term would smooth the APPROACH (less
    overshoot getting there) but isn't needed for "reaches target and holds
    it," which P+I alone satisfies -- the minimal design for the stated
    requirement, the same restraint this project applies elsewhere (e.g.
    tint_controller.py's rule-based-not-trained choice).

    WHY THIS REPLACES BangBangPlantResponse (kept, not deleted) AS THE
    DEFAULT. M10 Phase 2 found that no (ff_weight, deadband_k) tuning ever
    beat a static schedule, and diagnosed why: a bang-bang receiver only
    ever asks "did the setpoint cross a threshold," never "by how much" --
    so a forecast-driven setpoint shift can move WHEN the switch fires but
    never HOW HARD it runs, and firing early buys no proportional benefit.
    A proportional receiver doesn't have that ceiling: commanding a
    setpoint further from the current temperature genuinely commands more
    capacity (up to the real, state-dependent supply-air limit
    CabinModel.deliverable_cooling_w/heating_w already enforces) -- which is
    exactly the "command 19 to reach 26 faster, then relax it back" strategy
    this milestone exists to let the advisor actually use. Whether it does
    is a Phase-2-style re-tune, deliberately NOT assumed to transfer from
    the bang-bang result -- see ROADMAP.md's M12 section.

    ANTI-WINDUP, IMPLEMENTED, NOT ASSUMED UNNECESSARY. A naive PI integral
    accumulates without limit while its output sits saturated at the
    physical capacity ceiling (e.g. a large, sustained error early in a hot
    scenario) -- then overshoots badly correcting for that windup once the
    error shrinks. Standard fix: conditional integration, checked here
    against this project's own real, state-dependent saturation bounds
    (CabinModel.deliverable_cooling_w/heating_w -- the SAME bounds
    CabinModel._actuate() itself clamps against, not a second, independently
    guessed limit) -- the integral only accumulates further in a direction
    that ISN'T already saturated, so recovery is never delayed once the
    error starts pointing back the other way.

    `hvac.plant_response_kp_w_per_k`/`_ki_w_per_k_per_s` are `[ASSUMPTION]`:
    no literature source for these two numbers, chosen by simple
    process-control reasoning then VERIFIED (not just asserted) by tracing a
    real scenario minute by minute -- see ROADMAP.md's M12 section for the
    trace and what it showed.
    """

    def __init__(self, model: CabinModel):
        self.model = model
        self.kp = model.cfg["hvac"]["plant_response_kp_w_per_k"]
        self.ki = model.cfg["hvac"]["plant_response_ki_w_per_k_per_s"]
        self._integral = 0.0
        self._dt_s = 60.0
        """Matches run_controller()'s fixed per-call cadence (one call per
        simulated minute) -- respond() has no dt parameter of its own to
        keep its signature identical to BangBangPlantResponse's, so this is
        the one place that convention is assumed rather than passed in."""

    def respond(self, t_air_c: float, setpoint_c: float) -> float:
        # Sign convention matches CabinModel's (q_hvac > 0 heats, < 0 cools):
        # error > 0 means too COLD (want heat), error < 0 means too HOT
        # (want cooling) -- setpoint_c - t_air_c, not the other way round.
        error = setpoint_c - t_air_c
        q_unclamped = self.kp * error + self.ki * self._integral

        cool_limit = self.model.deliverable_cooling_w(t_air_c)   # <= 0
        heat_limit = self.model.deliverable_heating_w(t_air_c)   # >= 0
        pushing_further_into_cooling_limit = q_unclamped <= cool_limit and error < 0.0
        pushing_further_into_heating_limit = q_unclamped >= heat_limit and error > 0.0
        if not (pushing_further_into_cooling_limit or pushing_further_into_heating_limit):
            self._integral += error * self._dt_s

        # Not clamped here -- CabinModel._actuate() clamps whatever it's
        # given against these exact same deliverable_cooling_w/heating_w
        # bounds already, so clamping twice would be redundant, not safer.
        return self.kp * error + self.ki * self._integral


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


SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN = 50
"""M12 attempt 2's tuned result (src/tune_advisor_m12.py) -- how often
run_controller() should actually APPLY a fresh AnticipatorySetpointAdvisor
recommendation to PlantResponse, not how often recommend_setpoint() itself
gets called (always every minute regardless -- see run_controller()'s own
docstring on why those two are deliberately decoupled). Not a field on the
advisor class below because it describes the EVALUATION LOOP's cadence, not
the advisor's own decision logic -- every caller comparing this advisor
against ThermostatController (src/compare_controllers.py's compare(),
src/app.py) passes this explicitly via run_controller(...,
advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN) so the live
demo and the M5 harness never silently diverge from what was actually tuned.

WHY 50, NOT M10 PHASE 2's DEFAULT OF 1. Phase 2 (attempt 4, sweep_update_
interval() in src/tune_advisor.py) found spacing updates out was harmful --
but against BangBangPlantResponse, which has no settling time of its own for
an outer loop to respect ("nothing here has M8's physical-actuator/bandwidth
constraint"). PlantResponse's PI loop has a real one: integral time
Kp/Ki = 4000/4.0 = 1000 s (~17 min). Updating every simulated minute means
the advisor keeps moving the target before the inner PI loop gets anywhere
near the last one it was given -- a textbook cascaded-control timescale
mismatch, diagnosed after M12's first re-tune attempt (ff_weight/max_shift_k/
deadband_k alone) came back WORSE than the stale bang-bang-tuned values it
was meant to replace (TEST: comfort-ok collapsed from Phase 2's 21/23 to
2/23, worse-on-both 0/23 to 6/23 -- see src/tune_advisor_m12.py). Sweeping
advisor_update_interval_min against the new plant confirmed the mechanism
directly: worse-on-both hits exactly 0/31 on VAL at every interval from 40
minutes up, i.e. once the outer loop is slower than the inner loop's own
settling time, the two stop fighting entirely. 50 is the VAL winner in that
region (11/31 both-better, 0/31 worse-on-both), confirmed once on TEST:
mean +0.04%, median +0.00%, both-better 4/23, worse-on-both 0/23,
comfort-ok 20/23, total degree-hours 15.46 -> 15.45 K*h -- matching Phase
2's own old-plant headline (4/23 both-better, 0/23 worse-on-both, 21/23
comfort-ok) almost exactly. See ROADMAP.md's M12 section for the full
two-attempt narrative."""


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
    max_shift_k: float = 8.0
    """[ASSUMPTION]. Phase 1-2 (bang-bang plant) shipped 4.0 -- "was already
    known not to matter much once >= ~1-2 C" there, since the final clamp to
    the comfort range dominated regardless of how much authority this bound
    granted. M12 widened it to 8.0 alongside hvac.advisor_setpoint_min_c/
    max_c (18-30 C, well past the 22-26 C comfort range) so a shift can
    actually reach that wider range -- confirmed in M12 attempt 1's grid
    that 8.0 and 12.0 tied exactly (still not the lever that mattered; see
    deadband_k's docstring for what did)."""
    deadband_k: float | None = 1.0
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

    M10 PHASE 2 SHIPPED (bang-bang plant, SUPERSEDED by M12 below, kept for
    the record): `ff_weight=0.0` (the forecast is not used -- turning it on
    never helped, at any deadband tested), `deadband_k=1.5` (the VAL
    grid's own top-ranked point by this project's standing
    n_both_better-first discipline: 10/31 VAL scenarios both-better, only
    1/31 worse-on-both -- confirmed on TEST at 4/23 both-better, 0/23
    worse-on-both, mean energy delta -0.2%, essentially a wash rather than
    a win). This IS effectively a disclosed null result for the forecast's
    contribution under that architecture, not a tuned improvement -- stated
    plainly rather than dressed up, matching M9's own precedent (SHAP-
    diagnosed remedies tested, neither adopted, disclosed as a limitation).

    M12 RE-TUNE, TWO ATTEMPTS -- full numbers and the diagnosis in
    src/tune_advisor_m12.py's module docstring and ROADMAP.md's M12 section;
    summary here. ATTEMPT 1 (this field alone, jointly with max_shift_k,
    ff_weight, at the Phase-2 update cadence of every minute): a 27-point
    grid, confirmed on TEST at max_shift_k=8.0/deadband_k=1.0 -- came back a
    REGRESSION, not a win: comfort-ok collapsed from Phase 2's 21/23 to
    2/23, worse-on-both 0/23 to 6/23, even though mean energy ticked
    slightly positive (+0.16%). Diagnosed as a cascaded-control timescale
    mismatch against PlantResponse's PI loop (~17 min integral time,
    Kp/Ki=1000s) -- see SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN's own docstring
    above for the mechanism. ATTEMPT 2 fixed it by slowing
    advisor_update_interval_min instead of widening this deadband further:
    confirmed worse-on-both hits exactly 0/31 on VAL at every interval from
    40 minutes up, i.e. once the outer loop stops updating faster than the
    inner PI can settle, the two stop fighting entirely.

    SHIPPED (M12): `ff_weight=0.0` (unchanged -- still sharply harmful at
    any nonzero value, confirmed again in attempt 1, not re-litigated in
    attempt 2), `deadband_k=1.0`, `max_shift_k=8.0`,
    `advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN` (50,
    applied by every caller that compares this advisor against
    ThermostatController -- see that constant's own docstring). TEST:
    mean +0.04%, median +0.00%, both-better 4/23, worse-on-both 0/23,
    comfort-ok 20/23, total degree-hours 15.46 -> 15.45 K*h -- matching
    Phase 2's own old-plant headline almost exactly (4/23, 0/23, 21/23).
    Energy is flat, not a saving -- another disclosed parity result, not a
    win dressed up as one, but a GENUINE recovery from attempt 1's
    regression, arrived at by diagnosing the actual mechanism rather than
    searching the same three parameters harder."""

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

        # M12: clamp to hvac.advisor_setpoint_min_c/max_c -- this advisor's
        # OWN commanding authority, deliberately wider than
        # comfort.sliding_setpoint_low_c/high_c (the passenger-facing
        # comfort target ThermostatController/CabinModel.setpoint() use).
        # Under M10-M11's bang-bang PlantResponse the distinction was moot;
        # under M12's proportional PlantResponse it is the mechanism this
        # milestone exists to let the advisor use -- see PlantResponse's own
        # docstring and hvac.advisor_setpoint_min_c's config comment for why.
        # NOTE: this widens docs/serial_protocol.md's SETPOINT_RANGE_C
        # (22.0-26.0) beyond what that document currently promises -- a real
        # deployment's protocol would need updating to match before this
        # authority could actually reach real hardware; flagged, not fixed
        # here, since M10's software-only decision means nothing here talks
        # to a real board regardless (see ROADMAP.md's M8/M10 sections).
        hv = self.model.cfg["hvac"]
        lo, hi = hv["advisor_setpoint_min_c"], hv["advisor_setpoint_max_c"]
        return float(np.clip(setpoint_now + shift, lo, hi))


@dataclass
class StationPrecoolAdvisor:
    """M13: pre-cools the cabin ahead of a KNOWN, schedule-certain station
    stop -- boarding brings a real heat/humidity disturbance (open doors,
    fresh-air load, passenger heat -- see CabinModel.step()), and this
    mechanism asks the plant for more cooling BEFORE it arrives, banking
    thermal headroom so the disturbance is absorbed rather than felt. The
    concrete claim: a passenger sitting through a station stop under this
    mechanism should see LESS temperature excursion than the same stop
    under AnticipatorySetpointAdvisor alone -- verified directly on a real
    scenario, not just asserted (see tests/test_controllers.py).

    DELIBERATELY SEPARATE FROM AnticipatorySetpointAdvisor's ff_weight
    BLEND, not a variant of it. That blend answers "does a generic
    ML-forecast term improve AGGREGATE energy/comfort across random
    scenarios" -- both M10 Phase 2 and M12 found the answer is no (ff_weight
    stays shipped at 0.0). This answers a narrower, different question: "can
    the system visibly, demonstrably anticipate one SPECIFIC, schedule-known
    event" -- yes, and unlike a general forecast, arrival timing here is
    exact (occupancy.py's time_to_next_station_min), not a prediction with
    its own error to blend against. Reusing ff_weight for this would
    reintroduce exactly the aggregate-metric tension that made ff_weight=0.0
    correct for the OTHER question; keeping this as an independent, additive
    term means it can be added or removed without touching M12's already
    TEST-confirmed (ff_weight, deadband_k, max_shift_k,
    advisor_update_interval_min) result at all.

    NOT THROTTLED BY advisor_update_interval_min, unlike
    AnticipatorySetpointAdvisor's own shift -- and deliberately so.
    SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN=50 exists because a FEEDBACK-driven
    shift (reacting to CURRENT measured error, every minute) fights
    PlantResponse's own ~17 min PI settling time (see that constant's
    docstring). This term carries none of that risk: it is a fixed function
    of time_to_next_station_min, a SCHEDULE quantity with no measured error
    in the loop at all -- there is nothing for it to "chase" or overcorrect,
    so src/evaluate.py's run_controller() applies it fresh every minute,
    independent of whatever cadence the general advisor is throttled to.

    SELF-RELAXING WITHOUT SPECIAL-CASE LOGIC. occupancy.py's
    time_to_next_station_min counts down toward the NEXT station only
    (`st.arrive_min > t`, strictly) -- at the exact arrival minute it has
    already jumped to counting toward the station AFTER that one, typically
    a much larger number. precool_shift_c() therefore returns to 0.0
    automatically the instant a station is reached, with no separate
    "recovery" branch needed: traced directly against occupancy.py's own
    array, not assumed from reading the formula.

    `shift_k` and `lead_min` (`hvac.station_precool_shift_k` /
    `station_precool_lead_min`, -1.0 K / 15 min) are both [ASSUMPTION], and
    both are the RE-MEASURED values, not the originals. `shift_k` is applied
    as a CONSTANT across the whole lead window, not a ramp -- a ramp only
    reaches full strength at the moment of arrival, exactly when the time to
    actually cool down has run out; a step gives the plant the entire window
    at full authority instead (a 5/10/15-minute ramp was tested and kept
    46-78% less of the temperature benefit, so this is measured too).

    THE ORIGINAL VALUES (-3.0 K over a 30 min lead) WERE WRONG, AND THE
    MEASUREMENT SAID SO ON BOTH METRICS AT ONCE. Anchored to the team's own
    illustrative "22 C pulled to 19 C" example rather than derived, they cost
    2.96% MORE energy AND 22.5% MORE degree-hours than not precooling at all
    across the 23 TEST scenarios -- 0/23 better on both, 13/23 worse on both.
    Diagnosed, not tuned around: 3.0 K exceeds `comfort.band_k` (2.0 K), so
    whenever the plant actually REACHED the precool target the cabin was
    already in a cold-side comfort breach. The mechanism was buying a short
    warm-side breach at each stop by paying for a long cold-side one before
    it. Sized below `band_k` instead, the same mechanism flips to a real (if
    modest) win: +6.8% degree-hours for 0.80% energy at the shipped
    -1.0 K/15 min, TEST-confirmed on the same 23 scenarios.

    `lead_min` was previously hardcoded to `simulation.control_horizon_min`
    (30 min) -- M4's FORECAST horizon, borrowed rather than derived. Split
    into its own parameter because the question here is how long the PLANT
    needs to act, not how far the model can see: dead_time_min (2) +
    tau_act_min (5) + PlantResponse's PI settling (~17) is ~24 min. Swept
    10/15/24 against -1.0 and -1.5 K shifts -- longer leads buy more comfort
    at proportionally more energy (24 min/-1.5 K reaches +9.0% for 1.73%),
    and 15 min sits at the efficiency knee. Picked for that ratio, not for
    the largest absolute comfort number on the sweep.

    WORTH MORE THAN THE COMFORT NUMBER ITSELF: checked against the naive
    alternative (a constant setpoint bias held all journey, no event
    targeting), event-targeted precool STRICTLY DOMINATES -- a -0.5 K
    constant bias buys +5.7% comfort for 2.00% energy, while this buys
    +6.8% comfort for 0.80%, i.e. more comfort at ~2.7x lower energy cost.
    That comparison, not the raw percentage, is the clearest evidence in
    this project that anticipating a specific KNOWN event beats a blanket
    setpoint change -- and it is the honest form of the claim, since the
    energy cost is real and disclosed rather than described as a saving.

    NEVER PRECOOLS THE JOURNEY'S FINAL STATION -- a real fix, not a tuning
    choice, found by tracing WHY the cabin stayed cold and drifted further
    from setpoint after minute 125 on a real scenario rather than
    recovering: `lead_min`'s window for the LAST station engaged as usual,
    but the journey simply ENDED a few minutes later, before the cabin had
    any remaining time to recover -- the plant was left holding a
    setpoint shifted for a disturbance that, once reached, nobody stays
    aboard to feel the benefit of recovering from. Confirmed on all 23 TEST
    scenarios, not just the one that surfaced it: skipping the final
    station is a strict improvement over precooling it -- degree_hours
    -13.9% and energy -1.4% in aggregate, better-or-equal on BOTH on
    23/23 scenarios, zero cost at every OTHER station (their peak
    temperatures are bit-identical with or without this fix). Detected via
    `inputs.time_to_next_station_min == inputs.time_to_last_station_min`
    (both counting toward the SAME, final arrival) -- a single value
    comparison, not internal state, keeping this class a pure function of
    `ControllerInputs` like everything else in this module.
    """

    lead_min: float
    shift_k: float

    def precool_shift_c(self, inputs: ControllerInputs) -> float:
        t2s = inputs.time_to_next_station_min
        if t2s == inputs.time_to_last_station_min:
            return 0.0
        if 0.0 < t2s <= self.lead_min:
            return self.shift_k
        return 0.0
