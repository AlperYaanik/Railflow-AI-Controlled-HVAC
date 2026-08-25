"""Runs a controller through a scenario and scores it: energy (kWh) and
comfort (degree-hours outside the band).

METRIC UNIT NOTE. Comfort is reported in DEGREE-HOURS, not degree-minutes.
"Exceedance Degree-Hours" (Salimi et al., Indoor Air, 2021) and "degree-hours"
more generally are the standard, citable units in the HVAC comfort literature
-- degree-minutes is not a unit this project should be inventing. The
underlying computation stays at native (per-minute) resolution; only the
reported unit changes, by dividing by 60.

`run_controller()` is the ONE place a controller gets driven through a real
scenario minute by minute. tests/test_controllers.py imports it rather than
keeping its own copy of this loop, for the same reason LiveFeatureBuilder
exists as shared code rather than being reimplemented per caller: two
implementations of "how do you step this simulation" would risk silently
drifting apart, and that class of bug (a batch/live or a duplicated-logic
mismatch) has already been found and fixed twice in this project.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.cabin_model import CabinInputs, CabinModel
from src.config import DATA_DIR, load_config
from src.controllers import ControllerInputs, PlantResponse
from src.occupancy import Service, simulate
from src.weather import to_minutes


def degree_hours_outside_band(
    t_air_c: pd.Series, setpoint_c: pd.Series, band_k: float, dt_min: float = 1.0,
) -> float:
    """Integral of exceedance beyond the comfort band, in K*h.

    Not a count of minutes outside the band -- that would treat a 0.1 K miss
    the same as a 10 K miss, and a brief excursion the same as a sustained
    one. This is the integral over time of HOW FAR outside, which is what
    "Exceedance Degree-Hours" (Salimi et al., Indoor Air, 2021) and the wider
    degree-hours convention in HVAC comfort literature actually measure.
    """
    excess = (t_air_c - setpoint_c).abs() - band_k
    return float(excess.clip(lower=0).sum() * dt_min / 60.0)


def run_controller(
    controller_factory: Callable[[CabinModel], object],
    cfg: dict | None = None,
    city: str = "cairo",
    date: str = "2024-07-20",
    depart_hour: float = 8.0,
    direction: str = "down",
    pattern: str = "semi_express",
    load_factor: float = 1.0,
    advisor_update_interval_min: int = 1,
    tint_controller: object | None = None,
    station_precool_advisor: object | None = None,
) -> pd.DataFrame:
    """Drives one setpoint advisor through one real journey, minute by minute.

    `controller_factory(model)` builds the advisor against a freshly
    constructed CabinModel, so ThermostatController and
    AnticipatorySetpointAdvisor (or anything matching their
    `.recommend_setpoint(ControllerInputs) -> float` interface, a setpoint in
    degrees C) can be driven identically without this function needing to
    know which one it's holding. A single PlantResponse -- the stand-in for
    the real, third-party Control Unit this project never touches or
    redesigns (see src/controllers.py's M10 reframing note) -- is constructed
    once per run and shared by every advisor, so the ONLY thing that differs
    between arms is the setpoint each one recommends, never the mechanism
    that turns a setpoint into watts. That symmetry is what lets M5/M6
    isolate the advisor's actual contribution instead of conflating it with
    a difference in dispatch law.

    `advisor_update_interval_min` (default 1 = every prior behaviour,
    unchanged -- kept as this function's own raw default specifically so a
    standalone run, e.g. this module's own __main__ demo below, is
    unaffected by whatever a comparison harness tunes this to): the
    EFFECTIVE, PlantResponse-facing setpoint only updates every N minutes,
    held fixed in between, rather than chasing a fresh recommendation every
    single minute. `recommend_setpoint()` is still CALLED every minute
    regardless -- skipping the call itself would also skip advancing
    AnticipatorySetpointAdvisor's LiveFeatureBuilder history (its lag/EWMA
    features are defined in calendar minutes, not in "advisor decisions"),
    corrupting the features on the minutes it IS consulted. Only which of
    those per-minute recommendations actually reaches PlantResponse is
    throttled.

    M10 Phase 2 vs M12 FOUND OPPOSITE ANSWERS HERE, both honestly, against
    different plants. Phase 2 (BangBangPlantResponse) confirmed 1 was
    already optimal -- "nothing here has M8's physical-actuator/bandwidth
    constraint," so spacing updates out only added stale chatter. M12
    (PlantResponse's PI loop, with a real ~17 min settling time) found the
    opposite: a 1-minute cadence fights the inner loop's own settling
    (diagnosed as a cascaded-control timescale mismatch after M12's first
    re-tune attempt came back a regression), and every comparison that
    actually uses AnticipatorySetpointAdvisor against PlantResponse should
    pass `advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN`
    (src/controllers.py) explicitly rather than rely on this function's own
    default -- see that constant's docstring for the full mechanism and
    src/compare_controllers.py's compare() for where it's actually applied.

    `tint_controller` (M11, default None = no tinting, tint_level always
    0.0 -- bit-for-bit today's pre-M11 behaviour): an object matching
    ReactiveTintController/AnticipatoryTintAdvisor's
    `.recommend_tint(ControllerInputs) -> float` interface
    (src/tint_controller.py). Unlike `controller_factory`, this is a ready
    instance, not a factory -- neither tint controller has per-scenario
    state to construct fresh (no LiveFeatureBuilder-equivalent), so there's
    nothing a factory would buy here. Consulted every minute using the SAME
    ControllerInputs already built for the HVAC advisor.

    `station_precool_advisor` (M13, default None = no precool, byte-identical
    to pre-M13 behaviour): a StationPrecoolAdvisor (src/controllers.py),
    same ready-instance shape as `tint_controller`. UNLIKE the general
    advisor's own recommendation, its shift is added to `effective_setpoint_c`
    EVERY MINUTE, bypassing `advisor_update_interval_min` entirely -- see
    StationPrecoolAdvisor's own docstring for why that's safe here (a
    schedule-driven term, not a feedback one, so none of the cascaded-control
    risk that throttle exists for applies). The combined total is re-clipped
    to `hvac.advisor_setpoint_min_c/max_c` -- the same authority bound the
    general advisor's own recommendation was already clipped to -- so the
    two additive terms can never together exceed what a single term alone
    could already command.
    """
    cfg = cfg if cfg is not None else load_config()
    horizon = cfg["simulation"]["control_horizon_min"]

    wx = pd.read_csv(DATA_DIR / f"weather_{city}_summer.csv", parse_dates=["timestamp"])
    service = Service(direction=direction, pattern=pattern, load_factor=load_factor)
    profile = simulate(service, cfg)
    # M11: tunnels (or any other sun-blocking route feature -- see
    # config/cabin_params.yaml's tunnel_zones_min) mask GHI to ~0 for both
    # the current minute and the forecast horizon, so a tunnel shows up to
    # every consumer of ghi_w_m2/ghi_fcst_h (the HVAC advisor, the tint
    # controllers) as "no sun," with no separate in_tunnel field needed
    # anywhere downstream -- see src/tint_controller.py's module docstring.
    #
    # ONLY APPLIED WHEN tint_controller IS PROVIDED, deliberately -- the
    # `in_tunnel` column itself is still always computed/reported (harmless,
    # useful for inspection), but M2 already decided tunnel effects were out
    # of scope for the core cabin model ("add fidelity that no reviewer will
    # check, and cost hours" -- ROADMAP.md's M2 section). M11 reintroduces
    # them narrowly, for the specific demo they're the point of, not as a
    # silent universal physics change that would shift M5/M6/M10's
    # already-verified numbers for scenarios that were never about tinting.
    # Confirmed this distinction matters, not assumed: an earlier version
    # applied the mask unconditionally and flipped one M10 TEST-split
    # scenario from "not worse on both" to "worse on both" purely from two
    # tunnel zones neither the HVAC advisor nor PlantResponse has anything
    # to do with -- caught by tests/test_compare_controllers.py going red,
    # not by inspection.
    in_tunnel_raw = profile.in_tunnel if profile.in_tunnel else [False] * len(profile)
    in_tunnel = in_tunnel_raw if tint_controller is not None else [False] * len(profile)
    # round(), not int() -- data_generator.py rounds depart_hour to the nearest
    # minute when starting a scenario; truncating here instead would silently
    # start this replay up to a minute earlier than the scenario it's meant to
    # match. Harmless in effect (weather is interpolated and slowly varying, see
    # weather.to_minutes), but a real cross-file inconsistency, not just style.
    start = pd.Timestamp(date) + pd.to_timedelta(round(depart_hour * 60), unit="min")
    w = to_minutes(wx, start, len(profile) + horizon)

    model = CabinModel(cfg)
    advisor = controller_factory(model)
    plant = PlantResponse(model)  # one shared instance per run -- see docstring
    state = model.initial_state(model.setpoint(float(w["t_out_c"].iloc[0])))

    effective_setpoint_c = None  # holds the last-applied recommendation between updates
    rows = []
    for t in range(len(profile)):
        fcst_t = t + horizon
        ghi_now = 0.0 if in_tunnel[t] else float(w["ghi_w_m2"].iloc[t])
        ghi_fcst = 0.0 if (fcst_t < len(in_tunnel) and in_tunnel[fcst_t]) else float(w["ghi_w_m2"].iloc[fcst_t])
        inputs = ControllerInputs(
            t_air_c=state.t_air_c, t_mass_c=state.t_mass_c, t_out_c=float(w["t_out_c"].iloc[t]),
            ghi_w_m2=ghi_now, n_pax=profile.n_pax[t],
            door_open=profile.door_open[t], q_hvac_actual_w=state.q_hvac_actual_w,
            time_to_next_station_min=profile.time_to_next_station_min[t],
            expected_boarding=profile.expected_boarding[t],
            t_out_fcst_h=float(w["t_out_c"].iloc[fcst_t]),
            ghi_fcst_h=ghi_fcst,
            time_to_last_station_min=profile.time_to_last_station_min[t],
        )
        # Called every minute regardless of the update interval -- see
        # advisor_update_interval_min's docstring on why the CALL and the
        # APPLICATION are deliberately decoupled.
        recommended = advisor.recommend_setpoint(inputs)
        if effective_setpoint_c is None or t % advisor_update_interval_min == 0:
            effective_setpoint_c = recommended
        # M13: precool is additive and unthrottled -- see
        # station_precool_advisor's own docstring paragraph above for why it
        # bypasses advisor_update_interval_min while effective_setpoint_c
        # itself does not. Re-clipped to the same authority bound
        # AnticipatorySetpointAdvisor's own recommendation was already
        # clipped to, so this can't grant MORE total authority than one term
        # alone already had -- only lets both terms actually combine.
        precool_shift_c = (station_precool_advisor.precool_shift_c(inputs)
                            if station_precool_advisor is not None else 0.0)
        hv = cfg["hvac"]
        advised_setpoint_c = min(max(effective_setpoint_c + precool_shift_c,
                                      hv["advisor_setpoint_min_c"]), hv["advisor_setpoint_max_c"])
        cmd = plant.respond(inputs.t_air_c, advised_setpoint_c)
        tint_level = tint_controller.recommend_tint(inputs) if tint_controller is not None else 0.0
        r = model.step(state, CabinInputs(
            t_out_c=inputs.t_out_c, ghi_w_m2=inputs.ghi_w_m2, n_pax=inputs.n_pax,
            door_open=inputs.door_open, q_hvac_cmd_w=cmd, tint_level=tint_level,
            rh_out_pct=float(w["rh_out_pct"].iloc[t]),
        ), dt_s=60.0)
        # setpoint stays the fixed EN13129 scoring reference used to judge
        # EVERY arm identically (comfort metric, degree_hours) -- this is
        # unrelated to advised_setpoint_c, which is the new, genuinely
        # per-arm quantity (what THIS advisor recommended this minute).
        # Easy to conflate the two; they answer different questions.
        setpoint = model.setpoint(inputs.t_out_c)
        rows.append({
            "t": t, "t_air_c": r.t_air_c, "t_mass_c": r.t_mass_c, "setpoint_c": setpoint,
            "advised_setpoint_c": advised_setpoint_c, "precool_shift_c": precool_shift_c,
            "err_c": r.t_air_c - setpoint, "n_pax": profile.n_pax[t],
            "door_open": profile.door_open[t], "in_tunnel": in_tunnel[t],
            "tint_level": tint_level, "q_solar_w": r.q_solar_w,
            # cmd_w is now a SIMULATED PlantResponse output, not "Railflow's
            # decision" -- kept under its old name/units since score() and
            # every existing consumer still key off it unchanged.
            "cmd_w": cmd, "q_actual_w": r.q_hvac_actual_w,
            "electrical_w": r.electrical_w,
            "w_air_g_kg": r.w_air_g_kg, "rh_air_pct": r.rh_air_pct,
            # w_air_g_kg is identical for every controller run on the same
            # scenario -- the moisture balance (M9) depends only on
            # n_pax/door_open/t_out_c/rh_out_pct, never on q_hvac (no coil
            # dehumidification term). rh_air_pct is NOT necessarily
            # identical, even though it's derived from w_air_g_kg -- %RH
            # also depends on t_air_c, which DOES differ between
            # controllers, so two controllers can show different %RH while
            # sharing identical underlying moisture. Confirmed by a failing
            # test before this comment was written, not assumed.
        })
    return pd.DataFrame(rows)


@dataclass
class ScenarioResult:
    energy_kwh: float
    degree_hours: float
    worst_excursion_k: float
    mean_err_k: float
    minutes: int


def score(trajectory: pd.DataFrame, cfg: dict) -> ScenarioResult:
    """Summarises a run_controller() trajectory into the M5 metrics."""
    band_k = cfg["comfort"]["band_k"]
    return ScenarioResult(
        energy_kwh=float(trajectory["electrical_w"].sum() / 60000.0),
        degree_hours=degree_hours_outside_band(
            trajectory["t_air_c"], trajectory["setpoint_c"], band_k),
        worst_excursion_k=float(trajectory["err_c"].abs().max()),
        mean_err_k=float(trajectory["err_c"].mean()),
        minutes=len(trajectory),
    )


if __name__ == "__main__":
    from src.controllers import ThermostatController

    cfg = load_config()
    traj = run_controller(lambda model: ThermostatController(model), cfg)
    result = score(traj, cfg)
    print(f"ThermostatController, Cairo 2024-07-20 08:00:")
    print(f"  energy       : {result.energy_kwh:.2f} kWh")
    print(f"  degree-hours : {result.degree_hours:.2f} K*h")
    print(f"  worst excursion : {result.worst_excursion_k:.2f} K")
    print(f"  mean error   : {result.mean_err_k:+.2f} K")
