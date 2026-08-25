"""M7: live demo, Streamlit. Runs the SAME simulator M4-M6 were built and
tested on -- no separate demo path. `run_controller()` (src/evaluate.py),
`ThermostatController`/`AnticipatorySetpointAdvisor` (src/controllers.py) are
used exactly as src/compare_controllers.py uses them, so this app's numbers
must match that script's for the same scenario. If they ever don't, that's
a bug in this file, not a new result.

Static-first, per ROADMAP.md's M7 guardrail: compute the whole trajectory
up front (a couple of seconds, see the profiling note in ROADMAP.md), then
let the user scrub or auto-play through it -- animating already-computed
data, not trying to step the physics live in lockstep with wall-clock time.
The latter would be far more fragile for a live presentation.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas, see src/train.py.
# Streamlit's own bootstrap may import pandas before this module's top-level
# code runs at all -- untested combination the first time this file is run;
# this ordering is the same best-effort defence the rest of the project
# uses, verified working end to end when this file was first launched (see
# ROADMAP.md's M7 section for the result of that check).

import sys
import time
from pathlib import Path

# streamlit runs this file directly rather than as `python -m`, so the
# project root (parent of src/) isn't on sys.path by default -- without
# this, `from src...` fails with ModuleNotFoundError regardless of which
# directory `streamlit run` was invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.compare_controllers import held_out_test_scenarios
from src.config import load_config
from src.controllers import (
    SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
    AnticipatorySetpointAdvisor,
    StationPrecoolAdvisor,
    ThermostatController,
)
from src.evaluate import run_controller, score
from src.features import LiveFeatureBuilder
from src.tint_controller import ReactiveTintController
from src.train import MODEL_PATH

st.set_page_config(page_title="Railflow", layout="wide")


@st.cache_resource
def get_booster():
    if not MODEL_PATH.exists():
        return None
    return lgb.Booster(model_file=str(MODEL_PATH))


@st.cache_data
def get_scenarios() -> pd.DataFrame:
    return held_out_test_scenarios()


@st.cache_data(show_spinner="Running both controllers through the simulator...")
def run_both(city, date, depart_hour, direction, pattern, load_factor):
    """Computes the full trajectory for both controllers on one scenario.
    Cached on the scenario tuple -- switching scenarios re-simulates,
    re-selecting the same one is instant.
    """
    cfg = load_config()
    booster = get_booster()
    # advisor_update_interval_min: M12's tuned cadence, applied to BOTH
    # controllers -- see SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN's docstring in
    # src/controllers.py (matches src/compare_controllers.py's compare(), so
    # this app's numbers keep matching that script's per this module's own
    # docstring guarantee).
    run_kwargs = dict(cfg=cfg, city=city, date=str(date), depart_hour=depart_hour,
                       direction=direction, pattern=pattern, load_factor=load_factor,
                       advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN)

    thermo_traj = run_controller(lambda m: ThermostatController(m), **run_kwargs)

    def antic_factory(model):
        builder = LiveFeatureBuilder(cfg, city=city, direction=direction, pattern=pattern,
                                      load_factor=load_factor, depart_hour=depart_hour)
        return AnticipatorySetpointAdvisor(model, booster, builder)

    antic_traj = run_controller(antic_factory, **run_kwargs)
    return thermo_traj, antic_traj, score(thermo_traj, cfg), score(antic_traj, cfg)


@st.cache_data(show_spinner="Running the window-tint comparison...")
def run_tint_comparison(city, date, depart_hour, direction, pattern, load_factor):
    """M11, simplified after M11-continued's finding: only no-tint vs
    REACTIVE (SPD-style) tint runs live here now. AnticipatoryTintAdvisor
    was checked across all 23 TEST scenarios and found to use MORE energy
    than reactive in 14/23, averaging +0.34% worse -- a fixed t+30min
    forecast fires a "ghost tunnel" dip a full 30 minutes before every real
    one, while actual sun is still hitting the window (see ROADMAP.md's
    "M11, continued" section for the full mechanism and numbers). The class
    itself (src/tint_controller.py) is untouched and still tested
    (tests/test_tint_controller.py) -- this only stops the live demo from
    presenting a comparison whose own "anticipatory wins" framing didn't
    survive proper checking. Same HVAC advisor (AnticipatorySetpointAdvisor)
    in both runs, so any energy difference is isolated to tinting.
    """
    cfg = load_config()
    booster = get_booster()
    run_kwargs = dict(cfg=cfg, city=city, date=str(date), depart_hour=depart_hour,
                       direction=direction, pattern=pattern, load_factor=load_factor,
                       advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN)

    def hvac_factory(model):
        builder = LiveFeatureBuilder(cfg, city=city, direction=direction, pattern=pattern,
                                      load_factor=load_factor, depart_hour=depart_hour)
        return AnticipatorySetpointAdvisor(model, booster, builder)

    no_tint = run_controller(hvac_factory, tint_controller=None, **run_kwargs)
    reactive = run_controller(hvac_factory, tint_controller=ReactiveTintController(), **run_kwargs)
    return no_tint, reactive, score(no_tint, cfg), score(reactive, cfg)


@st.cache_data(show_spinner="Running the station-precool comparison...")
def run_precool_comparison(city, date, depart_hour, direction, pattern, load_factor):
    """M13: same scenario, same HVAC advisor -- ONLY station_precool_advisor
    varies, so any difference is isolated to precooling's own effect, same
    isolation principle as M11's tint comparison above. Deliberately not
    applied to ThermostatController -- see compare_controllers.py's
    _compare_pair() docstring on why this is asymmetric by design: the
    claim is "the AI anticipates a known stop," not "precooling helps
    regardless of who's doing it."
    """
    cfg = load_config()
    booster = get_booster()
    run_kwargs = dict(cfg=cfg, city=city, date=str(date), depart_hour=depart_hour,
                       direction=direction, pattern=pattern, load_factor=load_factor,
                       advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN)

    def hvac_factory(model):
        builder = LiveFeatureBuilder(cfg, city=city, direction=direction, pattern=pattern,
                                      load_factor=load_factor, depart_hour=depart_hour)
        return AnticipatorySetpointAdvisor(model, booster, builder)

    without_precool = run_controller(hvac_factory, **run_kwargs)
    with_precool = run_controller(
        hvac_factory, **run_kwargs,
        station_precool_advisor=StationPrecoolAdvisor(
            lead_min=cfg["hvac"]["station_precool_lead_min"],
            shift_k=cfg["hvac"]["station_precool_shift_k"],
        ),
    )
    return without_precool, with_precool, score(without_precool, cfg), score(with_precool, cfg)


st.title("Railflow — anticipatory HVAC vs. a static setpoint schedule")
st.caption(
    "Same simulator and controllers M4–M6 were built and tested on — this demo "
    "reuses run_controller() and score() directly, not a separate reimplementation."
)

cfg = load_config()
booster = get_booster()
if booster is None:
    st.error(f"No trained model at {MODEL_PATH}. Run `python -m src.train` first.")
    st.stop()

# get_scenarios()/run_both() below hit data/scenarios_raw.parquet and
# data/weather_{city}_summer.csv directly (via held_out_test_scenarios()/
# run_controller()) with no existence check of their own -- same class of
# gap the booster check above already closes for the model file, applied to
# the two setup steps ahead of it (see README's Setup section: weather,
# then data_generator, then train). Caught by code inspection, not a user
# report; confirmed by reproducing the raw FileNotFoundError before adding
# this guard (see tests/test_app.py's two missing-data tests).
try:
    scenarios = get_scenarios()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}. Run `python -m src.weather` "
             f"and `python -m src.data_generator` first.")
    st.stop()

scenarios = scenarios.assign(
    label=scenarios.apply(
        lambda r: f"#{r.scenario_id} — {r.city}, {r.date}, {r.depart_hour:.1f}h, "
                  f"{r.direction}/{r.pattern}, load {r.load_factor}",
        axis=1,
    )
)

st.sidebar.header("Scenario")
st.sidebar.caption(
    "All 23 are from the M4 held-out test split (2024-08-18 to 2024-08-30) — "
    "dates the forecaster never trained or validated on."
)
# Defaults to scenario 140 (cairo, 2024-08-25, 8.93h) rather than whichever
# scenario_id happens to sort first: under M12's shipped tuning most
# scenarios are near-indistinguishable (10/23 bit-identical -- see
# tests/test_integration.py's test_both_m5_controllers_produce_genuinely_
# different_trajectories) or scenario_id=1's own near-zero -0.03% -- a weak
# first impression for a live demo opening on this page. 140 is the
# cleanest illustrative case in the actual TEST-split record: +0.29% energy,
# EXACTLY zero comfort cost (not a traded-away margin) -- not the largest
# number available (scenario 45 reaches +0.45%, scenario 6 the largest RMS
# divergence), picked for having no caveat to explain, not for being the
# biggest. Every other scenario, including every less flattering one,
# remains one click away in this same picker -- this changes only which
# view opens first, not what's honestly on the record (see ROADMAP.md's M12
# section and data/m5_test_split_comparison.csv for the full 23-scenario
# range, unedited).
_default_matches = scenarios.index[scenarios["scenario_id"] == 140].tolist()
default_index = _default_matches[0] if _default_matches else 0
choice = st.sidebar.selectbox("Pick a journey", scenarios["label"], index=default_index)
row = scenarios.loc[scenarios["label"] == choice].iloc[0]

st.header("Station precool (M13)")
st.caption(
    "Same scenario, same HVAC advisor — only station_precool_advisor differs. A station stop is a "
    "KNOWN, schedule-certain event (occupancy.py's time_to_next_station_min counts down exactly, "
    "not a prediction): doors open, passengers board, heat and humidity load rises. Deliberately not "
    "routed through the same forecast blend (ff_weight) as the general advisor below — M10/M12 both "
    "found that generic ML-forecast blend hurts aggregate energy/comfort; this targets one specific, "
    "demonstrable event directly instead."
)

try:
    without_precool_traj, with_precool_traj, without_precool_score, with_precool_score = (
        run_precool_comparison(row.city, row.date, row.depart_hour, row.direction, row.pattern, row.load_factor)
    )

    precool_saving_pct = 100.0 * (1.0 - with_precool_score.energy_kwh / without_precool_score.energy_kwh)
    ppcol1, ppcol2, ppcol3, ppcol4 = st.columns(4)
    ppcol1.metric("Energy — no precool", f"{without_precool_score.energy_kwh:.2f} kWh")
    ppcol2.metric("Energy — with precool", f"{with_precool_score.energy_kwh:.2f} kWh",
                  f"{precool_saving_pct:+.1f}%")
    ppcol3.metric("Comfort — no precool", f"{without_precool_score.degree_hours:.2f} K·h")
    ppcol4.metric("Comfort — with precool", f"{with_precool_score.degree_hours:.2f} K·h",
                  f"{with_precool_score.degree_hours - without_precool_score.degree_hours:+.2f} K·h",
                  delta_color="inverse")

    ppcol5, ppcol6 = st.columns(2)
    ppcol5.metric("Worst deviation from setpoint — no precool", f"{without_precool_score.worst_excursion_k:.2f} K")
    ppcol6.metric("Worst deviation from setpoint — with precool", f"{with_precool_score.worst_excursion_k:.2f} K",
                  f"{with_precool_score.worst_excursion_k - without_precool_score.worst_excursion_k:+.2f} K",
                  delta_color="inverse")
    st.caption(
        "\"Worst deviation\" is worst_excursion_k -- the largest |cabin temp − setpoint| gap anywhere "
        "in the WHOLE journey (same metric ScenarioResult/score() uses everywhere else in this app), "
        "not a number picked from around one specific stop."
    )

    fig_precool, (ax_pc_temp, ax_pc_set) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    door_t = with_precool_traj.loc[with_precool_traj["door_open"], "t"].tolist()
    for ax in (ax_pc_temp, ax_pc_set):
        if door_t:
            run_start = prev = door_t[0]
            first_label = True
            for m in door_t[1:] + [None]:
                if m is not None and m == prev + 1:
                    prev = m
                    continue
                ax.axvspan(run_start, prev + 1, color="gray", alpha=0.25,
                           label="station stop" if first_label else None)
                first_label = False
                if m is not None:
                    run_start = prev = m

    precool_band_k = cfg["comfort"]["band_k"]
    ax_pc_temp.fill_between(without_precool_traj["t"], without_precool_traj["setpoint_c"] - precool_band_k,
                             without_precool_traj["setpoint_c"] + precool_band_k,
                             color="green", alpha=0.12, label=f"comfort band (±{precool_band_k:g} K, scored)")
    ax_pc_temp.plot(without_precool_traj["t"], without_precool_traj["setpoint_c"], "--", color="gray",
                     linewidth=1, label="setpoint")
    ax_pc_temp.plot(without_precool_traj["t"], without_precool_traj["t_air_c"], color="#ff7f0e",
                     linewidth=1.6, label="AI, no precool")
    ax_pc_temp.plot(with_precool_traj["t"], with_precool_traj["t_air_c"], color="#2ca02c",
                     linewidth=1.6, label="AI + station precool")
    ax_pc_temp.set_ylabel("cabin air temp (°C)")
    ax_pc_temp.legend(loc="upper right", fontsize=8)
    ax_pc_temp.grid(alpha=0.3)

    ax_pc_set.plot(without_precool_traj["t"], without_precool_traj["advised_setpoint_c"], color="#ff7f0e",
                    linewidth=1.2, label="AI, no precool")
    ax_pc_set.plot(with_precool_traj["t"], with_precool_traj["advised_setpoint_c"], color="#2ca02c",
                    linewidth=1.2, label="AI + station precool")
    ax_pc_set.set_ylabel("recommended\nsetpoint (°C)")
    ax_pc_set.set_xlabel("minute")
    ax_pc_set.legend(loc="upper right", fontsize=8)
    ax_pc_set.grid(alpha=0.3)

    fig_precool.tight_layout()
    st.pyplot(fig_precool)
    st.caption(
        "Top panel: cabin temperature against the actual scored comfort band, not a peak-temperature "
        "number quoted in isolation. Bottom panel: what the AI actually asked for -- the step drops in "
        "the green line ARE the precool commands themselves, shown against the same no-precool "
        "baseline so the request (bottom) and its effect on temperature (top) can be read together. "
        "shift_k (config/cabin_params.yaml's hvac.station_precool_shift_k) is a single constant, not "
        "swept or trend-fitted -- see StationPrecoolAdvisor's docstring in src/controllers.py."
    )
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}.")

st.divider()
st.header("Anticipatory HVAC vs. a static setpoint schedule (M10-M12)")

try:
    thermo_traj, antic_traj, thermo_score, antic_score = run_both(
        row.city, row.date, row.depart_hour, row.direction, row.pattern, row.load_factor
    )
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}. Run `python -m src.weather` first.")
    st.stop()

n_minutes = len(thermo_traj)
st.sidebar.divider()
st.sidebar.header("Playback")
minute = st.sidebar.slider("Minute", 0, n_minutes - 1, n_minutes - 1)
auto_play = st.sidebar.checkbox("Auto-play from here")

col1, col2, col3, col4 = st.columns(4)
saving_pct = 100.0 * (1.0 - antic_score.energy_kwh / thermo_score.energy_kwh)
col1.metric("Energy — static schedule", f"{thermo_score.energy_kwh:.2f} kWh")
col2.metric("Energy — anticipatory", f"{antic_score.energy_kwh:.2f} kWh", f"{saving_pct:+.1f}%")
col3.metric("Comfort — static schedule", f"{thermo_score.degree_hours:.2f} K·h")
col4.metric("Comfort — anticipatory", f"{antic_score.degree_hours:.2f} K·h",
            f"{antic_score.degree_hours - thermo_score.degree_hours:+.2f} K·h",
            delta_color="inverse")


def render(up_to: int):
    t = thermo_traj.iloc[:up_to + 1]
    a = antic_traj.iloc[:up_to + 1]

    fig, (ax_temp, ax_cmd, ax_rh) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

    band_k = cfg["comfort"]["band_k"]
    ax_temp.fill_between(t["t"], t["setpoint_c"] - band_k, t["setpoint_c"] + band_k,
                          color="green", alpha=0.12, label=f"comfort band (±{band_k:g} K, scored)")
    ax_temp.plot(t["t"], t["setpoint_c"], "--", color="gray", linewidth=1, label="setpoint")
    ax_temp.plot(t["t"], t["t_air_c"], color="#d62728", linewidth=1.8, label="static schedule")
    ax_temp.plot(a["t"], a["t_air_c"], color="#2ca02c", linewidth=1.8, label="anticipatory")
    ax_temp.set_ylabel("cabin air temp (°C)")
    ax_temp.set_title(f"minute {up_to} / {n_minutes - 1}")
    ax_temp.legend(loc="upper right", fontsize=8)
    ax_temp.grid(alpha=0.3)

    # M10: this panel used to plot cmd_w (a watts command) -- Railflow no
    # longer commands power (would require touching the real HVAC unit's
    # control electronics; see src/controllers.py's M10 note). It now shows
    # advised_setpoint_c, the recommendation each advisor actually produces.
    # NOT a continuous overlay of the dashed setpoint reference above, since
    # M12: advised_setpoint_c only updates every SHIPPED_ADVISOR_UPDATE_
    # INTERVAL_MIN (50) minutes for BOTH advisors (see that constant's
    # docstring in src/controllers.py), so it staircases away from the
    # continuously-updating setpoint_c reference by up to ~0.4 C between
    # updates -- confirmed by tracing a real run, not assumed. An earlier
    # version of this comment claimed an exact overlay; that was true only
    # before M12 introduced the shared update-interval throttle.
    ax_cmd.plot(t["t"], t["advised_setpoint_c"], color="#d62728", linewidth=1.2,
                label="static schedule")
    ax_cmd.plot(a["t"], a["advised_setpoint_c"], color="#2ca02c", linewidth=1.2,
                label="anticipatory (AI-recommended)")
    ax_cmd.set_ylabel("recommended cabin setpoint (°C)")
    ax_cmd.legend(loc="upper right", fontsize=8)
    ax_cmd.grid(alpha=0.3)

    # One line per controller, like the panels above -- checked, not assumed,
    # that this is actually necessary. The moisture SOURCE terms
    # (w_air_g_kg) don't depend on q_hvac (no coil dehumidification term),
    # but the saturation cap added to step() DOES depend on t_air_c, which
    # does depend on q_hvac -- so on a real scenario the two controllers'
    # absolute moisture ends up close but not always bit-identical
    # (tests/test_evaluate.py::test_saturation_cap_can_make_two_controllers_moisture_diverge
    # locks in "close", not "identical" -- an earlier version of that test
    # claimed identical and was wrong). On top of that, %RH is moisture
    # relative to what the air COULD hold at its current temperature, and
    # the two controllers reach different temperatures regardless -- so
    # their %RH readings differ more visibly than the underlying moisture
    # does. A single shared line was tried first and was wrong (confirmed
    # by a failing test, not caught by inspection), so both are drawn here
    # rather than picking one arbitrarily.
    ax_rh.plot(t["t"], t["rh_air_pct"], color="#d62728", linewidth=1.2, label="static schedule")
    ax_rh.plot(a["t"], a["rh_air_pct"], color="#2ca02c", linewidth=1.2, label="anticipatory")
    ax_rh.axhline(65.0, color="gray", linestyle=":", linewidth=1, label="ISO 19659-2 limit (65%)")
    ax_rh.set_ylabel("cabin RH (%)")
    ax_rh.set_xlabel("minute")
    ax_rh.legend(loc="upper right", fontsize=8)
    ax_rh.grid(alpha=0.3)

    fig.tight_layout()
    return fig


placeholder = st.empty()

if auto_play:
    for m in range(minute, n_minutes):
        placeholder.pyplot(render(m))
        time.sleep(0.03)
else:
    placeholder.pyplot(render(minute))

st.caption(
    "Cabin humidity (bottom panel): the two controllers' underlying moisture stays close but not "
    "always identical (it's mostly driven by passengers, doors, and outdoor weather rather than "
    "which controller is running — but a temperature-dependent saturation cap can nudge the two "
    "trajectories apart once cooling makes them diverge). %RH differs more visibly than the "
    "moisture itself, since it's moisture relative to what the air can hold at its own temperature, "
    "and the two controllers reach different temperatures. Neither line credits the AC for removing "
    "moisture while cooling, so both read as an upper bound during active cooling, not a corrected "
    "true value. See ROADMAP.md's M9 section."
)

st.divider()
st.header("Window tinting (M11)")
st.caption(
    "Same scenario, same HVAC advisor (AnticipatorySetpointAdvisor) in both runs — only the tint "
    "controller changes, so any energy difference here is isolated to tinting's effect, not mixed in "
    "with a different HVAC decision. Rule-based, not a trained model (src/tint_controller.py) — "
    "reacts to CURRENT GHI, the same behaviour a real SPD (Suspended Particle Device) panel's own "
    "built-in light sensor already gives for free, no forecast involved."
)

try:
    no_tint_traj, reactive_traj, no_tint_score, reactive_score = run_tint_comparison(
        row.city, row.date, row.depart_hour, row.direction, row.pattern, row.load_factor
    )

    tcol1, tcol2 = st.columns(2)
    tcol1.metric("Energy — no tint", f"{no_tint_score.energy_kwh:.2f} kWh")
    reactive_pct = 100.0 * (1.0 - reactive_score.energy_kwh / no_tint_score.energy_kwh)
    tcol2.metric("Energy — SPD-controlled tint", f"{reactive_score.energy_kwh:.2f} kWh",
                 f"{reactive_pct:+.1f}%")

    fig_tint, ax_tint = plt.subplots(figsize=(11, 2.6))
    tunnel_t = reactive_traj.loc[reactive_traj["in_tunnel"], "t"].tolist()
    if tunnel_t:
        # One axvspan per CONTIGUOUS tunnel run, not one per minute.
        run_start = prev = tunnel_t[0]
        first_label = True
        for m in tunnel_t[1:] + [None]:  # None is a sentinel that flushes the last run
            if m is not None and m == prev + 1:
                prev = m
                continue
            ax_tint.axvspan(run_start, prev + 1, color="gray", alpha=0.25,
                             label="tunnel" if first_label else None)
            first_label = False
            if m is not None:
                run_start = prev = m
    ax_tint.plot(reactive_traj["t"], reactive_traj["tint_level"], color="#ff7f0e", linewidth=1.4,
                 label="SPD-controlled tint")
    ax_tint.set_ylabel("tint level\n(0=clear, 1=dark)")
    ax_tint.set_xlabel("minute")
    ax_tint.set_ylim(-0.05, 1.05)
    ax_tint.legend(loc="upper right", fontsize=8)
    ax_tint.grid(alpha=0.3)
    fig_tint.tight_layout()
    st.pyplot(fig_tint)

    fig_power, ax_power = plt.subplots(figsize=(11, 2.6))
    tunnel_t_power = reactive_traj.loc[reactive_traj["in_tunnel"], "t"].tolist()
    if tunnel_t_power:
        run_start = prev = tunnel_t_power[0]
        first_label = True
        for m in tunnel_t_power[1:] + [None]:
            if m is not None and m == prev + 1:
                prev = m
                continue
            ax_power.axvspan(run_start, prev + 1, color="gray", alpha=0.25,
                              label="tunnel" if first_label else None)
            first_label = False
            if m is not None:
                run_start = prev = m
    ax_power.plot(no_tint_traj["t"], no_tint_traj["electrical_w"] / 1000.0, color="#7f7f7f",
                  linewidth=1.2, label="no tint")
    ax_power.plot(reactive_traj["t"], reactive_traj["electrical_w"] / 1000.0, color="#ff7f0e",
                  linewidth=1.4, label="SPD-controlled tint")
    ax_power.set_ylabel("electrical draw (kW)")
    ax_power.set_xlabel("minute")
    ax_power.legend(loc="upper right", fontsize=8)
    ax_power.grid(alpha=0.3)
    fig_power.tight_layout()
    st.pyplot(fig_power)
    st.caption(
        "Bottom panel shows WHEN energy is actually being spent (`electrical_w`, the same quantity "
        "the kWh metrics above sum over the whole journey), not just the tint decision. The gray line "
        "sits above the orange one whenever tinting is actively blocking solar heat the untinted cabin "
        "would otherwise have to cool -- watch it widen outside the tunnels, where there's real sun to "
        "block, and close up inside them, where there's none."
    )
    st.caption(
        "AI-driven (anticipatory) tinting was built and tested (see ROADMAP.md's \"M11, continued\" "
        "section) -- checked across all 23 TEST scenarios, it used MORE energy than this simple "
        "SPD-style controller in 14/23, averaging +0.34% worse, not better. A fixed t+30min forecast "
        "fires a tint-clearing dip a full 30 minutes before every real tunnel, while actual sun is "
        "still hitting the window. Not shown here because it didn't hold up, not because it wasn't "
        "tried -- `AnticipatoryTintAdvisor` (src/tint_controller.py) still exists and is still tested."
    )
except FileNotFoundError as e:
    st.error(f"Missing data file: {e.filename}.")

with st.expander("What this scenario looked like going in"):
    st.dataframe(row.drop("label").to_frame().T, hide_index=True)
