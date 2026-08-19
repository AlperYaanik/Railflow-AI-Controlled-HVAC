"""M7: live demo, Streamlit. Runs the SAME simulator M4-M6 were built and
tested on -- no separate demo path. `run_controller()` (src/evaluate.py),
`ThermostatController`/`AnticipatoryController` (src/controllers.py) are
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
from src.controllers import AnticipatoryController, ThermostatController
from src.evaluate import run_controller, score
from src.features import LiveFeatureBuilder
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
    run_kwargs = dict(cfg=cfg, city=city, date=str(date), depart_hour=depart_hour,
                       direction=direction, pattern=pattern, load_factor=load_factor)

    thermo_traj = run_controller(lambda m: ThermostatController(m), **run_kwargs)

    def antic_factory(model):
        builder = LiveFeatureBuilder(cfg, city=city, direction=direction, pattern=pattern,
                                      load_factor=load_factor, depart_hour=depart_hour)
        return AnticipatoryController(model, booster, builder)

    antic_traj = run_controller(antic_factory, **run_kwargs)
    return thermo_traj, antic_traj, score(thermo_traj, cfg), score(antic_traj, cfg)


st.title("Railflow — anticipatory HVAC vs. today's on/off control")
st.caption(
    "Same simulator and controllers M4–M6 were built and tested on — this demo "
    "reuses run_controller() and score() directly, not a separate reimplementation."
)

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
choice = st.sidebar.selectbox("Pick a journey", scenarios["label"], index=0)
row = scenarios.loc[scenarios["label"] == choice].iloc[0]

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
col1.metric("Energy — on/off", f"{thermo_score.energy_kwh:.2f} kWh")
col2.metric("Energy — anticipatory", f"{antic_score.energy_kwh:.2f} kWh", f"{saving_pct:+.1f}%")
col3.metric("Comfort — on/off", f"{thermo_score.degree_hours:.2f} K·h")
col4.metric("Comfort — anticipatory", f"{antic_score.degree_hours:.2f} K·h",
            f"{antic_score.degree_hours - thermo_score.degree_hours:+.2f} K·h",
            delta_color="inverse")


def render(up_to: int):
    t = thermo_traj.iloc[:up_to + 1]
    a = antic_traj.iloc[:up_to + 1]

    fig, (ax_temp, ax_cmd) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})

    ax_temp.fill_between(t["t"], t["setpoint_c"] - 1.0, t["setpoint_c"] + 1.0,
                          color="green", alpha=0.12, label="thermostat hysteresis band")
    ax_temp.plot(t["t"], t["setpoint_c"], "--", color="gray", linewidth=1, label="setpoint")
    ax_temp.plot(t["t"], t["t_air_c"], color="#d62728", linewidth=1.8, label="on/off")
    ax_temp.plot(a["t"], a["t_air_c"], color="#2ca02c", linewidth=1.8, label="anticipatory")
    boarding = t.loc[t["expected_boarding"].fillna(0) > 0, "t"] if "expected_boarding" in t else []
    ax_temp.set_ylabel("cabin air temp (°C)")
    ax_temp.set_title(f"minute {up_to} / {n_minutes - 1}")
    ax_temp.legend(loc="upper right", fontsize=8)
    ax_temp.grid(alpha=0.3)

    ax_cmd.plot(t["t"], t["cmd_w"] / 1000.0, color="#d62728", linewidth=1.2, label="on/off")
    ax_cmd.plot(a["t"], a["cmd_w"] / 1000.0, color="#2ca02c", linewidth=1.2, label="anticipatory")
    ax_cmd.set_ylabel("HVAC command (kW)\n(+heat / −cool)")
    ax_cmd.set_xlabel("minute")
    ax_cmd.legend(loc="upper right", fontsize=8)
    ax_cmd.grid(alpha=0.3)

    fig.tight_layout()
    return fig


placeholder = st.empty()

if auto_play:
    for m in range(minute, n_minutes):
        placeholder.pyplot(render(m))
        time.sleep(0.03)
else:
    placeholder.pyplot(render(minute))

with st.expander("What this scenario looked like going in"):
    st.dataframe(row.drop("label").to_frame().T, hide_index=True)
