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
from src.controllers import ControllerInputs
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
) -> pd.DataFrame:
    """Drives one controller through one real journey, minute by minute.

    `controller_factory(model)` builds the controller against a freshly
    constructed CabinModel, so ThermostatController and AnticipatoryController
    (or anything matching their `.command(ControllerInputs) -> float`
    interface) can be driven identically without src/evaluate.py needing to
    know which one it's holding.
    """
    cfg = cfg if cfg is not None else load_config()
    horizon = cfg["simulation"]["control_horizon_min"]

    wx = pd.read_csv(DATA_DIR / f"weather_{city}_summer.csv", parse_dates=["timestamp"])
    service = Service(direction=direction, pattern=pattern, load_factor=load_factor)
    profile = simulate(service, cfg)
    # round(), not int() -- data_generator.py rounds depart_hour to the nearest
    # minute when starting a scenario; truncating here instead would silently
    # start this replay up to a minute earlier than the scenario it's meant to
    # match. Harmless in effect (weather is interpolated and slowly varying, see
    # weather.to_minutes), but a real cross-file inconsistency, not just style.
    start = pd.Timestamp(date) + pd.to_timedelta(round(depart_hour * 60), unit="min")
    w = to_minutes(wx, start, len(profile) + horizon)

    model = CabinModel(cfg)
    controller = controller_factory(model)
    state = model.initial_state(model.setpoint(float(w["t_out_c"].iloc[0])))

    rows = []
    for t in range(len(profile)):
        inputs = ControllerInputs(
            t_air_c=state.t_air_c, t_mass_c=state.t_mass_c, t_out_c=float(w["t_out_c"].iloc[t]),
            ghi_w_m2=float(w["ghi_w_m2"].iloc[t]), n_pax=profile.n_pax[t],
            door_open=profile.door_open[t], q_hvac_actual_w=state.q_hvac_actual_w,
            time_to_next_station_min=profile.time_to_next_station_min[t],
            expected_boarding=profile.expected_boarding[t],
            t_out_fcst_h=float(w["t_out_c"].iloc[t + horizon]),
            ghi_fcst_h=float(w["ghi_w_m2"].iloc[t + horizon]),
        )
        cmd = controller.command(inputs)
        r = model.step(state, CabinInputs(
            t_out_c=inputs.t_out_c, ghi_w_m2=inputs.ghi_w_m2, n_pax=inputs.n_pax,
            door_open=inputs.door_open, q_hvac_cmd_w=cmd,
        ), dt_s=60.0)
        setpoint = model.setpoint(inputs.t_out_c)
        rows.append({
            "t": t, "t_air_c": r.t_air_c, "t_mass_c": r.t_mass_c, "setpoint_c": setpoint,
            "err_c": r.t_air_c - setpoint, "n_pax": profile.n_pax[t],
            "door_open": profile.door_open[t], "cmd_w": cmd, "q_actual_w": r.q_hvac_actual_w,
            "electrical_w": r.electrical_w,
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
