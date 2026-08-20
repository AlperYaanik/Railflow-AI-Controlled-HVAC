"""Generate the training dataset by simulating many scenarios end to end.

Each scenario is one train journey: a (city, calendar date, departure hour,
Service) combination, simulated minute by minute through the same stack M1-M3
built — real interpolated weather, the timetable/occupancy profile, and the
two-node cabin model with its actuator.

Scenario dimensions and their measured effect on energy (see docs/PARAMETERS.md
and ROADMAP.md M4), which is why departure hour gets by far the widest sampling
range and why sampling is random rather than a fixed grid:

    departure hour   120% swing   sampled continuously across ENR's 04:00-23:00
                                   operating window
    city              ~25%        both included every run
    load factor        28%        drawn from the M3 catalogue values
    day (of 92)        ~15%       drawn uniformly across real calendar dates,
                                   NOT percentile-stratified -- see note below
    stopping pattern   8.9%       drawn from the M3 catalogue values

WINTER IS EXCLUDED. This is a scope decision, not a bug carried over: Egypt is
cooling-dominated (M1) and summer is the design case; heating essentially never
engages at realistic occupancy even with the COP fix in place (see
docs/PARAMETERS.md). Including it would roughly double the scenario grid on a
3-hour budget for a regime the model rarely needs. The generator does not
special-case winter data out -- it simply never asks for it, so adding it later
is a one-line change to `_load_weather`, not a redesign.

Why uniform random dates rather than percentile-stratified sampling (contrast
with the exploratory analysis in weather.day_at_percentile, which deliberately
over-samples extremes to test hypotheses): a forecaster's error should reflect
real operating frequency. Deliberately over-representing extreme days here
would bias the model's average error away from what it will actually see.

The controller commanding HVAC during generation is deliberately NOT the
reactive thermostat used for evaluation elsewhere. A model trained only on one
policy's closed-loop trajectory sees command tightly coupled to error and has
little independent (state, command) variation to learn the plant's response
from -- persistence of excitation, the standard system-identification argument
for why training data needs deliberate exploration. See StochasticController.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cabin_model import CabinInputs, CabinModel
from src.config import DATA_DIR, load_config
from src.occupancy import Service, simulate
from src.weather import to_minutes

OUTPUT_PATH = DATA_DIR / "scenarios_raw.parquet"

CITIES = ("cairo", "aswan")
DEPART_HOUR_RANGE = (4.0, 23.0)  # ENR's real daily operating window
LOAD_FACTORS = (0.4, 0.7, 1.0, 1.1)
PATTERNS = ("semi_express", "express")
DIRECTIONS = ("down", "up")

TARGET_ROWS = 30_000
"""Saturates a ~20-feature tabular problem (ROADMAP M4); more wastes budget."""


@dataclass
class ControlMode:
    name: str
    frac: float | None
    """Fixed command fraction in [-1, 1] for this mode, or None for 'reactive'
    (proportional-to-error, the realistic default most minutes should use)."""


MODES = [
    ControlMode("reactive", None),
    ControlMode("full_cool", -1.0),
    ControlMode("full_heat", 1.0),
    ControlMode("off", 0.0),
    ControlMode("random", None),  # resolved to a fresh random level per hold
]
MODE_WEIGHTS = [0.55, 0.10, 0.05, 0.15, 0.15]


class StochasticController:
    """Persistence-of-excitation command policy for data generation.

    Spends most minutes in a realistic proportional-to-error mode, with a
    randomised "aggressiveness" per scenario, but periodically holds a forced
    mode (full power, off, or a random fixed level) for a few minutes at a
    time. This is what gives the dataset genuine (state, command) diversity
    rather than one narrow closed-loop trajectory repeated with different
    exogenous inputs.

    Also jitters the setpoint by a small per-scenario bias, so the model does
    not overfit to one exact sliding-setpoint curve.
    """

    def __init__(self, model: CabinModel, rng: np.random.Generator):
        self.model = model
        self.rng = rng
        self.gain_divisor = rng.uniform(0.8, 3.0)
        self.setpoint_bias = float(np.clip(rng.normal(0.0, 0.5), -1.5, 1.5))
        self._mode: ControlMode = MODES[0]
        self._mode_random_frac = 0.0
        self._hold_until = -1

    def _draw_mode(self, minute: int) -> None:
        self._mode = self.rng.choice(MODES, p=MODE_WEIGHTS)
        if self._mode.name == "random":
            self._mode_random_frac = float(self.rng.uniform(-1.0, 1.0))
        self._hold_until = minute + int(self.rng.integers(3, 21))

    def setpoint(self, t_out_c: float) -> float:
        return self.model.setpoint(t_out_c) + self.setpoint_bias

    def command(self, minute: int, t_air_c: float, t_out_c: float) -> float:
        if minute >= self._hold_until:
            self._draw_mode(minute)

        if self._mode.name == "reactive":
            # Sign convention matches the dispatch below: frac < 0 means too
            # hot -> cool, frac > 0 means too cold -> heat. So the error has
            # to be (setpoint - t_air), not (t_air - setpoint).
            err = self.setpoint(t_out_c) - t_air_c
            frac = float(np.clip(err / self.gain_divisor, -1.0, 1.0))
        elif self._mode.name == "random":
            frac = self._mode_random_frac
        else:
            frac = self._mode.frac

        hv = self.model.cfg["hvac"]
        return frac * hv["cooling_capacity_w"] if frac < 0 else frac * hv["heating_capacity_w"]


def _usable_dates(weather: pd.DataFrame) -> list:
    """Calendar dates safe to start a journey from.

    Excludes the last cached date: to_minutes() falls back to constant
    extrapolation past the end of the archive, which would flat-line the tail
    of a late-departure journey starting on the final day.
    """
    dates = sorted(weather["timestamp"].dt.date.unique())
    return dates[:-1] if len(dates) > 1 else dates


def generate_scenario(
    scenario_id: int,
    city: str,
    weather: pd.DataFrame,
    date,
    depart_hour: float,
    service: Service,
    cfg: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate one journey minute by minute. Returns a per-minute DataFrame."""
    profile = simulate(service, cfg)
    start = pd.Timestamp(date) + pd.to_timedelta(round(depart_hour * 60), unit="min")
    w = to_minutes(weather, start, len(profile))

    model = CabinModel(cfg)
    controller = StochasticController(model, rng)
    state = model.initial_state(controller.setpoint(float(w["t_out_c"].iloc[0])))

    rows = []
    for t in range(len(profile)):
        t_out = float(w["t_out_c"].iloc[t])
        ghi = float(w["ghi_w_m2"].iloc[t])
        rh_out = float(w["rh_out_pct"].iloc[t])
        cmd = controller.command(t, state.t_air_c, t_out)

        r = model.step(state, CabinInputs(
            t_out_c=t_out, ghi_w_m2=ghi,
            n_pax=profile.n_pax[t], door_open=profile.door_open[t],
            q_hvac_cmd_w=cmd, rh_out_pct=rh_out,
        ), dt_s=60.0)

        rows.append({
            "scenario_id": scenario_id, "city": city, "date": date,
            "depart_hour": depart_hour, "direction": service.direction,
            "pattern": service.pattern, "load_factor": service.load_factor,
            "minute": t,
            "t_air_c": r.t_air_c, "t_mass_c": r.t_mass_c,
            "t_out_c": t_out, "ghi_w_m2": ghi, "rh_out_pct": rh_out,
            "n_pax": profile.n_pax[t], "door_open": profile.door_open[t],
            "time_to_next_station_min": profile.time_to_next_station_min[t],
            "expected_boarding": profile.expected_boarding[t],
            "setpoint_c": controller.setpoint(t_out),
            "q_hvac_cmd_w": cmd, "q_hvac_actual_w": r.q_hvac_actual_w,
            "electrical_w": r.electrical_w, "cop": r.cop,
            "w_air_g_kg": r.w_air_g_kg, "rh_air_pct": r.rh_air_pct,
            # Not used as a forecaster feature (FEATURE_COLUMNS is an
            # explicit allow-list, see src/features.py) -- captured here so
            # it exists in scenarios_raw.parquet for anyone who wants to
            # look at it later, same reasoning M9 gave for tracking it in
            # CabinModel in the first place: not retraining the forecaster
            # on it, just no longer throwing it away.
        })

    return pd.DataFrame(rows)


def generate_dataset(
    cfg: dict | None = None,
    target_rows: int = TARGET_ROWS,
    seed: int = 20260818,
) -> pd.DataFrame:
    """Random scenario sampling until `target_rows` is reached. Reproducible via `seed`."""
    cfg = cfg if cfg is not None else load_config()
    rng = np.random.default_rng(seed)

    weather = {city: pd.read_csv(DATA_DIR / f"weather_{city}_summer.csv",
                                  parse_dates=["timestamp"]) for city in CITIES}
    dates = {city: _usable_dates(weather[city]) for city in CITIES}

    frames, total_rows, scenario_id = [], 0, 0
    while total_rows < target_rows:
        city = rng.choice(CITIES)
        date = rng.choice(dates[city])
        depart_hour = float(rng.uniform(*DEPART_HOUR_RANGE))
        service = Service(
            direction=str(rng.choice(DIRECTIONS)),
            pattern=str(rng.choice(PATTERNS)),
            load_factor=float(rng.choice(LOAD_FACTORS)),
        )

        df = generate_scenario(scenario_id, city, weather[city], date,
                                depart_hour, service, cfg, rng)
        frames.append(df)
        total_rows += len(df)
        scenario_id += 1

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    data = generate_dataset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)

    print(f"scenarios : {data['scenario_id'].nunique()}")
    print(f"rows      : {len(data)}")
    print(f"cities    : {data['city'].value_counts().to_dict()}")
    print(f"dates     : {data['date'].nunique()} unique calendar days")
    print(f"T_air     : {data['t_air_c'].min():.1f} - {data['t_air_c'].max():.1f} C")
    print(f"depart_hr : {data['depart_hour'].min():.1f} - {data['depart_hour'].max():.1f}")
    print(f"written to {OUTPUT_PATH}")
