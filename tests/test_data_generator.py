"""Tests for the scenario data generator.

The reactive-sign test exists because of a real bug found by testing rather
than assuming: the first version of StochasticController computed
`err = t_air_c - setpoint`, which is positive when the cabin is too HOT, and
the downstream dispatch sent positive fractions to HEATING. So the "realistic"
control mode commanded heat whenever the cabin overheated -- for 57.6% of rows
in the first generated dataset. It was caught by checking the mode-weight
distribution against the configured weights (full_heat should have been ~5%,
not 58%), not by any crash or NaN. Fixed to `err = setpoint - t_air_c`.
"""

import numpy as np
import pandas as pd
import pytest

from src.cabin_model import CabinModel
from src.config import DATA_DIR, load_config
from src.data_generator import (
    CITIES,
    ControlMode,
    StochasticController,
    generate_dataset,
    generate_scenario,
)
from src.occupancy import Service, simulate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def model(cfg):
    return CabinModel(cfg)


@pytest.fixture(scope="module")
def small_dataset(cfg, require_weather):
    """One shared, modest dataset for the tests that need real generated rows."""
    return generate_dataset(cfg, target_rows=3000, seed=1)


# --------------------------------------------------------- the actual bug


def test_reactive_mode_cools_when_hot(model):
    controller = StochasticController(model, np.random.default_rng(0))
    controller._mode = ControlMode("reactive", None)
    controller._hold_until = 10**6  # never redraw during this call

    cmd = controller.command(minute=0, t_air_c=40.0, t_out_c=30.0)
    assert cmd < 0.0, "cabin is 14 K above setpoint (26 C) but command is not cooling"


def test_reactive_mode_heats_when_cold(model):
    controller = StochasticController(model, np.random.default_rng(0))
    controller._mode = ControlMode("reactive", None)
    controller._hold_until = 10**6

    cmd = controller.command(minute=0, t_air_c=5.0, t_out_c=10.0)
    assert cmd > 0.0, "cabin is well below setpoint but command is not heating"


def test_reactive_command_scales_with_distance_from_setpoint(model):
    controller = StochasticController(model, np.random.default_rng(0))
    controller._mode = ControlMode("reactive", None)
    controller._hold_until = 10**6
    controller.gain_divisor = 20.0  # deliberately wide, so neither case saturates to +/-1

    small_error = controller.command(minute=0, t_air_c=27.0, t_out_c=25.0)
    large_error = controller.command(minute=0, t_air_c=35.0, t_out_c=25.0)
    assert abs(large_error) > abs(small_error)
    assert abs(large_error) < controller.model.cfg["hvac"]["cooling_capacity_w"], (
        "test setup invalid: command saturated, so this isn't testing scaling"
    )


def test_reactive_command_is_zero_at_setpoint(model):
    controller = StochasticController(model, np.random.default_rng(0))
    controller._mode = ControlMode("reactive", None)
    controller._hold_until = 10**6

    sp = controller.setpoint(30.0)
    assert controller.command(minute=0, t_air_c=sp, t_out_c=30.0) == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------- generated-data integrity


def test_no_missing_or_infinite_values(small_dataset):
    numeric = small_dataset.select_dtypes(include=[np.number])
    assert small_dataset.isna().sum().sum() == 0
    assert not np.isinf(numeric.to_numpy()).any()


def test_temperature_stays_physically_plausible(small_dataset):
    """Generous bounds -- this is a blow-up guard, not a tight physics check."""
    assert small_dataset["t_air_c"].between(-10, 60).all()
    assert small_dataset["t_mass_c"].between(-10, 60).all()


def test_actuator_limits_are_never_exceeded(small_dataset, model):
    """Loose bound: the legitimate limit is whichever of start/end-of-row
    T_air is looser, since each 60s row is 6 substeps and T_air moves within
    it. A naive end-of-row-only check produces false positives -- this is the
    same subtlety M2's test_delivered_power_is_clamped_as_the_cabin_cools
    needed. See docs/PARAMETERS.md.
    """
    df = small_dataset.sort_values(["scenario_id", "minute"]).reset_index(drop=True)
    t_air_start = df.groupby("scenario_id")["t_air_c"].shift(1).fillna(df["t_air_c"])

    cool_limit = np.minimum(
        df["t_air_c"].apply(model.deliverable_cooling_w),
        t_air_start.apply(model.deliverable_cooling_w),
    )
    heat_limit = np.maximum(
        df["t_air_c"].apply(model.deliverable_heating_w),
        t_air_start.apply(model.deliverable_heating_w),
    )

    tol = 5.0  # W, substep integration slack
    cool_violation = (df["q_hvac_actual_w"] < -1.0) & (df["q_hvac_actual_w"] < cool_limit - tol)
    heat_violation = (df["q_hvac_actual_w"] > 1.0) & (df["q_hvac_actual_w"] > heat_limit + tol)
    assert cool_violation.sum() == 0, f"{cool_violation.sum()} cooling-limit violations"
    assert heat_violation.sum() == 0, f"{heat_violation.sum()} heating-limit violations"


def test_command_before_actuation_never_exceeds_rated_capacity(small_dataset, cfg):
    """The commanded fraction is clipped to [-1, 1] before scaling by capacity."""
    hv = cfg["hvac"]
    assert small_dataset["q_hvac_cmd_w"].max() <= hv["heating_capacity_w"] + 1e-6
    assert small_dataset["q_hvac_cmd_w"].min() >= -hv["cooling_capacity_w"] - 1e-6


def test_cooling_still_dominates_command_direction(small_dataset):
    """Egypt is cooling-dominated (M1). A majority-heating dataset would be
    the exact symptom of the sign bug this file guards against.
    """
    cooling_energy = small_dataset.loc[small_dataset["q_hvac_actual_w"] < 0, "q_hvac_actual_w"].abs().sum()
    heating_energy = small_dataset.loc[small_dataset["q_hvac_actual_w"] > 0, "q_hvac_actual_w"].sum()
    assert cooling_energy > heating_energy


# ------------------------------------------------------------- scenario shape


def test_each_scenario_has_one_city_and_service(small_dataset):
    """Dimension columns must be constant within a scenario -- protects
    features.py's per-scenario grouping from a future refactor that mixes them.
    """
    dims = ["city", "date", "depart_hour", "direction", "pattern", "load_factor"]
    nunique = small_dataset.groupby("scenario_id")[dims].nunique()
    assert (nunique == 1).all().all()


def test_minutes_are_contiguous_within_each_scenario(small_dataset):
    """0..N-1, strictly increasing -- what the lag/shift features in M4 rely on."""
    for _, group in small_dataset.groupby("scenario_id"):
        minutes = group.sort_values("minute")["minute"].to_numpy()
        assert (minutes == np.arange(len(minutes))).all()


def test_departure_hour_spans_most_of_the_operating_window(small_dataset):
    assert small_dataset["depart_hour"].min() < 8.0
    assert small_dataset["depart_hour"].max() > 18.0


def test_only_cached_cities_appear(small_dataset):
    assert set(small_dataset["city"].unique()) <= set(CITIES)


def test_winter_dates_never_appear(small_dataset, cfg):
    """Documented scope decision (docs/PARAMETERS.md): summer only."""
    from src.config import DATA_DIR

    for city in small_dataset["city"].unique():
        summer = pd.read_csv(DATA_DIR / f"weather_{city}_summer.csv", parse_dates=["timestamp"])
        valid_dates = set(summer["timestamp"].dt.date)
        used_dates = set(small_dataset.loc[small_dataset["city"] == city, "date"])
        assert used_dates <= valid_dates


# --------------------------------------------------------------- reproducibility


def test_same_seed_reproduces_the_same_dataset(cfg, require_weather):
    a = generate_dataset(cfg, target_rows=1500, seed=42)
    b = generate_dataset(cfg, target_rows=1500, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_generally_differ(cfg, require_weather):
    a = generate_dataset(cfg, target_rows=1500, seed=1)
    b = generate_dataset(cfg, target_rows=1500, seed=2)
    assert not a["t_air_c"].equals(b["t_air_c"])


# ------------------------------------------------------------- single scenario


def test_generate_scenario_matches_the_occupancy_profile_length(cfg, require_weather):
    weather = pd.read_csv(DATA_DIR / "weather_cairo_summer.csv", parse_dates=["timestamp"])
    service = Service(direction="down", pattern="semi_express", load_factor=1.0)
    profile = simulate(service, cfg)
    df = generate_scenario(0, "cairo", weather, weather["timestamp"].iloc[8].date(),
                            8.0, service, cfg, np.random.default_rng(0))
    assert len(df) == len(profile)
