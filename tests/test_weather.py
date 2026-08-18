"""Tests for weather loading and the hourly-to-minute interpolation."""

import pandas as pd
import pytest

from src.config import DATA_DIR
from src.weather import COLUMN_NAMES, to_minutes


@pytest.fixture(scope="module")
def hourly():
    path = DATA_DIR / "weather_cairo_summer.csv"
    if not path.exists():
        pytest.skip("weather cache missing. Run:  python -m src.weather")
    return pd.read_csv(path, parse_dates=["timestamp"])


def test_cached_weather_has_the_expected_columns(hourly):
    for column in COLUMN_NAMES.values():
        assert column in hourly.columns


def test_cached_weather_is_hourly_and_complete(hourly):
    gaps = hourly["timestamp"].diff().dropna().unique()
    assert len(gaps) == 1, "weather series has gaps or mixed spacing"
    assert hourly.isna().sum().sum() == 0


def test_interpolation_produces_one_row_per_minute(hourly):
    start = hourly["timestamp"].iloc[8]
    out = to_minutes(hourly, start, 180)
    assert len(out) == 180
    spacing = out["timestamp"].diff().dropna().unique()
    assert len(spacing) == 1


def test_interpolation_reproduces_the_hourly_anchors(hourly):
    """On the hour, the interpolated value must equal the measured one."""
    start = hourly["timestamp"].iloc[8]
    out = to_minutes(hourly, start, 180).set_index("timestamp")

    for offset in (0, 60, 120):
        stamp = start + pd.Timedelta(f"{offset}min")
        measured = hourly.set_index("timestamp").loc[stamp, "t_out_c"]
        assert out.loc[stamp, "t_out_c"] == pytest.approx(measured, abs=1e-6)


def test_interpolation_removes_the_staircase(hourly):
    """The point of the exercise: no step changes between minutes.

    Held constant within the hour, T_out jumps by up to 2.4 K at hour
    boundaries, which lands on the cabin as a ~2.5 kW load step that is purely
    an artifact of data resolution.
    """
    start = hourly["timestamp"].iloc[6]
    out = to_minutes(hourly, start, 12 * 60)

    stepped = hourly.set_index("timestamp")["t_out_c"]
    largest_hourly_step = stepped.diff().abs().max()
    largest_minute_step = out["t_out_c"].diff().abs().max()

    assert largest_minute_step < largest_hourly_step / 30.0, (
        f"minute-to-minute step {largest_minute_step:.3f} K is not much smaller "
        f"than the hourly step {largest_hourly_step:.3f} K"
    )


def test_interpolated_values_stay_within_the_measured_range(hourly):
    """Linear interpolation must not overshoot the data."""
    start = hourly["timestamp"].iloc[0]
    out = to_minutes(hourly, start, 20 * 60)

    for column in ("t_out_c", "ghi_w_m2", "rh_out_pct", "wind_m_s"):
        assert out[column].min() >= hourly[column].min() - 1e-6
        assert out[column].max() <= hourly[column].max() + 1e-6


def test_irradiance_never_goes_negative(hourly):
    start = hourly["timestamp"].iloc[0]
    out = to_minutes(hourly, start, 20 * 60)
    assert out["ghi_w_m2"].min() >= 0.0


def test_interpolation_is_monotonic_between_two_anchors(hourly):
    """Between consecutive hourly readings the ramp must be straight."""
    series = hourly.set_index("timestamp")["t_out_c"]
    # Find an hour where temperature is clearly rising.
    diffs = series.diff().dropna()
    stamp = diffs.idxmax()
    start = stamp - pd.Timedelta("1h")

    out = to_minutes(hourly, start, 61)["t_out_c"].to_numpy()
    steps = out[1:] - out[:-1]
    assert (steps > 0).all()
    assert steps.max() - steps.min() < 1e-6, "ramp between anchors is not linear"


def test_a_journey_length_window_can_be_requested(hourly):
    """The full Cairo-Alexandria journey must fit without running off the end."""
    start = hourly["timestamp"].iloc[8]
    out = to_minutes(hourly, start, 166)
    assert len(out) == 166
    assert out.isna().sum().sum() == 0


def test_zero_or_negative_length_is_rejected(hourly):
    with pytest.raises(ValueError):
        to_minutes(hourly, hourly["timestamp"].iloc[0], 0)
