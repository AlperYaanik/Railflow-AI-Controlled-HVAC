"""Tests for weather loading and the hourly-to-minute interpolation."""

import pandas as pd
import pytest

from src.config import DATA_DIR
from src.weather import (
    COLUMN_NAMES,
    daily_summary,
    day_at_percentile,
    humidity_ratio,
    relative_humidity_from_ratio,
    to_minutes,
)


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


# --------------------------------------------------------- day selection (#4)


def test_daily_summary_covers_every_day(hourly):
    summary = daily_summary(hourly)
    assert len(summary) == hourly["timestamp"].dt.date.nunique()
    assert (summary["t_max"] >= summary["t_mean"]).all()


def test_percentile_selection_is_ordered(hourly):
    """A higher percentile must not select a cooler day."""
    summary = daily_summary(hourly)
    previous = -999.0
    for pct in (0, 25, 50, 75, 100):
        day = day_at_percentile(hourly, pct)
        t_max = summary.loc[day, "t_max"]
        assert t_max >= previous
        previous = t_max


def test_percentile_extremes_pick_the_extreme_days(hourly):
    summary = daily_summary(hourly)
    assert summary.loc[day_at_percentile(hourly, 0), "t_max"] == summary["t_max"].min()
    assert summary.loc[day_at_percentile(hourly, 100), "t_max"] == summary["t_max"].max()


def test_percentile_out_of_range_is_rejected(hourly):
    with pytest.raises(ValueError):
        day_at_percentile(hourly, 150)


def test_the_day_used_in_early_analysis_was_unusually_mild(hourly):
    """Documents why percentile selection exists.

    Every analysis before this used 15 July 2024, which turns out to sit near
    the 20th percentile of daily maximum temperature — so every figure reported
    from it was drawn from a cooler-than-typical day.
    """
    summary = daily_summary(hourly)
    target = pd.Timestamp("2024-07-15").date()
    if target not in summary.index:
        pytest.skip("15 July not in this weather window")

    rank = (summary["t_max"] < summary.loc[target, "t_max"]).mean()
    assert rank < 0.35, (
        f"15 July sits at the {rank:.0%} percentile — the premise of this test "
        "is that it was mild, so if this changes the docs need revisiting"
    )


# ------------------------------------------------------------- humidity (#2)


def test_humidity_ratio_rises_with_temperature_and_humidity(hourly):
    assert humidity_ratio(35, 50) > humidity_ratio(25, 50)
    assert humidity_ratio(30, 80) > humidity_ratio(30, 30)


def test_humidity_ratio_is_zero_in_perfectly_dry_air():
    assert humidity_ratio(40, 0) == pytest.approx(0.0)


def test_humidity_ratio_matches_a_known_psychrometric_point():
    """25 C, 50% RH is about 9.9 g/kg on any psychrometric chart."""
    assert humidity_ratio(25.0, 50.0) == pytest.approx(9.9, abs=0.3)


@pytest.mark.parametrize("t_c,rh_pct", [
    (26.0, 55.0), (45.0, 20.0), (10.0, 90.0), (0.0, 50.0), (35.0, 0.0), (20.0, 100.0),
])
def test_relative_humidity_from_ratio_is_the_exact_inverse_of_humidity_ratio(t_c, rh_pct):
    """The algebraic inverse of the same Magnus-formula relationship, not a
    separate approximation -- must round-trip exactly (well within float
    precision), across dry/saturated/hot/cold edges, not just one typical
    point. CabinModel's simplified humidity state (M9) reports its output
    through this function, so a drift here would silently mis-report every
    humidity number the simulator produces.
    """
    w = humidity_ratio(t_c, rh_pct)
    assert relative_humidity_from_ratio(t_c, w) == pytest.approx(rh_pct, abs=1e-6)


def test_egyptian_air_is_usually_drier_than_the_cabin_target(hourly):
    """Originally justified not modelling a humidity state at all -- M9 added
    a simplified one (CabinModel.step()'s moisture balance), but this finding
    still matters there: it's WHY the ventilation exchange term in that
    balance is a net-drying effect most hours, not a humidifying one, for
    THIS climate specifically. This argument would not hold on a coastal route.
    """
    indoor = humidity_ratio(26.0, 55.0)
    outdoor = hourly.apply(
        lambda r: humidity_ratio(float(r["t_out_c"]), float(r["rh_out_pct"])), axis=1
    )
    assert (outdoor > indoor).mean() < 0.5, (
        "outdoor air is more humid than the cabin target for most hours — "
        "the sensible-only assumption needs revisiting for this location"
    )
