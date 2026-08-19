"""Fetch real historical weather from the Open-Meteo Archive API.

Free, no API key. Cached to data/ as CSV; re-running skips what already exists.
"""

import math
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_DIR, load_config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
]

COLUMN_NAMES = {
    "time": "timestamp",
    "temperature_2m": "t_out_c",
    "relative_humidity_2m": "rh_out_pct",
    "shortwave_radiation": "ghi_w_m2",
    "wind_speed_10m": "wind_m_s",
}


def fetch(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "auto",
        },
        timeout=60,
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"]).rename(columns=COLUMN_NAMES)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def cache_path(city: str, season: str) -> Path:
    return DATA_DIR / f"weather_{city}_{season}.csv"


def get(city: str, season: str, location: dict, refresh: bool = False) -> pd.DataFrame:
    path = cache_path(city, season)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["timestamp"])

    window = location["seasons"][season]
    df = fetch(location["latitude"], location["longitude"], window["start"], window["end"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def to_minutes(df: pd.DataFrame, start: pd.Timestamp, minutes: int) -> pd.DataFrame:
    """Resample hourly weather onto a 1-minute grid by linear interpolation.

    The archive is hourly but the simulation steps every minute. Held constant
    within the hour, outdoor temperature becomes a staircase: on 15 Jul 2024 in
    Cairo the largest hour-to-hour step is 2.40 K, which lands on the cabin as a
    2.5 kW instantaneous load jump — a quarter of a station event, and entirely
    an artifact of the data's resolution.

    That matters beyond realism: a model trained on stepped weather can learn
    the step timing as if it were signal.

    GHI is interpolated the same way. Hourly irradiance is really an average
    over the hour, so interpolation is an approximation either way, but a smooth
    ramp is closer to the truth than a staircase.
    """
    if minutes <= 0:
        raise ValueError("minutes must be positive")

    series = df.set_index("timestamp").sort_index()
    grid = pd.date_range(start=start, periods=minutes, freq="1min")

    # Union the two indices so interpolation has the surrounding hourly anchors,
    # then keep only the minute grid.
    union = series.index.union(grid)
    out = (
        series.reindex(union)
        .interpolate(method="time", limit_direction="both")
        .reindex(grid)
    )
    out.index.name = "timestamp"
    return out.reset_index()


def humidity_ratio(t_c: float, rh_pct: float, pressure_kpa: float = 101.325) -> float:
    """Humidity ratio [g water / kg dry air], via the Magnus saturation formula.

    Originally added only to justify NOT modelling a humidity state (latent
    load from ventilation only exists where the outdoor humidity ratio
    EXCEEDS the indoor target -- below that, fresh air dehumidifies the
    cabin, true most of the time in Egypt's desert climate). Now also the
    building block CabinModel's simplified moisture balance uses directly
    (see ROADMAP.md's M9) -- the same function, no duplicated formula.
    """
    p_sat = 0.61094 * math.exp(17.625 * t_c / (t_c + 243.04))
    p_vap = max(0.0, min(rh_pct, 100.0)) / 100.0 * p_sat
    return 0.622 * p_vap / (pressure_kpa - p_vap) * 1000.0


def relative_humidity_from_ratio(t_c: float, w_g_kg: float, pressure_kpa: float = 101.325) -> float:
    """Inverse of humidity_ratio(): RH% from a humidity ratio at a given temperature.

    Exact algebraic inverse of the same Magnus-formula relationship above,
    not a separate approximation -- the two must round-trip. Used to report
    CabinModel's simulated cabin humidity in the unit a human (or an LCD
    display) actually reads, since w_g_kg alone isn't intuitive.
    """
    p_sat = 0.61094 * math.exp(17.625 * t_c / (t_c + 243.04))
    w = max(0.0, w_g_kg) / 1000.0
    p_vap = w * pressure_kpa / (0.622 + w)
    return 100.0 * p_vap / p_sat


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day aggregates, used to choose analysis days by percentile."""
    out = df.copy()
    out["date"] = out["timestamp"].dt.date
    return out.groupby("date").agg(
        t_max=("t_out_c", "max"),
        t_mean=("t_out_c", "mean"),
        ghi_max=("ghi_w_m2", "max"),
        rh_mean=("rh_out_pct", "mean"),
    )


def day_at_percentile(df: pd.DataFrame, percentile: float, by: str = "t_max"):
    """Pick a day by percentile of a daily statistic, rather than arbitrarily.

    Exists because every early analysis in this project used 15 July, which
    turned out to sit at the 18th-20th percentile of daily maximum temperature —
    so every reported figure came from an unusually mild day. Selecting by
    percentile makes the choice explicit and reproducible.

    Percentile is in [0, 100].
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ranked = daily_summary(df).sort_values(by)
    index = int(round((len(ranked) - 1) * percentile / 100.0))
    return ranked.index[index]


def download_all(refresh: bool = False) -> None:
    cfg = load_config()
    for location in cfg["locations"]:
        city = location["name"]
        for season in location["seasons"]:
            df = get(city, season, location, refresh=refresh)
            nan_count = int(df[list(COLUMN_NAMES.values())[1:]].isna().sum().sum())
            print(
                f"{city:8s} {season:6s}  {len(df):5d} rows  "
                f"{df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}  "
                f"T_out {df['t_out_c'].min():5.1f} to {df['t_out_c'].max():5.1f} C  "
                f"GHI max {df['ghi_w_m2'].max():4.0f} W/m2  "
                f"NaNs {nan_count}"
            )


if __name__ == "__main__":
    download_all()
