"""Fetch real historical weather from the Open-Meteo Archive API.

Free, no API key. Cached to data/ as CSV; re-running skips what already exists.
"""

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
