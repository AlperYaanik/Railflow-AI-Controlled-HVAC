"""Feature engineering for the multi-horizon-free forecaster.

Turns the raw per-minute trajectory from data_generator.py into
(features, target) rows: predict T_air at t+H from what is knowable at t.

Every lag/rolling/shift operation is done per scenario_id group. Getting this
wrong -- letting a lag or the target bleed across scenario boundaries -- would
silently corrupt the dataset without any error, so it is checked explicitly in
tests/test_features.py rather than just assumed from the groupby call looking
right.

Two categories of feature, and the line between them matters:

  KNOWN AT PREDICTION TIME  -- past command history, current state, cyclical
  time, and lookahead quantities a real controller legitimately has: the
  timetable (time to next station, expected boarding) and a weather forecast.
  Weather 30 minutes out is treated as known because it is exogenous and
  slowly varying -- forecasting it is a solved problem the project is not
  trying to solve, and the whole "anticipatory" premise depends on this
  lookahead being legitimate.

  NEVER A FEATURE  -- any future command (u(t+1..t+H)). At inference time
  those don't exist yet; the model is asked to forecast T_air GIVEN how a
  representative mix of control policies behaves over the horizon (data_generator
  deliberately mixes policies -- see its docstring), not given a hypothetical
  future command sequence.
"""

import numpy as np
import pandas as pd

from src.config import load_config

LAG_MINUTES = [1, 2, 3, 5, 10, 15, 20]
"""Command lags. Spans dead_time_min (2) through several tau_act_min (5
default) -- enough for the model to see the actuator's own lag state."""

TREND_LAG_MIN = 15
"""How far back to look for a simple recent-warming/cooling rate feature."""

FEATURE_COLUMNS = (
    ["t_air_c", "t_mass_c", "t_out_c", "ghi_w_m2", "n_pax", "door_open",
     "setpoint_c", "error_c", "t_air_trend_rate",
     "time_to_next_station_min", "expected_boarding",
     "t_out_fcst_h", "ghi_fcst_h",
     "hour_sin", "hour_cos",
     "q_hvac_actual_w", "q_actual_ewma",
     "city", "direction", "pattern", "load_factor"]
    + [f"q_actual_lag_{lag}" for lag in LAG_MINUTES]
)
"""Explicit allow-list train.py trains on. Deliberately excludes scenario_id,
date, depart_hour (folded into hour_sin/cos), minute, and every raw q_hvac_*
column at t+1..t+H -- those never exist as columns here, since every shift
in add_features() is either a LAG (positive shift, past) or the horizon
shift used only to build target_t_air_c / the two _fcst_h lookaheads."""

TARGET_COLUMN = "target_t_air_c"

CATEGORICAL_COLUMNS = ("city", "direction", "pattern")


def add_features(df: pd.DataFrame, cfg: dict | None = None, horizon_min: int | None = None) -> pd.DataFrame:
    """Adds features and the target column; drops rows without full history/future.

    Returns a new DataFrame. Input rows are unchanged except for added columns
    and the drop at the edges of each scenario.
    """
    cfg = cfg if cfg is not None else load_config()
    horizon_min = horizon_min if horizon_min is not None else cfg["simulation"]["control_horizon_min"]

    df = df.sort_values(["scenario_id", "minute"]).reset_index(drop=True)
    g = df.groupby("scenario_id", group_keys=False)

    for lag in LAG_MINUTES:
        df[f"q_actual_lag_{lag}"] = g["q_hvac_actual_w"].shift(lag)

    halflife = max(cfg["hvac"]["tau_act_min"], 0.1)
    df["q_actual_ewma"] = g["q_hvac_actual_w"].transform(
        lambda s: s.ewm(halflife=halflife).mean()
    )

    df["t_air_lag_trend"] = g["t_air_c"].shift(TREND_LAG_MIN)
    df["t_air_trend_rate"] = (df["t_air_c"] - df["t_air_lag_trend"]) / TREND_LAG_MIN

    hour_of_day = (df["depart_hour"] + df["minute"] / 60.0) % 24.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    df["error_c"] = df["t_air_c"] - df["setpoint_c"]

    # Lookahead: legitimate because weather is exogenous and known in advance,
    # not because the model gets to see the future of anything it's predicting.
    df["t_out_fcst_h"] = g["t_out_c"].shift(-horizon_min)
    df["ghi_fcst_h"] = g["ghi_w_m2"].shift(-horizon_min)

    df[TARGET_COLUMN] = g["t_air_c"].shift(-horizon_min)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    # Position WITHIN each scenario, not the raw 'minute' value -- shift()/ewm()
    # above are already position-based (correct regardless of what 'minute'
    # happens to start at), so the drop condition has to match that, not
    # silently assume 'minute' is 0-indexed from scenario start.
    max_lag = max(LAG_MINUTES + [TREND_LAG_MIN])
    position_in_scenario = g.cumcount()
    has_history = position_in_scenario >= max_lag
    has_future = df[TARGET_COLUMN].notna()
    out = df.loc[has_history & has_future].reset_index(drop=True)

    out.attrs["horizon_min"] = horizon_min
    return out


if __name__ == "__main__":
    from src.config import DATA_DIR

    raw = pd.read_parquet(DATA_DIR / "scenarios_raw.parquet")
    cfg = load_config()
    feat = add_features(raw, cfg)

    print(f"raw rows      : {len(raw)}")
    print(f"feature rows  : {len(feat)}  (dropped {len(raw) - len(feat)} at scenario edges)")
    print(f"horizon (min) : {feat.attrs['horizon_min']}")
    print(f"feature count : {len(FEATURE_COLUMNS)}")
    print(f"target range  : {feat[TARGET_COLUMN].min():.1f} - {feat[TARGET_COLUMN].max():.1f} C")
    print()
    print("NaN per feature column:")
    print(feat[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].isna().sum().to_string())
