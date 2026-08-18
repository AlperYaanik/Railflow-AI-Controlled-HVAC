"""Tests for feature engineering, focused on leakage.

The risk in this file is not a crash, it's a silent optimistic bias: a lag or
a target that accidentally reaches across a scenario boundary produces a
dataset that trains and evaluates fine but doesn't mean what it claims to.
Every test here is either a leakage check or a correctness check tying a
feature back to the raw column it's derived from.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LAG_MINUTES,
    TARGET_COLUMN,
    TRAINED_EWMA_HALFLIFE_MIN,
    TREND_LAG_MIN,
    LiveFeatureBuilder,
    add_features,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _two_scenarios(n_a=80, n_b=80, seed=0) -> pd.DataFrame:
    """Two independent synthetic scenarios, deliberately with DIFFERENT
    trajectories, so any cross-boundary leak would be detectable rather than
    accidentally matching by coincidence.
    """
    rng = np.random.default_rng(seed)

    def make(scenario_id, n, t_air_base, q_base):
        minute = np.arange(n)
        return pd.DataFrame({
            "scenario_id": scenario_id, "city": "cairo", "date": pd.Timestamp("2024-07-01").date(),
            "depart_hour": 8.0, "direction": "down", "pattern": "semi_express", "load_factor": 1.0,
            "minute": minute,
            "t_air_c": t_air_base + 0.05 * minute + rng.normal(0, 0.1, n),
            "t_mass_c": t_air_base + 0.03 * minute,
            "t_out_c": 35.0 + 0.02 * minute,
            "ghi_w_m2": 500.0,
            "n_pax": 40,
            "door_open": False,
            "time_to_next_station_min": np.maximum(30 - minute, 0),
            "expected_boarding": 20,
            "setpoint_c": 26.0,
            "q_hvac_cmd_w": q_base,
            "q_hvac_actual_w": q_base + rng.normal(0, 5, n),
            "electrical_w": abs(q_base) / 2.5,
            "cop": 2.5,
        })

    a = make(0, n_a, t_air_base=30.0, q_base=-10000.0)
    b = make(1, n_b, t_air_base=40.0, q_base=20000.0)  # deliberately very different
    return pd.concat([a, b], ignore_index=True)


def test_no_lag_crosses_a_scenario_boundary(cfg):
    """The first LAG_MINUTES rows of scenario B must never see scenario A's tail.

    Scenario A ends near q_hvac_actual_w ~ -10000; scenario B starts at
    +20000 and only descends slowly toward its own history. If a lag leaked
    across the boundary, early rows of B would show a lag value near -10000,
    which cannot occur from B's own trajectory.
    """
    raw = _two_scenarios()
    feat = add_features(raw, cfg, horizon_min=5)

    b_start = feat[feat["scenario_id"] == 1].sort_values("minute")
    assert len(b_start) > 0
    for lag in LAG_MINUTES:
        col = f"q_actual_lag_{lag}"
        assert (b_start[col] > 0).all(), (
            f"{col} in scenario B is negative near its start — looks like it "
            "borrowed scenario A's tail"
        )


def test_no_target_crosses_a_scenario_boundary(cfg):
    """The last rows of scenario A must never get scenario B's t_air as target.

    Scenario A's t_air stays near 30-34 C; scenario B's is near 40-44 C. A
    leaked target would show up as a value far outside A's own range.
    """
    raw = _two_scenarios()
    horizon = 5
    feat = add_features(raw, cfg, horizon_min=horizon)

    a_rows = feat[feat["scenario_id"] == 0]
    assert len(a_rows) > 0
    assert a_rows[TARGET_COLUMN].max() < 36.0, (
        "scenario A's target reaches into scenario B's temperature range — "
        "the horizon shift crossed the scenario boundary"
    )


def test_target_equals_t_air_shifted_by_horizon_within_scenario(cfg):
    raw = _two_scenarios()
    horizon = 7
    feat = add_features(raw, cfg, horizon_min=horizon)

    raw_by_scenario = {sid: g.sort_values("minute").set_index("minute")["t_air_c"]
                        for sid, g in raw.groupby("scenario_id")}

    for _, row in feat.sample(min(30, len(feat)), random_state=0).iterrows():
        true_future = raw_by_scenario[row["scenario_id"]].loc[row["minute"] + horizon]
        assert row[TARGET_COLUMN] == pytest.approx(true_future)


def test_weather_lookahead_matches_the_true_future_weather(cfg):
    raw = _two_scenarios()
    horizon = 6
    feat = add_features(raw, cfg, horizon_min=horizon)

    raw_by_scenario = {sid: g.sort_values("minute").set_index("minute")["t_out_c"]
                        for sid, g in raw.groupby("scenario_id")}
    for _, row in feat.sample(min(20, len(feat)), random_state=1).iterrows():
        true_future = raw_by_scenario[row["scenario_id"]].loc[row["minute"] + horizon]
        assert row["t_out_fcst_h"] == pytest.approx(true_future)


def test_no_future_command_is_ever_a_feature(cfg):
    """Every lag column must correlate with the PAST, never the future.

    Direct check: q_actual_lag_k at row (scenario, minute=m) must equal
    q_hvac_actual_w at minute=(m-k), not (m+k).
    """
    raw = _two_scenarios()
    feat = add_features(raw, cfg, horizon_min=5)

    raw_by_scenario = {sid: g.sort_values("minute").set_index("minute")["q_hvac_actual_w"]
                        for sid, g in raw.groupby("scenario_id")}

    lag = LAG_MINUTES[-1]  # the largest lag is the strictest check
    col = f"q_actual_lag_{lag}"
    for _, row in feat.sample(min(30, len(feat)), random_state=2).iterrows():
        past_value = raw_by_scenario[row["scenario_id"]].loc[row["minute"] - lag]
        assert row[col] == pytest.approx(past_value)


def test_dropped_rows_are_exactly_the_scenario_edges(cfg):
    raw = _two_scenarios(n_a=80, n_b=80)
    horizon = 10
    feat = add_features(raw, cfg, horizon_min=horizon)
    max_lag = max(LAG_MINUTES + [TREND_LAG_MIN])

    for sid, group in feat.groupby("scenario_id"):
        assert group["minute"].min() >= max_lag
        raw_max_minute = raw[raw["scenario_id"] == sid]["minute"].max()
        assert group["minute"].max() <= raw_max_minute - horizon


def test_history_drop_uses_position_in_scenario_not_the_raw_minute_value(cfg):
    """Regression guard for a real fix: the drop condition originally checked
    `minute >= max_lag`, which happens to equal "position within scenario"
    for real generated data (minute is always 0-indexed from departure) but
    silently assumes it. A scenario whose 'minute' column starts somewhere
    other than 0 would have been handled wrong -- rows would survive the
    filter without actually having max_lag rows of real history behind them.
    """
    n = 40
    offset = 1000  # minute column does NOT start at 0
    raw = pd.DataFrame({
        "scenario_id": 7, "city": "cairo", "date": pd.Timestamp("2024-07-01").date(),
        "depart_hour": 8.0, "direction": "down", "pattern": "semi_express", "load_factor": 1.0,
        "minute": np.arange(n) + offset,
        "t_air_c": 28.0 + 0.1 * np.arange(n), "t_mass_c": 28.0, "t_out_c": 35.0,
        "ghi_w_m2": 400.0, "n_pax": 30, "door_open": False,
        "time_to_next_station_min": 20, "expected_boarding": 5, "setpoint_c": 26.0,
        "q_hvac_cmd_w": -10000.0, "q_hvac_actual_w": -10000.0, "electrical_w": 4000.0, "cop": 2.5,
    })
    feat = add_features(raw, cfg, horizon_min=5)

    max_lag = max(LAG_MINUTES + [TREND_LAG_MIN])
    # Correct behaviour: exactly the first max_lag and last horizon_min rows
    # of the 40-row scenario are dropped, by POSITION -- regardless of the
    # (irrelevant) absolute 'minute' values.
    assert len(feat) == n - max_lag - 5
    assert feat["minute"].min() == offset + max_lag
    assert feat[list(FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_error_c_matches_t_air_minus_setpoint(cfg):
    """np.allclose, not `series == pytest.approx(series)` -- the latter does
    not reliably do element-wise comparison against a pandas Series."""
    raw = _two_scenarios()
    feat = add_features(raw, cfg, horizon_min=5)
    assert np.allclose(feat["error_c"], feat["t_air_c"] - feat["setpoint_c"])


def test_hour_of_day_cyclical_features_are_bounded_and_periodic(cfg):
    raw = _two_scenarios()
    feat = add_features(raw, cfg, horizon_min=5)
    assert feat["hour_sin"].between(-1.0, 1.0).all()
    assert feat["hour_cos"].between(-1.0, 1.0).all()

    # Ties the feature back to its documented formula (depart_hour + minute/60,
    # mod 24) rather than assuming any particular surviving row lands exactly
    # on a cycle boundary -- minute=20 at depart_hour=0 is 20 minutes past
    # midnight, not midnight itself, so hour_sin there is NOT 0.
    n = 40  # comfortably more than max_lag(20) + horizon(5), or every row gets dropped
    depart_hour = 0.0
    at_zero = pd.DataFrame({
        "scenario_id": [99] * n, "city": "cairo", "date": pd.Timestamp("2024-07-01").date(),
        "depart_hour": depart_hour, "direction": "down", "pattern": "semi_express", "load_factor": 1.0,
        "minute": np.arange(n),
        "t_air_c": 26.0, "t_mass_c": 26.0, "t_out_c": 26.0, "ghi_w_m2": 0.0,
        "n_pax": 0, "door_open": False, "time_to_next_station_min": 30,
        "expected_boarding": 0, "setpoint_c": 26.0,
        "q_hvac_cmd_w": 0.0, "q_hvac_actual_w": 0.0, "electrical_w": 0.0, "cop": 2.5,
    })
    f2 = add_features(at_zero, cfg, horizon_min=5)
    check_minute = max(LAG_MINUTES + [TREND_LAG_MIN])
    row = f2[f2["minute"] == check_minute].iloc[0]

    expected_hour = (depart_hour + check_minute / 60.0) % 24.0
    assert row["hour_sin"] == pytest.approx(np.sin(2 * np.pi * expected_hour / 24.0))
    assert row["hour_cos"] == pytest.approx(np.cos(2 * np.pi * expected_hour / 24.0))

    # A row that IS exactly on a day boundary must still give the (0, 1) anchor.
    midnight_minute = 24 * 60 - depart_hour * 60  # wraps hour_of_day back to 0
    at_midnight = at_zero.copy()
    at_midnight["minute"] = np.arange(n) + int(midnight_minute) - n // 2
    f3 = add_features(at_midnight, cfg, horizon_min=5)
    wrap_row = f3[np.isclose(f3["minute"] % (24 * 60), midnight_minute % (24 * 60))]
    assert len(wrap_row) > 0
    assert wrap_row.iloc[0]["hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert wrap_row.iloc[0]["hour_cos"] == pytest.approx(1.0, abs=1e-9)


def test_ewma_lags_behind_a_step_change_rather_than_jumping(cfg):
    """After a step change in delivered power, EWMA must sit strictly between
    the old and new level -- confirms it is actually smoothing, not just
    copying the current value.
    """
    n = 60
    step_at = 30
    q = np.where(np.arange(n) < step_at, -5000.0, 25000.0)
    raw = pd.DataFrame({
        "scenario_id": 0, "city": "cairo", "date": pd.Timestamp("2024-07-01").date(),
        "depart_hour": 8.0, "direction": "down", "pattern": "semi_express", "load_factor": 1.0,
        "minute": np.arange(n),
        "t_air_c": 28.0, "t_mass_c": 28.0, "t_out_c": 35.0, "ghi_w_m2": 500.0,
        "n_pax": 40, "door_open": False, "time_to_next_station_min": 20,
        "expected_boarding": 10, "setpoint_c": 26.0,
        "q_hvac_cmd_w": q, "q_hvac_actual_w": q, "electrical_w": np.abs(q) / 2.5, "cop": 2.5,
    })
    feat = add_features(raw, cfg, horizon_min=5)
    just_after = feat[feat["minute"] == step_at + 2].iloc[0]
    assert -5000.0 < just_after["q_actual_ewma"] < 25000.0


def test_categorical_columns_are_typed_as_category(cfg):
    raw = _two_scenarios()
    feat = add_features(raw, cfg, horizon_min=5)
    for col in CATEGORICAL_COLUMNS:
        assert feat[col].dtype.name == "category"


def test_feature_columns_all_exist_and_carry_no_nan(cfg):
    raw = _two_scenarios(n_a=120, n_b=120)
    feat = add_features(raw, cfg, horizon_min=10)
    for col in FEATURE_COLUMNS:
        assert col in feat.columns, f"{col} missing from output"
    assert feat[list(FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_horizon_parameter_actually_changes_the_target(cfg):
    raw = _two_scenarios(n_a=100, n_b=100)
    short = add_features(raw, cfg, horizon_min=5)
    long = add_features(raw, cfg, horizon_min=25)
    assert short.attrs["horizon_min"] == 5
    assert long.attrs["horizon_min"] == 25
    assert len(long) < len(short), "a longer horizon must drop more rows at the tail"


def test_default_horizon_comes_from_config(cfg):
    raw = _two_scenarios()
    feat = add_features(raw, cfg)
    assert feat.attrs["horizon_min"] == cfg["simulation"]["control_horizon_min"]


# --------------------------------------------------- LiveFeatureBuilder (M5 prep)


def _simulate_with_reactive_thermostat(cfg, require_weather, city="cairo", depart_hour=9.0):
    """Runs the REAL M1-M3 stack (weather, occupancy, cabin model) under a
    plain reactive thermostat, and returns the raw per-minute trajectory --
    the same shape data_generator.py produces, so add_features() can be run
    on it directly. Used to compare against LiveFeatureBuilder on genuine
    physics, not just synthetic fixtures.
    """
    from src.cabin_model import CabinInputs, CabinModel
    from src.config import DATA_DIR
    from src.occupancy import Service, simulate
    from src.weather import to_minutes

    cfg_local = cfg
    model = CabinModel(cfg_local)
    wx = pd.read_csv(DATA_DIR / f"weather_{city}_summer.csv", parse_dates=["timestamp"])
    service = Service(direction="down", pattern="semi_express", load_factor=1.0)
    profile = simulate(service, cfg_local)
    horizon = cfg_local["simulation"]["control_horizon_min"]
    start = pd.Timestamp("2024-07-10") + pd.to_timedelta(int(depart_hour * 60), unit="min")
    w = to_minutes(wx, start, len(profile) + horizon)

    state = model.initial_state(model.setpoint(float(w["t_out_c"].iloc[0])))
    rows = []
    for t in range(len(profile)):
        t_out = float(w["t_out_c"].iloc[t])
        setpoint = model.setpoint(t_out)
        frac = min(1.0, max(0.0, (state.t_air_c - setpoint) / 1.5))
        r = model.step(state, CabinInputs(
            t_out_c=t_out, ghi_w_m2=float(w["ghi_w_m2"].iloc[t]),
            n_pax=profile.n_pax[t], door_open=profile.door_open[t],
            q_hvac_cmd_w=-model.cooling_capacity_w * frac,
        ), dt_s=60.0)
        rows.append({
            "scenario_id": 0, "city": city, "date": start.date(), "depart_hour": depart_hour,
            "direction": service.direction, "pattern": service.pattern,
            "load_factor": service.load_factor, "minute": t,
            "t_air_c": r.t_air_c, "t_mass_c": r.t_mass_c, "t_out_c": t_out,
            "ghi_w_m2": float(w["ghi_w_m2"].iloc[t]), "n_pax": profile.n_pax[t],
            "door_open": profile.door_open[t],
            "time_to_next_station_min": profile.time_to_next_station_min[t],
            "expected_boarding": profile.expected_boarding[t], "setpoint_c": setpoint,
            "q_hvac_cmd_w": -model.cooling_capacity_w * frac, "q_hvac_actual_w": r.q_hvac_actual_w,
            "electrical_w": r.electrical_w, "cop": r.cop,
        })
    return pd.DataFrame(rows), w, service


def test_live_feature_builder_matches_batch_add_features_exactly(cfg, require_weather):
    """The test that makes LiveFeatureBuilder trustworthy for M5.

    Runs the real simulator once, then builds features two ways on the exact
    same trajectory: in batch via add_features(), and incrementally via
    LiveFeatureBuilder fed the same per-minute values a controller would
    actually have (including the +H weather lookahead, pulled from the same
    interpolated series). They must match to the last bit, not just
    approximately -- a controller silently using a slightly-off feature
    vector would not crash, it would just make worse decisions.
    """
    raw, w, service = _simulate_with_reactive_thermostat(cfg, require_weather)
    horizon = cfg["simulation"]["control_horizon_min"]

    batch = add_features(raw, cfg)

    builder = LiveFeatureBuilder(
        cfg, city="cairo", direction=service.direction, pattern=service.pattern,
        load_factor=service.load_factor, depart_hour=9.0,
    )
    live_rows = {}
    for t in range(len(raw)):
        row = raw.iloc[t]
        out = builder.step(
            t_air_c=row["t_air_c"], t_mass_c=row["t_mass_c"], t_out_c=row["t_out_c"],
            ghi_w_m2=row["ghi_w_m2"], n_pax=row["n_pax"], door_open=row["door_open"],
            setpoint_c=row["setpoint_c"], q_hvac_actual_w=row["q_hvac_actual_w"],
            time_to_next_station_min=row["time_to_next_station_min"],
            expected_boarding=row["expected_boarding"],
            t_out_fcst_h=float(w["t_out_c"].iloc[t + horizon]),
            ghi_fcst_h=float(w["ghi_w_m2"].iloc[t + horizon]),
        )
        if out is not None:
            live_rows[t] = out

    # add_features() also drops the LAST `horizon` rows of the batch, because
    # training needs a ground-truth target and there is none past the end of
    # the recorded journey. LiveFeatureBuilder has no such constraint -- a
    # live controller wants a forecast at every minute it's asked for one,
    # whether or not a "future" exists yet to grade it against. So it legally
    # produces MORE rows than the batch (146 vs 116 here); the comparison is
    # over the minutes both sides actually define, not equal row counts.
    max_lag = max(LAG_MINUTES + [TREND_LAG_MIN])
    assert set(live_rows.keys()) == set(range(max_lag, len(raw)))
    assert set(batch["minute"]) == set(range(max_lag, len(raw) - horizon))
    assert set(batch["minute"]) < set(live_rows.keys())

    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    for _, batch_row in batch.iterrows():
        live_row = live_rows[int(batch_row["minute"])].iloc[0]
        for col in numeric_cols:
            assert live_row[col] == pytest.approx(batch_row[col], abs=1e-9), (
                f"minute {batch_row['minute']}, column '{col}': "
                f"live={live_row[col]!r} vs batch={batch_row[col]!r}"
            )
        for col in CATEGORICAL_COLUMNS:
            assert str(live_row[col]) == str(batch_row[col])


def test_live_feature_builder_returns_none_during_warmup(cfg):
    max_lag = max(LAG_MINUTES + [TREND_LAG_MIN])
    builder = LiveFeatureBuilder(cfg, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    results = []
    for t in range(max_lag + 5):
        out = builder.step(
            t_air_c=26.0, t_mass_c=26.0, t_out_c=35.0, ghi_w_m2=400.0,
            n_pax=30, door_open=False, setpoint_c=26.0, q_hvac_actual_w=-5000.0,
            time_to_next_station_min=20, expected_boarding=5,
            t_out_fcst_h=36.0, ghi_fcst_h=420.0,
        )
        results.append(out is not None)
    assert results == [False] * max_lag + [True] * 5


def test_live_feature_builder_output_is_ready_for_the_model(cfg):
    """Output must already be in the exact column order/dtypes the saved
    booster expects -- no further massaging needed at the call site."""
    max_lag = max(LAG_MINUTES + [TREND_LAG_MIN])
    builder = LiveFeatureBuilder(cfg, city="aswan", direction="up",
                                  pattern="express", load_factor=0.7, depart_hour=12.0)
    out = None
    for t in range(max_lag + 1):
        out = builder.step(
            t_air_c=28.0, t_mass_c=27.0, t_out_c=38.0, ghi_w_m2=600.0,
            n_pax=20, door_open=t == max_lag, setpoint_c=26.0, q_hvac_actual_w=-8000.0,
            time_to_next_station_min=10, expected_boarding=8,
            t_out_fcst_h=39.0, ghi_fcst_h=610.0,
        )
    assert out is not None
    assert list(out.columns) == list(FEATURE_COLUMNS)
    assert len(out) == 1
    for col in CATEGORICAL_COLUMNS:
        assert out[col].dtype.name == "category"
    assert out.isna().sum().sum() == 0


def test_live_feature_builder_ewma_halflife_ignores_a_swept_tau_act(cfg):
    """The exact property the M6-readiness audit added: LiveFeatureBuilder's
    EWMA halflife must NOT follow cfg["hvac"]["tau_act_min"] when a caller
    (M6's tau_act sweep, src/sweep_tau_act.py) overrides it for CabinModel's
    actuator. Without this, sweeping the actuator's physical lag would also
    silently reshape the feature the frozen forecaster was trained on --
    conflating genuine lag sensitivity with train/serve feature skew.
    """
    import copy

    swept = copy.deepcopy(cfg)
    swept["hvac"]["tau_act_min"] = 999.0  # deliberately far from any real value
    builder = LiveFeatureBuilder(swept, city="cairo", direction="down",
                                  pattern="semi_express", load_factor=1.0, depart_hour=8.0)
    assert builder.halflife == max(TRAINED_EWMA_HALFLIFE_MIN, 0.1)
    assert builder.halflife != 999.0


def test_add_features_ewma_halflife_ignores_a_swept_tau_act(cfg):
    """Same property, batch side -- add_features() must match
    LiveFeatureBuilder's default rather than the two silently diverging."""
    import copy

    raw = _two_scenarios()
    swept = copy.deepcopy(cfg)
    swept["hvac"]["tau_act_min"] = 999.0
    default_halflife_feat = add_features(raw, cfg, horizon_min=5)
    swept_cfg_feat = add_features(raw, swept, horizon_min=5)
    pd.testing.assert_series_equal(
        default_halflife_feat["q_actual_ewma"], swept_cfg_feat["q_actual_ewma"]
    )
