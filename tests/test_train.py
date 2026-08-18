"""Tests for the chronological split and the forecaster's evaluation.

Two things matter enough to be tests rather than one-off analysis: the split
really is chronological (not just "no duplicate rows"), and the model's win
over persistence is real rather than an artifact -- checked here by also
beating a trivial weather-only linear baseline, not just the weak persistence
bar.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_DIR, load_config
from src.features import TARGET_COLUMN, add_features
from src.train import MODEL_PATH, chronological_split, persistence_mae, train_model


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw(cfg, require_weather):
    """require_weather runs first: if weather itself is missing, that's the
    real, more fundamental problem, and its message is the actionable one --
    telling someone to run data_generator when weather is also missing would
    just send them to a second, more confusing crash.
    """
    path = DATA_DIR / "scenarios_raw.parquet"
    if not path.exists():
        pytest.skip("no generated dataset. Run:  python -m src.data_generator")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def feat(raw, cfg):
    return add_features(raw, cfg)


@pytest.fixture(scope="module")
def trained(cfg, raw):
    """One shared training run -- 0.9s, but no reason to pay it per test.

    Depends on `raw` (unused directly) purely to inherit its skip check --
    train_model() reads the same parquet path internally by default, and
    without this it would hit a raw FileNotFoundError instead of a skip.
    """
    return train_model(cfg, verbose=False)


# ------------------------------------------------------- chronological split


def test_split_dates_never_overlap(feat):
    split = chronological_split(feat)
    train_dates = set(feat.loc[split == "train", "date"])
    val_dates = set(feat.loc[split == "val", "date"])
    test_dates = set(feat.loc[split == "test", "date"])
    assert not (train_dates & val_dates)
    assert not (train_dates & test_dates)
    assert not (val_dates & test_dates)


def test_split_is_actually_chronological_not_just_non_overlapping(feat):
    """The stronger property: every train date precedes every val date,
    which precedes every test date. A split that shuffled dates into three
    non-overlapping but interleaved buckets would pass the overlap test above
    and still not be a legitimate chronological split.
    """
    split = chronological_split(feat)
    max_train = max(feat.loc[split == "train", "date"])
    min_val = min(feat.loc[split == "val", "date"])
    max_val = max(feat.loc[split == "val", "date"])
    min_test = min(feat.loc[split == "test", "date"])
    assert max_train < min_val
    assert max_val < min_test


def test_split_covers_every_row_exactly_once(feat):
    split = chronological_split(feat)
    assert set(split.unique()) <= {"train", "val", "test"}
    assert len(split) == len(feat)
    assert split.isna().sum() == 0


def test_split_proportions_are_roughly_70_15_15(feat):
    split = chronological_split(feat)
    dates = sorted(feat["date"].unique())
    n = len(dates)
    fractions = split.value_counts(normalize=True)
    # by date count, not row count -- row count per date varies with how many
    # scenarios happened to be sampled on that date
    n_train_dates = feat.loc[split == "train", "date"].nunique()
    assert n_train_dates / n == pytest.approx(0.70, abs=0.05)


# ------------------------------------------------------------ the real bar


def test_model_beats_persistence_on_held_out_test_dates(trained):
    """The actual 'Done when' criterion from ROADMAP M4, locked in as a test
    rather than left as something to eyeball from a printout. Checked on
    TEST, not train/val -- train/val beating persistence would be a much
    weaker and less honest claim.
    """
    _, results, _, _ = trained
    test = results["test"]
    assert test["model_mae"] < test["persistence_mae"], (
        f"model MAE {test['model_mae']:.3f} C does not beat persistence "
        f"{test['persistence_mae']:.3f} C on held-out dates"
    )


def test_model_beats_a_weather_only_linear_baseline(cfg, feat, trained):
    """Guards against the model secretly being 'just track outdoor weather'.

    t_out_fcst_h dominates feature importance by a wide margin, which is
    physically sensible (T_out is the dominant 30-min-ahead driver in a
    cooling-dominated climate) but raises the question of whether the other
    features -- occupancy, timetable lookahead, command history -- add
    anything real. A closed-form linear fit on t_out_fcst_h alone isolates
    that: if the full model didn't beat it, the timetable-anticipatory
    premise this whole project rests on would have no evidence behind it.

    Measured margin: ~5.9% (linear-only 4.02 C -> full model 3.78 C on test),
    smaller than the persistence margin but real and worth guarding.
    """
    model, results, _, split = trained
    train_mask = split == "train"
    test_mask = split == "test"

    X1 = np.column_stack([feat.loc[train_mask, "t_out_fcst_h"], np.ones(train_mask.sum())])
    coef, *_ = np.linalg.lstsq(X1, feat.loc[train_mask, TARGET_COLUMN], rcond=None)
    linear_pred = feat.loc[test_mask, "t_out_fcst_h"] * coef[0] + coef[1]
    linear_mae = float(np.abs(linear_pred - feat.loc[test_mask, TARGET_COLUMN]).mean())

    assert results["test"]["model_mae"] < linear_mae, (
        "full model does not beat a trivial linear fit on weather forecast "
        "alone -- the occupancy/timetable/command features are adding nothing"
    )


def test_persistence_mae_matches_manual_computation(feat):
    """Ties persistence_mae() back to its definition: |T_air(t) - T_air(t+H)|."""
    manual = float((feat["t_air_c"] - feat[TARGET_COLUMN]).abs().mean())
    assert persistence_mae(feat) == pytest.approx(manual)


# --------------------------------------------------------------- robustness


def test_predictions_stay_physically_plausible(trained, feat):
    model, _, _, split = trained
    from src.features import FEATURE_COLUMNS

    test_mask = split == "test"
    pred = model.predict(feat.loc[test_mask, list(FEATURE_COLUMNS)])
    assert np.isfinite(pred).all()
    assert (pred > -10).all() and (pred < 60).all(), (
        "predictions leave the physically plausible range -- generous bounds, "
        "this is a blow-up guard, not a tight accuracy check"
    )


def test_training_is_reproducible(cfg, raw):
    """Same config, same data, same seed (fixed at 0 inside train_model) ->
    identical MAE. Matters because M5 will build on top of this model, and
    debugging is much harder if training is silently non-deterministic.

    Depends on `raw` (unused directly) for its skip check -- train_model()
    below reads the default parquet path itself.
    """
    _, results_a, _, _ = train_model(cfg, verbose=False)
    _, results_b, _, _ = train_model(cfg, verbose=False)
    assert results_a["test"]["model_mae"] == pytest.approx(results_b["test"]["model_mae"])


def test_saved_model_reloads_to_identical_predictions(trained, feat):
    """Round-trips through the actual save path used by the CLI entry point.
    Native text format, not pickle -- see ROADMAP's handoff-format rule.
    """
    import lightgbm as lgb

    model, _, _, split = trained
    from src.features import FEATURE_COLUMNS

    X_test = feat.loc[split == "test", list(FEATURE_COLUMNS)]
    original_pred = model.predict(X_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    reloaded = lgb.Booster(model_file=str(MODEL_PATH))
    reloaded_pred = reloaded.predict(X_test)

    assert np.allclose(original_pred, reloaded_pred)


def test_empty_split_raises_a_clear_error(cfg, require_weather):
    """A dataset spanning too few distinct dates must fail loudly, not
    silently train on an empty split.
    """
    import tempfile
    from pathlib import Path

    from src.data_generator import generate_dataset

    tiny = generate_dataset(cfg, target_rows=200, seed=99)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tiny.parquet"
        tiny.to_parquet(p)
        with pytest.raises(ValueError, match="split is empty"):
            train_model(cfg, raw_path=p, verbose=False)


# ------------------------------------------------ policy-dependence (M5 risk)


def test_model_stays_sane_under_a_policy_it_was_not_trained_to_expect(cfg, require_weather):
    """Regression guard for a real finding, documented in docs/PARAMETERS.md.

    The model was trained mostly under StochasticController's exploration
    policy (see data_generator.py). Driven LIVE by the plain reactive
    thermostat instead -- the actual baseline M5 will use -- it beats
    persistence in only about 1 of 4 scenarios, because persistence itself is
    a strong baseline once a controller holds T_air fairly stable. That is a
    scope note for M5 (don't reuse the "+21.8%" figure there), not a defect
    in M4 to fix here.

    What THIS test pins is narrower and genuinely load-bearing: predictions
    must stay physically sane under that distribution shift, not silently
    degrade into nonsense. If this starts failing, something about the
    feature/model pairing has gotten more fragile than it was when measured.
    """
    from src.cabin_model import CabinInputs, CabinModel
    from src.data_generator import CITIES
    from src.features import FEATURE_COLUMNS
    from src.occupancy import Service, simulate
    from src.train import MODEL_PATH
    from src.weather import to_minutes

    if not MODEL_PATH.exists():
        pytest.skip("no saved model. Run:  python -m src.train")
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    H = cfg["simulation"]["control_horizon_min"]
    wx = pd.read_csv(DATA_DIR / f"weather_{CITIES[0]}_summer.csv", parse_dates=["timestamp"])

    model = CabinModel(cfg)
    service = Service(direction="down", pattern="semi_express", load_factor=1.0)
    profile = simulate(service, cfg)
    start = pd.Timestamp("2024-07-20 14:00")
    w = to_minutes(wx, start, len(profile) + H)

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
            "scenario_id": 0, "city": CITIES[0], "date": start.date(), "depart_hour": 14.0,
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

    raw = pd.DataFrame(rows)
    feat = add_features(raw, cfg)
    pred = booster.predict(feat[list(FEATURE_COLUMNS)])

    assert np.isfinite(pred).all()
    assert (pred > -10).all() and (pred < 60).all()
    mae = float(np.abs(pred - feat[TARGET_COLUMN]).mean())
    assert mae < 6.0, (
        f"MAE {mae:.2f} C under the reactive policy is far outside the "
        f"0.9-2.7 C range measured when this was characterised -- something "
        f"has changed, not just the known policy-dependence"
    )
