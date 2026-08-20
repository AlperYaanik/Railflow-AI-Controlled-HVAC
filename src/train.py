"""Train the T_air(t+H) forecaster and check it against a persistence baseline.

Prediction target: T_air at t+H, H = simulation.control_horizon_min (30 min
by default) -- the single horizon this project needs, per ROADMAP M4. Not a
multi-horizon model; M5's controller only ever consults one horizon.

The bar is deliberately weak on purpose: T_air(t+H) = T_air(t), i.e. "nothing
changes." Beating it is the whole point of M4 -- if the model can't beat
"predict no change," the lookahead features are not doing their job and that
is the first place to look (ROADMAP's own words).
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas, see note below

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROJECT_ROOT, load_config
from src.features import FEATURE_COLUMNS, TARGET_COLUMN, add_features

# WHY lightgbm IS IMPORTED FIRST, AND MUST STAY FIRST:
# On this Windows environment, pandas' native runtime, if loaded into the
# process before lightgbm's, corrupts something LightGBM's C API depends on --
# every call into LGBM_DatasetSetField then dies with an access violation, a
# hard native crash, not a catchable Python exception. Confirmed with a
# minimal repro that has nothing to do with this project's data: `import
# pandas` with pandas never subsequently used, followed by `import lightgbm`
# and a call to `lgb.Dataset(random_array).construct()`, crashes the same way.
# Reversing the import order avoids it completely. See tests/conftest.py --
# the same ordering has to be forced there too, because pytest may import
# other test modules that pull in pandas before it ever reaches this module.

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_t_air_forecaster.txt"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 of dates go to test


def chronological_split(df: pd.DataFrame) -> pd.Series:
    """Assigns train/val/test by CALENDAR DATE, never by row.

    Non-negotiable per ROADMAP M4. A row-random split would leak: consecutive
    minutes within one journey are highly autocorrelated (lag features span up
    to 20 minutes, the target itself is only 30 minutes ahead of the current
    row), so a random split puts near-duplicate rows on both sides and the
    reported MAE would be optimistic nonsense. Splitting by date is what
    guarantees a given day's weather never appears in more than one split --
    it's the ordering a real deployment would actually see: train on the past,
    evaluate on days that come after it.
    """
    dates = sorted(df["date"].unique())
    n = len(dates)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))

    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train:n_train + n_val])
    # everything else (dates[n_train + n_val:]) is test, by construction

    split = pd.Series("test", index=df.index)
    split[df["date"].isin(train_dates)] = "train"
    split[df["date"].isin(val_dates)] = "val"
    return split


def persistence_mae(df: pd.DataFrame) -> float:
    """Baseline MAE: predict T_air(t+H) = T_air(t)."""
    return float((df["t_air_c"] - df[TARGET_COLUMN]).abs().mean())


def train_model(cfg: dict | None = None, raw_path=None, verbose: bool = True):
    """Loads raw scenarios, builds features, splits, trains, and evaluates.

    Returns (booster, results_dict, feat_dataframe, split_series) so callers
    (tests, M5) can inspect predictions without re-running the pipeline.
    """
    cfg = cfg if cfg is not None else load_config()
    raw = pd.read_parquet(raw_path or DATA_DIR / "scenarios_raw.parquet")
    feat = add_features(raw, cfg)
    split = chronological_split(feat)

    X = feat[list(FEATURE_COLUMNS)]
    y = feat[TARGET_COLUMN]

    masks = {name: (split == name) for name in ("train", "val", "test")}
    for name, mask in masks.items():
        if mask.sum() == 0:
            raise ValueError(f"'{name}' split is empty -- not enough distinct dates")

    # Native Dataset/train API, not the sklearn wrapper -- avoids pulling in
    # scikit-learn as a dependency for what would otherwise be a thin shim.
    categorical = [c for c in FEATURE_COLUMNS if c in ("city", "direction", "pattern")]
    train_set = lgb.Dataset(X[masks["train"]], label=y[masks["train"]], categorical_feature=categorical)
    val_set = lgb.Dataset(X[masks["val"]], label=y[masks["val"]], reference=train_set)

    params = {
        "objective": "regression_l1", "metric": "l1",
        # learning_rate/num_leaves tuned by src/benchmark_models.py's M9
        # 9-point grid (val-selected, confirmed once on test): beats the
        # original 0.05/31 by a real, free 0.047 C test MAE, no interface
        # change. See ROADMAP.md's M9 section for the full comparison
        # against other model families -- LightGBM (this) was competitive
        # with, not decisively better than, several of them; this is a
        # same-family tune, not evidence LightGBM is uniquely correct.
        "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 20,
        "verbosity": -1, "seed": 0,
    }
    model = lgb.train(
        params, train_set, num_boost_round=500, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    results = {}
    for name, mask in masks.items():
        pred = model.predict(X[mask], num_iteration=model.best_iteration)
        results[name] = {
            "n": int(mask.sum()),
            "model_mae": float(np.abs(pred - y[mask]).mean()),
            "persistence_mae": persistence_mae(feat[mask]),
            "date_range": (min(feat.loc[mask, "date"]), max(feat.loc[mask, "date"])),
        }
        results[name]["improvement_pct"] = (
            100.0 * (1.0 - results[name]["model_mae"] / results[name]["persistence_mae"])
        )

    if verbose:
        n_dates = {name: feat.loc[mask, "date"].nunique() for name, mask in masks.items()}
        print(f"train/val/test dates : {n_dates['train']}/{n_dates['val']}/{n_dates['test']}")
        for name in ("train", "val", "test"):
            r = results[name]
            print(f"  {name:5} n={r['n']:6d}  dates {r['date_range'][0]}..{r['date_range'][1]}  "
                  f"model MAE {r['model_mae']:.3f} C  persistence MAE {r['persistence_mae']:.3f} C  "
                  f"({r['improvement_pct']:+.1f}%)")

    return model, results, feat, split


if __name__ == "__main__":
    cfg = load_config()
    model, results, feat, split = train_model(cfg)

    print()
    print("Feature importance (gain), top 10:")
    imp = pd.Series(model.feature_importance(importance_type="gain"),
                     index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(imp.head(10).to_string())

    test = results["test"]
    passed = test["model_mae"] < test["persistence_mae"]
    print()
    print(f"{'PASS' if passed else 'FAIL'}: test MAE {test['model_mae']:.3f} C "
          f"{'<' if passed else '>='} persistence {test['persistence_mae']:.3f} C")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"saved to {MODEL_PATH}")
