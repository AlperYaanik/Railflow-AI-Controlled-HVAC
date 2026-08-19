"""M9 part 2: does a different model (or different LightGBM hyperparameters)
forecast T_air(t+H) better than what M4 shipped?

Reuses M4's exact pipeline end to end -- add_features(), chronological_split(),
FEATURE_COLUMNS, TARGET_COLUMN, persistence_mae() -- rather than a second,
parallel feature-engineering path. Only two things are new here: the extra
model families, and a one-hot encoding of the three categorical columns
(city, direction, pattern) for the libraries that don't take pandas
categoricals natively the way LightGBM does. The SAME one-hot encoding is
used for every non-LightGBM model (linear, random forest, XGBoost, CatBoost)
even though XGBoost/CatBoost could each take categoricals a different,
native way -- one shared preprocessing path keeps the comparison about the
model, not about which library got the more favourable encoding.

Same chronological-split discipline as everywhere else in this project:
every model sees the same train/val/test dates, tuned (where applicable)
against val, scored once on test.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas, see src/train.py

import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import DATA_DIR, PROJECT_ROOT, load_config
from src.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, add_features
from src.train import MODEL_PATH, chronological_split, persistence_mae

BENCHMARK_DIR = PROJECT_ROOT / "models" / "benchmark"
RESULTS_PATH = DATA_DIR / "m9_model_benchmark.csv"

LGBM_GRID = [
    {"num_leaves": nl, "learning_rate": lr}
    for nl in (15, 31, 63)
    for lr in (0.02, 0.05, 0.1)
]
"""Small, not exhaustive -- 9 points around the shipped (31, 0.05) config,
enough to say whether the shipped hyperparameters are already reasonable
without turning this into a multi-hour sweep for a tabular problem this
size."""


def _prep_data():
    cfg = load_config()
    raw = pd.read_parquet(DATA_DIR / "scenarios_raw.parquet")
    feat = add_features(raw, cfg)
    split = chronological_split(feat)
    X = feat[list(FEATURE_COLUMNS)]
    y = feat[TARGET_COLUMN]
    masks = {name: (split == name) for name in ("train", "val", "test")}
    return feat, X, y, masks


def _one_hot(X: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """Fits on TRAIN only (avoids leaking val/test categories into the
    encoding), applies to all rows. Numeric columns pass through unchanged.
    """
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(X.loc[masks["train"], list(CATEGORICAL_COLUMNS)])
    dummies = pd.DataFrame(
        enc.transform(X[list(CATEGORICAL_COLUMNS)]),
        columns=enc.get_feature_names_out(list(CATEGORICAL_COLUMNS)),
        index=X.index,
    )
    return pd.concat([X[numeric_cols].astype(float), dummies], axis=1)


def _score(pred: np.ndarray, y: pd.Series, feat: pd.DataFrame, mask: pd.Series) -> dict:
    return {
        "n": int(mask.sum()),
        "model_mae": float(np.abs(pred - y[mask]).mean()),
        "persistence_mae": persistence_mae(feat[mask]),
    }


def _train_lightgbm(X, y, feat, masks, params: dict) -> dict:
    categorical = list(CATEGORICAL_COLUMNS)
    train_set = lgb.Dataset(X[masks["train"]], label=y[masks["train"]], categorical_feature=categorical)
    val_set = lgb.Dataset(X[masks["val"]], label=y[masks["val"]], reference=train_set)
    full_params = {"objective": "regression_l1", "metric": "l1", "min_child_samples": 20,
                    "verbosity": -1, "seed": 0, **params}
    model = lgb.train(
        full_params, train_set, num_boost_round=500, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    pred = {name: model.predict(X[mask], num_iteration=model.best_iteration)
            for name, mask in masks.items()}
    return model, pred


def run() -> pd.DataFrame:
    feat, X, y, masks = _prep_data()
    Xoh = _one_hot(X, masks)

    rows = []
    artifacts = {}

    # -- LightGBM, shipped hyperparameters (num_leaves=31, learning_rate=0.05) --
    t0 = time.time()
    model, pred = _train_lightgbm(X, y, feat, masks, {"num_leaves": 31, "learning_rate": 0.05})
    rows.append({"model": "lightgbm_shipped", "train_s": time.time() - t0,
                 **{f"{n}_{k}": v for n, mask in masks.items()
                    for k, v in _score(pred[n], y, feat, mask).items()}})
    artifacts["lightgbm_shipped"] = ("lgbm", model)

    # -- LightGBM, small hyperparameter grid, picked by VAL MAE --
    best_val_mae, best_params, best_model, best_pred = None, None, None, None
    t0 = time.time()
    for params in LGBM_GRID:
        m, p = _train_lightgbm(X, y, feat, masks, params)
        val_mae = float(np.abs(p["val"] - y[masks["val"]]).mean())
        if best_val_mae is None or val_mae < best_val_mae:
            best_val_mae, best_params, best_model, best_pred = val_mae, params, m, p
    rows.append({"model": f"lightgbm_tuned{best_params}", "train_s": time.time() - t0,
                 **{f"{n}_{k}": v for n, mask in masks.items()
                    for k, v in _score(best_pred[n], y, feat, mask).items()}})
    artifacts["lightgbm_tuned"] = ("lgbm", best_model)

    # -- Linear Regression --
    t0 = time.time()
    lin = LinearRegression()
    lin.fit(Xoh[masks["train"]], y[masks["train"]])
    pred = {n: lin.predict(Xoh[mask]) for n, mask in masks.items()}
    rows.append({"model": "linear_regression", "train_s": time.time() - t0,
                 **{f"{n}_{k}": v for n, mask in masks.items()
                    for k, v in _score(pred[n], y, feat, mask).items()}})
    artifacts["linear_regression"] = ("sklearn", lin)

    # -- Random Forest, small hyperparameter grid, picked by VAL MAE --
    # sklearn's near-default settings (unlimited depth, min_samples_leaf=5)
    # were tried first and memorised the ~15k-row training set outright
    # (train MAE 0.85 C against a test MAE of 4.46 C -- confirmed textbook
    # overfitting by checking the train/val gap, not assumed from a bad test
    # score alone). Given the same fair val-selected search as LightGBM
    # above, rather than one hand-picked config.
    RF_GRID = [
        {"max_depth": d, "min_samples_leaf": leaf}
        for d in (6, 8, 10) for leaf in (20, 50, 100)
    ]
    t0 = time.time()
    best_val_mae, best_rf, best_pred = None, None, None
    for params in RF_GRID:
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0, **params)
        m.fit(Xoh[masks["train"]], y[masks["train"]])
        p = {n: m.predict(Xoh[mask]) for n, mask in masks.items()}
        val_mae = float(np.abs(p["val"] - y[masks["val"]]).mean())
        if best_val_mae is None or val_mae < best_val_mae:
            best_val_mae, best_rf, best_pred = val_mae, m, p
    rows.append({"model": f"random_forest{best_rf.get_params()['max_depth'], best_rf.get_params()['min_samples_leaf']}",
                 "train_s": time.time() - t0,
                 **{f"{n}_{k}": v for n, mask in masks.items()
                    for k, v in _score(best_pred[n], y, feat, mask).items()}})
    artifacts["random_forest"] = ("sklearn", best_rf)

    # -- XGBoost --
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        early_stopping_rounds=30, eval_metric="mae", random_state=0)
    xgb.fit(Xoh[masks["train"]], y[masks["train"]],
            eval_set=[(Xoh[masks["val"]], y[masks["val"]])], verbose=False)
    pred = {n: xgb.predict(Xoh[mask]) for n, mask in masks.items()}
    rows.append({"model": "xgboost", "train_s": time.time() - t0,
                 **{f"{n}_{k}": v for n, mask in masks.items()
                    for k, v in _score(pred[n], y, feat, mask).items()}})
    artifacts["xgboost"] = ("xgboost", xgb)

    # -- CatBoost --
    t0 = time.time()
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                            loss_function="MAE", random_seed=0, verbose=False,
                            early_stopping_rounds=30)
    cb.fit(Xoh[masks["train"]], y[masks["train"]],
           eval_set=(Xoh[masks["val"]], y[masks["val"]]))
    pred = {n: cb.predict(Xoh[mask]) for n, mask in masks.items()}
    rows.append({"model": "catboost", "train_s": time.time() - t0,
                 **{f"{n}_{k}": v for n, mask in masks.items()
                    for k, v in _score(pred[n], y, feat, mask).items()}})
    artifacts["catboost"] = ("catboost", cb)

    out = pd.DataFrame(rows)
    for name in ("train", "val", "test"):
        out[f"{name}_improvement_pct"] = 100.0 * (
            1.0 - out[f"{name}_model_mae"] / out[f"{name}_persistence_mae"]
        )
    out = out.sort_values("test_model_mae").reset_index(drop=True)
    return out, artifacts


def save_artifacts(artifacts: dict, winner: str) -> None:
    """Saves every trained model (for reproducibility/inspection), then
    labels the winner distinctly rather than silently overwriting
    MODEL_PATH -- adopting a non-LightGBM winner into the live app/
    controller needs an interface change (predict() signature, input
    encoding) that this script does not make on its own. See the
    benchmark's printed summary and ROADMAP.md's M9 section.
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    for name, (kind, obj) in artifacts.items():
        if kind == "lgbm":
            obj.save_model(str(BENCHMARK_DIR / f"{name}.txt"))
        elif kind == "xgboost":
            obj.save_model(str(BENCHMARK_DIR / f"{name}.json"))
        elif kind == "catboost":
            obj.save_model(str(BENCHMARK_DIR / f"{name}.cbm"))
        else:
            import joblib
            joblib.dump(obj, BENCHMARK_DIR / f"{name}.joblib")
    print(f"all {len(artifacts)} trained models saved under {BENCHMARK_DIR}")
    print(f"winner: {winner}")


if __name__ == "__main__":
    print("training 6 candidates (persistence is the pre-existing zero-effort baseline, "
          "not trained here) -- reusing M4's exact feature/split pipeline...\n")
    results, artifacts = run()

    print(results[["model", "train_s", "test_model_mae", "test_persistence_mae",
                    "test_improvement_pct"]].to_string(index=False))

    winner = results.iloc[0]["model"]
    print(f"\nbest test MAE: {winner}")
    shipped_mae = results.loc[results["model"] == "lightgbm_shipped", "test_model_mae"].iloc[0]
    print(f"shipped LightGBM test MAE for comparison: {shipped_mae:.4f} C "
          f"(currently deployed at {MODEL_PATH})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nfull results saved to {RESULTS_PATH}")

    save_artifacts(artifacts, winner)
