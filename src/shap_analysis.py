"""M9: SHAP feature attribution for the T_air(t+H) forecaster.

Planned from the project's original 5-day plan ("SHAP summary plot, ~30 min
with LightGBM") but never actually built until now -- `train.py`'s own
`__main__` block prints LightGBM's native `feature_importance(gain)`
instead, which answers a different and less complete question. Gain credits
a feature for being useful WHEREVER it appears in a split, with no sense of
how much it actually moves any GIVEN prediction or in which direction; a
feature can rank high on gain from a few very informative early splits while
mattering little to most individual predictions. SHAP (`TreeExplainer`,
exact for tree ensembles, not an approximation) answers the sharper
question: for THIS prediction, how much did EACH feature push it away from
the dataset's average, and which way.

Checked before trusting the result, not assumed to just work: LightGBM's
native `categorical_feature` support (`city`, `direction`, `pattern` here)
has a documented history of breaking SHAP's `TreeExplainer`
("could not convert string to float", shap/shap#170) on some version
combinations. Confirmed correct here, not just non-crashing, before this
file was written: SHAP's own additivity property --
`sum(shap_values, axis=1) + expected_value == booster.predict(X)` --
reproduced this project's actual categorical-feature model to within 1e-14
on 200 real test-split rows.

Runs on the M4 TEST split, the same held-out dates every other headline
number in this project is reported against -- explaining a model on the
data it trained on risks explaining memorisation, not learned signal.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas, see src/train.py

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DATA_DIR, load_config
from src.features import FEATURE_COLUMNS, add_features
from src.train import MODEL_PATH, chronological_split

PLOT_PATH = DATA_DIR / "m9_shap_summary.png"
IMPORTANCE_PATH = DATA_DIR / "m9_shap_importance.csv"


def held_out_test_features() -> pd.DataFrame:
    """The M4 test-split rows, feature-engineered, in FEATURE_COLUMNS order
    -- reuses add_features()/chronological_split() rather than a second
    feature-building path, same reasoning as every other script here."""
    cfg = load_config()
    raw = pd.read_parquet(DATA_DIR / "scenarios_raw.parquet")
    feat = add_features(raw, cfg)
    split = chronological_split(feat)
    return feat.loc[split == "test", list(FEATURE_COLUMNS)]


def compute(booster: lgb.Booster | None = None, X: pd.DataFrame | None = None):
    """Returns (shap_values, X). Both arguments overridable for tests that
    want a smaller X than the full test split."""
    if booster is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"no saved model at {MODEL_PATH}. Run: python -m src.train")
        booster = lgb.Booster(model_file=str(MODEL_PATH))
    if X is None:
        X = held_out_test_features()

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)
    return shap_values, X


def importance_table(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP value| per feature -- SHAP's own global-importance
    ranking, directly comparable to train.py's gain-based one but computed
    from a different, per-prediction-additive quantity. Where they disagree
    is itself informative (see ROADMAP.md's M9 section)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap_c": mean_abs})
            .sort_values("mean_abs_shap_c", ascending=False)
            .reset_index(drop=True))


def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, out_path=PLOT_PATH):
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle("M9 -- SHAP feature attribution, T_air(t+30min) forecaster, M4 test split",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    X = held_out_test_features()
    print(f"computing SHAP values on {len(X)} M4 test-split rows...")
    shap_values, X = compute(X=X)

    table = importance_table(shap_values, X)
    print("\nranked by mean |SHAP value| (degrees C of average prediction impact):")
    print(table.head(15).to_string(index=False))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(IMPORTANCE_PATH, index=False)
    print(f"\nsaved to {IMPORTANCE_PATH}")

    path = plot_summary(shap_values, X)
    print(f"saved {path}")
