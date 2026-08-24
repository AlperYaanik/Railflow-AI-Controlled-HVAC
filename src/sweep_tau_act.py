"""M6: characterises how much the M5 result depends on tau_act -- the
actuator lag GENUINELY UNKNOWN for any real vehicle (config/cabin_params.yaml
tags it [ASSUMPTION], swept rather than asserted). Sweeps
cfg["hvac"]["tau_act_min"] and re-runs both of M5's comparisons
(src/compare_controllers.py) at each point, on the SAME 23 held-out test
scenarios throughout -- only the actuator lag changes between rows, so any
trend is attributable to that one variable.

WHAT WAS EXPECTED GOING IN, AND WHAT THE SWEEP ACTUALLY FOUND (run this
before trusting the summary that follows -- it's the whole point of a
sweep): industrial dead-time-compensation literature reports predictive
control's advantage over reactive control growing with delay, the same
argument a Smith predictor or MPC dead-time compensation is built on. The
real 23-scenario sweep partially confirmed this (on/off energy saving rises
from +0.2% at tau_act=0 to a peak of +4.6% at tau_act=10, then flattens) but
falsified the sharper zero-lag hypothesis this module was ALSO built to
check: that compare_feedforward_contribution()'s gap (anticipatory vs
proportional-only) would shrink toward ~0 at tau_act_min=0, since the
forecast would supposedly have nothing left to anticipate once the actuator
has no lag to hide. It does not shrink -- it stays flat at roughly -5% to
-6% across the ENTIRE sweep, confirmed further by forcing dead_time_min=0
too (true zero total lag) on a smaller probe. Not a bug: the forecast's
value turns out to come substantially from anticipating KNOWN FUTURE
DISTURBANCES (expected_boarding, time_to_next_station_min, the weather
forecast) that exist independently of actuator dynamics, which the original
hypothesis didn't account for. Full writeup, and the more useful finding
this sweep actually produced (AnticipatorySetpointAdvisor's comfort is far
less sensitive to the unknown tau_act than either baseline), in ROADMAP.md's M6
section -- read that before drawing conclusions from a re-run of this
module, not just this docstring.

TAU_ACT_SWEEP_MIN tops out at 20 min, safely under control_horizon_min's 30
-- if a sweep value ever approached or exceeded the forecast horizon, even
the partial monotonic trend above would not be guaranteed to hold.

WHY OVERRIDING cfg IS SAFE HERE, SPECIFICALLY. compare()/
compare_feedforward_contribution() both thread `cfg` through to CabinModel
(the actuator physics that must see the swept value) AND to
LiveFeatureBuilder (the forecaster's feature engineering, which must NOT --
see TRAINED_EWMA_HALFLIFE_MIN's docstring in features.py). LiveFeatureBuilder
already defaults its EWMA halflife to the frozen training-time value
regardless of what cfg says, so sweeping cfg["hvac"]["tau_act_min"] here
changes ONLY the actuator, not the forecaster's feature distribution --
exactly the fix made during the M6-readiness audit, before this module was
written.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import copy

import pandas as pd

from src.compare_controllers import (
    compare,
    compare_feedforward_contribution,
    held_out_test_scenarios,
    summarize,
)
from src.config import DATA_DIR, load_config

TAU_ACT_SWEEP_MIN = (0.0, 2.0, 5.0, 10.0, 20.0)
"""[ASSUMPTION] range, not a claim about any real vehicle's actual actuator
-- see config/cabin_params.yaml's tau_act_min comment. Chosen to stay
comfortably under the 30-minute forecast horizon; see module docstring."""

RESULTS_PATH = DATA_DIR / "m6_tau_act_sweep.csv"


def sweep(
    tau_act_values_min=TAU_ACT_SWEEP_MIN,
    cfg: dict | None = None, scenarios: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-runs both M5 comparisons at each tau_act_min. One row per swept
    value; columns cover both the on/off headline (compare()) and the
    feedforward-contribution ablation (compare_feedforward_contribution()),
    so a single table supports both of M6's claims: the headline number's
    lag-sensitivity, and the zero-lag correctness check.
    """
    cfg = cfg if cfg is not None else load_config()
    scenarios = scenarios if scenarios is not None else held_out_test_scenarios()

    rows = []
    for tau in tau_act_values_min:
        swept_cfg = copy.deepcopy(cfg)
        swept_cfg["hvac"]["tau_act_min"] = tau

        onoff = summarize(compare(swept_cfg, scenarios))
        ff = summarize(
            compare_feedforward_contribution(swept_cfg, scenarios),
            baseline="proportional", candidate="anticipatory",
        )
        rows.append({
            "tau_act_min": tau,
            "onoff_mean_saving_pct": onoff["mean_energy_saving_pct"],
            "onoff_median_saving_pct": onoff["median_energy_saving_pct"],
            "onoff_n_energy_better": onoff["n_energy_better"],
            "onoff_n_worse_on_both": onoff["n_worse_on_both"],
            "onoff_total_baseline_dh": onoff["total_baseline_degree_hours"],
            "onoff_total_candidate_dh": onoff["total_candidate_degree_hours"],
            "ff_mean_saving_pct": ff["mean_energy_saving_pct"],
            "ff_total_baseline_dh": ff["total_baseline_degree_hours"],
            "ff_total_candidate_dh": ff["total_candidate_degree_hours"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = sweep()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("M6 tau_act sweep -- 23 held-out test-split scenarios per point")
    print()
    print(results.to_string(index=False))
    print()
    print(f"saved to {RESULTS_PATH}")
