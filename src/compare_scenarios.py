"""M14: the actual experiment -- does anticipation pay under disturbances
large enough to be worth anticipating?

WHAT THIS ANSWERS, AND WHAT IT DELIBERATELY RISKS. Every prior tuning round
in this project (M10 Phase 2, M12) measured the advisor on the real fetched
weather of the 23 TEST scenarios and found `ff_weight > 0` -- using M4's
forecast at all -- makes energy and comfort WORSE. This module re-runs that
same question under src/scenarios.py's explicit disturbances, testing one
falsifiable hypothesis: the forecast loses on mild days because there is
nothing worth anticipating, and should win once there is.

It can come back negative. If `ff_weight > 0` still loses under a 6 K
outdoor ramp visible 30 minutes ahead, then this architecture cannot
demonstrate anticipation benefit, and that is the finding -- recorded the
same way M10 Phase 2's parity and M11-continued's tint result were. The
scenarios that make the AI look worse stay in the printed table.

BOTH ARMS GET THE IDENTICAL ENVIRONMENT. The same `scenario` object is
passed to both `run_controller()` calls, so the disturbance is not a thing
done TO the baseline -- it is the weather they both fly through. Anything
that separates them afterwards came from the control policy.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pandas as pd

from src import scenarios as scenarios_mod
from src.compare_controllers import _load_booster, held_out_test_scenarios
from src.config import DATA_DIR, load_config
from src.controllers import (
    SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
    AnticipatorySetpointAdvisor,
    ThermostatController,
)
from src.evaluate import run_controller, score
from src.features import LiveFeatureBuilder

RESULTS_PATH = DATA_DIR / "m14_scenario_comparison.csv"

FF_WEIGHT_GRID = (0.0, 0.3, 0.6)
"""0.0 is what ships today (forecast ignored). 0.3 is the least-bad nonzero
point M10 Phase 2's own grid found. 0.6 probes further in case the optimum
moves substantially once disturbances are large -- deliberately not a fine
sweep, since the question here is directional ("does the sign flip?"), not
a re-tune."""


def _run_pair(cfg, booster, row, scenario, ff_weight):
    run_kwargs = dict(cfg=cfg, city=row.city, date=str(row.date), depart_hour=row.depart_hour,
                      direction=row.direction, pattern=row.pattern, load_factor=row.load_factor,
                      advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
                      scenario=scenario)

    def ai_factory(model):
        builder = LiveFeatureBuilder(cfg, city=row.city, direction=row.direction,
                                     pattern=row.pattern, load_factor=row.load_factor,
                                     depart_hour=row.depart_hour)
        return AnticipatorySetpointAdvisor(model, booster, builder, ff_weight=ff_weight)

    base = score(run_controller(lambda m: ThermostatController(m), **run_kwargs), cfg)
    ai = score(run_controller(ai_factory, **run_kwargs), cfg)
    return base, ai


def compare_all(cfg=None, scenarios_to_run=None, ff_weights=FF_WEIGHT_GRID) -> pd.DataFrame:
    cfg = cfg if cfg is not None else load_config()
    booster = _load_booster()
    test_rows = held_out_test_scenarios()
    scenarios_to_run = scenarios_to_run if scenarios_to_run is not None else scenarios_mod.ALL

    records = []
    for scenario in scenarios_to_run:
        for ff_weight in ff_weights:
            agg = {"base_energy": 0.0, "ai_energy": 0.0, "base_dh": 0.0, "ai_dh": 0.0,
                   "base_ramp": 0.0, "ai_ramp": 0.0, "base_peak": 0.0, "ai_peak": 0.0,
                   "base_switch": 0, "ai_switch": 0, "both_better": 0, "worse_on_both": 0}
            for row in test_rows.itertuples():
                base, ai = _run_pair(cfg, booster, row, scenario, ff_weight)
                agg["base_energy"] += base.energy_kwh
                agg["ai_energy"] += ai.energy_kwh
                agg["base_dh"] += base.degree_hours
                agg["ai_dh"] += ai.degree_hours
                agg["base_ramp"] += base.power_ramp_w_per_min
                agg["ai_ramp"] += ai.power_ramp_w_per_min
                agg["base_peak"] = max(agg["base_peak"], base.peak_power_kw)
                agg["ai_peak"] = max(agg["ai_peak"], ai.peak_power_kw)
                agg["base_switch"] += base.n_switching_events
                agg["ai_switch"] += ai.n_switching_events
                energy_ok = ai.energy_kwh <= base.energy_kwh + 1e-9
                comfort_ok = ai.degree_hours <= base.degree_hours + 1e-9
                agg["both_better"] += int(energy_ok and comfort_ok)
                agg["worse_on_both"] += int((not energy_ok) and (not comfort_ok))

            n = len(test_rows)
            records.append({
                "scenario": scenario.name,
                "ff_weight": ff_weight,
                "energy_saving_pct": 100.0 * (1.0 - agg["ai_energy"] / agg["base_energy"]),
                "comfort_improvement_pct": (100.0 * (1.0 - agg["ai_dh"] / agg["base_dh"])
                                            if agg["base_dh"] > 0 else 0.0),
                "ramp_reduction_pct": (100.0 * (1.0 - agg["ai_ramp"] / agg["base_ramp"])
                                       if agg["base_ramp"] > 0 else 0.0),
                "peak_reduction_pct": (100.0 * (1.0 - agg["ai_peak"] / agg["base_peak"])
                                       if agg["base_peak"] > 0 else 0.0),
                "base_switching": agg["base_switch"], "ai_switching": agg["ai_switch"],
                "both_better": agg["both_better"], "worse_on_both": agg["worse_on_both"],
                "n_scenarios": n,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    cfg = load_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"M14: {len(scenarios_mod.ALL)} scenarios x {len(FF_WEIGHT_GRID)} ff_weights "
          f"x {len(held_out_test_scenarios())} TEST journeys, both arms each...")
    results = compare_all(cfg)
    results.to_csv(RESULTS_PATH, index=False)

    pd.set_option("display.width", 200)
    show = results.copy()
    for col in ("energy_saving_pct", "comfort_improvement_pct", "ramp_reduction_pct",
                "peak_reduction_pct"):
        show[col] = show[col].map(lambda v: f"{v:+.2f}%")
    print(show.to_string(index=False))

    print(f"\nsaved to {RESULTS_PATH}")
    print("\nPositive % = AI better than the static baseline. Read the ff_weight=0.0 rows as "
          "'what ships today' and the nonzero ones as 'forecast turned on'. If no nonzero row "
          "beats its own ff_weight=0.0 row in the same scenario, the forecast still does not "
          "earn its keep and M14's honest answer is negative -- see ROADMAP.md's M14 section.")
