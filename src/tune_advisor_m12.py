"""M12: re-tunes AnticipatorySetpointAdvisor against the new PI-based
PlantResponse -- src/tune_advisor.py's four attempts all tuned against the
retired BangBangPlantResponse, and M10 Phase 2's own diagnosis was explicit
that a bang-bang receiver structurally could not benefit from the forecast.
M12 replaced that receiver with one that responds proportionally to how far
a commanded setpoint sits from the current temperature -- which is exactly
the mechanism a forecast-driven "overshoot the setpoint now, relax it later"
strategy needs to have any effect at all. Same discipline throughout: VAL-
split grid, ranked by n_both_better, confirmed ONCE on TEST.

ATTEMPT 1 -- (ff_weight, max_shift_k, deadband_k), RESULT: A REGRESSION, NOT
A WIN, WORSE THAN THE STALE BANG-BANG-TUNED VALUES IT WAS MEANT TO REPLACE.

Rationale going in: under BangBangPlantResponse, max_shift_k never mattered
past ~1-2 C (src/tune_advisor.py's attempt 1) because the plant only cared
whether a threshold was crossed, never by how far. Under PlantResponse's
proportional response, how far the commanded setpoint sits from the comfort
range is now the whole mechanism -- so M12 also widened the advisor's own
commanding authority (hvac.advisor_setpoint_min_c/max_c, 18-30 C,
config/cabin_params.yaml) well past the 22-26 C comfort range, specifically
so a strategy like "command 19 C to get more capacity now, then relax back
toward 26 C" is expressible at all. max_shift_k is what lets the advisor
actually reach into that widened range.

27-point grid_search_ff_shift_deadband(): ff_weight in {0.0, 0.3, 0.6} x
max_shift_k in {4.0, 8.0, 12.0} x deadband_k in {1.0, 1.5, 2.0}, VAL split
(31 scenarios). Confirmed the ff_weight finding from Phase 2 transfers
unchanged: ANY nonzero ff_weight is sharply worse (ff_weight=0.3 alone
reaches 27/31 VAL scenarios worse-on-both), so the forecast still does not
earn its keep under this plant either -- not re-litigated further. But the
VAL winner among ff_weight=0.0 points (max_shift_k=8.0, deadband_k=1.0, tied
with max_shift_k=12.0) was itself weak -- only 1/31 VAL scenarios
both-better, 4/31 comfort-ok -- and TEST-confirming it made the true picture
clear by comparison against the SAME TEST split run with the untouched,
already-known-stale Phase-2 values (ff_weight=0.0, max_shift_k=4.0,
deadband_k=1.5) through the SAME new plant:

  stale values,   new plant : mean +0.12%  both-better 0/23  worse-on-both 6/23  comfort-ok  4/23
  attempt-1 winner, new plant: mean +0.16%  both-better 0/23  worse-on-both 6/23  comfort-ok  2/23
  (for reference) Phase 2, OLD plant: mean -0.2%  both-better 4/23  worse-on-both 0/23  comfort-ok 21/23

Energy ticks marginally positive, but comfort collapsed (21/23 comfort-ok
under the old plant down to 2-4/23 here) and worse-on-both went from 0/23 to
6/23 -- a genuine regression, not the "parity" Phase 2 found, and the grid's
own top-ranked point isn't even clearly better than just leaving the stale
values in place. Widening (max_shift_k, deadband_k) further was not the
answer to whatever's actually wrong.

DIAGNOSIS: a cascaded-control timescale mismatch, not a bad grid point.
PlantResponse's PI integral time is Kp/Ki = 4000/4.0 = 1000 s (~17 min --
see config/cabin_params.yaml's plant_response_ki_w_per_k_per_s comment). The
advisor recomputes its recommended setpoint every SIMULATED MINUTE from
CURRENT t_air_c error. Before the inner PI loop has settled anywhere near
the setpoint it was given a minute ago, the outer advisor loop moves the
target again -- the textbook failure mode of a cascaded control loop whose
outer stage updates faster than its inner stage can settle, rather than
much slower. Phase 2's own attempt 4 (sweep_update_interval() below --
originally run against BangBangPlantResponse) found spacing updates out
was harmful there, but for a reason specific to that plant: "nothing here
has M8's physical-actuator/bandwidth constraint" (see that function's
docstring) -- a bang-bang receiver has no settling time for an outer loop to
respect. PlantResponse now has a real one, so that conclusion is not assumed
to transfer -- it is a different, untested question, addressed directly by
ATTEMPT 2 below rather than assumed either way.

ATTEMPT 2 -- (advisor_update_interval_min, deadband_k), ff_weight=0.0 and
max_shift_k=8.0 held fixed (attempt 1's own ff_weight/max_shift_k findings:
any nonzero ff_weight is sharply harmful regardless of update cadence tested
so far, and max_shift_k=8.0 tied max_shift_k=12.0 exactly -- neither is the
open question here). grid_search_update_interval()'s 18-point grid:
advisor_update_interval_min in {1, 5, 10, 15, 20, 30} (spanning the
actuator's own dead_time_min+tau_act_min ~7 min lag up through and beyond
the PI's ~17 min integral time) x deadband_k in {1.0, 1.5, 2.0} (attempt 1's
own range, re-tested since a slower cadence changes how long each shift
stays in effect). Result documented at the bottom of this docstring once
run -- see ROADMAP.md's M12 section for whichever of these two attempts
this project ultimately shipped, and why.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pandas as pd

from src.compare_controllers import compare, held_out_test_scenarios, held_out_val_scenarios, summarize
from src.config import DATA_DIR, load_config

FF_WEIGHT_GRID = (0.0, 0.3, 0.6)
MAX_SHIFT_K_GRID = (4.0, 8.0, 12.0)
DEADBAND_K_GRID = (1.0, 1.5, 2.0)

UPDATE_INTERVAL_GRID = (1, 5, 10, 15, 20, 30)
ATTEMPT_2_FF_WEIGHT = 0.0
ATTEMPT_2_MAX_SHIFT_K = 8.0

RESULTS_PATH = DATA_DIR / "m12_advisor_tune.csv"
INTERVAL_RESULTS_PATH = DATA_DIR / "m12_advisor_tune_update_interval.csv"

_RANK_COLUMNS = ["n_both_better", "mean_energy_saving_pct"]
_RANK_ASCENDING = [False, False]


def grid_search_ff_shift_deadband(cfg: dict | None = None, scenarios: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attempt 1 -- see module docstring. Kept runnable for reproducibility;
    __main__ below does not re-run it by default since its result (a
    regression, not a win) is already recorded in the docstring and in
    RESULTS_PATH from the run this project actually shipped from."""
    cfg = cfg if cfg is not None else load_config()
    scenarios = scenarios if scenarios is not None else held_out_val_scenarios()

    rows = []
    for ff_weight in FF_WEIGHT_GRID:
        for max_shift_k in MAX_SHIFT_K_GRID:
            for deadband_k in DEADBAND_K_GRID:
                s = summarize(compare(
                    cfg, scenarios,
                    ff_weight=ff_weight, max_shift_k=max_shift_k, deadband_k=deadband_k,
                ))
                rows.append({
                    "ff_weight": ff_weight, "max_shift_k": max_shift_k, "deadband_k": deadband_k,
                    "mean_energy_saving_pct": s["mean_energy_saving_pct"],
                    "median_energy_saving_pct": s["median_energy_saving_pct"],
                    "n_both_better": s["n_both_better"],
                    "n_worse_on_both": s["n_worse_on_both"],
                    "n_comfort_equal_or_better": s["n_comfort_equal_or_better"],
                    "n_energy_better": s["n_energy_better"],
                    "n_scenarios": s["n_scenarios"],
                })
    return pd.DataFrame(rows)


def pick_winner_ff_shift_deadband(grid: pd.DataFrame) -> tuple[float, float, float]:
    ranked = grid.sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING)
    top = ranked.iloc[0]
    return float(top["ff_weight"]), float(top["max_shift_k"]), float(top["deadband_k"])


def grid_search_update_interval(cfg: dict | None = None, scenarios: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attempt 2 -- see module docstring for the cascaded-control-timescale
    hypothesis this tests. ff_weight/max_shift_k held at attempt 1's own
    findings (ATTEMPT_2_FF_WEIGHT=0.0, ATTEMPT_2_MAX_SHIFT_K=8.0); the two
    open dimensions are advisor_update_interval_min (the new hypothesis) and
    deadband_k (re-tested since a slower cadence changes how long each shift
    stays in effect, so attempt 1's deadband_k ranking need not still hold)."""
    cfg = cfg if cfg is not None else load_config()
    scenarios = scenarios if scenarios is not None else held_out_val_scenarios()

    rows = []
    for update_interval in UPDATE_INTERVAL_GRID:
        for deadband_k in DEADBAND_K_GRID:
            s = summarize(compare(
                cfg, scenarios,
                advisor_update_interval_min=update_interval,
                ff_weight=ATTEMPT_2_FF_WEIGHT, max_shift_k=ATTEMPT_2_MAX_SHIFT_K, deadband_k=deadband_k,
            ))
            rows.append({
                "advisor_update_interval_min": update_interval, "deadband_k": deadband_k,
                "mean_energy_saving_pct": s["mean_energy_saving_pct"],
                "median_energy_saving_pct": s["median_energy_saving_pct"],
                "n_both_better": s["n_both_better"],
                "n_worse_on_both": s["n_worse_on_both"],
                "n_comfort_equal_or_better": s["n_comfort_equal_or_better"],
                "n_energy_better": s["n_energy_better"],
                "n_scenarios": s["n_scenarios"],
            })
    return pd.DataFrame(rows)


def pick_winner_update_interval(grid: pd.DataFrame) -> tuple[int, float]:
    ranked = grid.sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING)
    top = ranked.iloc[0]
    return int(top["advisor_update_interval_min"]), float(top["deadband_k"])


if __name__ == "__main__":
    cfg = load_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    val_scenarios = held_out_val_scenarios()
    print(f"M12 attempt 2: {len(UPDATE_INTERVAL_GRID)}x{len(DEADBAND_K_GRID)} "
          f"(advisor_update_interval_min, deadband_k) grid on {len(val_scenarios)} VAL scenarios "
          f"(ff_weight={ATTEMPT_2_FF_WEIGHT}, max_shift_k={ATTEMPT_2_MAX_SHIFT_K} held fixed)...")
    grid = grid_search_update_interval(cfg, val_scenarios)
    grid.to_csv(INTERVAL_RESULTS_PATH, index=False)

    ranked = grid.sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING).reset_index(drop=True)
    print(ranked.to_string(index=False))

    update_interval, deadband_k = pick_winner_update_interval(grid)
    winner_row = ranked.iloc[0]
    print(f"\nVAL winner: advisor_update_interval_min={update_interval}, deadband_k={deadband_k}  "
          f"(mean saving {winner_row['mean_energy_saving_pct']:+.2f}%, "
          f"both-better {int(winner_row['n_both_better'])}/{int(winner_row['n_scenarios'])}, "
          f"worse-on-both {int(winner_row['n_worse_on_both'])}/{int(winner_row['n_scenarios'])})")

    print(f"\nsaved full grid to {INTERVAL_RESULTS_PATH}")

    print("\nConfirming ONCE on TEST split (never re-tuned against it)...")
    test_scenarios = held_out_test_scenarios()
    test_summary = summarize(compare(
        cfg, test_scenarios, advisor_update_interval_min=update_interval,
        ff_weight=ATTEMPT_2_FF_WEIGHT, max_shift_k=ATTEMPT_2_MAX_SHIFT_K, deadband_k=deadband_k,
    ))
    print(f"TEST: mean saving {test_summary['mean_energy_saving_pct']:+.2f}%  "
          f"median {test_summary['median_energy_saving_pct']:+.2f}%  "
          f"both-better {test_summary['n_both_better']}/{test_summary['n_scenarios']}  "
          f"worse-on-both {test_summary['n_worse_on_both']}/{test_summary['n_scenarios']}  "
          f"comfort-ok {test_summary['n_comfort_equal_or_better']}/{test_summary['n_scenarios']}")
