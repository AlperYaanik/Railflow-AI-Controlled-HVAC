"""M10 Phase 2: re-tunes AnticipatorySetpointAdvisor for the new
setpoint-recommendation law -- the replacement for the retired
(gain_k, ff_weight) tune this project used twice before (M5, then M9 after
the forecaster's own retune). Same discipline both times used: a grid
searched on VAL, ranked by n_both_better (scenarios strictly better on both
energy AND comfort) rather than mean energy saving alone -- chasing the mean
rewards a few large-swing scenarios and can hide a less robust result, the
exact failure mode this project's own M5 correction log already documents --
then the winning point confirmed ONCE on TEST, never re-tuned against it.

ATTEMPT 1 (ff_weight x max_shift_k, deadband reusing thermostat_hysteresis_k
throughout, same as Phase 1 shipped) CAME BACK NEGATIVE OR FLAT AT EVERY ONE
OF 25 POINTS -- not a near-miss, a real structural problem, diagnosed rather
than shrugged off. At ff_weight=0.0 the result was EXACTLY IDENTICAL to
ThermostatController (0.000000% saving, 0/31 both-better, every
max_shift_k). Traced one scenario minute by minute: advised_setpoint_c
differs from the static schedule on 125/166 minutes (by up to ~2.9 C), yet
cmd_w and t_air_c come out bit-identical to ThermostatController throughout.
Mechanism: the advisor's own deadband and PlantResponse's switching
half-band were both `thermostat_hysteresis_k / 2` -- the same number reused
to avoid inventing a second unsourced threshold (this project's usual
preference) -- so a shift only ever became nonzero at the exact moment
|current_error| already exceeded the threshold PlantResponse itself
switches on under the STATIC setpoint. The shift only ever confirmed a
decision PlantResponse was already making, never pre-empted one. Above
ff_weight=0.0 it got worse, not better, monotonically (ff_weight=1.0: 25/31
scenarios worse on both energy and comfort) -- consistent with a bang-bang
receiver that has no proportional response: nudging its switch point either
does nothing (see above) or moves it in a way that adds runtime without a
matching benefit, since PlantResponse commits to full capacity the instant
it switches regardless of how far past the threshold the setpoint sits.

ATTEMPT 2 decoupled the advisor's deadband from PlantResponse's via a new
`deadband_k` parameter (src/controllers.py), on the theory that a narrower
deadband would let a forecast-driven shift move BEFORE PlantResponse's own
switching point rather than only ever confirming it after the fact. Grid:
ff_weight in {0.0, 0.3, 0.6, 0.9, 1.0} x deadband_k in
{0.0, 0.25, 0.5, 0.75, 1.0}. STILL NEGATIVE OR FLAT EVERYWHERE, but real
movement: at ff_weight=0.0, mean saving improved from an exact 0.000000% (a
mathematical identity with ThermostatController, attempt 1) to -0.11% at
deadband_k=1.0 with both-better UP to 10/31 (vs. 0/31) -- and, checked
carefully rather than assumed, the trend across deadband_k=0.00..1.00 ran
the OPPOSITE direction from the original hypothesis: WIDER did better than
narrower, not narrower better than wider (deadband_k=1.00: -0.11%/10-better;
0.75: -0.20%/6; 0.50: -0.26%/7; 0.25: -0.30%/8; 0.00: -0.32%/7). (Also: the
"deadband_k=1.0 should reproduce attempt 1's baseline-identical result" note
in an earlier version of this docstring was WRONG -- deadband_k is compared
against thermostat_hysteresis_k directly, i.e. deadband_k=2.0 reproduces the
default, not deadband_k=1.0=half of it. Caught rechecking the numbers, not
assumed correct.)

Re-diagnosed from this: at ff_weight=0, the advisor's only signal is
current_error -- information PlantResponse's own hysteresis already reacts
to natively. A narrow deadband just re-derives a noisier, more twitchy
version of a decision PlantResponse was going to make anyway (more spurious
setpoint perturbation, not more genuine anticipation), so narrower making
things WORSE is consistent with the same "never stops nudging" failure mode
the original (pre-M10) deadband was built to fix. Mathematically, as
deadband_k -> infinity the advisor's shift is always deadbanded to exactly
0, converging to an EXACT identity with ThermostatController (0.00%) from
below -- so within [0, 1.0] the curve is approaching, not exceeding, parity,
and attempt 2's grid never actually tested wide enough to see whether it
keeps improving past that, plateaus, or crosses into positive territory
before converging back to 0.

ATTEMPT 3 (this file, current) tests two things attempt 2 didn't:
deadband_k values well past the original default (thermostat_hysteresis_k=
2.0), to see the actual shape of that curve rather than assuming it from
five points that never bracketed it; and ff_weight values between 0 and 0.3
(both attempts only ever tested 0.0 and 0.3, a wide gap, given every value
>=0.3 tested so far was clearly worse). The deadband is also now read as a
CONFIDENCE FILTER, not just a noise floor: a wide deadband combined with
some nonzero ff_weight only lets a LARGE, forecast-driven signal through
(small current-error nudges get zeroed same as before), which the narrow-
deadband grid in attempt 2 never isolated from the noisy small-signal case.
`max_shift_k` still held fixed at 4.0 (attempt 1 showed it barely matters
once >= ~1-2 C, since the final result clamps to the 4.0-C-wide sliding-
setpoint envelope regardless).

WHY VAL, NEVER TEST, FOR THE SEARCH ITSELF. held_out_val_scenarios()
(src/compare_controllers.py) returns the VAL-split scenarios -- dates
M4's forecaster was validated (early-stopped) against but this controller
tuning has never touched. Searching on VAL and confirming once on TEST is
what makes the confirmation step meaningful: if the grid were searched on
TEST directly, "confirmed on TEST" would just be restating the number the
search already picked to maximise, the same leakage the chronological
train/val/test split exists to prevent everywhere else in this project.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pandas as pd

from src.compare_controllers import compare, held_out_test_scenarios, held_out_val_scenarios, summarize
from src.config import DATA_DIR, load_config

FF_WEIGHT_GRID = (0.0, 0.075, 0.15, 0.225, 0.3)
DEADBAND_K_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
MAX_SHIFT_K = 4.0
"""Held fixed -- see module docstring. Attempt 1's grid showed this barely
matters once >= ~1-2 C, since the result is clamped to the 4.0-C-wide
sliding-setpoint envelope regardless."""

SHIPPED_FF_WEIGHT = 0.0
SHIPPED_DEADBAND_K = 1.5

UPDATE_INTERVAL_GRID_MIN = (1, 2, 5, 10, 15, 20)
"""ATTEMPT 4 -- a dimension none of attempts 1-3 ever varied: how often the
EFFECTIVE (PlantResponse-facing) setpoint updates, as opposed to holding the
last recommendation fixed for a few minutes at a time (src/evaluate.py's
`advisor_update_interval_min`, added for exactly this test). Prompted by a
real, well-grounded question, independent of this project's own reasoning:
does recomputing every single minute just add chatter a bang-bang receiver
can't usefully react to before the next change arrives -- structurally the
same argument that already set M8's 30 s telemetry cadence
(docs/serial_protocol.md Sec 5: the actuator's own lag, not the link's
update rate, is the real bottleneck). Range chosen around the actuator's own
assumed response (`dead_time_min` + `tau_act_min` ~= 2 + 5 = 7 min,
`config/cabin_params.yaml`): 1 (today's shipped behaviour, included as the
reference point) up to 20 (comfortably under the 30 min forecast horizon,
matching M6's own tau_act sweep ceiling for the same reason).

Held fixed at the shipped (ff_weight, deadband_k) point while sweeping this,
deliberately -- isolates whether update cadence alone moves the needle
before spending compute on a full re-cross of all three dimensions.

RESULT: spacing it out makes things WORSE, not better, and not by a little
-- interval=1 (today's shipped default) is the clear VAL winner (mean
-0.04%, 10/31 both-better); every wider interval is worse, bottoming out at
interval=20 (mean -1.90%, only 2/31 both-better, 15/31 worse-on-both -- the
single worst point found across all four tuning attempts). Confirmed on
TEST at interval=1: mean -0.17%, median +0.00%, 4/23 both-better, 0/23
worse-on-both -- consistent with attempts 1-3's shipped-point numbers, as
it should be, since interval=1 IS the shipped default.

WHY THE M8 ANALOGY DOESN'T TRANSFER, DIAGNOSED RATHER THAN LEFT AS A
SURPRISE: M8's 30 s cadence is about how often a value needs to CROSS A
COMMUNICATION LINK to a physically slow actuator -- sending faster than the
actuator can respond wastes bandwidth, not compute. Here there is no link
and no bandwidth cost: recomputing the recommendation every minute is pure
arithmetic, and PlantResponse's own hysteresis (not the advisor's update
rate) is what already prevents actuator chatter -- that's what the
deadband and hysteresis band are FOR. Holding the recommendation fixed
doesn't reduce chatter PlantResponse wasn't going to have anyway; it only
makes the recommendation stale relative to the cabin's actual, continuously
evolving state, which the current_error term is specifically there to
track. Staleness has a cost and no offsetting benefit here -- confirmed
empirically, not merely argued.
"""

RESULTS_PATH = DATA_DIR / "m10_advisor_tune.csv"
UPDATE_INTERVAL_RESULTS_PATH = DATA_DIR / "m10_advisor_update_interval_sweep.csv"

_RANK_COLUMNS = ["n_both_better", "mean_energy_saving_pct"]
_RANK_ASCENDING = [False, False]
"""n_both_better first (this project's standing discipline -- see module
docstring), mean_energy_saving_pct as the tiebreaker. Both descending: more
both-better scenarios wins, then higher mean saving wins."""


def grid_search(cfg: dict | None = None, scenarios: pd.DataFrame | None = None) -> pd.DataFrame:
    """Runs compare() at every (ff_weight, deadband_k) grid point on the
    given scenarios (VAL by default), one row per point."""
    cfg = cfg if cfg is not None else load_config()
    scenarios = scenarios if scenarios is not None else held_out_val_scenarios()

    rows = []
    for ff_weight in FF_WEIGHT_GRID:
        for deadband_k in DEADBAND_K_GRID:
            s = summarize(compare(
                cfg, scenarios,
                ff_weight=ff_weight, max_shift_k=MAX_SHIFT_K, deadband_k=deadband_k,
            ))
            rows.append({
                "ff_weight": ff_weight, "deadband_k": deadband_k,
                "mean_energy_saving_pct": s["mean_energy_saving_pct"],
                "median_energy_saving_pct": s["median_energy_saving_pct"],
                "n_both_better": s["n_both_better"],
                "n_worse_on_both": s["n_worse_on_both"],
                "n_comfort_equal_or_better": s["n_comfort_equal_or_better"],
                "n_energy_better": s["n_energy_better"],
                "n_scenarios": s["n_scenarios"],
            })
    return pd.DataFrame(rows)


def pick_winner(grid: pd.DataFrame) -> tuple[float, float]:
    ranked = grid.sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING)
    top = ranked.iloc[0]
    return float(top["ff_weight"]), float(top["deadband_k"])


def sweep_update_interval(cfg: dict | None = None, scenarios: pd.DataFrame | None = None) -> pd.DataFrame:
    """ATTEMPT 4 -- see UPDATE_INTERVAL_GRID_MIN's docstring. Holds the
    shipped (ff_weight, deadband_k) fixed, sweeps advisor_update_interval_min
    alone on the given scenarios (VAL by default)."""
    cfg = cfg if cfg is not None else load_config()
    scenarios = scenarios if scenarios is not None else held_out_val_scenarios()

    rows = []
    for interval in UPDATE_INTERVAL_GRID_MIN:
        s = summarize(compare(
            cfg, scenarios,
            ff_weight=SHIPPED_FF_WEIGHT, max_shift_k=MAX_SHIFT_K, deadband_k=SHIPPED_DEADBAND_K,
            advisor_update_interval_min=interval,
        ))
        rows.append({
            "advisor_update_interval_min": interval,
            "mean_energy_saving_pct": s["mean_energy_saving_pct"],
            "median_energy_saving_pct": s["median_energy_saving_pct"],
            "n_both_better": s["n_both_better"],
            "n_worse_on_both": s["n_worse_on_both"],
            "n_comfort_equal_or_better": s["n_comfort_equal_or_better"],
            "n_energy_better": s["n_energy_better"],
            "n_scenarios": s["n_scenarios"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config()

    val_scenarios = held_out_val_scenarios()
    print(f"M10 Phase 2, attempt 3: {len(FF_WEIGHT_GRID)}x{len(DEADBAND_K_GRID)} "
          f"(ff_weight, deadband_k) grid on {len(val_scenarios)} VAL scenarios "
          f"({val_scenarios['date'].min()}..{val_scenarios['date'].max()}), "
          f"max_shift_k held at {MAX_SHIFT_K}...")
    grid = grid_search(cfg, val_scenarios)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULTS_PATH, index=False)

    ranked = grid.sort_values(_RANK_COLUMNS, ascending=_RANK_ASCENDING).reset_index(drop=True)
    print(ranked.to_string(index=False))

    ff_weight, deadband_k = pick_winner(grid)
    winner_row = ranked.iloc[0]
    print(f"\nVAL winner: ff_weight={ff_weight}, deadband_k={deadband_k}  "
          f"(mean saving {winner_row['mean_energy_saving_pct']:+.1f}%, "
          f"both-better {int(winner_row['n_both_better'])}/{int(winner_row['n_scenarios'])}, "
          f"worse-on-both {int(winner_row['n_worse_on_both'])}/{int(winner_row['n_scenarios'])})")

    print(f"\nsaved full grid to {RESULTS_PATH}")

    print("\nConfirming ONCE on TEST split (never re-tuned against it)...")
    test_scenarios = held_out_test_scenarios()
    test_summary = summarize(compare(
        cfg, test_scenarios, ff_weight=ff_weight, max_shift_k=MAX_SHIFT_K, deadband_k=deadband_k,
    ))
    print(f"TEST: mean saving {test_summary['mean_energy_saving_pct']:+.1f}%  "
          f"median {test_summary['median_energy_saving_pct']:+.1f}%  "
          f"both-better {test_summary['n_both_better']}/{test_summary['n_scenarios']}  "
          f"worse-on-both {test_summary['n_worse_on_both']}/{test_summary['n_scenarios']}  "
          f"comfort-ok {test_summary['n_comfort_equal_or_better']}/{test_summary['n_scenarios']}")
