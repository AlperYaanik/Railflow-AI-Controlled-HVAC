"""M5's headline comparisons, run on every scenario in M4's held-out TEST
split -- the "X% less energy at equal comfort" numbers ROADMAP.md's M5
section reports.

TWO DIFFERENT COMPARISONS LIVE HERE, ANSWERING TWO DIFFERENT QUESTIONS. Both
share the same per-scenario loop (_compare_pair) so there is exactly one
place that knows how to run two controllers through a scenario and score
them -- a second implementation of that loop is exactly the kind of
duplicated-logic drift this project has already been bitten by twice (see
evaluate.py's and features.py's module docstrings on why run_controller/
LiveFeatureBuilder are shared code for the same reason).

  compare() -- AnticipatorySetpointAdvisor vs ThermostatController. "Does
  smart advice beat what real rail HVAC does today?" Both are evaluated
  through the SAME PlantResponse (src/controllers.py), so this isolates the
  advisor's setpoint choice, not a difference in dispatch mechanism -- see
  the M10 note in src/controllers.py. ThermostatController is an honest
  baseline (on/off with a sourced 1.5-2 K hysteresis band, not an
  unrealistically narrow strawman -- see docs/PARAMETERS.md). This is the
  deployment-relevant number -- and, since M10 Phase 2's tuning (see
  AnticipatorySetpointAdvisor's own docstring), the honest answer is
  "no better than, not worse than" -- a disclosed parity finding, not the
  pre-M10 "+3.9%" claim.

  compare_feedforward_contribution() -- the shipped AnticipatorySetpointAdvisor
  (ff_weight=0.0, itself Phase 2's tuned result) vs the SAME advisor/deadband
  with the forecast deliberately forced ON (ff_weight=0.3). "Does M4's
  forecaster earn its keep on top of what's shipped?" Re-pointed in M10
  Phase 2 to demonstrate directly that the answer is no -- turning the
  forecast on costs both energy and comfort under this architecture, a
  diagnosed consequence of PlantResponse's lack of proportional response,
  not a forecaster quality problem. An ablation, not a deployment baseline.
  NOT a real PID -- no integral or derivative term -- and must never be
  labelled as one; see its own docstring.

Reporting only whichever of the two looks better in a given moment would be
exactly the kind of selective reporting this project's honesty rules exist
to prevent. They answer different questions, so both belong in the record;
which one heads a given slide should follow from which question that slide
is asking, not from which number is larger.

WHY THE TEST SPLIT, SPECIFICALLY (both comparisons). Both controllers are
compared on the same 23 scenarios M4 never trained or validated on
(chronological, dates 2024-08-18 to 2024-08-30 -- see src/train.py's
chronological_split). Running this on train-split scenarios would let a
forecaster that memorised its training trajectories look better than it
actually is; the whole point of a held-out split is that this is the honest
number.

held_out_test_scenarios() reuses train.py's chronological_split() rather than
re-deriving "which dates are test" here -- same duplicated-logic concern as
above. It's applied directly to the raw scenario table, not to
add_features()'s output -- checked equivalent (identical test scenario_ids
either way, since every scenario here is long enough to survive
add_features()'s edge-trimming) and cheaper: no need to build the full
feature table just to ask which calendar dates are held out.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

from typing import Callable

import pandas as pd

from src.config import DATA_DIR, load_config
from src.controllers import SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN, AnticipatorySetpointAdvisor, ThermostatController
from src.evaluate import run_controller, score
from src.features import LiveFeatureBuilder
from src.train import MODEL_PATH, chronological_split

RESULTS_PATH = DATA_DIR / "m5_test_split_comparison.csv"
FEEDFORWARD_RESULTS_PATH = DATA_DIR / "m5_feedforward_contribution.csv"


def _held_out_scenarios(split_name: str) -> pd.DataFrame:
    """One row per `split_name`-split scenario_id: the exact (city, date,
    depart_hour, direction, pattern, load_factor) tuple data_generator.py
    drew for it.

    No `cfg` parameter -- which scenarios are held out depends only on
    calendar dates in scenarios_raw.parquet (see chronological_split), never
    on any physical parameter. An earlier version took `cfg` anyway and did
    nothing with it -- found during the M6-readiness audit: a parameter that
    silently has zero effect is worse than no parameter, since it invites
    the false assumption that overriding cfg (e.g. for an M6 tau_act sweep)
    would change which scenarios get evaluated.
    """
    raw = pd.read_parquet(DATA_DIR / "scenarios_raw.parquet")
    split = chronological_split(raw)
    ids = raw.loc[split == split_name, "scenario_id"].unique()

    cols = ["scenario_id", "city", "date", "depart_hour", "direction", "pattern", "load_factor"]
    return (raw[raw["scenario_id"].isin(ids)][cols]
            .drop_duplicates("scenario_id")
            .sort_values("scenario_id")
            .reset_index(drop=True))


def held_out_test_scenarios() -> pd.DataFrame:
    """The 23 TEST-split scenarios every headline M5/M6 number is reported
    against. See _held_out_scenarios() for what "held out" means here."""
    return _held_out_scenarios("test")


def held_out_val_scenarios() -> pd.DataFrame:
    """M10 Phase 2: the VAL-split scenarios used to tune
    (ff_weight, max_shift_k) -- never the TEST split, so the confirm-once-on-
    TEST step afterward is measuring generalisation, not re-reading the same
    scenarios the tuning already fit itself to. Same shape as
    held_out_test_scenarios(), same non-goal of taking a `cfg` parameter."""
    return _held_out_scenarios("val")


def _thermostat_factory(row):
    """Ignores `row` -- ThermostatController has no per-scenario state to
    match, unlike AnticipatorySetpointAdvisor's LiveFeatureBuilder below.
    Takes `row` anyway so _compare_pair can call every baseline/candidate
    factory the same way regardless of which controller it wraps."""
    def factory(model):
        return ThermostatController(model)
    return factory


def _anticipatory_factory(cfg, booster, row, **kwargs):
    """Builds a fresh LiveFeatureBuilder matched to THIS scenario's city/
    direction/pattern/load_factor/depart_hour -- not the single fixed
    combination every unit test in test_controllers.py happens to use.
    Getting this wrong would silently feed the model the wrong categorical
    features for every scenario except that one (see this milestone's audit:
    confirmed LightGBM's saved pandas_categorical mapping correctly realigns
    LiveFeatureBuilder's single-row categories, but that guarantee is
    worthless if the row itself claims the wrong city).

    `**kwargs` forwards to AnticipatorySetpointAdvisor -- e.g. ff_weight=0.0
    for compare_feedforward_contribution()'s pure-proportional baseline.
    """
    def factory(model):
        builder = LiveFeatureBuilder(
            cfg, city=row.city, direction=row.direction, pattern=row.pattern,
            load_factor=row.load_factor, depart_hour=row.depart_hour,
        )
        return AnticipatorySetpointAdvisor(model, booster, builder, **kwargs)
    return factory


def _load_booster() -> lgb.Booster:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"no saved model at {MODEL_PATH}. Run: python -m src.train")
    return lgb.Booster(model_file=str(MODEL_PATH))


RowFactory = Callable[[object], Callable[[object], object]]
"""A function of one scenario row that returns a controller_factory(model)
-- the shape both _thermostat_factory and _anticipatory_factory(...) share,
so _compare_pair can drive either (or any future controller matching this
shape) without knowing which one it's holding."""


def _compare_pair(
    cfg: dict, scenarios: pd.DataFrame,
    baseline_factory: RowFactory, candidate_factory: RowFactory,
    baseline: str, candidate: str,
    advisor_update_interval_min: int = 1,
    candidate_station_precool_advisor: object | None = None,
) -> pd.DataFrame:
    """Runs two controllers on every scenario, returns one row per scenario
    with both controllers' scores side by side and the deltas between them.

    The ONE place this loop is implemented -- compare() and
    compare_feedforward_contribution() both call this rather than each
    keeping their own copy (see the module docstring on why). `baseline`/
    `candidate` name the column stems in the returned DataFrame (e.g.
    "thermo"/"antic", or "proportional"/"anticipatory") so each comparison's
    output is self-describing rather than every column always saying
    "thermo" regardless of what's actually being compared.

    `advisor_update_interval_min` forwards to run_controller() (see its own
    docstring) -- applied identically to BOTH arms, same symmetry principle
    as sharing one PlantResponse mechanism.

    `candidate_station_precool_advisor` (M13) is DELIBERATELY asymmetric,
    unlike every other parameter here -- applied to the candidate arm only,
    never the baseline. The claim M13 exists to demonstrate is specifically
    "the AI anticipates a known stop, the dumb static schedule does not" --
    giving BOTH arms the same precool would test a different, uninteresting
    question (does precooling help regardless of who's doing it) and erase
    the exact contrast this mechanism is for.
    """
    rows = []
    for row in scenarios.itertuples():
        run_kwargs = dict(cfg=cfg, city=row.city, date=str(row.date), depart_hour=row.depart_hour,
                           direction=row.direction, pattern=row.pattern, load_factor=row.load_factor,
                           advisor_update_interval_min=advisor_update_interval_min)
        b = score(run_controller(baseline_factory(row), **run_kwargs), cfg)
        c = score(run_controller(candidate_factory(row), **run_kwargs,
                                  station_precool_advisor=candidate_station_precool_advisor), cfg)

        rows.append({
            "scenario_id": row.scenario_id, "city": row.city, "date": str(row.date),
            "depart_hour": round(row.depart_hour, 2), "direction": row.direction,
            "pattern": row.pattern, "load_factor": row.load_factor,
            f"{baseline}_energy_kwh": b.energy_kwh, f"{candidate}_energy_kwh": c.energy_kwh,
            "energy_saving_pct": 100.0 * (1.0 - c.energy_kwh / b.energy_kwh),
            f"{baseline}_degree_hours": b.degree_hours, f"{candidate}_degree_hours": c.degree_hours,
            "degree_hours_delta": c.degree_hours - b.degree_hours,
            f"{baseline}_worst_k": b.worst_excursion_k, f"{candidate}_worst_k": c.worst_excursion_k,
        })
    return pd.DataFrame(rows)


def compare(
    cfg: dict | None = None, scenarios: pd.DataFrame | None = None,
    advisor_update_interval_min: int = SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
    station_precool_advisor: object | None = None,
    **advisor_kwargs,
) -> pd.DataFrame:
    """AnticipatorySetpointAdvisor vs ThermostatController -- "does smart
    control beat what real rail HVAC does today?" See the module docstring.

    `**advisor_kwargs` forwards to AnticipatorySetpointAdvisor -- e.g.
    `ff_weight=...`, `max_shift_k=...` for M10/M12's VAL-split tuning
    (`src/tune_advisor.py`, `src/tune_advisor_m12.py`). Empty by default, so
    every existing caller keeps getting the class's own shipped defaults,
    unchanged. `advisor_update_interval_min` forwards to run_controller() --
    see its own docstring; defaults to SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN
    (M12's tuned cadence, applied to BOTH arms -- see that constant's
    docstring in src/controllers.py for why 1-minute updates fight
    PlantResponse's own PI settling time), not run_controller()'s own raw
    default of 1 -- that raw default stays 1 specifically so a standalone
    ThermostatController-only run (no comparison happening) is unaffected.

    `station_precool_advisor` (M13, default None = today's M12 behaviour,
    unchanged) forwards to the candidate (AnticipatorySetpointAdvisor) arm
    ONLY -- see _compare_pair()'s own docstring for why that asymmetry is
    the point, not an oversight.
    """
    cfg = cfg if cfg is not None else load_config()
    booster = _load_booster()
    scenarios = scenarios if scenarios is not None else held_out_test_scenarios()
    return _compare_pair(
        cfg, scenarios,
        baseline_factory=_thermostat_factory,
        candidate_factory=lambda row: _anticipatory_factory(cfg, booster, row, **advisor_kwargs),
        baseline="thermo", candidate="antic",
        advisor_update_interval_min=advisor_update_interval_min,
        candidate_station_precool_advisor=station_precool_advisor,
    )


def compare_feedforward_contribution(
    cfg: dict | None = None, scenarios: pd.DataFrame | None = None,
    advisor_update_interval_min: int = SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN,
) -> pd.DataFrame:
    """The shipped AnticipatorySetpointAdvisor (ff_weight=0.0 -- itself
    Phase 2's own VAL-grid/TEST-once tuned result, not an arbitrary choice;
    see the class's docstring in src/controllers.py) vs the SAME advisor and
    deadband with the forecast deliberately forced ON (ff_weight=0.3, the
    least-bad nonzero point that same grid found). "Does M4's forecaster
    earn its keep on top of what's actually shipped?" A different question
    from compare(): not "beats today's rail HVAC" but "the forecast
    specifically is worth having, here" -- re-pointed in M10 Phase 2 (it
    used to compare the shipped, forecast-using controller against a
    forced-proportional baseline; today the shipped controller IS the
    proportional one, so this now demonstrates directly, rather than merely
    asserts, that deliberately turning the forecast back on makes things
    worse under this architecture -- see ROADMAP.md's M10 section.

    `advisor_update_interval_min` defaults to SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN
    (M12), applied to BOTH arms -- see tests/test_compare_controllers.py's
    test_forcing_the_forecast_on_does_not_help for whether the ff_weight=0.3
    finding still holds at this slower cadence (originally found only at the
    fast, every-minute cadence -- not assumed to carry over unchecked).

    NOT A PID CONTROLLER. Neither arm has an integral or derivative term.
    Calling this "PID" on a slide would be the same kind of unearned claim
    this project already renamed once (see controllers.py: "Smith
    predictor" -> "learned-model anticipatory controller"). A real P+I+D
    baseline is a separate, not-yet-built controller.
    """
    cfg = cfg if cfg is not None else load_config()
    booster = _load_booster()
    scenarios = scenarios if scenarios is not None else held_out_test_scenarios()
    return _compare_pair(
        cfg, scenarios,
        baseline_factory=lambda row: _anticipatory_factory(cfg, booster, row),
        candidate_factory=lambda row: _anticipatory_factory(cfg, booster, row, ff_weight=0.3),
        baseline="shipped", candidate="with_forecast",
        advisor_update_interval_min=advisor_update_interval_min,
    )


def summarize(results: pd.DataFrame, baseline: str = "thermo", candidate: str = "antic") -> dict:
    """Aggregate stats behind a headline number -- distribution, not just a
    mean, because a single average can hide the exact policy-dependence risk
    M4's own audit already found once (docs/PARAMETERS.md).

    `baseline`/`candidate` must match the column stems in `results` (see
    _compare_pair) -- defaults suit compare()'s output; pass
    baseline="shipped", candidate="with_forecast" for
    compare_feedforward_contribution()'s output instead. The OUTPUT dict's
    keys are fixed regardless of which comparison produced `results`, so a
    caller doesn't need to know which one it's holding.
    """
    dh_b, dh_c = f"{baseline}_degree_hours", f"{candidate}_degree_hours"
    comfort_ok = results[dh_c] <= results[dh_b] + 1e-9
    energy_better = results["energy_saving_pct"] > 0
    return {
        "n_scenarios": len(results),
        "mean_energy_saving_pct": float(results["energy_saving_pct"].mean()),
        "median_energy_saving_pct": float(results["energy_saving_pct"].median()),
        "std_energy_saving_pct": float(results["energy_saving_pct"].std()),
        "min_energy_saving_pct": float(results["energy_saving_pct"].min()),
        "max_energy_saving_pct": float(results["energy_saving_pct"].max()),
        "n_energy_better": int(energy_better.sum()),
        "n_comfort_equal_or_better": int(comfort_ok.sum()),
        "n_both_better": int((energy_better & comfort_ok).sum()),
        "n_worse_on_both": int((~energy_better & ~comfort_ok).sum()),
        "mean_degree_hours_delta": float(results["degree_hours_delta"].mean()),
        "total_baseline_degree_hours": float(results[dh_b].sum()),
        "total_candidate_degree_hours": float(results[dh_c].sum()),
    }


def _print_summary(label: str, s: dict, baseline_name: str, candidate_name: str) -> None:
    print(f"{label} -- {s['n_scenarios']} held-out test-split scenarios "
          f"(2024-08-18 to 2024-08-30, both cities, both patterns/directions, all load factors)")
    print(f"  energy saving  mean {s['mean_energy_saving_pct']:+.1f}%   "
          f"median {s['median_energy_saving_pct']:+.1f}%   "
          f"std {s['std_energy_saving_pct']:.1f}pp   "
          f"range [{s['min_energy_saving_pct']:+.1f}%, {s['max_energy_saving_pct']:+.1f}%]")
    print(f"  scenarios with any energy saving        : {s['n_energy_better']}/{s['n_scenarios']}")
    print(f"  scenarios with equal-or-better comfort  : {s['n_comfort_equal_or_better']}/{s['n_scenarios']}")
    print(f"  scenarios better on BOTH energy+comfort : {s['n_both_better']}/{s['n_scenarios']}")
    print(f"  scenarios worse on BOTH                 : {s['n_worse_on_both']}/{s['n_scenarios']}")
    print(f"  total degree-hours: {baseline_name} {s['total_baseline_degree_hours']:.2f} K*h  "
          f"vs {candidate_name} {s['total_candidate_degree_hours']:.2f} K*h")
    print()


if __name__ == "__main__":
    cfg = load_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = compare(cfg)
    results.to_csv(RESULTS_PATH, index=False)
    _print_summary("M5 comparison 1/2: Anticipatory vs ThermostatController (on/off)",
                    summarize(results), "thermostat", "anticipatory")
    print(f"saved to {RESULTS_PATH}")
    print()

    ff_results = compare_feedforward_contribution(cfg)
    ff_results.to_csv(FEEDFORWARD_RESULTS_PATH, index=False)
    _print_summary("M5 comparison 2/2: shipped (ff_weight=0) vs forecast forced ON (NOT a real PID -- see docstring)",
                    summarize(ff_results, baseline="shipped", candidate="with_forecast"),
                    "shipped", "with_forecast")
    print(f"saved to {FEEDFORWARD_RESULTS_PATH}")
