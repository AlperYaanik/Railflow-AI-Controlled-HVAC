# Railflow-AI-Controlled-HVAC

Timetable-anticipatory, latency-compensated HVAC control for railway cabins.

A train knows something a building never can: its timetable. HVAC responds to
commands with a significant delay, so a reactive thermostat is always late — it
overheats when passengers board and overcools after they leave. This project
predicts conditions ahead of time and acts early instead.

Modelled on the **Cairo–Alexandria** intercity corridor (Egyptian National
Railways, mainline / EN 13129). Built for the Siemens Mobility Fit4Rail
programme.

## Setup

```bash
pip install -r requirements.txt
```

Nothing under `data/` or `models/` is committed — all of it is regenerable and
gitignored (weather cache, the generated dataset, the trained forecaster).
Build it once, in order, before running anything that needs the model:

```bash
python -m src.weather          # real 2024 hourly weather for Cairo/Aswan (Open-Meteo, free, no key)
python -m src.data_generator   # ~30k rows across ~190 simulated journeys
python -m src.train            # trains and saves the forecaster (a few seconds)
```

Tests and scripts that need weather/data/model skip with an actionable message
(e.g. `Run: python -m src.weather`) until the corresponding step above has run.

## Verify the install

```bash
python -m pytest tests/ -q
```

223 tests. Physics (free-float convergence, derived time constants, actuator
behaviour, numerical convergence), timetable/occupancy invariants, feature
engineering (leakage checks, batch/live equivalence), the forecaster
(chronological split, beats persistence), the two controllers (regression
guards for two real bugs found during development — see `docs/PARAMETERS.md`),
the M5 comparison harness, and the M6 actuator-lag sweep.

## Look at the model

```bash
python -m src.config             # every derived physical quantity (UA, C, both time constants)
python -m src.occupancy          # service catalogue, station-by-station passenger profile
python -m src.train               # retrains, reports test MAE vs the persistence baseline
python -m src.compare_controllers # the M5 headline: AnticipatoryController vs on/off, and a
                                   # feedforward-contribution ablation, on the held-out test split
python -m src.sweep_tau_act       # M6: re-runs both comparisons across a range of actuator lag
python -m src.plot_tau_act_sweep  # produces data/m6_tau_act_sweep.png from the sweep above
```

**No time constant is written in `config/cabin_params.yaml`**; both are
consequences of the declared physical parameters (`C_air`, `C_mass`, `UA`,
`hA`), and the physics tests check the simulator against the derivation. The
actuator lag (`tau_act`) is genuinely unknown and is never asserted either —
see M6.

## Layout

| Path | What it is |
|---|---|
| `config/cabin_params.yaml` | Every physical parameter, each with a source tag |
| `src/config.py` | Config loading and derived quantities |
| `src/weather.py` | Open-Meteo download and cache |
| `src/cabin_model.py` | Two-node cabin thermal model + HVAC actuator |
| `src/occupancy.py` | Timetable, passenger loading, door events, lookahead |
| `src/data_generator.py` | Generates the training dataset by simulating many journeys |
| `src/features.py` | Feature engineering — batch (`add_features`) and live (`LiveFeatureBuilder`), kept provably identical |
| `src/train.py` | Trains the LightGBM forecaster (T_air 30 min ahead), chronological split |
| `src/controllers.py` | `ThermostatController` (on/off baseline) and `AnticipatoryController` (forecast-driven) |
| `src/evaluate.py` | Drives a controller through a real scenario minute by minute; scores energy + comfort |
| `src/compare_controllers.py` | The M5 comparison harness — on/off headline and a feedforward-contribution ablation |
| `src/sweep_tau_act.py`, `src/plot_tau_act_sweep.py` | M6 — sweeps the unknown actuator lag rather than asserting a value |
| `tests/` | Physics, occupancy, features, training, controllers, and cross-layer integration tests |
| `ROADMAP.md` | Milestones M0–M8, objective-driven |
| `docs/PARAMETERS.md` | **Why every coefficient has the value it has, plus a corrections log** |

## What the numbers do and do not claim

There is **no real train data in this project** and none is available — Egyptian
National Railways publishes no ridership or HVAC telemetry, and EN 13129 /
EN 14750 are paywalled and publish no thermal time constants.

So:

- Results are stated as *"X% within the same comfort band on a parameterised
  cabin model"*, never as savings on a real train.
- No actuator lag is asserted. It is swept across 0–20 minutes and the
  *dependence* is reported.
- Thermal time constants are derived from declared `C` and `UA` values, not
  written down.
- Every parameter carries a confidence rating, and the lowest-confidence ones
  are shown by sensitivity analysis to be the ones that matter least.

`docs/PARAMETERS.md` includes a corrections log of what was got wrong and fixed
along the way, because the reasoning matters more than the final numbers.

## Status

**M0–M6 complete.** Repository hygiene and sourced parameters (M0–M1); the
cabin thermal model and physics tests (M2); the timetable/occupancy layer
(M3); a forecaster that beats persistence on genuinely held-out future dates
(M4); the controller comparison (M5) — a modest, honestly-reported energy
saving plus a much larger, more consistent comfort improvement, after a
first attempt that came back negative and was diagnosed and fixed; and the
actuator-lag sensitivity sweep (M6), which found the original "savings grow
monotonically with lag" hypothesis only partly true and a more useful
finding (comfort robustness to that unknown) in its place. Full results and
the reasoning behind every correction are in `ROADMAP.md` and
`docs/PARAMETERS.md`.

**Next:** M7 (a live Streamlit demo running the same simulator) and M8
(handover — the UART protocol document for the hardware side, and a
validated serial bridge).

## License

MIT — see [LICENSE](LICENSE).
