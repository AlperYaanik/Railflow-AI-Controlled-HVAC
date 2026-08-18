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

Weather data is **not** committed — it is regenerable and `data/` is gitignored.
Fetch it once before running anything:

```bash
python -m src.weather
```

That pulls real 2024 hourly weather for Cairo and Aswan from the Open-Meteo
Archive API (free, no API key) and caches it to `data/`. Tests that need weather
skip with an explanatory message until you do.

## Verify the install

```bash
python -m pytest tests/ -q
```

81 tests. They cover the physics (free-float convergence, derived time
constants, actuator behaviour, numerical convergence), the timetable and
occupancy invariants, and the seams between the three layers.

## Look at the model

```bash
python -m src.config
```

Prints every derived quantity — heat capacities, both `UA` terms, the internal
coupling, and both thermal time constants. **No time constant is written in the
config**; they are consequences of the declared physical parameters, and the
tests check the simulator against them.

```bash
python -m src.occupancy
```

Prints the service catalogue and the station-by-station passenger profile.

## Layout

| Path | What it is |
|---|---|
| `config/cabin_params.yaml` | Every physical parameter, each with a source tag |
| `src/config.py` | Config loading and derived quantities |
| `src/weather.py` | Open-Meteo download and cache |
| `src/cabin_model.py` | Two-node cabin thermal model + HVAC actuator |
| `src/occupancy.py` | Timetable, passenger loading, door events, lookahead |
| `tests/` | Physics, occupancy and cross-layer integration tests |
| `ROADMAP.md` | Milestones M0–M8, objective-driven |
| `docs/PARAMETERS.md` | **Why every coefficient has the value it has** |

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

M0–M3 complete: repository hygiene, sourced parameters and real weather, the
cabin thermal model, and the timetable/occupancy layer. Next is the dataset and
forecasting model (M4), then the controller comparison that produces the
headline result (M5).

## License

MIT — see [LICENSE](LICENSE).
