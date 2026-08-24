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

283 tests, all passing — the earlier 3 `xfail` (M5/M6 directional
comparisons pinned to the pre-M10 watts-dispatch law) are gone, not because
they were deleted, but because M10 Phase 2's re-tune earned honest,
currently-passing assertions in their place (see `ROADMAP.md`'s M10
section). Physics (free-float convergence, derived time constants, actuator
behaviour, numerical convergence, the M9 humidity moisture balance including
its saturation cap), timetable/occupancy invariants (M11: including
tunnel-zone wiring), feature engineering (leakage checks, batch/live
equivalence), the forecaster (chronological split, beats persistence), the
setpoint advisors and shared plant response (regression guards for two real
bugs found during development — see `docs/PARAMETERS.md`), M11's window-tint
controllers (solar-gain physics, tunnel masking, anticipatory-vs-reactive
timing),
the M5 comparison harness, the M6 actuator-lag sweep, the M7 Streamlit
demo (headless `AppTest`, including that a missing model/dataset/weather
file each produce a clean actionable error rather than a crash), the M8 serial
protocol (frame encode/decode, range validation, checksum corruption, and
a real send/receive round trip over a virtual loopback), the M9 model
benchmark's data/encoding mechanics (the full multi-model comparison
itself takes minutes and isn't run in this suite — see `ROADMAP.md`), and
M9's SHAP attribution (including a regression guard for a real, documented
LightGBM/SHAP categorical-feature incompatibility class).

## Look at the model

```bash
python -m src.config             # every derived physical quantity (UA, C, both time constants)
python -m src.occupancy          # service catalogue, station-by-station passenger profile
python -m src.train               # retrains, reports test MAE vs the persistence baseline
python -m src.compare_controllers # the M5 headline: AnticipatorySetpointAdvisor vs on/off, and a
                                   # feedforward-contribution ablation, on the held-out test split
python -m src.sweep_tau_act       # M6: re-runs both comparisons across a range of actuator lag
python -m src.plot_tau_act_sweep  # produces data/m6_tau_act_sweep.png from the sweep above
python -m src.benchmark_models    # M9: LightGBM (tuned) vs linear/random forest/XGBoost/CatBoost,
                                   # same chronological split -- needs pip install -r requirements.txt
                                   # (adds scikit-learn/xgboost/catboost, benchmark-only dependencies)
python -m src.shap_analysis       # M9: SHAP feature attribution on the test split -- produces
                                   # data/m9_shap_summary.png; needs shap (requirements.txt)
```

**No time constant is written in `config/cabin_params.yaml`**; both are
consequences of the declared physical parameters (`C_air`, `C_mass`, `UA`,
`hA`), and the physics tests check the simulator against the derivation. The
actuator lag (`tau_act`) is genuinely unknown and is never asserted either —
see M6.

## Run the live demo

```bash
streamlit run src/app.py
```

Opens both controllers running side by side on the same 23 held-out
test-split scenarios `compare_controllers.py` uses — pick a journey, scrub
or auto-play through it, compare energy and comfort. For a plain-English
walkthrough of what's on screen (not written for developers), see
`docs/USER_MANUAL.md`. If the live demo can't be shown on presentation day,
`docs/demo_recording_script.md` is the fallback recording script.

## Hardware handover (M8, reframed by M10)

This project's own compute never touches a real HVAC unit's control
electronics — doing so would void the manufacturer's warranty, so it isn't
an implementation choice this project could make differently. Instead it
sends one outbound *recommendation* over UART: a target cabin setpoint
(°C) — the same kind of input a passenger or technician could enter on the
unit's own thermostat, never a direct power/compressor command. The frame
format, every field's units and range, and exactly what the board side must
do with it (including link-loss behaviour) is fully specified in
`docs/serial_protocol.md`, written so the board can be implemented from that
document alone. `src/serial_bridge.py` is the reference sender:

```bash
python -m src.serial_bridge   # sends one frame over pyserial's built-in
                               # in-memory loopback -- no real port, no
                               # virtual-COM driver, nothing hardware-specific
```

Swap `"loop://"` for a real port name (`"COM3"`, `"/dev/ttyUSB0"`) to send
to actual hardware; nothing else about the call changes.

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
| `src/controllers.py` | `PlantResponse` (shared plant reaction), `ThermostatController` (static-schedule advisor) and `AnticipatorySetpointAdvisor` (forecast-driven advisor) |
| `src/evaluate.py` | Drives a controller through a real scenario minute by minute; scores energy + comfort |
| `src/compare_controllers.py` | The M5 comparison harness — on/off headline and a feedforward-contribution ablation |
| `src/sweep_tau_act.py`, `src/plot_tau_act_sweep.py` | M6 — sweeps the unknown actuator lag rather than asserting a value |
| `src/app.py` | M7 — Streamlit live demo, runs the same simulator with both controllers side by side |
| `src/serial_bridge.py` | M8 — the UART sender to whoever owns the board; validated against pyserial's built-in loopback |
| `src/benchmark_models.py` | M9 — LightGBM (tuned) vs linear regression, random forest, XGBoost, CatBoost, on the same chronological split |
| `src/shap_analysis.py` | M9 — SHAP feature attribution on the forecaster, verified correct (not just non-crashing) against LightGBM's categorical features |
| `tests/` | Physics (including the M9 humidity moisture balance), occupancy, features, training, controllers, the app, the serial bridge, the model benchmark, SHAP, and cross-layer integration tests |
| `ROADMAP.md` | Milestones M0–M9, objective-driven |
| `docs/PARAMETERS.md` | **Why every coefficient has the value it has, plus a corrections log** |
| `docs/serial_protocol.md` | The M8 handover document — full frame spec for the board side, no source reading required |
| `docs/USER_MANUAL.md` | Plain-English guide to running the live demo, for a non-developer audience |
| `docs/demo_recording_script.md` | Timed walkthrough script for the M7 fallback recording |

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

**M7 core complete, one manual step outstanding.** The live Streamlit demo
runs end to end, verified in a real browser and covered by headless
`AppTest` tests, including that a missing model/dataset/weather file each
produce a clean actionable error rather than a crash. **Not yet done:** the
screen-recording fallback itself still needs a person to record it —
`docs/demo_recording_script.md` has the exact walkthrough to follow.

**M8 complete.** `docs/serial_protocol.md` fully specifies the one-way
UART command frame (format, units, ranges, cadence, and required
receiver/link-loss behaviour) for whoever implements the board side, from
the document alone. `src/serial_bridge.py` is the reference sender,
validated round-trip against pyserial's built-in in-memory loopback — no
hardware or virtual-COM driver needed to verify it.

**M9 in progress.** A simplified cabin humidity state is built, tested, and
wired all the way through to the live demo (`CabinModel.step()`'s moisture
balance — no coil dehumidification, disclosed as an upper bound whenever the
AC is cooling; a saturation cap added after a real test caught %RH exceeding
100%). `src/benchmark_models.py` compared the shipped LightGBM forecaster
against linear regression, random forest, XGBoost, and CatBoost on the same
chronological split — model family barely matters here (all within a 1.6%
MAE band) — and the tuned LightGBM hyperparameters it found are adopted into
the shipped model, with the anticipatory advisor's `ff_weight` re-tuned
against the retrained model and M5/M6 regenerated to match (see ROADMAP.md's
M5 section for the pre-M10 headline number, now marked superseded — see
below). `src/shap_analysis.py` adds SHAP feature attribution on top —
verified correct against LightGBM's categorical features (not just
non-crashing), confirming the forecaster leans on forecast/trend signal
rather than current state alone. A follow-up diagnostic pass (bias/variance,
residuals, slice-based errors, a learning curve) found the forecaster
systematically under-predicts extreme heat — the same documented failure
mode as AI weather models underestimating record temperatures — and tested
two literature-grounded remedies (monotonic constraints, an explicit
saturation feature); neither held up under this project's own
val-then-test-once discipline, so **neither was adopted** — a disclosed,
known limitation rather than a papered-over one (see ROADMAP.md's M9 §4b).
Still open: the physical prototype, its own separately-scoped effort,
blocked on real hardware measurements.

**M10 complete, both phases.** The team was told the real HVAC unit's control
electronics cannot be touched or modified without voiding the manufacturer's
warranty, so the system's output was reframed: it now recommends a cabin
setpoint (°C) instead of commanding power. `ThermostatController` turned out
to already be, structurally, "setpoint → hysteresis → watts" — splitting
that mechanism into its own `PlantResponse` class gave a stand-in for the
real Control Unit this project never touches, and let `AnticipatoryController`
(renamed `AnticipatorySetpointAdvisor`) reuse the exact same forecast+blend
machinery while its final step changed from a watts dispatch to a setpoint
recommendation.

**Phase 2's honest result: parity, not a win.** Re-tuning `(ff_weight,
deadband_k)` on the VAL split, confirmed once on TEST — the same discipline
that produced the pre-M10 "+3.9%" headline — found that no configuration
beats the static baseline once the setpoint recommendation is fed through a
realistic on/off receiving unit (three attempts, 80 configurations tested,
all flat or negative on VAL). Diagnosed, not shrugged off: a bang-bang unit
has no proportional response, so a forecast-driven setpoint shift can only
move WHEN the switch fires, never HOW HARD it runs. Shipped result
(`ff_weight=0.0`, `deadband_k=1.5`): TEST mean energy delta −0.2% (median
+0.0%), 0/23 scenarios worse on both energy and comfort, aggregate
degree-hours slightly lower despite per-scenario comfort improving on only
19/23 — a disclosed null/parity finding for the forecast's contribution
under this architecture, following the same honesty this project already
applied to M9's SHAP-diagnosed, ultimately-not-adopted remedies. M4's
forecaster itself is unaffected (+21% vs. persistence, unchanged) — this is
a limit of what a bang-bang actuator can do with a good forecast, not a
forecasting problem. A fourth attempt tested whether the setpoint
recommendation updating every minute was itself the problem (spacing it out
to reduce "chatter," the same reasoning behind M8's 30 s telemetry cadence)
— confirmed the opposite: spacing it out makes things worse, since nothing
here has M8's physical-actuator/bandwidth constraint. Verified with the full
test suite and the pinned `ThermostatController` regression staying
byte-identical throughout. Full mechanism, all four tuning attempts, and
what would actually be needed for the forecast to pay off are in
ROADMAP.md's M10 section.

**M11: AI-controlled window tinting, built and demoed.** First-round judges
flagged that auto-dimming SPD glass reacting to its own sensor isn't novel;
the differentiator is anticipatory control using the same forecast lookahead
already built for HVAC. `AnticipatoryTintAdvisor` reacts to forecasted GHI
(`ghi_fcst_h`) instead of current GHI, so it clears the glass ahead of a
tunnel/shaded zone instead of only after entering it — verified directly,
not just eyeballed off a chart (`tests/test_tint_controller.py`). Tinting is
now wired into the actual solar-gain physics (`CabinInputs.tint_level`
slides effective SHGC between a clear and a fully-tinted value), so its
energy benefit is real, not cosmetic: on the demo's default scenario, energy
drops from 36.95 kWh (no tint) to 34.49–34.67 kWh (reactive/anticipatory,
roughly tied on this particular scenario). Deliberately rule-based rather
than a trained model, matching the team's own explicit scope guidance for
this milestone. Full detail in ROADMAP.md's M11 section.

**M12: the receiving Control Unit becomes a black box that converges and
holds — a proportional-integral model, replacing the bang-bang one.** A team
discussion reframed the real HVAC unit one level further: it doesn't matter
*how* it reaches a commanded setpoint, only that it does, and holds there —
unlocking a strategy the old on/off model couldn't express at all
("command 19°C to converge faster, then relax back toward 26°C"), since a
bang-bang receiver only ever asks whether a threshold was crossed, never by
how much. `PlantResponse` is now PI (`Kp=4000`, `Ki=4.0`, gains swept
empirically after an initial guess oscillated, with anti-windup checked
against the model's own real saturation bounds) — verified by trace, not
assumed: `degree_hours` on the pinned reference scenario dropped from 42.79
to 0.23 K·h. Re-tuning the advisor against the new plant took two honest
attempts: the first (re-tuning `ff_weight`/`max_shift_k`/`deadband_k` alone)
came back a genuine **regression** — comfort-ok collapsed from Phase 2's
21/23 to 2/23 — diagnosed as a cascaded-control timescale mismatch between
the advisor's every-minute updates and the new plant's own ~17-minute
settling time. The second attempt fixed it directly by slowing the
advisor's update cadence to 50 minutes, which eliminated the mismatch
entirely (0/31 scenarios worse on both energy and comfort at every interval
tested from 40 minutes up) and, confirmed once on TEST, landed almost
exactly back at Phase 2's own old-plant headline (4/23 both-better, 0/23
worse-on-both, 20/23 comfort-ok) — a disclosed parity result reached by
diagnosing the real mechanism, not by widening the same three knobs harder.
Full two-attempt narrative in ROADMAP.md's M12 section and
`src/tune_advisor_m12.py`.

**M13: station precool, closing a real gap between the architecture and the
team's own demo scenario.** Checking M12's shipped tuning against a concrete
"pull the HMI setpoint to 19°C ahead of a stop" example found that
`ff_weight=0.0` (both M10 Phase 2's and M12's own tuned result) means the
advisor is pure feedback — it never anticipates a scheduled future event,
including a station stop it already has exact timing for
(`time_to_next_station_min`). New `StationPrecoolAdvisor`: an independent,
additive, schedule-driven shift (not a variant of `ff_weight`, so it doesn't
touch M12's already TEST-confirmed result), applied to the AI arm only —
never the static baseline, preserving the exact contrast being demonstrated.
Verified on a real trace, honestly: peak cabin temperature around a boarding
stop drops 25.19°C→24.45°C (−0.73°C), at a real energy cost (+6.2% on the
scenarios checked) — a genuine, demonstrable anticipation effect, disclosed
as modest rather than oversold as the full "22 stays at 22" framing. Full
detail in ROADMAP.md's M13 section.

## License

MIT — see [LICENSE](LICENSE).
