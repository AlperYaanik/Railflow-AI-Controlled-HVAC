# Railflow — Roadmap

Objective-driven plan for a railway cabin HVAC control prototype.
Organised by **what must be true**, not by calendar days.

---

## Guiding principle: small scope, honest foundations

Two axes are often confused. They are independent:

- **Scope** — how much we build. Kept deliberately small.
- **Quality** — whether what we build is real. Kept high.

We optimise for *small scope + honest foundations*. The failure mode we are explicitly avoiding is **small scope + faked foundations** — hardcoded numbers, invented data, a chart shaped by hand to look convincing. That version is faster for one afternoon and worthless the day after the presentation, because it cannot be extended.

The demo audience has ~20 minutes and will not inspect deeply. That constrains the **presentation surface**, not the foundations. This project is intended to continue after the presentation, so the foundations are built to be extended.

**Total effort: ~13 hours.**

---

## The idea in one sentence

> **Timetable-anticipatory, latency-compensated HVAC control for railway cabins.**

A building's HVAC cannot know its future occupancy. A train knows its timetable — it knows that in four minutes, 200 people board at the next station. Because HVAC responds to commands with a significant delay, a reactive thermostat is always late: it overheats when passengers board and overcools after they leave. Acting on *predicted* conditions instead of *current* ones removes that lag.

---

## Non-goals

Listed explicitly, because keeping scope small requires naming what we are not doing:

- ❌ **EnergyPlus / Sinergym** — building simulators; no train model exists, and the physics (thermal mass, occupant density, door cycles) is wrong for a rail cabin
- ❌ **Modelica / BOPTEST** — toolchain cost exceeds the entire project budget
- ❌ **Reinforcement learning** — the problem is forecasting + control, not policy search
- ❌ **Transfer learning from building HVAC** — transfers the easy part (weather response), none of the hard part (passenger dynamics), and with no real data the target domain would also be synthetic
- ❌ **Deep learning** — a ~30k-row tabular problem. Gradient boosting is faster, more accurate here, and trains in seconds on CPU
- ❌ **Cloud training** — nothing here needs a GPU. One local environment, the same machine that runs the demo
- ❌ **Hardware debugging** — the board is owned by someone else; our scope ends at a documented protocol

---

## Milestones

### M0 — The repository is not lying to itself
**Effort:** 0.5h

Two existing defects to fix before anything else:

- `.gitignore` contains a bare `*.ipynb`, which ignores the entire `notebooks/` folder. Notebooks currently **cannot be committed**. The intent was almost certainly `.ipynb_checkpoints/`.
- `notebooks/02_preprocesing.ipynb` is misspelled → `02_preprocessing.ipynb`.

Also: populate the empty `requirements.txt`. Inference-time dependencies stay minimal (`lightgbm`, `pandas`, `numpy`, `pyyaml`, `streamlit`, `pyserial`).

**Done when:** `git status` shows notebooks as trackable, and a clean clone installs from `requirements.txt` without error.

---

### M1 — Every physical parameter has a source
**Effort:** 2h

The credibility of every downstream number rests here.

**Build:**
- `config/cabin_params.yaml` — geometry, effective thermal capacity `C_eff`, envelope `UA`, fresh-air rate, HVAC capacity, COP curve, comfort band. **Every value carries a source comment** (EN standard, datasheet, or a stated geometric assumption).
- `src/weather.py` — real historical weather from the Open-Meteo Archive API (free, no API key): temperature, relative humidity, global horizontal irradiance, wind. Cached to `data/`.

**Decided — vehicle class and operating context.** Mainline / regional (**EN 13129**), Egyptian National Railways, modelled on the Cairo–Alexandria corridor (~208 km, real station list). Long runs, all-seated, few stops. The anticipatory-control effect therefore leans on **large boarding/alighting events at major stations** and on **solar and weather ramps over a long journey**, rather than on frequent door cycles.

**Egypt is a cooling-dominated climate**, so summer is the design case and heating is close to irrelevant. Two locations are fetched: **Cairo** (network hub, realistic operating point) and **Aswan** (thermal stress case).

*`UIC 553-1` specifies climatic chamber testing at −25 °C and **+45 °C**. Worth noting: the measured 2024 data reaches **46.4 °C in Cairo and 48.1 °C in Aswan** — real conditions on this corridor exceed the standard's upper test bound. That is a legitimate and quotable observation about why HVAC control margin matters here.*

**Critical rule:** the thermal time constant is **never written into the config**. It is *derived* (see M2).

**Done when:** the config is complete, every line has a justification, and real weather CSVs are on disk.

---

### M2 — A cabin simulator you can trust
**Effort:** 3h — *the core of the project*

**Build:** `src/cabin_model.py` — a **two-node (2R2C)** grey-box network:

```
C_air  · dT_air/dt  = Q_pass + Q_solar + Q_trans + Q_inf + Q_door + Q_hvac + hA·(T_mass − T_air)
C_mass · dT_mass/dt = hA·(T_air − T_mass)
```

**Why two nodes rather than one lumped node.** Tested before building, not assumed: lumping the interior mass into the air node means 68 boarding passengers heat ~3.0 MJ/K instead of 0.18 MJ/K. Under an **isothermal** boarding test (outdoor equal to cabin start, so passengers are the only forcing), the two-node model gives a **2.12 K** air rise over an 8-minute dwell against **0.71 K** for a single node — **3.0× at the end of the dwell, 8.6× in the first two minutes**. Against a ±2 K comfort band, the two-node disturbance is worth controlling and the single-node one sits well inside the band. A lumped model would put the project's central effect below the threshold that matters. The extra cost is one parameter (`hA`) and ~30 lines.

> **A confounded test, caught on review.** The first version of this comparison ran `T_out = 40 °C` against a 24 °C cabin and reported a 9.65 K "boarding spike." At those conditions the outdoor gradient supplied 15.7 kW against the passengers' 4.8 kW — **3.3× more heat than the thing being measured**. The ratio between topologies happened to survive (both models see the same forcing), but the headline number was mostly envelope gain wearing a passenger costume. The isothermal figures above are the honest ones.

| Term | Basis |
|---|---|
| `Q_pass` | passenger count × (sensible + latent metabolic load) |
| `Q_trans` | `UA · (T_out − T_in)` |
| `Q_solar` | measured GHI × glazing area × SHGC |
| `Q_inf` | fresh-air mass flow × `c_p` × ΔT |
| `Q_door` | step infiltration during station dwell |
| `Q_hvac` | actuator output — dead time, first-order lag, capacity limit, COP(`T_out`) |

**On the time constant — the most important design decision in this project.**

There is no publicly citable time constant for a rail cabin. EN 13129 and EN 14750 are paywalled and publish none. Therefore we never assert one. In an RC model it is not a free parameter — it is a consequence. For the two-node network, the two time constants are `−1/λ` for the eigenvalues of the 2×2 state matrix:

```
dT_air/dt  = −(UA + hA)/C_air  · T_air + hA/C_air  · T_mass
dT_mass/dt =        hA/C_mass  · T_air − hA/C_mass · T_mass
```

We assert `C_air`, `C_mass`, `UA`, and `hA` (each individually defensible from geometry and materials); both τ fall out. Every input can be challenged separately, and the answer to "where does that number come from?" is a derivation rather than a guess.

Derived values: **τ_fast ≈ 1.2–1.3 min** (air), **τ_slow ≈ 78–97 min** (whole cabin), varying with occupancy.

> **A shortcut that was tried and rejected.** The familiar single-node form `C_eff / UA` is only valid when `hA ≫ UA`. Here `hA ≈ 1500 W/K` against `UA ≈ 740–1040 W/K`, so it materially under-reports the slow constant. The physics test in M2 caught this by comparing the simulator against the derived value — exactly what that test exists for. The lesson generalises: a derivation is only worth its credibility if something checks it.

**On the least defensible parameter.** `C_mass` is an `[ASSUMPTION]` built from a mass budget (4.34 MJ/K if every kilogram coupled; we take ~65%). Rather than defend the number, we measured how much it matters: sweeping it across **1.6–4.34 MJ/K** moves τ_slow from 45 to 120 min but changes τ_fast by **<3%**, the boarding spike by **<10%**, and pull-down time **not at all**. Everything the controller sees is governed by `C_air` and `hA`. A regression test locks this in, so the weakest input is demonstrably the one that matters least.

**Sanity against the real world.** `hA` rests on 6 W/m²K, mid-range for indoor natural convection (2–10) and conservative given forced HVAC circulation. Sweeping it 2–15 W/m²K moves τ_fast between 0.6 and 1.9 min — the two-node conclusion holds across the entire plausible range, and *strengthens* at the low end. Stated precisely: the boarding disturbance spans 1.4–3.2 K across that sweep against a ±2 K band, so it is *comparable to* the band and exceeds it below roughly `h = 8` — it is not unconditionally larger, and the presentation should not claim it is. The 40 kW cooling capacity sits at the top of the commercial range for a single coach (units are typically 23–32 kW, up to ~40 kW for high-capacity stock), which is the right place for a 46–48 °C design condition.

**Full rationale for every coefficient — including the sweeps, the rejected alternatives, a calibration priority list, and a corrections log — is in [`docs/PARAMETERS.md`](docs/PARAMETERS.md).**

The **actuator** lag (`tau_act` — compressor spin-up, damper travel, coil inertia) is genuinely unknown and is handled separately in M6.

**Done when** the physics sanity tests pass (`tests/test_physics.py`, 20 tests):
- **Free-float:** with HVAC off the cabin tends to `T_out` from both above and below; at equilibrium nothing drifts
- **Linearity:** doubling passengers doubles the sensible load; solar is linear in GHI and uses only the sunlit glazing fraction
- **Derived time constants:** the simulated 63% step-response time matches `system_time_constants()` within 15%, and the fast air mode is ≥5× faster than the slow mode
- **Boarding spike:** 68 passengers over an 8-min dwell move air temperature >5 K, with air leading the interior mass
- **Actuator:** capacity clamps, dead time delays delivery, the lag converges first-order, and `tau_act = 0` degenerates to instant delivery
- **Energetics:** COP falls with `T_out` and honours its floor; electrical draw is coil load over COP; zero draw when off
- **Numerical health:** a genuine convergence study against a 0.25 s reference (the 10 s substep costs 0.017 K over an hour of hard forcing), call-granularity consistency, and finite state under extreme forcing
- **Robustness:** results are insensitive to `C_mass`, and a single-node model is shown to understate the boarding spike — so the topology choice cannot be silently undone later

The time-constant test is the one that matters most: if the simulator disagrees with the derivation, either the model or the config is wrong. It earned its place immediately — it is what exposed the 29% error in the single-node shortcut.

---

### M3 — The cabin behaves like it is on a railway
**Effort:** 2h

**Build:** `src/occupancy.py` — station list, dwell times, boarding/alighting with morning and evening peaks, door-opening infiltration events.

**Deliberately excluded:** speed-induced infiltration, tunnel effects, solar orientation factors. These add fidelity that no reviewer will check, and cost hours.

**Modelled as a service, not a time-of-day curve.** A metro runs continuously and its loading varies through the day; an intercity train is a discrete run whose loading is a property of *which service it is*. So load factor scales a whole run. The dataset uses both directions × three load factors (0.4 / 0.7 / 1.0).

**Done when:** 39 occupancy tests pass — the route empties at the terminus and occupancy stays non-negative at every load factor including 0 and 1.3; the base pattern reproduces the config exactly at load factor 1.0; doors open only while stopped; and the lookahead features lead the boarding event they describe (without which the controller has nothing to anticipate).

> **What M3 found.** Two problems, both now recorded in [`docs/PARAMETERS.md`](docs/PARAMETERS.md). First, door infiltration was expressed as a per-stop total divided by the route's mean dwell — so editing one station's timetable moved `door_ua` by 36% at *every* station. Fixed by restating it as a rate. Second, and more important: **the bang-bang baseline's own oscillation is larger than the station disturbances it faces** (72 of 166 minutes outside the comfort band, against a worst station excursion of +1.73 K). That traces to the actuator having no supply-air rate limit. It is written up as an open issue that must be settled before M5, because comparing against a baseline that oscillates for a modelling reason rather than a physical one would be a straw man.

---

### M4 — A forecaster that provably beats doing nothing
**Effort:** 3h

**Build:**
- `src/data_generator.py` — scenario loop over seasons × occupancy profiles × stochastic setpoint policies. Target **~30k rows** (this saturates a ~20-feature tabular problem; more is wasted time).
- `src/features.py` — lagged commands `u(t-1..t-k)`, command EWMA (half-life ≈ τ), cyclical time encoding, and the **lookahead features that are the entire point**: time to next station, expected boarding count, and the +30 min weather outlook.
- `src/train.py` — LightGBM predicting `T_in` at a single horizon `t+H`, where `H` is the control horizon.

**Non-negotiable:** **chronological** train/validation/test split. A random split leaks future information through the lag features and produces a meaningless score.

**Done when:** test MAE beats a persistence baseline (`T_in(t+H) = T_in(t)`). If it does not, the lookahead features are not wired in correctly — that is the first place to look.

---

### M5 — The result
**Effort:** 2h

**Build:** `src/controllers.py` and `src/evaluate.py`
- **Baseline:** on/off thermostat (staged on/off is common in real rail HVAC, so this is an honest comparison, not a straw man)
- **Proposed:** latency-compensated predictive control — acts on the *predicted* `T_in` at `t+H` rather than the current value

**Metrics:** energy (kWh) and comfort (degree-minutes outside band). Evaluated on a **held-out weather period and an unseen occupancy profile.**

**Done when:** you can state one sentence — *"X% less energy at equal comfort"* — and point to the plot behind it.

---

### M6 — Honest about what we don't know
**Effort:** 1h — *highest value per hour in the plan*

`tau_act` is unknown. Rather than picking a number, **sweep it**: `tau_act ∈ {0, 2, 5, 10, 20} min`, re-running the M5 evaluation for each.

The claim becomes:

> Energy saving from anticipatory control increases monotonically with actuator lag. At `tau_act = 0` the controller correctly degenerates to the baseline. We do not claim a specific lag for any vehicle — we characterise the dependence on it.

Two reasons this earns its hour:
- **It is also a correctness test.** If the predictive controller does *not* converge to baseline behaviour at zero lag, there is a bug, not a result.
- **It is Q&A armour.** "How fast does the HVAC actually respond?" is a likely question from a rail engineer, and it is the one question that could visibly stall the presentation. A fifteen-second answer — *"I don't claim a value; here is the dependence"* — turns the weakest point into the moment the work looks most careful.

**Done when:** a single sweep plot exists — saving vs `tau_act`.

---

### M7 — Something to show in 20 minutes
**Effort:** 2h

**Build:** `src/app.py` — Streamlit, running the **same simulator** live with both controllers side by side.

No separate demo infrastructure: the simulator is the data generator, the evaluation harness, *and* the demo engine. One artifact, three uses.

**Guardrail:** build the static comparison plot **first** — it is the guaranteed fallback and takes minutes. Upgrade to the live dashboard only once it exists. If Streamlit consumes more than two hours, ship the static version.

**Done when:** the loop runs end to end on the demo machine, **and a screen recording exists as a fallback** in case the live demo fails in front of an audience.

---

### M8 — Handover-ready
**Effort:** 1h

- `docs/serial_protocol.md` — the UART frame definition (fixed-order fields: `T_in, T_out, N_pass, setpoint, Q_cmd`), units, ranges, and cadence. This is the deliverable for whoever owns the board.
- `src/serial_bridge.py` — sender validated against a virtual COM loopback, so it is testable without hardware.
- Expand `README.md`: what this is, how to run it, and what the numbers do and do not claim.

**Done when:** someone else can implement the board side from the document alone, without asking questions.

---

## What must never be cut

If time runs short, cut in this order: the live dashboard (→ static plots), the multi-season data (→ one season), the serial bridge implementation (→ protocol document only).

These five are the project. Everything else is presentation:

1. The grey-box simulator (**M2**)
2. The physics sanity tests (**M2**)
3. The chronological split (**M4**)
4. The baseline-vs-predictive comparison (**M5**)
5. The `tau_act` sweep (**M6**)

---

## Honesty rules

These are what make the work defensible under questioning:

- **Never** claim "X% savings on a real train." Say: *"X% within the same comfort band on a parameterised cabin model, across an actuator-lag range of 0–20 minutes."* The second version is both more defensible and more impressive to an engineer.
- Never assert a time constant. Derive it, or sweep it.
- Never present a number whose source you cannot name.
- State plainly that the data is synthetic and that no real train data was available. This is a stated limitation, not a weakness to conceal — reviewers find hidden ones anyway.

---

## Continuing after the presentation

The foundations above are built to be extended. Natural next steps, roughly in order of value:

1. **Calibrate against real data** — if any Siemens telemetry becomes available, fit `C_eff` and `UA` to it. The model is already parameterised for exactly this.
2. **Proper MPC** — replace the heuristic predictive controller with a real receding-horizon optimiser over the learned forward model.
3. **Sequence models** — a GRU or 1D-TCN over a 60-minute window learns the lag implicitly, without hand-built EWMA features. A fair comparison against the LightGBM baseline is a genuine result either way.
4. **Humidity and full PMV comfort** — the current comfort metric is temperature-only; PMV is the standard rail comfort measure.
5. **Multi-zone** — per-car or per-zone control, with coupling between zones.
6. **Richer physics** — the terms deliberately excluded in M3: speed infiltration, tunnels, solar orientation.
