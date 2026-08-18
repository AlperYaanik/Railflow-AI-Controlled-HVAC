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
| `Q_hvac` | actuator output — dead time, first-order lag, **supply-air limit**, COP(`T_out`) |

**The actuator is supply-air limited, not simply capacity limited.** Rated capacity cannot be dumped into the cabin air as sensible power — it arrives through a finite air stream that cannot be colder than the coil allows:

```
Q_cool_max(T_air) = min( rated,  ṁ_supply · c_p · (T_air − T_supply_min) )
```

Authority is therefore **state-dependent**: 40 kW during pull-down, **25.7 kW at setpoint**, zero at supply temperature. Without this the model permitted 13.3 K/min of air cooling, and the baseline oscillated for a numerical rather than a physical reason — which would have made the M5 comparison a straw man. Fixing it removed the systematic overcooling (mean error −1.35 K → +0.01 K) and cut time outside the comfort band from 72/166 minutes to 38/166.

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

**Modelled as a service, not a time-of-day curve.** A metro runs continuously and its loading varies through the day; an intercity train is a discrete run whose loading is a property of *which service it is*. So load factor scales a whole run. The catalogue spans **both directions × two stopping patterns × four load factors** — 16 services.

Load factor **1.1** is included because it is the point where the route peaks at exactly 80 passengers, the **EN 13129 design load** (all seats occupied). Without it the standard's own worst case was never simulated: the base pattern peaks at 73.

The **express** pattern follows the real **Talgo 2027** service — Cairo, Sidi Gaber, Alexandria — running the route in about 2 h 30 against the semi-express 2 h 45. It uses 8.9% less energy with half the door-minutes, and stops M4 from assuming a fixed journey length.

**Done when:** 39 occupancy tests pass — the route empties at the terminus and occupancy stays non-negative at every load factor including 0 and 1.3; the base pattern reproduces the config exactly at load factor 1.0; doors open only while stopped; and the lookahead features lead the boarding event they describe (without which the controller has nothing to anticipate).

> **What M3 found, and what it forced.** Door infiltration was expressed as a per-stop total divided by the route's mean dwell — so editing one station's timetable moved `door_ua` by 36% at *every* station. Fixed by restating it as a rate. More seriously, the baseline's own oscillation turned out to be larger than the station disturbances it faced, which traced back to the actuator having no supply-air limit. That was fixed in M2 before proceeding (see above), and it **reversed one of our conclusions**: with a valid measurement, passengers dominate the disturbance (+7.65 kWh) rather than doors (+0.34 kWh). That is a better result for the project than the one it replaced — the dominant disturbance is the one the timetable can actually predict. Full detail in [`docs/PARAMETERS.md`](docs/PARAMETERS.md).

**Cross-layer tests.** `tests/test_integration.py` checks the seams between M1, M2 and M3, which is where most bugs have actually lived: that supply air exceeds fresh air, that the unit still holds the comfort band at the hottest measured conditions *given state-dependent authority*, that the control horizon covers the actuator delay, that 1-minute logging resolves the fast mode, and that every service × city combination runs without going unphysical. Two findings are locked in as tests — that passengers dominate doors, and that the disturbance decomposition is monotonic (it was not, before the supply-air fix).

---

### M4 — A forecaster that provably beats doing nothing
**Effort:** 3h

**Build:**
- `src/data_generator.py` — scenario loop over the dimensions below × stochastic setpoint policies. Target **~30k rows** (this saturates a ~20-feature tabular problem; more is wasted time).

**Scenario dimensions, ranked by measured effect on energy.** This ranking was produced by experiment, not guessed, and it changed the plan:

| Dimension | Effect | Status |
|---|---|---|
| **Departure hour** | **120%** | **Must sample** — 04:00 uses 18.2 kWh, 16:00 uses 40.1 kWh |
| City (Cairo / Aswan) | ~25% | Both included |
| Load factor (0.4 → 1.1) | 28% | Spanned by the catalogue |
| Day (across 92 summer days) | ~15% | Sample by percentile, never a fixed day |
| Stopping pattern | 8.9% | Both included |

Departure hour was not on any list of known issues and turns out to dominate everything else — ENR runs 37 trains daily between 04:00 and 23:00, so the whole range is real service. A model trained only on morning departures would miss most of the problem.

*This also settles a deferred question: tying load factor to time of day is **not worth building**, because departure hour already carries a 4× larger effect through solar and outdoor temperature.*

**Built:**
- `src/data_generator.py` — random scenario sampling (not a fixed grid) across departure hour × city × day × load factor × pattern × direction, until ~30k rows. **Winter is excluded** — a scope decision, not an oversight: Egypt is cooling-dominated and heating essentially never engages at realistic occupancy even with the M2 heating-COP fix in place. The controller commanding HVAC during generation is deliberately **not** the reactive thermostat used for evaluation — a `StochasticController` spends most minutes in a proportional-to-error mode (randomised aggressiveness per scenario) but periodically forces full-power, off, or a random fixed level for a few minutes at a time. This is standard system-identification practice: a model trained on one policy's closed-loop trajectory has little independent (state, command) variation to learn the plant's response from.
- `src/features.py` — command lags (1–20 min) and an EWMA (half-life = `tau_act_min`) of *delivered* power (`q_hvac_actual_w`, not the raw command — the physically causal driver), cyclical hour-of-day, current state, and the lookahead features: time to next station, expected boarding, and weather **30 minutes ahead** pulled from the scenario's own future trajectory (legitimate because weather is exogenous and known in advance — forecasting it is a different, solved problem this project isn't attempting). Every lag/shift/EWMA operation is grouped by `scenario_id` so nothing bleeds across journeys.
- `LiveFeatureBuilder` (also in `src/features.py`, added once it became clear M5 would need it) — builds the identical feature row `add_features()` computes in batch, one minute at a time, for a controller driving the simulator live rather than reading a finished table. Added because the batch function fundamentally can't be reused as-is: it shifts forward *and* backward across a whole finished DataFrame, which a live controller — with only the history accumulated so far — cannot do. Without one shared implementation, M5 would have had to reconstruct this by hand, and a live-vs-batch mismatch would be invisible: it wouldn't crash, it would just quietly feed the model rows it was never validated against. `tests/test_features.py::test_live_feature_builder_matches_batch_add_features_exactly` runs the real M1–M3 simulator once and checks both paths against each other to the last bit, over the range where both are defined — proof, not assumption, of what makes this trustworthy for M5 to build on.
- `src/train.py` — LightGBM (native `Dataset`/`train` API, not the sklearn wrapper, to avoid pulling in scikit-learn as a dependency), predicting `T_air` at `t+H` where `H = simulation.control_horizon_min` (30 min), read from config rather than hardcoded.

**Non-negotiable, and implemented as stated:** chronological split **by calendar date**, not by row and not by scenario. Dates are sorted and partitioned 70/15/15 into train/val/test, so a given day's weather can never appear on both sides of the split — the leak a random split would produce, since lag features span up to 20 minutes and the target itself is only 30 minutes ahead of the current row.

**Done when:** test MAE beats the persistence baseline. **It does, on genuinely held-out future dates:**

| Split | Dates | Model MAE | Persistence MAE | Improvement |
|---|---|---|---|---|
| Train | 54 | 3.451 °C | 5.023 °C | +31.3% |
| Val | 12 | 3.374 °C | 5.114 °C | +34.0% |
| **Test** | **11** | **3.780 °C** | **4.837 °C** | **+21.8%** |

The improvement is smaller on test than train/val — an honest generalisation gap that a random split would have hidden, not a red flag: test isn't hotter or harder by any measure checked (T_out and persistence-MAE are actually slightly *easier* on test), so this reads as ordinary date-to-date variance across a modest 11-day sample, worth knowing rather than averaging away.

**Is the model doing more than tracking the weather forecast?** `t_out_fcst_h` dominates feature importance by a wide margin (gain 13027, next-highest 3490) — physically sensible, since outdoor temperature is the dominant 30-minute-ahead driver in a cooling-dominated climate. Checked rather than assumed: a closed-form linear fit on `t_out_fcst_h` alone gets test MAE 4.019 °C; the full model reaches 3.780 °C, a further **5.9% on top of weather alone**. Smaller than the margin over persistence, but real, and it's the actual evidence behind this project's timetable-anticipatory premise — without it, "occupancy and timetable features help" would be an assertion, not a finding.

> **What M4 found — two bugs, one in the new code, one in the environment.**
>
> **A sign inversion in the exploration policy.** The reactive control mode computed `err = t_air_c − setpoint`, positive when the cabin is too *hot* — and the dispatch sent positive fractions to *heating*. So the "realistic" mode fought overheating with heat, for 57.6% of rows in the first generated dataset. Not caught by a crash: caught by checking the command-mode distribution against the configured weights (full_heat should have been ~5%, not 58%) and by `cooling_energy > heating_energy` failing on a cooling-dominated climate. Fixed to `err = setpoint − t_air_c`; both directions now have dedicated regression tests (`test_reactive_mode_cools_when_hot`, `test_reactive_mode_heats_when_cold`).
>
> **A native crash from a pandas/LightGBM import-order conflict.** `lgb.Dataset(...).construct()` died with an access violation deep in `LGBM_DatasetSetField` — a hard native crash, not a catchable Python exception. Root-caused rather than worked around blindly: bisection (shape, dtype, contiguity, ownership, value range, value order — all ruled out one at a time) eventually isolated it to **import order** — `import pandas` with pandas never subsequently used, followed by `import lightgbm` and one `Dataset.construct()` call on a random array, reproduces the crash on this Windows environment every time; reversing the order avoids it completely. Fixed in `src/train.py` (`lightgbm` imported first) and in `tests/conftest.py` (forced session-wide, since pytest could otherwise import a pandas-using test file before `test_train.py`). Worth knowing for the 22 August handoff: whoever runs this on their own machine may need the same fix if they hit the same crash.

**Cross-layer tests.** `tests/test_integration.py` ties two of M3's findings forward into what the generator actually produces — the EN 13129 design-load case (`load_factor=1.1`) and both stopping patterns are asserted present in a real generated sample, not just assumed to still be there — plus a wiring check that M4's forecast horizon is actually read from config rather than a second hardcoded value drifting away from the dead-time/`tau_act` invariant M1–M3 already established.

**Tests: 175 → 178** (39 new: 17 in `test_data_generator.py`, 14 in `test_features.py`, 11 in `test_train.py`, minus overlap, plus 3 appended to `test_integration.py`). Verified on a fresh clone with no `data/` at all: every M4 test now skips with an actionable message (`Run: python -m src.weather`) instead of a raw `FileNotFoundError` — the same gap M0 fixed for notebooks and M3 fixed for the weather-dependent integration tests, recurring here because `data_generator.py` sits one layer deeper than what those earlier fixes covered.

> **A follow-up check found something M5 needs to know before it's built.** Driving the saved model *live* (minute-by-minute, through the actual M2/M3 simulator, not the pre-built features table) with the plain reactive thermostat — M5's actual baseline — it beat persistence in only 1 of 4 test scenarios, against 4/4 under the policy it was trained on. Not a bug: verified the live reconstruction against batch `add_features()` on the identical trajectory and got bit-for-bit identical predictions. Full writeup and what it means for M5 in [`docs/PARAMETERS.md`](docs/PARAMETERS.md) and in M5's own section below.

---

### M5 — The result
**Effort:** 2h

**Grounded in external research before building anything** (control theory, rail HVAC literature, HVAC comfort metrics), because the earlier language for the "proposed" controller didn't survive scrutiny. Findings and their direct design consequences:

| Research finding | Consequence for this milestone |
|---|---|
| A true Smith predictor is a specific single-loop architecture (delay-free model + delay model + feedback-corrected mismatch) built for an analytic plant model | **Renamed.** The controller is a "learned-model anticipatory controller" — same spirit as Smith-predictor/MPC dead-time compensation, not that structure. Calling it a Smith predictor would be a claim a control-theory-literate reviewer could correctly dispute |
| A real train HVAC MPC study: pure feedback control has "long transition time, system shock, and load mismatch"; predictive control combined with **feedforward *and* feedback with dynamic correction** reached **13.44% daily energy saving** | The controller blends the M4 forecast (feedforward) with current error (feedback) — not a pure feedforward "trust the model" design, which M4's own audit shows would be exposed to the forecaster's known policy-dependence with nothing to catch it |
| HVAC comfort literature uses **degree-hours** ("Exceedance Degree-Hours", Salimi et al., *Indoor Air*, 2021), not degree-minutes | Metric implemented as `sum(max(0, \|T_air − setpoint\| − band_k)) / 60` — computed at native minute resolution, **reported in K·h** |
| A real vehicle A/C patent documents a 1.5 K hysteresis band; AIRAH's commercial guidance gives 2 K as the standard energy-saving target | `hvac.thermostat_hysteresis_k = 2.0` (sourced, config), replacing an earlier ad-hoc 0.5–1.0 K band that would have made the baseline cycle unrealistically often |
| General building MPC: 10–50% (commonly 20–30%) energy savings; the train-specific study above: 13.44% | Calibration for the headline number — a result in the 10–30% range is defensible and literature-consistent; far outside it needs scrutiny before being reported |

**Built:** `src/controllers.py` — `ThermostatController` (baseline) and `AnticipatoryController` (proposed) — and `src/evaluate.py` (`degree_hours_outside_band`, `run_controller`, `score`).

> **Two real bugs, both invisible from reading the code, both found only by running a real simulation minute by minute rather than checking an aggregate result.**
>
> **A sign bug in `AnticipatoryController`** — the exact same class of mistake already made once in `data_generator.py`'s `StochasticController`: error computed as `t_air − setpoint` (positive when hot) feeding a dispatch that sends positive fractions to heating. Fixing only that produced **no visible change** in the test output, because a **second, independent** sign bug in the dispatch line (`-cooling_capacity * frac` instead of `frac * cooling_capacity`, given `frac` is already signed) was silently cancelling the first fix out. Two bugs stacked to look like "the fix did nothing" — caught by tracing `frac` and the delivered command minute by minute, not by re-running the aggregate test and shrugging at an unchanged number. On a 36 °C day, the broken version ended at 43 °C (fighting itself with heat); fixed, it ends at 24.9 °C.
>
> **`ThermostatController`, run dual-mode** (heat below the lower threshold, cool above the upper one — what a generic thermostat would do), **oscillated between full heat and full cool 18 times in a 166-minute journey**, `T_air` swinging 21.8–30 °C — an 8+ K limit cycle, not the intended ±1 K band. Mechanism: this cabin's air node is fast (`τ_fast` ≈ 1.1–1.3 min, the same property M2 built the two-node model to capture) relative to the rated capacities, so a full-power command overshoots the *opposite* switching threshold before the actuator's own dead-time/lag can arrest it — a "textbook" hysteresis band sized for a slower plant doesn't hold here. Fixed by making the baseline **cooling-only**, matching every other baseline already built in this project (M2/M3's modulating and on/off analysis, `data_generator.py`'s exploration policy) and the documented cooling-dominated climate. Both are now permanent regression tests in `tests/test_controllers.py`, not just something caught once and moved past.

**Metrics:** energy (kWh) and comfort (**degree-hours** outside band, sourced above). Evaluated on a **held-out weather period and an unseen occupancy profile** — already available without extra work: the M4 test split's 23 held-out scenarios (dates 2024-08-18 to 2024-08-30) already span all 4 load factors, both stopping patterns, both directions and both cities.

**`ff_weight` and `gain_k`** (the feedforward/feedback blend ratio and control gain) are `[ASSUMPTION]` — no literature source gives an exact blend ratio for this specific forecaster/plant pairing. Swept across `ff_weight ∈ {0, 0.3, 0.6, 0.9, 1.0}` in tests rather than asserted correct, the same treatment given to every other under-sourced parameter in this project (`docs/PARAMETERS.md`'s calibration priority list). `ff_weight = 0` is also a correctness check: the blend must degenerate to plain proportional feedback exactly, verified against a hand-computed value.

**Cross-layer tests** (`tests/test_integration.py`): `thermostat_hysteresis_k` (the controller's switching deadband) and `comfort.band_k` (the scoring metric's threshold) are confirmed to be independently wired, not accidentally the same number doing two jobs; the two controllers are confirmed to produce genuinely different trajectories on the same scenario (RMS difference > 0.5 K — a sameness bug would mean one of them isn't doing what it claims); and both controllers' scores land in a physically plausible range on a first real scenario.

**Done when:** you can state one sentence — *"X% less energy at equal comfort"* — and point to the plot behind it. **Not yet reached** — the pieces above are built and tested; running the full comparison across the M4 held-out split to produce the headline number is the next step.

> **Inherited risk from M4, still unresolved, still governs how the headline number must be measured.** M4's forecaster beats persistence robustly (+21.8–28.5% across independent seeds) **when evaluated the way it was trained** — against the stochastic exploration policy. Tested live against the plain reactive thermostat (`ThermostatController`, this milestone's actual baseline) and it beat persistence in only 1 of 4 scenarios, because persistence itself is a strong baseline once a controller holds `T_air` fairly stable. Full detail in [`docs/PARAMETERS.md`](docs/PARAMETERS.md). **The headline number must come from `AnticipatoryController` vs `ThermostatController` directly, on the reactive baseline's own trajectories — never from reusing the "+21.8%" forecaster-vs-persistence figure.** If the controller comparison disappoints, the likely fix is training data generated under the reactive policy, not a different model.

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
