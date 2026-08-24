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

Derived values: **τ_fast ≈ 1.1–1.3 min** (air), **τ_slow ≈ 72–97 min** (whole cabin), varying with occupancy.

> **A shortcut that was tried and rejected.** The familiar single-node form `C_eff / UA` is only valid when `hA ≫ UA`. Here `hA ≈ 1500 W/K` against `UA ≈ 740–1174 W/K`, so it materially under-reports the slow constant. The physics test in M2 caught this by comparing the simulator against the derived value — exactly what that test exists for. The lesson generalises: a derivation is only worth its credibility if something checks it.

**On the least defensible parameter.** `C_mass` is an `[ASSUMPTION]` built from a mass budget (4.34 MJ/K if every kilogram coupled; we take ~65%). Rather than defend the number, we measured how much it matters: sweeping it across **1.6–4.34 MJ/K** moves τ_slow from 42 to 111 min but changes τ_fast by **<3%**, the boarding spike by **<10%**, and pull-down time **not at all**. Everything the controller sees is governed by `C_air` and `hA`. A regression test locks this in, so the weakest input is demonstrably the one that matters least.

**Sanity against the real world.** `hA` rests on 6 W/m²K, mid-range for indoor natural convection (2–10) and conservative given forced HVAC circulation. Sweeping it 2–15 W/m²K moves τ_fast between 0.6 and 1.8 min — the two-node conclusion holds across the entire plausible range, and *strengthens* at the low end. Stated precisely: the boarding disturbance spans 1.4–3.0 K across that sweep against a ±2 K band, so it is *comparable to* the band and exceeds it below roughly `h = 8` — it is not unconditionally larger, and the presentation should not claim it is. The 40 kW cooling capacity sits at the top of the commercial range for a single coach (units are typically 23–32 kW, up to ~40 kW for high-capacity stock), which is the right place for a 46–48 °C design condition.
>
> *(The `UA`/τ figures above were re-derived during the M8-era cross-milestone audit — this section had been missed by the earlier fresh-air rate correction and still held pre-correction numbers, e.g. `UA ≈ 740–1040 W/K` and τ_slow up to 120 min. `docs/PARAMETERS.md` has the full regenerated sensitivity tables and the same note.)*

**Full rationale for every coefficient — including the sweeps, the rejected alternatives, a calibration priority list, and a corrections log — is in [`docs/PARAMETERS.md`](docs/PARAMETERS.md).**

The **actuator** lag (`tau_act` — compressor spin-up, damper travel, coil inertia) is genuinely unknown and is handled separately in M6.

> **M10 note.** This physics is unchanged by the M10 reframing (see M10's own section, below M9) — `Q_hvac` still flows exactly as modelled here. What changed is *who decides* the value fed into it: before M10, Railflow's own control law computed `Q_hvac` directly; since M10, a simulated `PlantResponse` (a stand-in for the real, third-party unit's own onboard controller) computes it in reaction to a setpoint Railflow recommends. `tau_act`/`dead_time_min` still describe the actuator this equation drives — only their *ownership* is reinterpreted (M6's section has the detail).

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

**Done when:** you can state one sentence — *"X% less energy at equal comfort"* — and point to the plot behind it. **Reached, but the first attempt didn't survive contact with data — recorded below rather than smoothed over.**

> **M10 status: everything from here to the end of this section describes the pre-M10 watts-dispatch law — superseded, kept for the record.** `AnticipatoryController` (the class every number below was measured on) was renamed `AnticipatorySetpointAdvisor` and its dispatch tail rewritten to recommend a cabin setpoint instead of a watts command, so Railflow's own compute never has to touch a real HVAC unit's control electronics (see M10's own section, after M9, for the full why). `gain_k` — tuned below alongside `ff_weight` — is retired, replaced by `max_shift_k`/`deadband_k`. **Phase 2 has since re-run this exact tuning discipline under the new law — see the result at the end of this section, after the historical narrative below, which is kept visible rather than deleted, the same way every earlier superseded number in this section already is.**

> **The first full run across the M4 test split came back NEGATIVE.** `AnticipatoryController` used *more* energy than `ThermostatController` on 15/23 scenarios, and was strictly worse on both energy **and** comfort on 6/23 — the "inherited risk" below, materialising. Diagnosis, not guesswork: on the canonical scenario, `ThermostatController` is fully off 32.5% of the time; the raw proportional law had no floor and was off only 0.6% of the time — it never stopped nudging, paying continuous low-power compressor cost the baseline's real off-periods avoid entirely. **Fix:** added a deadband to `AnticipatoryController`, reusing `hvac.thermostat_hysteresis_k` (already sourced for the baseline, so no second unsourced number) rather than inventing a threshold, applied as a continuous shrink-to-zero rather than a hard on/off jump so it can't reintroduce the chattering `ThermostatController` hit the first time dual-mode hysteresis was tried on this fast plant. **Validated on the val split first** (mean energy saving −4.5%→+3.4%, dominated scenarios 6/23→0/31) **before re-running the test split once** for the actual headline number — tuning against the same split being reported on would be exactly the leakage the chronological split exists to prevent.
>
> **Superseded by a physics correction — numbers below are current.** `ventilation.fresh_air_m3_h_per_passenger` was corrected 15→20 (the real EN 13129 standard rate; 15 was the *reduced/extreme-condition* figure — see `config/cabin_params.yaml` and the corrections log). More fresh air means more ventilation heat load in an Egyptian summer, which changes the physics both the dataset and the trained forecaster are built on — so `scenarios_raw.parquet` was regenerated, the forecaster retrained (test MAE 4.339 °C vs persistence 5.499 °C, **+21.1%**, essentially unchanged from the pre-correction +21.8%), and both comparisons below re-run once on the resulting test split. This is not a retuning pass — `ff_weight`, `gain_k` and the deadband are untouched — it's the same controller measured against corrected ground truth.

> **Superseded twice more since — numbers below are current.** (1) M9 retrained the forecaster with tuned LightGBM hyperparameters (`num_leaves`/`learning_rate` 31/0.05 → 15/0.02, found by `src/benchmark_models.py`'s grid, zero interface change). (2) M9 then re-tuned `ff_weight` 0.6 → 0.45 against that retrained model specifically — a 25-point (gain_k, ff_weight) grid searched on VAL, ranked by scenarios-strictly-better-on-both rather than mean alone (which rewards a few large-swing scenarios over a robust result), confirmed ONCE on TEST. The same (gain_k=3.0, ff_weight=0.45) point won independently against both the old and the new model — not assumed to transfer, checked. `gain_k` itself is unchanged.
>
> **Headline result — M4 held-out test split, 23 scenarios, 2024-08-18 to 2024-08-30, both cities, both patterns/directions, all 4 load factors:** `AnticipatoryController` uses **3.9% less energy on average** than `ThermostatController` (median +3.6%, range −0.0% to +7.5%, std 2.4pp) — up from the pre-retune +0.9%, and this time with **no comfort trade-off**: comfort-equal-or-better held at **21/23** (unchanged), scenarios better on **both** energy and comfort rose **13/23 → 19/23**, and **zero scenarios worse on both**, throughout. Total degree-hours 42.79 → 25.14 K·h. A strict improvement over the pre-retune point, not a traded-away margin — the same discipline that caught the earlier deadband fix's real trade-offs would have caught one here too, and didn't find one. Reproducible via `python -m src.compare_controllers`; `tests/test_compare_controllers.py` locks in the shape of the result as a regression guard, not the exact decimals.

> **A second comparison, answering a different question: does the ML forecast itself earn its keep?** `compare()` above measures "beats today's real rail HVAC (on/off)". `compare_feedforward_contribution()` measures something narrower: the shipped `AnticipatoryController` (`ff_weight=0.45`) against the *same* controller and deadband forced to `ff_weight=0.0` — pure proportional feedback, no M4 forecast involved at all. **Not a real PID** (no integral or derivative term); mislabelling it as one would repeat the exact overclaim this milestone already renamed once ("Smith predictor" → "learned-model anticipatory controller"). Result, same 23 test-split scenarios, post-retune: the forecast **costs 3.7% more energy on average** (down from 6.9% — expected, since a lower `ff_weight` means the shipped controller now leans more on feedback than before, so the forecast's own marginal contribution is naturally smaller in both directions) but still **cuts total degree-hours by 62%** (66.72 → 25.14 K·h, was 70%) and is equal-or-better on comfort in **23/23 scenarios** — still the cleanest, most consistent result in this milestone. Mechanism unchanged: pure proportional feedback only reacts once `T_air` has already left the deadband; adding the forecast lets the controller act earlier, at a modest continuous energy cost — "AI buys ride quality," not primarily energy. Reproducible via `python -m src.compare_controllers` (prints both comparisons); regression-tested in `tests/test_compare_controllers.py`.
>
> **Inherited risk from M4 — flagged before the run, and it was right.** M4's forecaster beats persistence robustly (+21.8–28.5% across independent seeds) **when evaluated the way it was trained** — against the stochastic exploration policy. Tested live against the plain reactive thermostat (`ThermostatController`, this milestone's actual baseline) and it beat persistence in only 1 of 4 scenarios, because persistence itself is a strong baseline once a controller holds `T_air` fairly stable. Full detail in [`docs/PARAMETERS.md`](docs/PARAMETERS.md). **The headline number came from `AnticipatoryController` vs `ThermostatController` directly, on the reactive baseline's own trajectories — never from reusing the "+21.8%" forecaster-vs-persistence figure.** The controller comparison *did* disappoint at first, exactly as this note warned — but the actual fix was not the one predicted here (retraining under the reactive policy). It was a one-line change to the controller's own control law. Left both predictions on the record: the risk assessment was right that this was fragile; the specific guess at the fix was wrong. Worth remembering that distinction next time a risk note tries to predict its own resolution.

> **M10 Phase 2 result — the "+3.9%" above is now superseded too, and the honest replacement is different IN KIND, not just in number.** Once Railflow's output became a setpoint recommendation instead of a watts command, the identical tuning discipline that produced +3.9% above — a VAL-split grid search, ranked by `n_both_better`, confirmed once on TEST — was re-run on `(ff_weight, deadband_k)`. **Three attempts, 80 total configurations, all measured on the honest VAL split: not one beat the static baseline.** Diagnosed, not shrugged off: `PlantResponse` (the simulated real receiving unit — see M10's own section) has no proportional response, fully on or fully off with nothing between, so a forecast-driven setpoint shift can only move WHEN the switch fires, never HOW HARD it runs — firing early on a forecast buys no proportional benefit to offset its added runtime. Full mechanism and all three attempts' numbers are in `src/tune_advisor.py`'s module docstring and `AnticipatorySetpointAdvisor`'s own docstring (`src/controllers.py`).
>
> **Shipped as of M10 Phase 2: `ff_weight=0.0`, `deadband_k=1.5`** — the search's own top-ranked point (10/31 VAL scenarios both-better, 1/31 worse-on-both), confirmed on the same TEST split at **mean energy delta −0.2%** (median +0.0%, range −1.2% to +0.5%, std 0.5pp), **0/23 scenarios worse on both**, and total degree-hours slightly LOWER in aggregate (41.43 vs `ThermostatController`'s 42.79 K·h) even though per-scenario comfort is only equal-or-better on 19/23. **This is parity, not a win** — stated plainly rather than dressed up, the same honesty this project already applied to M9's SHAP-diagnosed remedies (tested, neither adopted, disclosed rather than hidden). `ff_weight=0.0` means the forecast currently contributes nothing positive under this architecture; deliberately forcing it back on (`ff_weight=0.3`) makes energy strictly worse in **every one of 23** test-split scenarios (`compare_feedforward_contribution()`, re-pointed in Phase 2 specifically to demonstrate this directly rather than merely assert it) — a clean, repeatable, diagnosed finding, not noise. Reproducible via `python -m src.compare_controllers`; `tests/test_compare_controllers.py` locks in the current shape as a regression guard.
>
> **What this means for the project's claim.** M4's forecaster itself is unaffected by any of this — it still beats persistence by +21% at the raw t+H prediction level, a claim about prediction accuracy, not about what a bang-bang actuator can do with that prediction. What M10 Phase 2 found is narrower and more honest than the pre-M10 story: recommending a setpoint — the only channel the warranty constraint leaves available — to a realistic, simple on/off receiving unit does not translate the forecaster's accuracy into a measurable energy or comfort win. A real, tested limit of this specific deployment path, not of the forecast or the physics underneath it.
>
> **A fourth attempt, prompted by a real question rather than run speculatively: does recomputing the recommendation every single minute just add chatter, when the actuator can't react that fast anyway?** The same reasoning that sets M8's 30 s telemetry cadence (`docs/serial_protocol.md` §5). Tested directly by adding `advisor_update_interval_min` (`src/evaluate.py`) — holds the *effective*, PlantResponse-facing setpoint fixed for N minutes at a time instead of chasing a fresh recommendation every minute. **Result: the opposite of the hypothesis.** Spacing it out makes things WORSE, monotonically-ish, and by a lot — every-minute updates (`interval=1`, today's default) is the clear VAL winner; `interval=20` is the single worst point found across all four tuning attempts (mean −1.90%, 15/31 scenarios worse on both). Diagnosed, not left as a surprise: M8's cadence reasoning is about not over-communicating to a *physically slow actuator over a link* — there is no link and no bandwidth cost here, and `PlantResponse`'s own hysteresis (not the advisor's update rate) is what already prevents actuator chatter. Holding the recommendation fixed doesn't reduce chatter that was never going to happen; it only makes the advice stale relative to the cabin's continuously evolving actual state. Confirms, rather than changes, the shipped default. Full table in `src/tune_advisor.py`'s module docstring.

---

### M6 — Honest about what we don't know
**Effort:** 1h — *highest value per hour in the plan*

`tau_act` is unknown. Rather than picking a number, **sweep it**: `tau_act ∈ {0, 2, 5, 10, 20} min`, re-running both of M5's comparisons for each (`src/sweep_tau_act.py`, `src/plot_tau_act_sweep.py`). Grounded before running it, not asserted: industrial dead-time-compensation literature reports predictive control's advantage over reactive control growing with delay — the same argument a Smith predictor is built on — provided the delay stays within the forecast horizon. Ours is 30 min, comfortably above the sweep's 20 min ceiling.

> **Re-run twice since — after the `fresh_air_m3_h_per_passenger` correction, and again after M9's model retune + `ff_weight` re-tune (0.6→0.45).** Same methodology throughout, same sweep points, each pass measured against the currently-shipped controller. The table and conclusions below reflect the current controller; earlier passes are superseded, not separate findings — see git history for the pre-retune numbers if ever needed for comparison.

**What was predicted going in, and what the real sweep (23 held-out test-split scenarios at every point) actually showed:**

| Predicted | Found |
|---|---|
| Energy saving increases *monotonically* with lag | **Negative** at `tau_act=0` (**−3.9%**) but already **positive** at `tau_act=2` (**+1.6%**) — narrower and shallower than the pre-retune shape (was negative through both 0 *and* 2 min), because a lower `ff_weight` leans more on feedback and is less exposed to a fast actuator overshooting on a stale forecast. Rises **+3.9% → +4.8% → +5.1%** from 5→20, monotonic but with visibly diminishing returns at the top end (+0.99pp from 5→10, only +0.29pp from 10→20). Still a **weaker, conditional** claim than a clean monotonic curve would be — it does not hold if the real actuator turns out to be fast |
| At `tau_act=0`, `compare_feedforward_contribution()`'s gap (anticipatory vs proportional-only) shrinks toward ~0 | Still doesn't shrink toward 0 — now **strongly positive** at `tau_act=0` (**+7.4%**, the forecast saves energy over pure feedback there) and consistently **negative** from `tau_act=2` up (**+0.6%** then **−3.7% to −4.0%**). The *comfort* contribution is the one number in this milestone that has stayed unambiguous across every re-run, this one included: the forecast improves comfort at every single swept point, by a wide margin, every time |

**Not a bug — the original framing undersold the forecaster, and remains true after both corrections.** "Converges to baseline at zero lag" assumed the forecast's only job was compensating for actuator delay. It isn't: `expected_boarding`, `time_to_next_station_min` and the weather forecast let it anticipate *known future disturbances* independent of how fast the actuator responds.

**Comfort, re-examined honestly rather than carried over from the last run — and the shape changed in a way worth stating plainly.** At the *middle* of the sweep (`tau_act` 2, 5, 10 min), `AnticipatoryController` has lower total degree-hours than on/off, often by a wide margin (`tau_act=5`: 25.1 vs 42.8 K·h). But comfort dominance now **reverses at both ends of the range, not just the slow one**: at `tau_act=0` the retuned controller is *worse* on comfort than on/off (123.6 vs 117.4 K·h — it wasn't before the retune) as well as at `tau_act=20` (29.1 vs 25.6 K·h, consistent with the earlier finding). The honest version is now **"comfort-dominant in the middle of the plausible lag range, not at either extreme — always ahead of a forecast-free controller, roughly tied or slightly behind plain on/off at both the fast and slow ends."** Scenarios worse on both energy and comfort are concentrated hard at one end this time: **9/23 at `tau_act=0`** (up from 2/23 pre-retune — a real regression at that specific point, not noise), just **1/23 at `tau_act=2`**, and **0/23 from 5 min on**. The controller's `ff_weight`/`gain_k` were tuned at the training-time default (`tau_act_min=5`) — this sweep is exactly what shows the cost of that anchor: better where the tuning was aimed, worse at the extreme furthest from it.

Two reasons this earned its hour, even though the specific claim it set out to check was wrong:
- **It is still a correctness-relevant check — it just corrected the check itself.** The zero-lag "degenerates to baseline" test rested on a premise (the forecast is *only* about actuator delay) that the data falsified outright. Running it anyway surfaced what the forecaster actually does, which is a real result, not a wasted hour.
- **It is Q&A armour** — arguably better armour than the original claim would have been. "How fast does the HVAC actually respond?" now gets *"I don't claim a value; here's the dependence, and here's why our controller is comparatively insensitive to getting it wrong"* — a stronger answer than a bare monotonic curve, because it directly addresses the risk of the assumption being wrong rather than just describing it.

> **Before sweeping: the forecaster is frozen, but one of its features isn't — fixed.** `LiveFeatureBuilder`'s `q_actual_ewma` halflife and `add_features()`'s (training-time) halflife used to read `cfg["hvac"]["tau_act_min"]` directly (`src/features.py`). Sweeping that same config field to change the *actuator's* physical lag (`CabinModel.tau_act_s`, `src/cabin_model.py`) would have *also* silently changed the EWMA smoothing window fed to the forecaster — which was trained once, at `tau_act_min = 5`, and is not retrained per sweep point. That would conflate two different effects: genuine actuator-lag sensitivity (what the sweep measures) with train/serve feature skew (an artefact of reusing a frozen model on a shifted feature distribution). Found during the M6-readiness audit, before any sweep code was written, and fixed before it could bite: both functions now take an explicit `ewma_halflife_min` parameter, defaulting to `TRAINED_EWMA_HALFLIFE_MIN` (`src/features.py`, captured once at import from `cabin_params.yaml`) instead of reading `cfg` live. Zero behaviour change for every existing caller — verified equal to the old live-lookup for the current config. **When M6's sweep is written:** pass the swept `cfg` to `CabinModel` only; pass the default (or `TRAINED_EWMA_HALFLIFE_MIN` explicitly) to `LiveFeatureBuilder`.

**Done when:** a single sweep plot exists — saving vs `tau_act`. **Reached** — `data/m6_tau_act_sweep.png` (two panels: energy saving, and the comfort-robustness finding above), reproducible via `python -m src.sweep_tau_act && python -m src.plot_tau_act_sweep`. Per-point numbers in `data/m6_tau_act_sweep.csv`.

> **M10 status.** `tau_act` now describes the real, third-party receiving unit's own actuator lag — something Railflow doesn't control and is deliberately modelling as unknown, rather than Railflow's own commanded actuator's lag (see M10's own section, after M9). `src/sweep_tau_act.py`/`src/plot_tau_act_sweep.py` needed **zero code changes** for this — both only call `src/compare_controllers.py`'s `compare()`/`compare_feedforward_contribution()`, whose signatures the M10 refactor didn't touch. Only the *interpretation*, and the table/numbers above (measured on the retired watts-dispatch law), are superseded.
>
> **M10 Phase 2 result — regenerated against the tuned `(ff_weight=0.0, deadband_k=1.5)` law.** The pre-M10 sweep's headline finding ("comfort-dominant in the middle of the plausible lag range") no longer applies — there is no longer a strong lag-dependence to characterise, which is itself the finding. Energy delta stays within **+0.19% to −0.17%** across the *entire* swept range (`tau_act` = 0, 2, 5, 10, 20 min) — parity, not just at the training-time default but essentially everywhere the actuator lag could plausibly sit. Aggregate comfort (total degree-hours) is slightly BETTER for the advisor at every point **except `tau_act=0`**, where it is slightly worse (118.5 vs baseline's 117.4 K·h) — worse-on-both scenarios are 3/23 at `tau_act=0`, drop to 1/23 at `tau_act=2`, and are 0/23 at the shipped `tau_act=5` default, then reappear at 3/23 for both `tau_act=10` and `tau_act=20`. The feedforward-contribution ablation (`ff_mean_saving_pct`) is negative at every single swept point (−0.45% to −0.99%) — confirming, across the full plausible actuator-lag range and not just at one default, that turning the forecast on does not help regardless of how fast or slow the real unit's response turns out to be. Reproducible via `python -m src.sweep_tau_act && python -m src.plot_tau_act_sweep`; current numbers in `data/m6_tau_act_sweep.csv`.

---

### M7 — Something to show in 20 minutes
**Effort:** 2h

**Build:** `src/app.py` — Streamlit, running the **same simulator** live with both controllers side by side.

No separate demo infrastructure: the simulator is the data generator, the evaluation harness, *and* the demo engine. One artifact, three uses.

**Guardrail:** build the static comparison plot **first** — it is the guaranteed fallback and takes minutes. Upgrade to the live dashboard only once it exists. If Streamlit consumes more than two hours, ship the static version.

**Predictions made before building, checked after:** (1) the app's displayed numbers for a given scenario must equal `compare_controllers.py`'s `compare()` output for that identical scenario — same code path, not a reimplementation, so any mismatch would be a bug in the app, not a new result; (2) the known Windows lightgbm/pandas import-order crash (`docs/PARAMETERS.md`'s corrections log) had never been tested under `streamlit run` specifically. Both checked, not assumed:

> **A real bug found on first run — `ModuleNotFoundError: No module named 'src'`.** `streamlit run src/app.py` executes the file directly rather than as `python -m`, so the project root isn't on `sys.path` the way every other entry point in this project gets it for free. Fixed with an explicit `sys.path.insert(0, ...)` at the top of `app.py`, before any `from src...` import. The lightgbm/pandas crash did **not** reproduce — inference-only usage (`Booster.predict()`, no `Dataset` construction) appears to be a different code path than the one that crashes, though `import lightgbm` first is still kept as the same defensive ordering the rest of the project uses.

**Built:** `src/app.py` (static-first, per the guardrail above — precomputes the full trajectory for both controllers via `run_controller()`, then a minute slider/auto-play animates the already-computed data rather than trying to step the physics live in lockstep with wall-clock time). Scenario picker draws from the same 23 held-out test-split scenarios `compare_controllers.py` uses, not an invented one.

> **M10 update.** The middle panel used to plot `cmd_w` (a watts command). Since M10 it plots `advised_setpoint_c` (°C) for both controllers instead — the recommendation each advisor actually produces, not a power command (see M10's own section for why). `ThermostatController`'s line here exactly overlays the dashed setpoint reference already drawn in the panel above it, an honest visual: the baseline's own advice is just the static schedule, nothing more.

**Verified in a real browser**, not just by reading the code: launched via `streamlit run`, cross-checked the displayed energy/comfort numbers against `compare()` directly for two different scenarios (exact match both times), confirmed the scenario picker and minute-slider interactions both correctly re-trigger the simulation, confirmed no console or server errors. **Then made permanent** as `tests/test_app.py` (Streamlit's own headless `AppTest` framework, no browser needed) — the same cross-check against `compare()`, plus a check that a missing model file produces a clean, actionable error rather than a crash. Two more non-obvious things found only by running that test and watching it fail for the wrong reason first: `AppTest` re-executes `app.py`'s source fresh each run rather than reusing an already-imported module object, so patching has to target `src.train.MODEL_PATH` (the actual source of the name) not `src.app`'s copy of it; and `get_booster()`'s `@st.cache_resource` cache is a single slot shared across every `AppTest` run in one process, so a real model loaded by an earlier test would silently mask the missing-model test unless explicitly cleared.

**A second gap found afterward, by code inspection rather than a user report:** the missing-model check had no equivalent for the other two setup steps. `get_scenarios()` (reads `data/scenarios_raw.parquet`) and `run_both()` (reads `data/weather_{city}_summer.csv` via `run_controller()`) had no existence check at all — a demo machine that skipped `python -m src.weather` / `python -m src.data_generator` (or had a partial, gitignore-respecting copy of `data/`) would crash with a raw `FileNotFoundError` traceback instead of the clean message the model case already got. Reproduced directly first (patched `DATA_DIR` and called `held_out_test_scenarios()`/`run_controller()` outside Streamlit, confirmed the raw crash), then locked in as two more failing tests in `tests/test_app.py` before the fix, both now passing after it: `get_scenarios()` and `run_both()` are each wrapped in `try/except FileNotFoundError`, reporting the specific missing path and the setup command that produces it, same `st.error()` + `st.stop()` shape as the model check. Same caching gotcha applied here as the model case: both are `@st.cache_data`, so the tests clear `st.cache_data` before patching or an earlier test's real cached result would mask the missing-file path.

**Done when:** the loop runs end to end on the demo machine — **reached**, `streamlit run src/app.py` (or `.claude/launch.json`'s `railflow-demo` config), all of `tests/test_app.py`'s three missing-file guards (model, scenarios, weather) passing (229 at the time this was written; see the top-level test count in README.md, which is the one place this project keeps that number so it can't drift out of sync across documents again).

**Screen-recording fallback — partially done.** Three automated recording paths were tried and tested, not assumed: the Chrome-extension browser (would have used its GIF recorder) wasn't connected in this environment; this project's own sandboxed browser preview pane isn't compositing frames (`screenshot failed: the Browser pane is not displayed`); no local `ffmpeg` is installed to assemble a video from still frames. All three confirmed unavailable by trying them directly. Since none of that is something code can fix, and a jury-facing recording benefits from a human narrating it anyway, the deliverable instead is `docs/demo_recording_script.md` — an exact, timed (~75s) walkthrough script written against the app's real UI, using the built-in Windows Xbox Game Bar (`Win`+`G`, no install) as the recording tool. **Still needed before presenting live:** someone actually records it following that script.

---

### M8 — Handover-ready
**Effort:** 1h

- `docs/serial_protocol.md` — the UART frame definition (fixed-order fields: `T_in, T_out, N_pass, setpoint, Q_cmd`), units, ranges, and cadence. This is the deliverable for whoever owns the board.
- `src/serial_bridge.py` — sender validated against a virtual COM loopback, so it is testable without hardware.
- Expand `README.md`: what this is, how to run it, and what the numbers do and do not claim.

**Design decisions made and written down, not left implicit:** one-way only (Railflow → board; a status-reporting return channel is explicitly out of scope, not overlooked); ASCII CSV framing with an NMEA-0183-style XOR checksum (hand-verifiable from a bare terminal capture, no tooling needed to debug); a 30 s nominal cadence justified against the plant's own fastest mode (`tau_fast`, 66 s at the default internal convection coefficient, still 35 s even at the fastest extreme of that coefficient's own sensitivity sweep) and the actuator's dead time + lag (`dead_time_min` + `tau_act_min`, 2 + 5 min) — the actuator, not the link, is always the bottleneck.

> **M10 update — `setpoint_c` is now the primary field; `q_cmd` is demoted.** At the time M8 was built, `Q_cmd` was sent as a normalized capacity fraction (reusing `AnticipatoryController.last_frac`) rather than raw watts, specifically so the board's real actuator capacity never had to match Railflow's modelled one — and the link-loss watchdog fell back to `q_cmd = 0` after 3 missed intervals. Since M10 (see its own section, below), Railflow's compute cannot touch a real HVAC unit's control electronics at all, so `q_cmd` is no longer something the board should act on — it's kept only as an optional, simulation-only diagnostic. `setpoint_c` (already a field in the frame — no protocol change needed) is now the decision the frame exists to deliver: the same kind of value a passenger or technician could enter on the unit's own thermostat. The watchdog's safe fallback is correspondingly now "the board reverts to its own local/default setpoint," not `q_cmd = 0`. Full detail in `docs/serial_protocol.md` §1, §4, §4.1, §7.6.

**Built and verified, not just documented.** `src/serial_bridge.py` implements `encode_frame()`/`decode_frame()`/`checksum()` plus a `SerialBridge` sender class. Validated against pyserial's own built-in `"loop://"` in-memory loopback — confirmed working in this environment before writing anything against it, not assumed from pyserial's docs — which needs no virtual-COM driver install and no admin rights, satisfying "validated against a virtual COM loopback" without any hardware or system-level dependency. `tests/test_serial_bridge.py` locks the protocol document's own worked checksum example (§6) in as a regression check specifically so the document and the code can never silently drift apart, plus range-validation (every field rejected at both edges, accepted at both boundaries), checksum-corruption rejection, and the sequence counter's wrap-at-256 behaviour.

**One real mistake caught before it shipped:** the protocol document's first worked checksum example was hand-guessed (`*5A`) rather than computed — wrong. Caught by actually running the computation instead of trusting the guess (`0x3F`, not `0x5A`), fixed in the document, and locked in as `test_checksum_matches_the_protocol_documents_worked_example` so a document/code mismatch here is now a test failure, not something a board implementer discovers the hard way.

**Done when:** someone else can implement the board side from the document alone, without asking questions — **reached**.

> **Scope decision, made after M8 was already built: this protocol will not drive real hardware for the rail-cabin simulation.** The full-scale train model (M0–M7's results, the Streamlit demo) is presented as software only — live in the demo or its recording, never wired to a physical ESP32. `docs/serial_protocol.md`/`src/serial_bridge.py` stay in the repo, not deleted: as a design record (the reasoning, the checksum convention, the range/cadence justification method are all still real work worth keeping visible) and as a **template**, not a drop-in, for whatever the physical prototype actually needs. Checked directly, not assumed reusable as-is: the field *semantics* are rail-scale-specific (`Q_cmd` as heating/cooling capacity assumes a real compressor, which the prototype's plain fan isn't; `N_pass` assumes a headcount, which PIR can't produce; the 30 s cadence and every field's range were derived from the rail cabin's own `tau_fast`/weather bounds) and the direction is one-way, while the prototype's "continuous decision mechanism" needs a live loop (sensors → PC → decision → actuator, repeatedly) — a second frame type at minimum. The prototype's own protocol is scoped as a new, separate artifact once real measurements exist (M9 §3), reusing this one's mechanism, not its numbers.
>
> **M10 reinforces this decision; it doesn't reopen it.** The reframing that made `setpoint_c` the protocol's primary field (see above) exists for the same underlying reason this scope decision was made: Railflow's compute doesn't get to touch real HVAC control hardware. A setpoint recommendation is, if anything, an even better fit for "software only, never wired to a physical ESP32" than a capacity-fraction command was — there's less reason to ever change this decision, not more. M9 §3's physical prototype remains separately scoped and untouched by any of this.

---

### M9 — Humidity, a benchmark, SHAP, and the physical prototype
**Status:** (1) humidity — reached. (2) benchmark + retune — reached. (3) physical prototype — not started, blocked on real measurements from the team. (4) SHAP + diagnosis — reached. Everything except (3) needed nothing new from anyone; (3) does.

**1. Humidity — a simplified, honestly-bounded moisture state. Built.**
Researched before building anything: a *full* psychrometric model needs a coil Apparatus Dew Point and Bypass Factor — two more `[ASSUMPTION]`-tagged numbers this project doesn't have, and modeling them cascades into a features/retraining decision the same way the fresh-air correction did. Not worth it this close to presenting.

What got built instead: `CabinState` gained a third state, `w_air_g_kg` (humidity ratio), evolved by a moisture balance in `CabinModel.step()` — passenger generation (`occupancy.latent_heat_w_per_pax` converted via a new sourced constant, `thermal_mass.latent_heat_vaporization_j_kg`, a unit conversion of an existing number, not a new domain assumption) plus ventilation/door exchange against outdoor humidity ratio (`weather.humidity_ratio()`, previously only used to *justify not* modeling humidity, now the moisture balance's own building block — same Magnus formula, no duplicated logic). A new inverse function, `weather.relative_humidity_from_ratio()`, reports the state as RH% for anything human-facing (an LCD, a log); confirmed to round-trip `humidity_ratio()` exactly (`tests/test_weather.py`, six points spanning dry/saturated/hot/cold). Deliberately **no coil dehumidification term** — disclosed in `step()`'s own comment, right where `rh_air_pct` is computed, as an upper-bound overestimate whenever the AC is actively cooling: the energy accounting already charges for passenger latent load as if the coil removes it (unchanged), but the humidity *state* doesn't reflect that removal, since modeling the coupling properly is exactly the ADP/Bypass-Factor complexity being deferred.

**Tested before building, then re-tested against what actually got built — the two didn't quite agree, and the discrepancy taught something real.** Before building anything: a back-of-envelope calculation (passenger moisture only, temperature *held fixed*, no ventilation) suggested +3.33 g/kg / 54.8%→70.3% RH over the existing 8-minute boarding scenario — comfortably crossing ISO 19659-2's 65% RH cabin limit, the same shape of argument that justified the two-node *thermal* model in M2. After building the real coupled model and running the *same* scenario properly isolated (matching `test_boarding_spike_is_visible_in_air_temperature`'s own isothermal/iso-humid convention, not an arbitrary outdoor/indoor gap): **+1.94 g/kg absolute (14.64→16.58), but only 55.0%→55.4% in RH%.** The absolute moisture rise is real and not tiny (`tests/test_physics.py::test_boarding_raises_cabin_humidity_ratio` locks in ">1.0 g/kg", deliberately not the exact figure). The RH% barely moves because **temperature rises at the same time**, and warmer air holds more moisture before the same RH% is reached — the two effects partially cancel in the RH-reading, not in the underlying moisture. Worth stating plainly rather than quietly using whichever number sounds better: the pre-build estimate was a genuine upper bound (isolating moisture from its own temperature coupling), not a prediction, and the coupled model is the one to trust and to cite.

**1b. Wiring humidity through to where it's actually visible — found missing during a whole-project review, fixed the same day.** Built-and-tested is not the same as *used*: `evaluate.py::run_controller()` never captured `w_air_g_kg`/`rh_air_pct` from `CabinModel.step()`'s output, and `CabinInputs.rh_out_pct` was never fed real weather (every call sat on the 50% default) -- so despite existing and passing its own tests, humidity was invisible in the demo, the trajectory data, and the generated dataset. Fixed in `run_controller()` and `data_generator.py` (both now pass the scenario's real `rh_out_pct` and capture the two new fields) and surfaced in `src/app.py` as a third chart panel.

Two more real things were found only by testing the wiring, not by inspecting the code:

- **A single shared humidity line was the first design, and it was wrong.** `w_air_g_kg` genuinely IS identical for two controllers on the same scenario (the moisture balance has no `q_hvac` term). But `%RH` is moisture *relative to what the air can hold at its current temperature*, and the two controllers reach different temperatures -- so their `%RH` readings differ even though the underlying moisture doesn't. Caught by a failing test (`test_both_controllers_share_moisture...`), not by inspection. `app.py`'s humidity panel now plots one line per controller, like every other panel, with the caption explaining why they're close but not equal.
- **`%RH` could exceed 100%.** Nothing in the original moisture balance stopped `w_air_g_kg` from exceeding what the air can physically hold as vapour at its own temperature -- confirmed by a real test on real Cairo/Aswan weather, not a synthetic edge case. Fixed with a saturation cap inside `step()`'s substep loop (`w_air_g_kg` clamped to `humidity_ratio(t_air_c, 100.0)` each substep) -- passive condensation (a window or wall sweating), a cheap, different mechanism from the still-undone coil dehumidification term, disclosed the same way. This introduced a further, smaller effect worth naming: since the cap depends on temperature and `w_air_g_kg` is a persistent state, one substep where the cap binds differently for two controllers forks their moisture trajectories forward from that point -- so two controllers' absolute moisture can end up *close* on a real scenario but not always bit-identical (`tests/test_evaluate.py::test_saturation_cap_can_make_two_controllers_moisture_diverge` bounds this at <2 g/kg rather than asserting exact equality).

**2. Benchmark: other models, and other hyperparameters. Built and run.**
`src/benchmark_models.py` reuses M4's exact pipeline (`add_features()`, `chronological_split()`, `FEATURE_COLUMNS`) end to end — only the model families are new, plus a one-hot encoding of the three categorical columns shared identically across every non-LightGBM model, so the comparison is about the model, not about which library got the more favourable preprocessing. Six candidates, each tuned against VAL and scored once on TEST, same discipline as everywhere else in this project: the shipped LightGBM config, a 9-point LightGBM hyperparameter grid, linear regression, a 9-point random forest grid, XGBoost, and CatBoost.

**The finding: model family barely matters here.** All six land in a tight band, test MAE 4.27–4.34 °C — a **1.6% relative spread**, with a properly-regularised random forest edging out a tuned LightGBM by 0.02 °C (noise-level, not a real difference) and even plain linear regression close behind at 4.31 °C. The feature engineering is carrying the result, not the model family — a genuinely useful thing to know, not the finding that was expected going in.

**One real diagnosis along the way, not a clean result reported at face value.** Random forest's first pass (near sklearn-default hyperparameters) scored far worse — until checking train-vs-test MAE showed why: train MAE 0.85 °C against a test MAE of 4.46 °C, textbook overfitting on the ~15k-row training set. Re-run with a proper VAL-selected regularisation grid (matching the same discipline LightGBM's hyperparameter search already used, not a hand-picked fix based on peeking at test score, which would have been exactly the leakage this project's methodology exists to prevent) — closed the gap entirely and made it the nominal winner.

**Recommendation, not just a leaderboard: keep LightGBM (tuned). Adopted.** Random forest's 0.02 °C edge cost roughly 80–300× the training time (random forest's full grid: ~137 s; LightGBM's: ~1.8 s) for a gain inside this comparison's own noise floor — no cross-validation confidence interval was computed, this is a single chronological split. The tuned LightGBM hyperparameters (`num_leaves=15`, `learning_rate=0.02`, down from the shipped `31`/`0.05`) beat the shipped config by a real, free 0.047 °C with **zero interface change** — same `lgb.Booster`, same `.predict()` call `AnticipatoryController` already makes. Switching model family entirely (to catboost/xgboost/random forest) would need `src/controllers.py`/`src/app.py` changes for a gain this benchmark can't distinguish from noise — not worth it. `src/train.py` now ships these hyperparameters; the model at `models/lgbm_t_air_forecaster.txt` was retrained on them, and every downstream number (M5, M6) was regenerated against the retrained model, not left stale — see M5's section for what moved, and note this is also what made the gain_k/ff_weight re-tune below necessary rather than optional.

All six trained models are saved under `models/benchmark/` (gitignored, same as every other model artifact) for inspection; full results in `data/m9_model_benchmark.csv` (also gitignored).

> **M10 forward-pointer.** The `gain_k`/`ff_weight` re-tune this section made necessary (see M5's section) belonged to the retired watts-dispatch law. `gain_k` no longer exists; `AnticipatorySetpointAdvisor`'s equivalent tunable is `max_shift_k`, currently an untuned `[ASSUMPTION]` placeholder. Re-earning this re-tune for `(ff_weight, max_shift_k)`, same VAL-grid/TEST-once discipline, is explicit Phase-2 scope — see M10's own section.

**3. The physical prototype — its own train process, its own protocol, its own continuous decision loop. Deliberately not a repurposed M9(2) or M8.**
Scoped clearly, after M8 was already built and the rail simulation's own hardware plan was dropped (see M8's note above): the prototype gets a **separate model**, trained on a **separate dataset**, generated from a **separate config** (`config/prototype_params.yaml` — the existing pipeline is entirely config-driven, so this is a new config file, not new code, *once every physical parameter in it is sourced the same way every rail-cabin parameter was*: enclosure dimensions and material, the 5V fan's real airflow/capacity, sensor placement) — not the M4 forecaster retargeted, a genuinely different system-identification exercise at a genuinely different scale. It also needs a **separate protocol**: not M8's (see that section's note — direction, field semantics, ranges and cadence are all rail-scale-specific), but a new one built the same way M8's was, once real measurements make it possible to derive real numbers instead of guessing them. And it needs a **continuous decision loop**, not a one-shot command: sensors → PC → decision → actuator, repeating, which is a live-loop bridge, not the single-frame sender M8 built.

Every piece of this is blocked on the same thing: real physical measurements from whoever is building the hardware — a written request already sent to the team (enclosure material/dimensions, fan airflow, the fan-as-ventilation and PIR-as-event-trigger reframing decisions), no reply yet. A decision on which forecast features a prototype can actually supply is also still open: the lookahead features (`expected_boarding`, `time_to_next_station_min`) assume a timetable a benchtop demo has no analogue for.

*(M10 confirms this scope is unaffected by the rail-cabin output reframing: still its own, separate model/dataset/config/protocol, still blocked on the same real measurements, still not reusing M8's — or M10's — field semantics as a drop-in.)*

**4. SHAP feature attribution — in the original 5-day plan ("~30 min with LightGBM"), never actually built until a whole-project review flagged the gap. Built.**
`train.py`'s own `__main__` block already prints LightGBM's native `feature_importance(gain)`, which answered a narrower question than it looked like: gain credits a feature for being useful *wherever* it appears in a split, with no sense of how much it moves any *given* prediction or in which direction. `src/shap_analysis.py` adds `shap.TreeExplainer` (exact for tree ensembles, not sampled/approximated) on the M4 test split — the same held-out dates every other headline number here is reported against, since explaining a model on its own training data risks explaining memorisation.

**Checked before trusting the result, not assumed to work.** LightGBM's native `categorical_feature` support (this model uses it for `city`/`direction`/`pattern`) has a documented history of breaking SHAP's `TreeExplainer` outright (`shap/shap#170`, "could not convert string to float") on some version combinations. Tested directly against this project's real categorical columns before writing a line of the analysis script: not only did it run, SHAP's own additivity property (`sum(shap_values) + expected_value == raw prediction`) reproduced the model's actual output to **1e-14** — floating-point precision, not "close enough." Locked in as a permanent regression test (`tests/test_shap_analysis.py`) so a future library upgrade that reintroduces the incompatibility fails loudly instead of silently.

**The finding: strong agreement at the top, real disagreement further down — and the disagreement is informative, not noise.** SHAP's top 3 by mean |impact| match `feature_importance(gain)`'s top 3 exactly: `t_out_fcst_h`, `setpoint_c`, `t_out_c` — both methods agree the forecasted disturbance dominates. Below that they diverge: `t_mass_c` ranks 4th by gain but drops to 9th by SHAP (a few very-high-gain splits, likely covering rare extreme-thermal-mass scenarios, that don't move *most* predictions much — visible in the summary plot as a handful of far-right outlier points, not the bulk of the distribution); `ghi_fcst_h` (forecasted solar) doesn't crack gain's top 10 at all but ranks 7th by SHAP — used less dramatically per split, but more *consistently* across rows. And the bottom of the ranking is itself a finding worth stating: `t_air_c` and `error_c` — current cabin temperature and current error from setpoint, i.e. "just look at where things are right now" — are the two **least** important of all 28 features (locked in as `tests/test_shap_analysis.py::test_forecast_features_rank_above_current_state_alone`, not just eyeballed off the plot). The model leans on forecast and trend information, not "smart persistence" — direct evidence for this project's actual premise, not merely consistent with it.

Plot at `data/m9_shap_summary.png` (committed, same convention as `data/m6_tau_act_sweep.png` — `.gitignore` only excludes `data/*.csv`/`*.pkl`/`*.parquet`, not plot images); reproducible via `python -m src.shap_analysis`. Ranked table at `data/m9_shap_importance.csv` (gitignored, regenerable).

**4b. Following up on what SHAP found: a real diagnosis, two tested remedies, neither adopted — a validated negative result, not a wasted afternoon.**

The SHAP summary plot alone doesn't say *why* a feature matters or *where* the model is wrong — a deeper pass applied established model-diagnosis methodology (not ad-hoc eyeballing) to the shipped forecaster: bias/variance via the train/val/test gap, residual analysis, slice-based error breakdown by city/direction/pattern/temperature-regime, a learning curve (does more data help?), and SHAP *dependence* plots (the functional form per feature, not just its aggregate importance).

**What the diagnosis found, each connected to a named, real phenomenon rather than left as an unexplained quirk:**
- **Systematic under-prediction** (mean residual −1.43 °C on the test split, left-skewed) concentrated in the hottest 52% of test rows. This is not unique to this project — it's the same documented failure mode as large AI weather models (GraphCast, Pangu-Weather), which systematically underestimate record-breaking temperatures for a structural reason: tree/boosting models predict per-leaf averages of whichever training points land together, so rare extremes get pulled toward their leaf's bulk — physics-based forecasts (HRES) still beat them specifically on record events. Mechanistically: this project's `regression_l1` objective fits the *median* per leaf; a left-skewed residual distribution has its median sitting below its mean, which is exactly the systematic bias measured.
- **A SHAP dependence plateau** for `t_out_c`/`t_out_fcst_h` across roughly 32–40 °C, then a sharp break above it — tested whether this was a data-sparsity artefact (it wasn't: that band holds over half of all training rows, the densest part of the dataset) before concluding it's more likely a real, learned reflection of `comfort.sliding_t_out_high_c = 40` — the cabin's sliding setpoint (and the HVAC's practical authority to hold it) has its own ceiling at exactly that temperature. This is the textbook control-theory phenomenon of actuator saturation (a hard, well-studied nonlinearity — quasi-linear response within authority, flat/nonlinear once saturated) showing up as an ML interpretability artefact.
- **The learning curve is already flat**: quadrupling the training data (13→54 days) improved test MAE by only ~4% relatively. The standard diagnosis of a flat learning curve is "more data of the same kind won't help much" — ruling out the intuitive "just generate a bigger dataset" fix before it could consume a day chasing a dead end.

**Two remedies, grounded in real precedent, both tested — one showed a real, if second-order, effect; one didn't:**
- **Monotonic constraints** (LightGBM's `monotone_constraints`, forcing the outdoor-temperature relationship to never reverse) have direct precedent in HVAC/building-energy modelling specifically (a 2025 chiller-energy-prediction study uses the same technique for the same reason). Discovered along the way: `monotone_constraints` is flatly incompatible with the `regression_l1` objective this project ships (LightGBM's fatal error, hit directly rather than assumed from docs) — its constraint enforcement needs a well-behaved Hessian, which L1/MAE doesn't have. Testable only under an L2 (squared-error) objective instead.
- **An explicit "HVAC saturation" feature** (degrees past `sliding_t_out_high_c`, engineered directly from the config value already driving the physical ceiling) — general actuator-control ML guidance recommends exactly this kind of distance-to-limit feature. Tested; it did not help (test MAE 4.315 °C vs the 4.292 °C baseline, marginally worse) — most likely because `setpoint_c` (already a feature) is already a deterministic function of this same threshold, so the explicit feature added redundant noise rather than new information. Literature-recommended is not the same as helpful *here* — dropped rather than kept on the strength of the precedent alone.

**The methodology caught its own false lead — which is the point of having one.** An initial exploratory comparison (L1 baseline vs. L2-objective vs. L2+monotonic), scored directly on the TEST split for a quick look, showed L2+monotonic winning clearly (test MAE 4.292→4.165 °C, bias −1.43→−0.15 °C). Redone properly before trusting it — a 9-point (objective × num_leaves × learning_rate × monotonic) grid selected on **VAL**, per this project's standing discipline (same shape as the gain_k/ff_weight search) — and **the L1 baseline won VAL outright** (3.956 °C vs every L2 variant's 4.11–4.15 °C). The earlier test-split result wasn't a real improvement; it was the model happening to fit the TEST window's specific weather better than VAL's, which is exactly the kind of split-specific illusion the val-then-test-once discipline exists to catch. **Not adopted. The shipped model (L1, `num_leaves=15`, `learning_rate=0.02`) is unchanged.**

The underlying weakness (real, diagnosed, connected to known ML/control phenomena above) remains a **disclosed, known limitation** rather than a papered-over one — the same honesty pattern as M9's own humidity dehumidification gap: characterised precisely, not hidden, not forced into a fix that doesn't hold up under the project's own evidence bar.

**Done when:** (1) — **reached**, humidity state built, tested, and disclosed (not merely "decided", the bar this line originally set). (2) — **reached**, six models benchmarked and checkpointed, honest finding on record (model family barely matters here); the tuned LightGBM hyperparameters are adopted into the shipped model, and the gain_k/ff_weight re-tune this made necessary is also applied — see M5's section for the regenerated headline numbers. (3) — blocked on real prototype measurements; not yet scoped to a concrete bar. (4) — **reached**, SHAP built, verified correct (not just non-crashing), and its finding is a real result, not a formality.

---

### M10 — Reframe the output: a setpoint recommendation, not a power command
**Effort:** ~3h (Phase 1 only — code restructuring + doc rewrites, no retuning)

**Why.** The team was told the real HVAC unit's control electronics cannot be touched or modified — doing so would void the manufacturer's warranty. Every controller through M9 ends its decision as `Q_cmd`: a watts command (`frac * capacity_w`) fed directly into the simulated actuator. Deploying that for real would require exactly the kind of physical/electrical access to the compressor that's now off the table. The fix: recommend a **desired cabin interior temperature** (a setpoint, °C) instead — the same kind of value a human or the unit's own existing thermostat/remote/BMS interface already accepts, with zero hardware modification.

**The core insight this reframing turned out to hinge on:** `ThermostatController` was *already*, structurally, "setpoint → hysteresis → watts" — it just derived its own setpoint internally instead of accepting one. Splitting that mechanism into its own class (`PlantResponse`) gives, almost for free, a stand-in for "the real, third-party Control Unit this project will never redesign" — which mirrors the warranty constraint exactly, rather than merely working around it.

**Build (Phase 1 — code + docs restructuring only, no retuning):**
- `src/controllers.py` — new `PlantResponse` class (the shared hysteresis-around-a-supplied-setpoint mechanism, extracted verbatim from `ThermostatController`'s old body); `ThermostatController` shrinks to a trivial static-schedule advisor (`.recommend_setpoint(inputs) -> float`), no longer owns hysteresis state; `AnticipatoryController` renamed `AnticipatorySetpointAdvisor`, dispatch tail changed from `frac * capacity_w` (watts) to a clipped setpoint shift (°C); `gain_k` retired in favour of `max_shift_k` (a different physical quantity — a setpoint-authority bound, not a capacity-fraction gain).
- `src/evaluate.py` — `run_controller()`'s loop gets an explicit two-stage composition: `advised_setpoint_c = advisor.recommend_setpoint(inputs)`, then `cmd = plant.respond(inputs.t_air_c, advised_setpoint_c)`, one `PlantResponse` instance shared across the whole run. New `advised_setpoint_c` column in the returned trajectory. `score()`/`ScenarioResult`/`degree_hours_outside_band()` untouched.
- `src/compare_controllers.py`, `src/app.py` — renamed-class references only; `app.py`'s middle chart panel now plots each controller's `advised_setpoint_c` (°C) instead of `cmd_w` (kW).
- `src/sweep_tau_act.py`, `src/plot_tau_act_sweep.py` — confirmed **zero code changes needed**; only `tau_act_min`'s interpretation shifts (see M6's section).
- `docs/serial_protocol.md`, `src/serial_bridge.py` — `setpoint_c` (already a field in the frame) reframed as the primary, acted-upon field; `q_cmd` demoted to an optional, simulation-only diagnostic. Wire **format** byte-for-byte unchanged — no protocol version bump needed.
- `README.md`, `docs/USER_MANUAL.md`, `docs/demo_recording_script.md`, `docs/PARAMETERS.md` — rewritten to describe the recommendation, not a command (see each doc's own M10 notes).

> **A methodological side-effect, not just a deployability fix.** Because every compared arm now shares the *identical* `PlantResponse` mechanism, M5/M6's comparison isolates the advisor's actual setpoint choice rather than conflating it with a difference in dispatch law (the old comparison pitted a bang-bang baseline against a proportional-fraction candidate — structurally different mechanisms). A second, unplanned simplification: the old law let `AnticipatoryController` dispatch real heating while `ThermostatController` was cooling-only by design — an asymmetry between the two arms that had nothing to do with what M5 was supposed to be measuring. Under the new law both arms share the same cooling-only `PlantResponse`; a positive setpoint shift just tolerates a warmer cabin longer, never dispatches heat. (Phase 2 should confirm scenarios needing real heating stay negligible under the new law too — see below.)

**M5/M6/M9(2)'s existing numbers — marked superseded, not deleted.** The +3.9% headline energy saving (M5), the `tau_act` sweep's shape (M6), and the `(gain_k=3.0, ff_weight=0.45)` re-tune (M9§2) all describe the retired watts-dispatch law. Kept on the record in their original sections, each now under an explicit "M10 status" note — the same way this project has already kept every earlier superseded number (the fresh-air-rate correction, the M9 LightGBM retune) rather than silently overwriting them.

**Verified, not assumed, before moving to docs:** `pytest tests/ -q` — 270 passed, 3 xfailed (exactly the three directional `compare_controllers` tests pinned to the retired law's numbers, marked `xfail` rather than deleted or left silently red — see `tests/test_compare_controllers.py`). Critically, `tests/test_evaluate.py::test_run_controller_matches_the_manually_traced_reference` — `ThermostatController`'s pinned regression (30.86 kWh / 0.47 K / 5.06 K) — passed **byte-identical**, confirming `PlantResponse`'s extraction changed nothing about the baseline's actual behaviour.

**Done when (Phase 1 only):** `PlantResponse`/`AnticipatorySetpointAdvisor` exist and are wired through `evaluate.py`, `compare_controllers.py`, `app.py` — **reached**. The pinned baseline regression is unchanged — **reached**. The app's middle panel shows setpoints, not watts — **reached**. No listed doc still claims Railflow's software directly commands HVAC power — **reached** (this section plus the touched M2/M5/M6/M7/M8/M9 notes plus README/PARAMETERS/USER_MANUAL/demo_recording_script). The three directional `compare_controllers` tests are `xfail`-marked, not silently red or deleted — **reached**.

---

### M10, continued — Phase 2 (complete)

Was deferred deliberately, at the user's own request, to a separate pass once Phase 1's architecture had settled. Now done — the headline finding is a disclosed parity result, not a win, arrived at honestly rather than forced:

1. **Regenerated M5's headline comparison — done.** Not a repeat of +3.9%: three grid-search attempts (25 + 25 + 30 = 80 configurations, all measured on the honest VAL split) found no configuration beats the static baseline. Full result in M5's own section (bottom) and `AnticipatorySetpointAdvisor`'s docstring (`src/controllers.py`).
2. **Re-ran the tuning grid search on VAL, confirmed once on TEST — done, but on `(ff_weight, deadband_k)`, not `(ff_weight, max_shift_k)` as originally planned.** `max_shift_k` turned out not to be the lever that mattered (attempt 1 showed it barely moves the result once ≥ ~1-2 C, since the recommendation is clamped to the sliding-setpoint envelope regardless); `deadband_k` — this advisor's own deadband, decoupled in Phase 2 from `PlantResponse`'s switching band it originally reused outright — turned out to be the one that did. Same `n_both_better`-first, TEST-confirmed-once discipline throughout, adjusted mid-course to actually test the right variable rather than mechanically following the original plan.
3. **Regenerated the M6 `tau_act` sweep — done**, against the new shipped `(ff_weight=0.0, deadband_k=1.5)`. See M6's own section for the current table.
4. **Cooling-only-under-both-arms harmlessness — addressed by existing evidence, not a new experiment.** `PlantResponse` is cooling-only by design (same documented reason every other baseline in this project is: Egypt is cooling-dominated, heating is rarely engaged — see `PlantResponse`'s own docstring). With the shipped `ff_weight=0.0` producing shifts that net out to near-parity with the static baseline, there is no meaningful suppressed-heating-demand signal to chase here; the pre-existing cooling-dominated-climate finding already covers it.
5. **`StochasticController` exploring setpoint-space directly — still open, now less promising, not pursued.** The root cause identified in Phase 2 is `PlantResponse`'s lack of proportional response (a structural property of the *plant*), not M4's forecast quality or training-data coverage (M4 already beats persistence by +21%) — better training data would not change a bang-bang unit's inability to act on "how urgent" a forecast is. Left open for whoever revisits this, with that caveat attached so it isn't re-tried blind.
6. **Widening the `[sliding_setpoint_low_c, high_c]` clamp — not pursued, same reasoning as (5).** Phase 2's grid varied `max_shift_k` (attempt 1) and found it didn't matter once past ~1-2 C — the advisor already had more setpoint-shifting authority than it could usefully spend. The limiting factor was never authority to shift further; it was that shifting a bang-bang switch's timing doesn't have the right kind of leverage on the outcome. Widening the clamp would not address that.

**What would actually be needed for the forecast to pay off, if anyone picks this up later:** a receiving unit with SOME proportional or staged response (even a coarse 2-3 stage compressor, not full continuous modulation) — something `PlantResponse` deliberately does not model, since it stands in for the least capable real unit this project could defensibly claim compatibility with (see `PlantResponse`'s own docstring, and `ThermostatController`'s original "on/off... is documented in vehicle A/C patents, so this is an honest baseline, not a straw man"). That would be a genuinely new architecture question, not a re-tune — flagged here, not attempted, since it changes what `PlantResponse` is claiming to represent.

---

### M11 — AI-controlled window tinting
**Effort:** ~2h, deliberately scoped small

**Why.** First-round judges flagged that auto-dimming SPD (Suspended Particle Device) glass reacting to its own light sensor isn't novel — it's an existing product. The team's differentiator: use the SAME lookahead machinery this project already built for HVAC (M4's forecast features) to pre-empt sun exposure before it arrives, not react to it after the fact. Explicit team guidance, followed deliberately here: *"no one is going to look at the code... we just need a simulation that shows our idea"* — this milestone is scoped to demonstrate the contrast clearly, not to build production tinting control.

**Build:**
- `src/cabin_model.py` — `CabinInputs.tint_level` (0.0 clear, 1.0 fully darkened, default 0.0 for every pre-M11 caller). Effective SHGC now slides between `solar_heat_gain_coefficient` (the pre-M11 fixed value, kept as the tint_level=0 ceiling — every earlier milestone's result reproduces bit-for-bit) and a new `solar_heat_gain_coefficient_tinted` (`config/cabin_params.yaml`, `[ASSUMPTION]`, no vehicle-specific datasheet sourced).
- `config/cabin_params.yaml` — `route.tunnel_zones_min`: two illustrative sun-blocking zones by minute-into-journey. Marked `[ASSUMPTION] / DEMO DEVICE` explicitly, not a claim about the real Cairo–Alexandria route (flat Nile Delta terrain; no tunnel was sourced for it) — stands in for whatever a real deployment's route would actually have (a tunnel, a station canopy, a shaded stretch).
- `src/occupancy.py` — `OccupancyProfile.in_tunnel`, computed from the config zones.
- `src/tint_controller.py` — `ReactiveTintController` (reacts to current `ghi_w_m2`, mimics stock SPD — the "not novel" baseline) and `AnticipatoryTintAdvisor` (reacts to `ghi_fcst_h`, the SAME t+H forecast feature the HVAC advisor already consumes). Both deliberately rule-based (linear GHI→tint map, `[ASSUMPTION]`-tagged reference point sourced against real fetched 2024 summer GHI maxima), not trained models — see the team guidance above.
- `src/evaluate.py` — `run_controller()` gains `tint_controller` (default `None`, unchanged behaviour) and applies the tunnel mask to `ghi_w_m2`/`ghi_fcst_h` for every consumer, so a tunnel needs no separate "in_tunnel" field anywhere downstream — it already reads as "no sun."
- `src/app.py` — a new "Window tinting" demo section: energy compared across no-tint/reactive/anticipatory (same HVAC advisor throughout, isolating tinting's own effect), and a tint-level-over-time chart with tunnel zones shaded.

**No separate baseline-vs-advisor rigor pass (unlike M5/M6) — deliberately.** This is a single demonstration, not a second tuned comparison study; matches the team's explicit "simplify, don't waste more time" guidance.

**Verified, not just plausible:** `tests/test_tint_controller.py` (10 tests) — tint_level=0.0 reproduces pre-M11 `q_solar_w` exactly; more tint never increases solar gain; a configured tunnel reaches the controller as `ghi_w_m2≈0`; tinting measurably lowers energy versus no tint; and, the actual demo claim — at a minute where the anticipatory advisor's t+H lookahead lands inside the tunnel but the current minute doesn't, anticipatory tint is already clearing while reactive tint is not. On the app's default scenario: energy no-tint 36.95 kWh → reactive 34.49 kWh (+6.7%) → anticipatory 34.67 kWh (+6.2%) — tinting's overall benefit is real and larger than M10's HVAC-setpoint result; anticipatory vs. reactive is close and scenario-dependent (this scenario happens to favour reactive slightly), same per-scenario variability already documented for the HVAC advisor. The chart tells the clearer story: reactive visibly drops to clear only AT each tunnel boundary; anticipatory drops ~30 min (`control_horizon_min`) *before* it, exactly the anticipation being demonstrated.

**Done when:** a working simulation shows tint reducing HVAC load and reacting ahead of a route feature it can forecast — **reached**.

---

## What must never be cut

If time runs short, cut in this order: the live dashboard (→ static plots), the multi-season data (→ one season), the serial bridge implementation (→ protocol document only).

These five are the project. Everything else is presentation:

1. The grey-box simulator (**M2**)
2. The physics sanity tests (**M2**)
3. The chronological split (**M4**)
4. The baseline-vs-predictive comparison (**M5**)
5. The `tau_act` sweep (**M6**)

*(M10 footnote: "protected" now means the comparison **mechanism** — `compare_controllers.py`'s harness, the chronological split, the sweep infrastructure — and its **regenerated** numbers, not the specific pre-M10 figures currently on record. Regenerating them under the new setpoint-recommendation architecture is exactly the Phase-2 work M10 defers — see that section for the list. Keeping this promise honest means finishing that work, not quietly reusing the old numbers.)*

---

## Honesty rules

These are what make the work defensible under questioning:

- **Never** claim "X% savings on a real train." Say: *"X% within the same comfort band on a parameterised cabin model, across an actuator-lag range of 0–20 minutes."* The second version is both more defensible and more impressive to an engineer.
- Never assert a time constant. Derive it, or sweep it.
- Never present a number whose source you cannot name.
- State plainly that the data is synthetic and that no real train data was available. This is a stated limitation, not a weakness to conceal — reviewers find hidden ones anyway.
- **Never claim Railflow's software directly commands HVAC power output.** Since M10, the honest claim is: *Railflow recommends a cabin setpoint; a separate, unmodified thermostatic control system decides the resulting power draw.* Railflow's own compute never touches a real HVAC unit's control electronics — see M10's own section for why, and `docs/serial_protocol.md` for what actually crosses the wire.

---

## Continuing after the presentation

The foundations above are built to be extended. Natural next steps, roughly in order of value:

1. **Calibrate against real data** — if any Siemens telemetry becomes available, fit `C_eff` and `UA` to it. The model is already parameterised for exactly this.
2. **Proper MPC** — replace the heuristic predictive controller with a real receding-horizon optimiser over the learned forward model.
3. **Sequence models** — a GRU or 1D-TCN over a 60-minute window learns the lag implicitly, without hand-built EWMA features. A fair comparison against the LightGBM baseline is a genuine result either way.
4. **Humidity and full PMV comfort** — the current comfort metric is temperature-only; PMV is the standard rail comfort measure.
5. **Multi-zone** — per-car or per-zone control, with coupling between zones.
6. **Richer physics** — the terms deliberately excluded in M3: speed infiltration, tunnels, solar orientation.
