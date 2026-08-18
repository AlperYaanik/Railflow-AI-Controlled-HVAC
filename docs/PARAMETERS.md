# Parameter Rationale

Why every coefficient in `config/cabin_params.yaml` has the value it has.

`cabin_params.yaml` carries a one-line `[tag]` per value so it stays scannable.
This document carries the reasoning: what was considered, what was measured,
what was rejected, and what would change the answer. It exists so that a future
session — or a reviewer asking "where does that number come from?" — does not
have to re-derive any of it.

**No thermal time constant appears in the config.** Both are consequences of
`C` and `UA`, computed in `src/config.py:system_time_constants()` and checked
against the simulator in `tests/test_physics.py`.

---

## How to read this

| Tag | Meaning |
|---|---|
| `[EN13129]` | Stated scope or requirement of EN 13129:2016 (mainline rolling stock) |
| `[UIC553]` | UIC leaflet 553-1, climatic chamber test conditions |
| `[GEOMETRY]` | Computed from coach dimensions declared in the config |
| `[TYPICAL]` | Conventional engineering value for this equipment class |
| `[ASSUMPTION]` | Our own estimate — replace these first if real data arrives |

**Confidence** below is our own rating: **High** (geometric or standards-backed),
**Medium** (conventional value, wide agreement), **Low** (reasoned estimate, no
source). Low-confidence values are only acceptable where a sensitivity sweep
shows they do not drive the result — which is why the sweeps exist.

---

## Geometry

| Parameter | Value | Confidence |
|---|---|---|
| `length_m` | 26.4 | High |
| `width_m` | 2.9 | High |
| `interior_height_m` | 2.4 | High |
| `saloon_volume_m3` | 150.0 | Medium |
| `envelope_area_m2` | 290.0 | Medium |
| `glazing_area_m2` | 22.0 | Medium |

26.4 m × 2.9 m is the standard UIC-profile mainline coach. Gross interior volume
would be ~184 m³; we take **150 m³** because vestibules, toilets, and ceiling
ducting are not part of the conditioned saloon.

`envelope_area_m2 = 290` is sides + roof + floor + end walls for that envelope.

`glazing_area_m2 = 22` counts windows on **both** sides. This matters — see
`sunlit_glazing_fraction` below, which exists specifically to stop that from
being double-counted.

---

## Envelope

| Parameter | Value | Confidence |
|---|---|---|
| `u_value_w_m2k` | 2.2 | Low |
| `solar_heat_gain_coefficient` | 0.45 | Low |
| `sunlit_glazing_fraction` | 0.5 | Low |

**`u_value_w_m2k = 2.2`** — a modern insulated rail car body sits in the
2.0–2.5 W/m²K range. Rail bodies are far worse than buildings: thin walls,
thermal bridges through the steel structure, and a large glazing fraction.
Gives `UA_envelope = 290 × 2.2 = 638 W/K`.

**`solar_heat_gain_coefficient = 0.45`** — tinted/laminated rail glazing.
Clear glass would be ~0.8; heavily tinted transit glazing ~0.35.

**`sunlit_glazing_fraction = 0.5`** — *added during the M1 audit, after
noticing a latent double-counting bug.* Only one side of the coach faces the sun
at a time, but `glazing_area_m2` counts both. Without this factor the model
would apply full GHI to the full glazed area. We do not model orientation (out
of scope by decision in M3), so this is the flat-rate stand-in.

Effective solar aperture: `22 × 0.45 × 0.5 = 4.95 m²`.
`test_solar_uses_only_the_sunlit_glazing_fraction` guards against regression.

---

## Thermal mass

| Parameter | Value | Confidence |
|---|---|---|
| `air_density_kg_m3` | 1.2 | High |
| `air_cp_j_kgk` | 1005.0 | High |
| `interior_mass_capacity_j_k` | 2.8e6 | **Low** |
| `interior_surface_area_m2` | 250.0 | Low |
| `internal_h_w_m2k` | 6.0 | Low |

`C_air = 150 × 1.2 × 1005 = 0.181 MJ/K`.

### `interior_mass_capacity_j_k` — the least defensible number in the project

Originally set to 1.6 MJ/K as a bare guess. Replaced with a mass budget:

| Item | Mass | c_p | Capacity |
|---|---|---|---|
| 80 seats @ 14 kg | 1120 kg | 1400 J/kgK | 1.57 MJ/K |
| Wall / ceiling panels | 900 kg | 1200 | 1.08 |
| Floor + subfloor | 1200 kg | 900 | 1.08 |
| Racks, tables, trim | 400 kg | 900 | 0.36 |
| Glazing | 300 kg | 840 | 0.25 |
| **Total if fully coupled** | | | **4.34 MJ/K** |

Not all of it couples on a one-hour timescale — deep subfloor and structure
behind panels lag badly — so we take **~65% → 2.8 MJ/K**.

**More important than the value: it barely matters.** Swept across the full
plausible range:

| `C_mass` | τ_fast | τ_slow | 8-min boarding rise | Pull-down 30→24 °C |
|---|---|---|---|---|
| 1.6 MJ/K | 1.14 min | 45.2 min | 2.25 K | 7 min |
| **2.8 MJ/K** | **1.16 min** | **77.7 min** | **2.12 K** | **7 min** |
| 4.34 MJ/K | 1.17 min | 119.5 min | 2.05 K | 7 min |

τ_fast moves <3%, the boarding spike <10%, pull-down not at all. Only τ_slow
(the mass soak) shifts, and nothing the controller sees depends on it —
everything is governed by `C_air` and `hA`. Locked in by
`test_results_are_insensitive_to_interior_mass`.

### `interior_surface_area_m2 × internal_h_w_m2k` → `hA = 1500 W/K`

Sets the air↔mass coupling, and therefore τ_fast — the sharpness of the
boarding response. 250 m² of seats, panels, floor and racks; 6 W/m²K is
mid-range for indoor natural convection (textbook 2–10) and **conservative**,
since forced HVAC circulation pushes the effective value toward the top.

Sensitivity, at the current `C_mass`:

| `h` [W/m²K] | `hA` [W/K] | τ_fast | τ_slow | 8-min boarding rise |
|---|---|---|---|---|
| 2.0 | 500 | 1.94 min | 139.2 min | 3.21 K |
| 4.0 | 1000 | 1.45 min | 93.0 min | 2.52 K |
| **6.0** | **1500** | **1.16 min** | **77.7 min** | **2.12 K** |
| 10.0 | 2500 | 0.82 min | 65.6 min | 1.67 K |
| 15.0 | 3750 | 0.61 min | 59.6 min | 1.40 K |

The two-node conclusion holds across the whole range and *strengthens* at low
`h`. **Stated honestly:** the boarding disturbance is 1.4–3.2 K against a ±2 K
comfort band, so it is *comparable to* the band across the range and exceeds it
below roughly `h = 8`. It is not unconditionally larger than the band, and the
presentation should not claim that it is.

---

## Ventilation

| Parameter | Value | Confidence |
|---|---|---|
| `fresh_air_m3_h_per_passenger` | 15.0 | Low |
| `min_fresh_air_m3_h` | 300.0 | Low |

EN 13129 mandates a per-passenger fresh air rate but **the figure is behind the
paywall**, so this is tagged `[ASSUMPTION]` rather than dressed up as
standards-derived. 15 m³/h/pax is a conventional mainline design value.

The 300 m³/h floor keeps a near-empty coach ventilated; it binds below 20
passengers, which is why τ is identical at 0 and 20 pax.

At design load: `1200 m³/h → 0.4 kg/s → UA_ventilation = 402 W/K`, comparable to
the entire envelope. Ventilation is not a minor term in a rail cabin.

---

## Occupancy

| Parameter | Value | Confidence |
|---|---|---|
| `seats` | 80 | Medium |
| `design_load_pax` | 80 | **High** `[EN13129]` |
| `sensible_heat_w_per_pax` | 70.0 | Medium |
| `latent_heat_w_per_pax` | 45.0 | Medium |

`design_load_pax` is the one genuinely standards-backed number here: EN 13129
defines design load for mainline vehicles as all seats occupied including
tip-up seats and wheelchair spaces. Unlike EN 14750 (urban), there is no
standing-passenger term — which is why this project leans on large boarding
events rather than crush loading.

70 W sensible + 45 W latent is a seated passenger at light activity (~115 W
total), standard ASHRAE-style values.

**Latent heat is deliberately not coupled to air temperature.** It is added to
the coil load — it costs energy — but does not drive `T_air`, because modelling
that properly needs a humidity state and full psychrometrics, which is out of
budget.

**This is justified by the climate, not by convenience.** Latent load from
ventilation only exists where the outdoor humidity ratio *exceeds* the indoor
target; below that, fresh air dehumidifies the cabin. Computed against a
26 °C / 55% RH target (11.5 g/kg) using `weather.humidity_ratio()`:

| City | Median outdoor | p95 | Hours exceeding indoor target |
|---|---|---|---|
| Cairo | 10.8 g/kg | 16.0 | **42.3%** |
| Aswan | 5.9 g/kg | 15.1 | **8.6%** |

So in Aswan the incoming air is drier than the cabin target more than 90% of the
time, and in Cairo the majority of the time. The latent load is dominated by
**passengers**, which the coil already carries.

**The limits of that argument, stated plainly:**

- **ISO 19659-2** sets train cabin limits of **27 °C and 65% RH**, and published
  field measurements record RH reaching **81.2%** in early morning — humidity is
  a real operational problem in rail, just not the binding one here.
- Cairo's p95 RH is 79% and its maximum 94%, so a humid tail exists.
- **Alexandria is coastal and humid.** We simulate Cairo and Aswan weather, not
  Alexandria's. That is convenient for this assumption and should be disclosed
  rather than glossed over.
- The argument is **Egypt-specific**. It would not transfer to a coastal or
  monsoon route, and the model should not be presented as if it would.

---

## Doors

| Parameter | Value | Confidence |
|---|---|---|
| `air_exchange_m3_per_min` | 28.0 | **Low** |

A **rate while the door is open** → `door_ua = 563 W/K`. Total exchange per stop
follows from the dwell length.

Sanity: 28 m³/min against a 150 m³ saloon is ~0.19 air changes per minute, so a
3-minute dwell turns over roughly half the cabin volume.

**Previously a bug.** This was expressed as a total *per stop* divided by the
route's mean dwell. That coupled the physics to the timetable: lengthening
Cairo's dwell from 8 to 20 minutes moved `door_ua` by **−36% at every station**,
including ones that had not changed. A rate is the correct form — how fast air
crosses an open doorway is a property of the doorway, not of the schedule.
`test_door_physics_is_independent_of_the_timetable` locks this.

**Still flagged:** 563 W/K approaches the entire envelope `UA` of 638 W/K, so an
open door nearly doubles the coupling to outdoors. That makes station stops a
large disturbance — which suits the project's premise, so it deserves scrutiny
rather than acceptance.

---

## Timetable and occupancy

| Parameter | Value | Confidence |
|---|---|---|
| Station dwell times | 3–8 min | **Low — see below** |
| `load_factor` (in `occupancy.py`) | 0.4 / 0.7 / 1.0 | **Low** |
| `dwell_scale` | 1.0 | **Low — exposed for sweeping** |
| Boarding / alighting counts | per config route | **Low** |

**No Egyptian National Railways ridership data is public.** Searched and
confirmed: the only concrete public figure is Ramses Station handling ~300,000
passengers/day, which is station-level, not per-train. Load factors are
therefore assumptions, handled by generating the dataset across a **spread**
(0.4 / 0.7 / 1.0 / 1.1) rather than committing to one value.

**Why 1.1 is in that list.** EN 13129 defines design load for mainline vehicles
as all seats occupied — **80 passengers**. The base route pattern peaks at 73, so
without a factor above 1.0 the standard's own worst case was never simulated.
1.1 is the factor at which the route peaks at exactly 80. It is also the worst
case in every metric: +28% energy and +0.75 K worst excursion against load
factor 0.4.

**Stopping patterns.** Real ENR services vary, so the catalogue includes both:

| Pattern | Calls | Journey | Door-minutes | Energy |
|---|---|---|---|---|
| Semi-express | 6 | 2 h 46 | 22 | baseline |
| **Express** (Talgo 2027) | **3** | **2 h 36** | **12** | **−8.9%** |

The express follows the real Talgo 2027 pattern — Cairo, Sidi Gaber,
Alexandria — which runs the route in about 2 h 30. Skipping calls also returns
their dwell time, landing at 2 h 36 without inventing a separate timetable. It
matters for M4 because the model must not assume a fixed journey length.

**Dwell times may be roughly 2× too long.** The literature puts scheduled dwells
at **30 s – 2 min**, with ~35 s to move 40 passengers through a door. Our
intermediate stations use 3–4 minutes. An 8-minute dwell at Cairo Ramses is
defensible for a terminus (crew, cleaning, scheduling), but the intermediate
stops are probably generous.

This matters because door heat scales with dwell:

| `dwell_scale` | Door-minutes | Door energy | Share of station heat |
|---|---|---|---|
| 0.50 | 13 | 1.87 kWh | 13% |
| **1.00** | **22** | **3.16 kWh** | **20%** |
| 1.50 | 31 | 4.45 kWh | 26% |

Since a longer dwell inflates exactly the disturbance our controller is meant to
fix, this is a bias **in our favour** and must not be left unexamined.
`dwell_scale` exists so it can be swept rather than trusted.

**Alighting is stored as a fraction, not a count.** The config lists absolute
alighting numbers, but those break under load-factor scaling — at 0.4 load a
fixed count can exceed the number of people aboard and drive occupancy negative.
`occupancy.py` converts them to fractions of those aboard, which keeps the route
balanced at any load factor and is lossless at the calibration point
(`test_base_pattern_reproduces_the_config_exactly`).

**Door vs passenger, quantified.** Which disturbance actually dominates:

| | During a stop (instantaneous) | Over the whole journey |
|---|---|---|
| Door | **8.6 kW** | 3.16 kWh |
| Passengers (sensible) | 5.1 kW | **12.86 kWh** |

So the *station spike* is door-dominated while *total energy* is
passenger-dominated. The disturbance being anticipated is a combined station
event, not simply "passengers board" — the presentation should say so.

---

## HVAC

| Parameter | Value | Confidence |
|---|---|---|
| `cooling_capacity_w` | 40000.0 | Medium |
| `heating_capacity_w` | 25000.0 | Medium |
| `cop_nominal` | 2.8 | Medium |
| `cop_degradation_per_k` | 0.04 | Low |
| `cop_reference_t_out_c` | 30.0 | Low |
| `cop_min` | 1.2 | Low |
| `heating_cop` | 1.0 | **High — physical identity, not fitted** |
| `dead_time_min` | 2.0 | **Low — swept in M6** |
| `tau_act_min` | 5.0 | **Low — swept in M6** |
| `supply_air_m3_h` | 4800.0 | Medium |
| `supply_air_min_temp_c` | 10.0 | Low |
| `supply_air_max_temp_c` | 45.0 | Low |
| `thermostat_hysteresis_k` | 2.0 | **Medium — two independent sources agree** |

### Thermostat hysteresis (M5)

`thermostat_hysteresis_k = 2.0` — the **controller's own switching deadband**,
not `comfort.band_k` (the scoring metric's threshold). The two are
deliberately different concepts that happen to both land near 2 K; a test
(`test_hysteresis_and_comfort_band_are_independent_config_values`) guards
against a future edit conflating them because the numbers currently match.

Two independent sources converge: a real vehicle A/C patent documents a
1.5 K hysteresis band (thresholds 3–4.5 K either side of a reference); AIRAH
(commercial HVAC guidance) gives 1 K as a minimum, **2 K as the standard
target** for systems prioritising energy savings, 3 K as a stretch target.
2.0 K is AIRAH's standard target, sitting close to the patent figure rather
than picked to match it.

**Found the hard way that this plant needs to stay single-mode.** A first
version of `ThermostatController` ran dual-mode — cooling above the upper
threshold, heating below the lower one, matching a generic thermostat. Traced
minute by minute (not just checked in aggregate), it oscillated between full
heat and full cool **18 times in a 166-minute journey**, `T_air` swinging
21.8–30 °C. This cabin's air node is fast (`τ_fast` ≈ 1.1–1.3 min) relative to
the rated capacities, so a full-power command overshoots the *opposite*
threshold before the actuator's own dead-time/lag can arrest it — a hysteresis
band sized for a slower plant doesn't hold here. Fixed by making the M5
baseline cooling-only, matching every other baseline already built in this
project. `thermostat_hysteresis_k` itself was not wrong; running it dual-mode
on this specific plant was.

### Anticipatory controller blend (M5)

`ff_weight = 0.6`, `gain_k = 3.0` — the feedforward (M4 forecast) / feedback
(current error) blend ratio and control gain. **`[ASSUMPTION]`**, and honestly
so: no literature source gives an exact blend ratio for this combination of a
LightGBM forecaster and this specific plant. A real train HVAC MPC study
(cited in ROADMAP's M5 section) confirms the *shape* — feedforward with
real-time feedback correction, not pure feedforward — but not a ratio.

Swept across `ff_weight ∈ {0, 0.3, 0.6, 0.9, 1.0}` in
`tests/test_controllers.py` rather than asserted correct — the same treatment
given to every other under-sourced parameter here (`hA`, `C_mass`, the
`StochasticController`'s `gain_divisor`). `ff_weight = 0` is also a
correctness check on the formula itself: it must degenerate to plain
proportional feedback exactly, verified against a hand-computed value, not
just "run without crashing."

**A deadband was added after the M5 headline comparison first came back
negative.** Running `AnticipatoryController` against `ThermostatController`
across the M4 test split (`src/compare_controllers.py`), the anticipatory
controller used *more* energy on 15/23 scenarios and was strictly worse on
both energy and comfort on 6/23. Diagnosed on the canonical scenario, not
guessed: `ThermostatController` is fully off 32.5% of the time; the raw
proportional law (no floor) was off only 0.6% of the time. With no deadband,
`raw` (the blended feedforward/feedback error) is essentially never exactly
zero — continuous solar gain, fresh-air load and passenger heat keep it
slightly nonzero almost always — so the controller pays continuous low-power
compressor cost that bang-bang's real off-periods simply don't. Fixed by
zeroing `raw` below `thermostat_hysteresis_k / 2` (reusing the baseline's own
sourced switching tolerance rather than inventing a second unsourced number),
applied as a continuous shrink-to-zero — `excess = |raw| − half_band`, zero
below it, `sign(raw) · excess` above — specifically to avoid a hard on/off
jump reintroducing the chattering `ThermostatController` hit the first time
dual-mode hysteresis was tried on this plant (see above). Validated on the
**val split** before the test split was re-run: mean energy saving
−4.5%→+3.4%, dominated scenarios (worse on both) 6/23→0/31. Confirmed once
on the test split for the actual headline number — see ROADMAP.md's M5
section for the full result and the honest caveat that total degree-hours is
not improved in aggregate.

**A follow-up ablation quantifies what the feedforward term (`ff_weight`)
itself buys.** `src/compare_controllers.py`'s `compare_feedforward_contribution()`
compares the shipped controller against the identical controller/deadband
with `ff_weight` forced to 0 (pure proportional feedback, no M4 forecast).
On the same 23 test-split scenarios: the forecast costs 6.2% more energy on
average but cuts total degree-hours 60.47→22.86 K·h (23/23 scenarios
equal-or-better). This is the cleanest result in the milestone and it
directly supports the design choice above ("why feedforward alone isn't
used" applies in reverse here too: pure feedback alone isn't the free
option either — it is measurably worse on comfort). **Not a PID comparison**
— `ff_weight=0` has no integral or derivative term, so do not present it as
one.

### Cooling capacity — raised from 30 kW during the M1 audit

30 kW is a normal European coach figure, and it **saturated** against Egyptian
weather. Steady-state peak load computed against the actual fetched data, at
design occupancy and the sliding setpoint:

| Case | Peak load | Saturated hours @ 30 kW | @ 40 kW |
|---|---|---|---|
| Cairo, 50% sunlit | 33.4 kW | 1.3% | 0% |
| Aswan, 50% sunlit | 34.8 kW | 5.6% | 0% |
| Cairo, 100% sunlit | 37.9 kW | 14.0% | 0% |
| Aswan, 100% sunlit | 39.3 kW | 23.3% | 0% |

A saturated unit would have made M5 meaningless: both controllers pin at the
limit during exactly the hours that matter, so the comparison would measure
nothing.

**That table assumed 40 kW was always available. It is not.** Re-run against the
supply-air limit, the unit cannot hold setpoint at peak — but the cabin settles
inside the comfort band anyway, because as it warms the load falls and authority
rises:

| City | Peak `T_out` | Setpoint | Load at setpoint | Available at setpoint | Equilibrium |
|---|---|---|---|---|---|
| Cairo | 46.4 °C | 26.0 °C | 30.1 kW | 25.7 kW | **27.7 °C** (+1.7 K) |
| Aswan | 48.1 °C | 26.0 °C | 30.4 kW | 25.7 kW | **27.8 °C** (+1.8 K) |

Both sit just inside the ±2 K band, with little margin. This is realistic — real
trains do run warm in extreme heat — but it is marginal, and a hotter day or a
higher load would push it out. Do not describe the unit as comfortably sized.

**Correction to an earlier claim.** This document previously said 40 kW sat at
the *top* of the commercial range, based on a source quoting 23–32 kW. A better
reference gives **12–15 tons of refrigeration for an 85-foot coach car**, i.e.
**42–53 kW**. Against that, 40 kW for a 26 m coach is **mid-to-low**, not high.
Both readings can be true — the lower figures are per-unit and installations
often carry two or three units — but the honest description is "within the
commercial range", not "at the top of it".

**More important than the rating: it is not all deliverable.** The rated figure
includes latent load and fresh-air pre-cooling and was never pure sensible power
into the cabin air. What actually reaches the air node is bounded by the
supply-air path — see the resolved issue above. At setpoint the usable figure is
about **25.7 kW**, not 40.

### Supply air

`supply_air_m3_h = 4800` — derived rather than guessed. Rail HVAC supplies
**20–30% outside air at design conditions**; fresh air at design load is
15 m³/h/pax × 80 = **1200 m³/h**, which at ~25% implies ~4800 m³/h total supply.

An independent source quotes **1200 m³/h for a 24 m intercity coach with 80
passengers at +45 °C** — an exact match to our fresh-air figure, arrived at
separately. That is the only genuine external confirmation of any ventilation
parameter in this project.

`supply_air_min_temp_c = 10.0` is the coil limit; colder risks frost and draught
complaints. EN 13129 governs draught limits but is paywalled, so this stays
`[ASSUMPTION]`.

### Supply fan — the energy floor, and a heat load

`supply_fan_sfp_kw_per_m3s = 2.0` → **2.67 kW continuous**.

EN 13779 / EN 16798-3 put constant-volume systems at **1.7 kW/(m³/s)** and
variable-volume at **2.4**, with regulatory values typically 2–3. Rail ducting is
compact and filtered, so mid-range rather than best-in-class.

**Only the supply fan is modelled.** The condenser fan is conventionally already
inside quoted COP/EER figures, so counting it separately would double up.

It matters twice over:

1. **Electrical floor.** The fan runs continuously because fresh air must be
   delivered whether or not cooling is called for. No controller can reduce it.
2. **Heat load.** The motor sits in the air stream, so its 2.67 kW lands in the
   cabin as sensible heat, raising the cooling demand it is supposed to serve.

Measured over a full Cairo–Alexandria run:

| | Energy | Share |
|---|---|---|
| Compressor | 16.21 kWh | 68.7% |
| **Supply fan** | **7.38 kWh** | **31.3%** |
| **Total** | **23.59 kWh** | |

Compressor energy itself rose from 14.12 to 16.21 kWh, because the fan heat has
to be removed.

**Why this had to be fixed before M5.** Omitting the fan inflates every
percentage saving, because the denominator is too small. Concretely: **a 10%
compressor saving is only 6.9% of the true total.** Reporting the first number
would have been wrong in the direction that flatters us.

### Weather interpolation

The archive is hourly; the simulation steps every minute. Held constant within
the hour, outdoor temperature is a staircase — the largest step on 15 Jul 2024
in Cairo is **2.40 K**, landing as a **2.5 kW instantaneous load jump**, a
quarter of a station event and entirely an artifact of data resolution.

`src/weather.py:to_minutes()` interpolates onto a 1-minute grid. Per-minute
steps fall from **2.40 K to 0.040 K**.

It also removed a **systematic bias**, which was the bigger problem. Flooring to
the hour used the temperature at the *start* of each hour, which on a warming
day is consistently too cool — so energy was **underestimated by 5.2%**
(23.59 → 24.83 kWh). A smoothing fix turned out to be an accuracy fix.

---

## KNOWN LIMITATION — the unit misses the band at extreme hours

Once supply-air limiting **and** fan heat are both accounted for, the modelled
40 kW unit cannot hold the ±2 K comfort band at the hottest measured conditions:

| City | Peak `T_out` | Equilibrium without fan | With fan | Band limit |
|---|---|---|---|---|
| Cairo | 46.4 °C | 27.67 °C | **28.69 °C** | 28.0 °C |
| Aswan | 48.1 °C | 27.80 °C | **28.82 °C** | 28.0 °C |

The fan alone is worth **+1.02 K** on equilibrium — enough on its own to push
the peak hours outside the band.

How often it actually matters:

| City | Summer hours outside band | Worst overshoot |
|---|---|---|
| Cairo | **0.5%** | +0.69 K |
| Aswan | **2.0%** | +1.23 K |

**This is accepted, not a defect.** Real trains do run warm in extreme heat, and
98–99.5% coverage is a realistic design point. Two tests pin it: one asserts the
band holds through typical conditions (<5% of hours exceeded), the other asserts
the worst-case overshoot stays under 1.5 K, so a regression would be caught.

It also **strengthens the project's argument**: at peak conditions there is no
spare authority left to react with, so acting early is the only remaining lever.

### COP — cooling and heating are separate models, not one curve

**Cooling:** `cop_cooling(T_out) = max(1.2, 2.8 − 0.04·(T_out − 30))` →
**2.20 at 45 °C**. 2.8 nominal at 30 °C is a design-point value for this
equipment class. The degradation slope and floor are shape assumptions, not
measurements — the floor exists to keep the model physical at extreme
temperatures rather than to represent a real cut-off.

**Heating:** `cop_heating(T_out) = 1.0`, unconditionally. **This used to reuse
the cooling formula** — `cop(t_out_c)` took only outdoor temperature and
applied one curve to both directions. That curve says "COP falls as it gets
hotter," correct for cooling and backwards for heating: fed a heating scenario
it read "COP rises as it gets colder," producing **COP 3.8 at 5 °C** — a
heat-pump reading applied to equipment that is not a heat pump.

Searched specifically for rail heating hardware rather than general HVAC. The
answer is unambiguous: rail cabin heating is **resistive** (tubular finned
elements), sold as a standalone product distinct from the vapour-compression
cooling unit, in the **7–36 kW** range — consistent with `heating_capacity_w`.
Trains have abundant line/generator power, so the reliability of a simple
resistive element is favoured over the marginal efficiency of a second
refrigerant loop. Joule heating converts electrical input to heat directly, so
**COP = 1.0 is a physical identity, not a fitted curve** — there is nothing to
tune and no outdoor-temperature dependence to model.

`cop(t_out_c, heating=...)` now dispatches between `cop_cooling()` and
`cop_heating()`, selected in `step()` by the **sign of delivered power**
(`q_hvac > 0`), never by outdoor temperature. Verified end to end: the same
reproduction that showed COP 3.8 at 5 °C now shows COP 1.00, and electrical
draw for 20 kW delivered heat rose from 7.93 kW to **22.67 kW** — the honest
cost was previously understated by roughly 2.9×.

**Why this sat unnoticed.** Every actuator test up to this point commanded
cooling only — dead time, lag, capacity clamp, COP, electrical draw were all
exercised in one direction. `tests/test_physics.py` now mirrors each of those
for heating, plus a direct regression guard
(`test_heating_and_cooling_disagree_at_the_same_temperature`) asserting the two
paths are never accidentally reunified.

**Severity, in practice: low.** Recomputing the actual breakeven — where
passenger heat plus the 2.67 kW fan load stop covering envelope and
ventilation loss, not just the sliding-setpoint curve's 20 °C boundary — against
our cached winter weather:

| Occupancy | Heating needed below | Cairo winter hours below it | Aswan |
|---|---|---|---|
| Empty train | 18.4 °C | 16.5% | 10.6% |
| 40 passengers | 15.5 °C | **0.0%** | **0.0%** |
| 68 passengers | 14.4 °C | 0.0% | 0.0% |

At any realistic occupancy, heating essentially never engages against our
cached data. The fix was worth making regardless — a wrong COP is wrong
whether or not the current dataset happens to exercise it — but it did not
block M4, and winter stays excluded from the M4 dataset by earlier decision.

### Actuator lag — the values that are deliberately not claimed

`dead_time_min = 2.0` and `tau_act_min = 5.0` are **defaults for single runs
only**. They represent compressor spin-up, damper travel, and coil inertia.

**No public source exists for these.** EN 13129 and EN 14750 are paywalled and
publish no time constants; UIC 553-1 specifies chamber test conditions, not
dynamics. Asserting a value here is the single thing most likely to collapse
under jury questioning.

**M6 therefore sweeps `tau_act ∈ {0, 2, 5, 10, 20} min`** and reports the
dependence instead of a point value. The `tau_act = 0` endpoint doubles as a
correctness check: the predictive controller must degenerate to the baseline
there, and `test_zero_tau_act_removes_the_lag` enforces the model side of it.

---

## Comfort

| Parameter | Value | Confidence |
|---|---|---|
| `setpoint_c` | 22.0 | Low |
| `sliding_t_out_low_c` / `high_c` | 20.0 / 40.0 | Low |
| `sliding_setpoint_low_c` / `high_c` | 22.0 / 26.0 | Low |
| `band_k` | 2.0 | Low |

EN 13129 defines interior temperature as a **sliding function of outdoor
temperature** rather than a fixed setpoint — you do not hold 22 °C when it is
45 °C outside, both for energy reasons and because the thermal shock on
boarding would be unpleasant. This is a simplified linear ramp expressing that
idea: 22 °C at 20 °C outdoor rising to 26 °C at 40 °C outdoor.

**It is not the standard's curve**, which is paywalled. Do not present it as
such.

`band_k = 2.0` (±2 K) defines the comfort metric — degree-minutes outside the
band — and is the yardstick the boarding disturbance is measured against.

---

## Simulation

| Parameter | Value | Confidence |
|---|---|---|
| `timestep_min` | 1 | High |
| `control_horizon_min` | 30 | Medium |
| `SUBSTEP_S` (in `cabin_model.py`) | 10.0 | **High — measured** |

### `SUBSTEP_S = 10.0` — chosen by convergence study, not by feel

τ_fast is ~1.2 min, so a 60 s explicit Euler step runs at `dt/τ ≈ 0.83` —
stable but badly inaccurate. Measured against a 0.25 s reference over one hour
of hard forcing (45 °C, 70 pax, 900 W/m², 25 kW cooling):

| Substep | Error |
|---|---|
| 1 s | −0.0012 K |
| 5 s | −0.0076 K |
| **10 s** | **−0.0156 K** |
| 30 s | −0.0476 K |
| 60 s | −0.0953 K |

10 s costs 0.016 K and ~8.6k iterations per simulated day, which is free.
`test_integration_has_converged_at_the_chosen_substep` verifies this, and
includes a guard that a 60 s step is measurably worse — so the test cannot
silently pass for the wrong reason.

`control_horizon_min = 30` is `H` in the predictive controller. It should
exceed dead time + a few actuator time constants; M6's sweep is what will show
whether 30 min is the right choice across the `tau_act` range.

---

## Route

Cairo → Benha → Tanta → Damanhour → Sidi Gaber → Alexandria Misr.

**The station list and stopping pattern are real** — this is the flagship
Egyptian National Railways intercity corridor, ~208 km. **Timings and passenger
counts are representative, not operator data**, and are tagged `[ASSUMPTION]`.
No ENR ridership or telemetry data is public; this was searched for and does
not exist in accessible form.

Occupancy balances to zero at the terminus and peaks at 73/80 (91% load
factor). Verified in the M1 audit — an unbalanced route would silently leave
passengers aboard and corrupt every downstream heat load.

---

## Locations

**Cairo** (30.0444, 31.2357) — network hub, realistic operating point.
**Aswan** (24.0889, 32.8998) — thermal stress case.

Real historical 2024 weather from the Open-Meteo Archive API (free, no key),
hourly, zero NaNs, hourly-contiguous.

| | T_out range | GHI max |
|---|---|---|
| Cairo summer | 21.0 – 46.4 °C | 993 W/m² |
| Cairo winter | 5.5 – 28.9 °C | 801 W/m² |
| Aswan summer | 24.6 – 48.1 °C | 1016 W/m² |
| Aswan winter | 4.6 – 31.5 °C | 891 W/m² |

**Two findings worth quoting in the presentation:**

1. UIC 553-1 specifies climatic chamber testing to **+45 °C**. Measured 2024
   conditions reach **46.4 °C (Cairo)** and **48.1 °C (Aswan)** — the corridor
   exceeds the standard's upper test bound.
2. UIC 553-1 specifies solar simulation at **800 W/m²**. Measured GHI reaches
   **907–1016 W/m²** — it exceeds the standard on irradiance too.

Egypt is cooling-dominated: even *winter* Cairo reaches 28.9 °C. Heating is
close to irrelevant. `heating_capacity_w` and `heating_cop` are sourced in the
HVAC section below regardless, since a wrong number is wrong whether or not the
current dataset happens to exercise it.

---

## RESOLVED — the supply-air limit

**Was:** the model applied rated capacity directly to the air node as sensible
power, permitting 13.3 K/min of air cooling. A bang-bang thermostat therefore
oscillated for a numerical reason rather than a physical one, which would have
made any M5 controller comparison a straw man.

**Fix:** cooling is now delivered through a finite air stream at a bounded
supply temperature:

```
Q_cool_max(T_air) = min( rated_capacity, m_dot_supply · c_p · (T_air − T_supply_min) )
```

Authority is now **state-dependent** — full during pull-down, tapering as the
cabin approaches supply temperature:

| `T_air` | Deliverable cooling | vs rated |
|---|---|---|
| ≥35 °C | 40.0 kW | 100% |
| 30 °C | 32.2 kW | 80% |
| **26 °C (setpoint)** | **25.7 kW** | **64%** |
| 20 °C | 16.1 kW | 40% |
| 10 °C | 0 kW | 0% |

Measured effect on the baseline over a full Cairo–Alexandria run:

| | Before | After |
|---|---|---|
| Mean error vs setpoint | −1.35 K | **+0.01 K** |
| Worst overcooling | ≈ −5.3 K | **−2.82 K** |
| Minutes outside ±2 K band | 72/166 | **38/166** |
| Energy | 16.25 kWh | 14.58 kWh |

The systematic overcooling is gone. Applied to both the command *and* the lag
output — the lag state otherwise carries an old high command forward as the
cabin cools and the stream can no longer support it.

### Consequence: the decomposition became measurable

Before the fix, the door/passenger decomposition returned doors-only as *larger*
than doors-plus-passengers, which is impossible. It is now monotonic and valid:

| Contribution | Worst excursion | Energy |
|---|---|---|
| Neither (weather only) | +1.94 K | 6.49 kWh |
| Doors only | +2.74 K | 6.92 kWh |
| Passengers only | +3.61 K | 14.22 kWh |
| Both | +3.73 K | 14.56 kWh |

**Passengers dominate both energy (+7.65 kWh vs doors' +0.34 kWh) and peak
excursion.** This supersedes an earlier claim that the station spike was
door-dominated — that came from an instantaneous comparison which ignored the
fact that doors are open only 13% of the journey. It is a better result for the
project than the old one: the dominant disturbance is the one the timetable can
actually predict.

### Consequence: the baseline choice still matters, and must be stated

With the plant fixed, the remaining oscillation is physical. But it still
depends heavily on how the baseline modulates:

| Baseline | Mean | Swing | Outside band | Energy |
|---|---|---|---|---|
| On/off | +0.01 K | 6.03 K | 38/166 | 14.58 kWh |
| 2-stage | −0.05 K | 6.03 K | 37/166 | 14.64 kWh |
| **Modulating** | +0.34 K | 5.26 K | **22/166** | **14.12 kWh** |

Staging barely helps; proportional modulation nearly halves the time outside
band. **M5 should compare against the modulating baseline**, otherwise a large
part of any measured gain is just modulation rather than anticipation. If the
on/off baseline is also reported, the split must be stated explicitly.

### What the worst excursion actually is

An intermediate hypothesis — that the peak was capacity-limited and therefore
identical across baselines — was **wrong**, and worth recording. The HVAC never
saturates on a Cairo day (0 of 166 minutes at its authority limit). The peak is
at **minute 2**, the origin boarding at Cairo Ramses: 68 passengers arrive at
once against a 2-minute actuator dead time.

Adding 30 minutes of pre-conditioning (an empty train standing at the platform
with HVAC running, which is what really happens) reduces it from +3.73 K to
+3.21 K but does not remove it — because it is not an artifact. Boarding-rate
literature gives ~35 s for 40 passengers per door, so 68 passengers really do
board in about a minute; resolving the exchange at arrival is a fair
approximation for a mainline coach.

Genuine excursions, modulating baseline, with pre-conditioning:

| Minute | Excursion | Where |
|---|---|---|
| 2 | +3.21 K | Cairo Ramses, origin boarding |
| 82 | +2.77 K | Tanta |
| 152 | +2.58 K | running |

**Both exceed the ±2 K comfort band.** The premise holds — but note it took a
corrected plant model and a corrected measurement to establish it, and the
earlier version of this claim did not survive scrutiny.

---

## Scenario coverage — what actually drives the result

Measured across the full catalogue (2 directions × 2 stopping patterns × 4 load
factors) on days selected by percentile, in both cities.

**Headline figures, as a distribution rather than a point:**

| | p10 | p50 | p90 | max |
|---|---|---|---|---|
| Energy per trip | 20.09 kWh | **24.94 kWh** | 30.89 kWh | 36.30 kWh |
| Worst excursion | — | **+4.69 K** | — | +7.75 K |
| Minutes outside band | — | **51/166** | — | 71/166 |

Fan share across everything: **28.7%**.

### Departure hour dominates everything else

The single most important scenario variable, and it was not on the list of
known issues at all. Same day, same service, varying only departure time:

| Departure | Energy | Worst excursion | Outside band |
|---|---|---|---|
| 04:00 | 18.19 kWh | +3.43 K | **6**/166 |
| 08:00 | 25.04 kWh | +4.66 K | 49/166 |
| 12:00 | 39.57 kWh | +9.95 K | 33/166 |
| **16:00** | **40.10 kWh** | **+11.78 K** | 38/166 |
| 20:00 | 27.85 kWh | +8.52 K | 57/166 |
| 22:00 | 24.14 kWh | +6.81 K | 41/166 |

**120% energy swing**, against **28%** for load factor and roughly 15% for day
percentile. ENR runs 37 trains daily between 04:00 and 23:00, so the whole range
is real service. An 04:00 departure is nearly problem-free; a 16:00 departure is
severely stressed.

**Consequence for M4: departure hour must be a sampled dimension.** Training
only on morning departures would miss most of the problem.

**Consequence for the optional `#3a`** (tying load factor to time of day):
**not worth building.** Departure hour already carries a 4× larger effect
through solar and outdoor temperature, and the catalogue already spans load
factors independently. Sampling departure hour subsumes it.

### Ranking the scenario dimensions

| Dimension | Effect on energy | Verdict |
|---|---|---|
| **Departure hour** | **120%** | Must sample in M4 |
| City (Cairo vs Aswan) | ~25% | Both already included |
| Load factor (0.4 → 1.1) | 28% | Already spanned |
| Stopping pattern | 8.9% | Both already included |
| Day percentile | ~15% | Sample across the 92 days |

### A prediction of mine that was too strong

I expected single-day figures to be **10–20% low**, because 15 July sits at the
20th percentile of daily maximum temperature. Measured against the full
catalogue, the old single-day energy figure (24.69 kWh) lands within **1%** of
the median (24.94 kWh) — the day percentile matters much less than I claimed,
because express services and light loads offset hot days.

The underlying point still stands, but for a different reason: the **spread** is
20–36 kWh. Quoting any single number as "the" result is wrong because of the
range, not because of a bias. Corrected here rather than left overstated.

---

## OPEN RISK FOR M5 — the forecaster's advantage is policy-dependent

Found by testing M4's saved model against a **live**, minute-by-minute
simulation loop rather than only against the pre-built features table —
specifically, by driving it with the plain reactive thermostat from M2/M3
instead of the `StochasticController` it was trained under. That live-testing
capability was originally a throwaway diagnostic script; it is now permanent,
tested infrastructure (`LiveFeatureBuilder` in `src/features.py`) precisely
*because* this investigation showed M5 would need it and that hand-rolling it
twice would be a real risk. This section documents what that infrastructure
found, not a gap in the infrastructure itself — the gap it exposed is in the
training distribution, not in whether M5 can call the model correctly.

**The headline result (test MAE 3.780 °C vs persistence 4.837 °C, +21.8%) is
real and reproducible** — confirmed across 3 independently regenerated 30k-row
datasets with different seeds, improvement 24.4–28.5% each time, all higher
than the committed dataset's figure. That is not in question.

**What's new:** evaluated live against the M2/M3 reactive thermostat — the
controller M5 will actually use as its baseline — across 4 scenarios (varied
date, departure hour):

| Policy | Beats persistence |
|---|---|
| `StochasticController` (training distribution) | **4/4 scenarios** |
| Reactive thermostat (M5's baseline) | **1/4 scenarios** |

Verified this wasn't a bug in the live-reconstruction script before trusting
it: batch `add_features()` applied to the identical trajectory produces
*bit-for-bit identical* predictions to the incremental version. The gap is
real, not a diagnostic error.

**The mechanism, as far as it's actually been established — no further than
this:** under the reactive thermostat, persistence MAE itself is very low
(0.7–2.2 °C across the 4 scenarios) — the controller holds `T_air` comparatively
stable, so "predict no change" is already a strong baseline and there is less
error left for *any* forecaster to remove. Under the stochastic policy,
persistence MAE is much higher (4.9–7.7 °C) because the exploration bursts
create large swings persistence badly fails to predict.

A finer breakdown — whether the model's edge concentrates specifically near
station disturbances, which would be the tidy story matching this project's
premise — was tested on one scenario and did **not** cleanly confirm it
(persistence stayed strong even near a station event in that case, on a small
20-vs-96-row split). **Recorded as inconclusive rather than stretched into a
cleaner claim than the evidence supports.**

**Why this doesn't block M5, and what M5 should do about it:**

- M4's own "Done when" criterion (ROADMAP) is about the forecaster in
  isolation, evaluated the way it was trained — which it satisfies, robustly.
- M5's actual deliverable is a **controller comparison** (baseline vs
  predictive), not a standalone forecaster MAE claim. The forecaster only
  needs to be *useful enough to inform better decisions*, not to minimize MAE
  against persistence under every possible policy that could generate its
  input trajectory.
- **Do not carry the "+21.8% beats persistence" framing into M5's headline
  result.** That number describes the forecaster's training distribution, not
  the deployed baseline. M5 needs its own honest measurement of what the
  predictive controller achieves against the reactive baseline, on the
  reactive baseline's own trajectories.
- If M5's predictive controller also disappoints once measured this way, the
  fix is more likely additional training scenarios generated *under* the
  reactive policy (closing the distribution gap) than a change to the
  features or the model class.

---

## Calibration priority

If real vehicle or operational data ever arrives, replace in this order.
Ranked by *(uncertainty × influence on the result)*, not by uncertainty alone:

| Rank | Parameter | Why |
|---|---|---|
| 1 | `tau_act_min`, `dead_time_min` | Genuinely unknown *and* directly drives the headline result. Currently swept rather than claimed — real values would replace a sweep with a number. |
| 2 | `internal_h_w_m2k` (→ `hA`) | Sets τ_fast and the boarding response. Swept 2–15 W/m²K; the conclusion survives but the magnitude moves by >2×. |
| 3 | `air_exchange_m3_per_min` | Large disturbance (`door_ua` ≈ 563 W/K, ~90% of envelope UA) resting on a pure estimate. |
| 4 | `u_value_w_m2k` | Sets `UA_envelope`; drives steady-state load and hence capacity sizing. |
| 5 | `fresh_air_m3_h_per_passenger` | Would become High confidence with EN 13129 access alone — no vehicle data needed. |
| 6 | `cooling_capacity_w` | Easy to obtain from any real vehicle datasheet. |
| — | `interior_mass_capacity_j_k` | **Deliberately last.** Lowest confidence in the project, but measured to be nearly irrelevant to every output that matters. |

---

## Reproducing the analysis

```bash
python -m src.config
```

Prints all derived quantities: `C_air`, `C_mass`, `C_eff`, both `UA` terms,
`hA`, and both time constants.

```bash
python -m pytest tests/ -q
```

23 physics tests. The ones that constrain parameter choices:

- `test_slow_mode_matches_derived_time_constant` — ties the simulator to the
  derivation. This is what exposed that the single-node shortcut `C_eff/UA` was
  materially wrong once `hA` was no longer ≫ `UA`.
- `test_integration_has_converged_at_the_chosen_substep` — validates `SUBSTEP_S`.
- `test_results_are_insensitive_to_interior_mass` — justifies leaving the
  lowest-confidence parameter alone.
- `test_single_node_would_understate_the_boarding_spike` — stops the 2R2C
  topology from being "simplified" away in a later session.

---

## Corrections log

Things that were wrong and were fixed. Recorded because the reasoning matters
more than the final numbers.

| What | Why it was wrong | Fix |
|---|---|---|
| `cooling_capacity_w = 30000` | European figure applied to Egypt; saturated on 1–23% of summer hours, which would have made the M5 controller comparison measure nothing | Raised to 40 kW after computing peak load against real weather |
| Solar used full `glazing_area_m2` | Counts both sides of the coach; only one faces the sun | Added `sunlit_glazing_fraction`, plus a guard test |
| τ derived as `C_eff / UA` | Only valid when `hA ≫ UA`; here `hA ≈ 1500` against `UA ≈ 740–1040`, so it materially under-reported τ_slow | Replaced with exact 2×2 eigenvalues. **The simulator was right; the formula was wrong** — the test caught it |
| Boarding-spike justification test | Ran `T_out = 40 °C` against a 24 °C cabin, so the outdoor gradient supplied 3.3× more heat than the passengers. Measured envelope gain and called it a boarding spike | Re-tested isothermally. Honest figures: 2.12 K (2R2C) vs 0.71 K (1R1C) over 8 min |
| `test_substepping_does_not_change_the_answer` | Tautological — both branches integrate at `SUBSTEP_S`, so it could not detect discretisation error despite claiming to prove convergence | Renamed to say what it checks; added a real convergence study with a failure guard |
| `interior_mass_capacity_j_k = 1.6e6` | A bare guess at 37% of a mass budget, with no rationale | Rebuilt from the budget (2.8 MJ/K), and swept to show it barely matters |
| Door infiltration as a per-stop total | Divided by the route's mean dwell, so editing one station's timetable changed `door_ua` by −36% at *every* station | Restated as a rate while open, decoupling physics from schedule; regression test added |
| M3 station-disturbance measurement | Measured temperature rise from wherever the cabin happened to sit. A badly tuned controller was overcooling ~4 K, so most of the "rise" was recovery, not disturbance — the same confounding as the M2 boarding test | Re-measured as excursion *above setpoint* under a baseline that actually holds setpoint |
| M3 door-vs-passenger decomposition | Returned doors-only (+2.65 K) as *larger* than doors-plus-passengers (+2.26 K), which is impossible for a monotonic system. The bang-bang limit cycle swamped the disturbance being measured | Fixed by the supply-air limit. Now monotonic and valid, and it reverses the earlier conclusion: passengers dominate, not doors |
| HVAC delivered rated capacity as sensible power | Permitted 13.3 K/min of air cooling, so the baseline oscillated numerically rather than physically | Supply-air path added: authority is bounded by `ṁ·c_p·(T_air − T_supply_min)` and is state-dependent |
| "40 kW is at the top of the commercial range" | Based on a 23–32 kW source. A better reference gives 12–15 tons (42–53 kW) for an 85 ft coach, making 40 kW mid-to-low | Corrected in the HVAC section |
| "Station spike is door-dominated" | An instantaneous comparison that ignored doors being open only 13% of the journey | Superseded by the valid decomposition: passengers dominate energy 22× over and peak excursion too |
| "The peak excursion is capacity-limited" | Hypothesised because all three baselines shared an identical +3.73 K max. Wrong — the HVAC never saturates (0 of 166 min). The peak was the origin boarding against the actuator dead time | Recorded rather than quietly dropped, because the hypothesis was tested and falsified |
| Fan power omitted entirely | Only compressor power was counted. The supply fan is 31% of total energy and its heat adds ~1 K to equilibrium. Omitting it inflates every percentage saving — a 10% compressor saving is 6.9% of the true total | Supply fan added as both an electrical floor and a cabin heat load |
| Weather held constant within the hour | A 2.40 K staircase step = 2.5 kW load jump, which M4 could have learned as signal. Worse, flooring to the hour used start-of-hour temperature, underestimating energy by 5.2% on a warming day | `to_minutes()` linear interpolation; per-minute steps fall to 0.040 K |
| `test_decomposition_is_monotonic` asserted the wrong invariant | Used *peak error*, which is not guaranteed monotonic under closed-loop control — adding a load shifts when the controller acts, moving the peak either way. It passed by coincidence and broke when fan heat was added | Re-asserted on **energy**, which has no such escape: more heat in, more work out |
| Equilibrium test ignored fan heat | Once fan heat was included, the peak-hour equilibrium exceeded the comfort band — the test had been passing on an incomplete load | Split into two tests: typical conditions must hold the band, extreme hours must miss it by less than a pinned tolerance |
| M4 exploration policy's reactive mode | `err = t_air_c − setpoint` is positive when too hot, and the dispatch sent positive fractions to heating — so the "realistic" control mode fought overheating with heat, 57.6% of rows in the first generated dataset | Caught by checking the command-mode distribution against configured weights, not a crash. Fixed to `err = setpoint − t_air_c`; both directions now have dedicated tests |
| `lgb.Dataset(...).construct()` crashed natively | Access violation deep in `LGBM_DatasetSetField` on this Windows environment. Bisected shape, dtype, contiguity, ownership, value range and value order — all ruled out — before isolating it to **import order**: `import pandas` (unused) before `import lightgbm` reproduces it on a trivial random array, every time | `lightgbm` imported first in `src/train.py`; forced session-wide in `tests/conftest.py`, since pytest could otherwise import a pandas-using test file first. Worth checking again on the team's own machines during the 22 Aug handoff |
| One `cop()` formula for both heating and cooling | Built for cooling ("COP falls as it gets hotter"); applied to heating it read backwards and produced COP 3.8 at 5 °C for what is actually a resistive heater, understating electrical cost ~2.9× | Split into `cop_cooling()` (unchanged) and `cop_heating()` = 1.0, a physical identity for resistive elements, not a fitted curve. Dispatched by sign of delivered power, never by `t_out_c`. Ten end-to-end heating tests added — the gap that let it sit unnoticed was that every prior actuator test only ever commanded cooling |
| "Degree-minutes outside band" as the M5 comfort metric | An invented unit — nowhere in HVAC comfort literature. Every comfort number produced in this project's analysis, before this correction, had actually been a minute-*count* anyway, a different and weaker metric than either name implies | Sourced properly: "degree-hours" (Salimi et al., *Indoor Air*, 2021) is the real, citable convention. `degree_hours_outside_band()` computes at native minute resolution and reports in K·h |
| `AnticipatoryController`'s error sign | `current_error = t_air_c − setpoint_c` (positive when hot) fed a dispatch built for negative-when-hot — the identical class of bug already fixed once in `data_generator.py`'s `StochasticController`. Fixing only this produced no visible change in testing | Flipped to `setpoint − t_air_c`. But see the next row — this fix alone was not sufficient, and its apparent no-op was itself a symptom worth recording |
| `AnticipatoryController`'s dispatch line | A second, independent sign bug: `-cooling_capacity_w * frac` instead of `frac * cooling_capacity_w`, given `frac` is already signed. This one was masking the error-sign fix directly above — two bugs stacked to look like "the fix did nothing," found by tracing `frac` and the delivered command minute by minute rather than trusting an unchanged aggregate result | Corrected to `frac * capacity`, matching the already-correct pattern in `data_generator.py`. A 46 °C-adjacent day that had ended at 43 °C (fighting itself with heat) now ends at 24.9 °C |
| `ThermostatController` run dual-mode | Heat below the lower threshold, cool above the upper one — what a generic thermostat does. Oscillated between full heat and full cool 18 times in 166 minutes, `T_air` swinging 21.8–30 °C, because this cabin's fast air node (`τ_fast` ≈ 1.1–1.3 min) overshoots the opposite threshold before the actuator's dead-time/lag can respond | Made cooling-only, matching every other baseline already built in this project and the documented cooling-dominated climate |
| `evaluate.run_controller`'s journey start time used `int(depart_hour * 60)` | `data_generator.py` uses `round(...)` for the same computation — a real cross-file inconsistency (up to 1 minute), found while auditing M1–M5 compatibility before trusting the M5 headline number. Harmless in practice (weather is interpolated and slowly varying), but a genuine divergence, not just style | Changed to `round(...)`, matching `data_generator.py` |
| `AnticipatoryController`'s first M5 headline run | Used *more* energy than `ThermostatController` on 15/23 test-split scenarios, strictly worse on both energy and comfort on 6/23. Root cause: the proportional law had no floor, so it was fully off only 0.6% of minutes vs the baseline's 32.5% — continuous low-power modulation costing more than the baseline's real off-periods | Added a deadband reusing `thermostat_hysteresis_k` (no new unsourced number), applied continuously to avoid reintroducing chattering. Validated on the val split (−4.5%→+3.4% mean saving, 6/23→0/31 dominated) before confirming once on test (+3.9% mean, 0/23 dominated) |
