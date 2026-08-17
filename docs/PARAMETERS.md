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
budget. This understates comfort degradation in humid conditions; Alexandria
(coastal) would be the place that shows up.

---

## Doors

| Parameter | Value | Confidence |
|---|---|---|
| `air_exchange_m3_per_stop` | 120.0 | **Low** |

Bulk air exchange over one dwell, spread evenly across the mean dwell time
(4.2 min over the route) → **`door_ua = 574 W/K` while open**.

Sanity: 120 m³ against a 150 m³ saloon is ~0.8 air changes per stop. Plausible
for doors open 3–8 minutes on a hot day.

**Flagged:** 574 W/K approaches the entire envelope `UA` of 638 W/K, so an open
door nearly doubles the coupling to outdoors. That makes station stops a large
disturbance — which suits the project's premise, so it deserves scrutiny rather
than acceptance. If the demo result looks suspiciously good, this is the first
number to re-examine.

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
| `dead_time_min` | 2.0 | **Low — swept in M6** |
| `tau_act_min` | 5.0 | **Low — swept in M6** |

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
nothing. **40 kW** gives ~15% margin on the worst case.

Cross-check against real products: commercial rail HVAC units run **23–32 kW**,
with up to ~40 kW for high-capacity stock. 40 kW is at the **top** of the
commercial range — the right place for a 46–48 °C design condition, but it
should be described as the top of the range, not a mid-range choice.

### COP

`COP(T_out) = max(1.2, 2.8 − 0.04·(T_out − 30))` → **2.20 at 45 °C**.

2.8 nominal at 30 °C is a design-point value for this equipment class. The
degradation slope and floor are shape assumptions, not measurements — the floor
exists to keep the model physical at extreme temperatures rather than to
represent a real cut-off.

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
close to irrelevant, which is why `heating_capacity_w` is left at a generic
value without further justification.

---

## Calibration priority

If real vehicle or operational data ever arrives, replace in this order.
Ranked by *(uncertainty × influence on the result)*, not by uncertainty alone:

| Rank | Parameter | Why |
|---|---|---|
| 1 | `tau_act_min`, `dead_time_min` | Genuinely unknown *and* directly drives the headline result. Currently swept rather than claimed — real values would replace a sweep with a number. |
| 2 | `internal_h_w_m2k` (→ `hA`) | Sets τ_fast and the boarding response. Swept 2–15 W/m²K; the conclusion survives but the magnitude moves by >2×. |
| 3 | `air_exchange_m3_per_stop` | Large disturbance (574 W/K, ~90% of envelope UA) resting on a pure estimate. |
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
