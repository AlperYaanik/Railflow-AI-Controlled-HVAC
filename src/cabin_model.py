"""Grey-box railway cabin thermal model with HVAC actuator dynamics.

Two-node (2R2C) lumped-capacitance network:

    outdoor ──UA_env──┐
                      ├── [T_air] ──hA── [T_mass]
    fresh air ─UA_vent┘      │
    door infiltration ───────┤
    passengers (sensible) ───┤
    solar ───────────────────┤
    HVAC ────────────────────┘

Why two nodes and not one: lumping the interior mass into the air node
understates the temperature spike from a boarding event by ~2x at the end of a
dwell and ~5x in the first two minutes. That spike is the phenomenon this
project exists to address, so the model has to represent it.

No time constant is asserted anywhere. Both are consequences of C and UA
values declared in config/cabin_params.yaml — see src/config.py.

Sign convention: q_hvac > 0 heats, q_hvac < 0 cools.

M9 adds a THIRD, separate state: cabin humidity ratio (CabinState.w_air_g_kg),
a simple moisture balance (passenger generation + ventilation/door exchange)
alongside the two temperature nodes -- deliberately NOT a fourth coupled node
in the network above, since moisture doesn't buffer into interior mass the
way heat does at the timescales this project cares about. Deliberately does
NOT model coil dehumidification (see step()'s comment right before it builds
StepResult) -- that needs an Apparatus Dew Point/Bypass Factor coil model
this project doesn't have the numbers or time budget for. Disclosed as an
upper bound during active cooling, not silently ignored.
"""

from collections import deque
from dataclasses import dataclass, field

from src.config import (
    air_heat_capacity,
    envelope_ua,
    interior_mass_capacity,
    internal_coupling_ua,
    load_config,
    ventilation_ua,
)
from src.weather import humidity_ratio, relative_humidity_from_ratio

SUBSTEP_S = 10.0
"""Internal integration step [s].

Chosen against the fast mode: tau_air is ~1.2 min at design load, so a 60 s
explicit Euler step would sit at dt/tau ~ 0.83 — stable but badly inaccurate.
At 10 s, dt/tau ~ 0.14. A full day is ~8.6k iterations, which costs nothing.
"""


@dataclass
class CabinInputs:
    """Exogenous conditions over one step."""

    t_out_c: float
    ghi_w_m2: float = 0.0
    n_pax: float = 0.0
    door_open: bool = False
    q_hvac_cmd_w: float = 0.0
    fan_on: bool = True
    """Supply fan state. Defaults on: a train in service ventilates continuously,
    whether or not cooling is called for."""
    rh_out_pct: float = 50.0
    """Outdoor relative humidity, for the moisture balance (see CabinState.w_air_g_kg).
    Defaults to a neutral mid-range value so every existing caller that doesn't pass
    it (M4-M8's data_generator.py/evaluate.py/tests) keeps working unchanged -- their
    humidity output just won't reflect real weather until they're updated to pass the
    weather CSV's actual rh_out_pct column, a deliberately separate follow-up (M9)."""


@dataclass
class CabinState:
    """Mutable cabin state. `q_hvac_actual_w` is delivered power, not commanded."""

    t_air_c: float
    t_mass_c: float
    w_air_g_kg: float
    """Cabin air humidity ratio [g water / kg dry air]. See step()'s moisture
    balance for what this does and, importantly, does NOT account for."""
    q_hvac_actual_w: float = 0.0
    _pipeline: deque = field(default_factory=deque, repr=False)

    def copy(self) -> "CabinState":
        s = CabinState(self.t_air_c, self.t_mass_c, self.w_air_g_kg, self.q_hvac_actual_w)
        s._pipeline = deque(self._pipeline)
        return s


@dataclass
class StepResult:
    """What happened over one step, for logging and evaluation."""

    t_air_c: float
    t_mass_c: float
    w_air_g_kg: float
    rh_air_pct: float
    """Cabin humidity, in both units step() computes it in and the one a human
    reads. See step()'s moisture balance docstring for the upper-bound caveat
    whenever the HVAC is actively cooling."""
    q_hvac_actual_w: float
    q_envelope_w: float
    q_ventilation_w: float
    q_door_w: float
    q_passengers_w: float
    q_solar_w: float
    q_latent_w: float
    q_fan_w: float
    compressor_w: float
    """Coil-side electrical draw. Compressor power when cooling (q_hvac < 0);
    resistive-heater power when heating (q_hvac > 0) — same slot, different
    equipment, selected by cop()."""
    fan_w: float
    electrical_w: float
    """Total electrical draw: coil (compressor or heater) plus supply fan."""
    cop: float


class CabinModel:
    """Simulates one coach. Construct once, step repeatedly."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else load_config()

        self.c_air = air_heat_capacity(self.cfg)
        self.c_mass = interior_mass_capacity(self.cfg)
        self.ua_env = envelope_ua(self.cfg)
        self.ha = internal_coupling_ua(self.cfg)

        env = self.cfg["envelope"]
        self.solar_aperture = (
            self.cfg["geometry"]["glazing_area_m2"]
            * env["solar_heat_gain_coefficient"]
            * env["sunlit_glazing_fraction"]
        )

        occ = self.cfg["occupancy"]
        self.pax_sensible_w = occ["sensible_heat_w_per_pax"]
        self.pax_latent_w = occ["latent_heat_w_per_pax"]

        tm = self.cfg["thermal_mass"]
        rho_cp = tm["air_density_kg_m3"] * tm["air_cp_j_kgk"]
        self.air_cp = tm["air_cp_j_kgk"]

        # Moisture balance (M9): cabin air mass, and passenger moisture
        # generation converted from the existing latent_heat_w_per_pax via
        # the latent heat of vaporization -- a unit conversion, not a new
        # domain assumption (see config's comment on both numbers).
        self.air_mass_kg = self.cfg["geometry"]["saloon_volume_m3"] * tm["air_density_kg_m3"]
        self.moisture_gen_g_s_per_pax = (
            self.pax_latent_w / tm["latent_heat_vaporization_j_kg"] * 1000.0
        )
        self.target_rh_pct = self.cfg["comfort"]["target_rh_pct"]

        hv = self.cfg["hvac"]
        self.cooling_capacity_w = hv["cooling_capacity_w"]
        self.heating_capacity_w = hv["heating_capacity_w"]
        self.dead_time_s = hv["dead_time_min"] * 60.0
        self.tau_act_s = hv["tau_act_min"] * 60.0

        # Supply-air path: rated capacity is only reachable if the air stream
        # can carry it. See config for the derivation of these figures.
        self.supply_ua = hv["supply_air_m3_h"] / 3600.0 * rho_cp
        self.supply_min_c = hv["supply_air_min_temp_c"]
        self.supply_max_c = hv["supply_air_max_temp_c"]

        # Supply fan: continuous electrical draw, and the same power lands in
        # the cabin as sensible heat because the motor sits in the air stream.
        self.supply_fan_w = (
            hv["supply_fan_sfp_kw_per_m3s"] * 1000.0 * hv["supply_air_m3_h"] / 3600.0
        )

        # Door infiltration, as a rate while the door is open. Deliberately
        # independent of the timetable: how fast air crosses an open doorway is
        # a property of the doorway, not of how long the train is scheduled to
        # sit there. Total exchange per stop then follows from the dwell length.
        self.door_ua = self.cfg["doors"]["air_exchange_m3_per_min"] / 60.0 * rho_cp

    # ---------------------------------------------------------------- helpers

    def initial_state(self, t_air_c: float, t_mass_c: float | None = None) -> CabinState:
        """Starts the cabin "in comfort" on humidity (target_rh_pct at t_air_c),
        the same unjustified-further convention already used for t_air_c itself --
        the caller picks a starting temperature with no further ceremony, so the
        starting humidity ratio follows the same pattern rather than needing its
        own new default-choosing logic."""
        w0 = humidity_ratio(t_air_c, self.target_rh_pct)
        return CabinState(t_air_c, t_air_c if t_mass_c is None else t_mass_c, w0)

    def cop_cooling(self, t_out_c: float) -> float:
        """Vapour-compression COP, degrading as outdoor temperature rises."""
        hv = self.cfg["hvac"]
        value = hv["cop_nominal"] - hv["cop_degradation_per_k"] * (
            t_out_c - hv["cop_reference_t_out_c"]
        )
        return max(hv["cop_min"], value)

    def cop_heating(self, t_out_c: float) -> float:
        """Resistive-heater COP: a physical identity, not a fitted curve.

        Rail cabin heating uses tubular finned resistive elements, a separate
        product from the cooling unit, not a reversible heat pump. Joule
        heating converts electrical input to heat directly, so COP = 1.0 has
        no outdoor-temperature dependence — `t_out_c` is accepted only so this
        has the same signature as cop_cooling() and can be selected by sign of
        the command rather than branched on at every call site.
        """
        return self.cfg["hvac"]["heating_cop"]

    def cop(self, t_out_c: float, heating: bool = False) -> float:
        """Dispatches to the cooling or heating COP.

        These were previously one formula applied to both modes. Fed a
        heating scenario, "COP falls as it gets hotter" reads backwards —
        "COP rises as it gets colder" — and at 5 C it produced COP 3.8 for
        what is actually a resistive heater, roughly 4x too cheap.
        """
        return self.cop_heating(t_out_c) if heating else self.cop_cooling(t_out_c)

    def setpoint(self, t_out_c: float) -> float:
        """EN 13129 uses a sliding interior target rather than a fixed one.

        This is a simplified linear ramp expressing that idea, not the
        standard's curve (which is paywalled).
        """
        c = self.cfg["comfort"]
        lo, hi = c["sliding_t_out_low_c"], c["sliding_t_out_high_c"]
        s_lo, s_hi = c["sliding_setpoint_low_c"], c["sliding_setpoint_high_c"]
        if t_out_c <= lo:
            return s_lo
        if t_out_c >= hi:
            return s_hi
        return s_lo + (s_hi - s_lo) * (t_out_c - lo) / (hi - lo)

    def deliverable_cooling_w(self, t_air_c: float) -> float:
        """Most sensible cooling the supply air stream can deliver right now [W, negative].

        Bounded by two independent things: what the coil can produce (rated
        capacity) and what the air stream can carry at its coldest allowed
        supply temperature. The second shrinks to zero as the cabin approaches
        supply temperature, which is what stops the model from cooling the
        cabin arbitrarily fast or below the coil temperature.
        """
        air_side = self.supply_ua * max(0.0, t_air_c - self.supply_min_c)
        return -min(self.cooling_capacity_w, air_side)

    def deliverable_heating_w(self, t_air_c: float) -> float:
        """Most sensible heating the supply air stream can deliver right now [W]."""
        air_side = self.supply_ua * max(0.0, self.supply_max_c - t_air_c)
        return min(self.heating_capacity_w, air_side)

    def _actuate(self, state: CabinState, q_cmd_w: float, dt_s: float) -> float:
        """Dead time, then first-order lag, then the supply-air capacity clamp.

        Returns delivered thermal power [W].
        """
        q_cmd_w = min(
            max(q_cmd_w, self.deliverable_cooling_w(state.t_air_c)),
            self.deliverable_heating_w(state.t_air_c),
        )

        # Transport delay: a FIFO exactly `depth` substeps deep. Pre-filling it
        # with zeros means a fresh run delivers nothing until the dead time has
        # actually elapsed, rather than short-circuiting on the first step.
        depth = int(round(self.dead_time_s / dt_s))
        if depth > 0:
            while len(state._pipeline) < depth:
                state._pipeline.append(0.0)
            state._pipeline.append(q_cmd_w)
            q_delayed = state._pipeline.popleft()
        else:
            q_delayed = q_cmd_w

        if self.tau_act_s <= 0.0:
            state.q_hvac_actual_w = q_delayed
        else:
            alpha = dt_s / self.tau_act_s
            state.q_hvac_actual_w += alpha * (q_delayed - state.q_hvac_actual_w)

        # Clamp the OUTPUT too, not only the command. The lag state carries the
        # old command forward, so a unit that was cooling hard while the cabin
        # was hot would keep delivering that power as the cabin cooled and the
        # air stream could no longer support it.
        state.q_hvac_actual_w = min(
            max(state.q_hvac_actual_w, self.deliverable_cooling_w(state.t_air_c)),
            self.deliverable_heating_w(state.t_air_c),
        )
        return state.q_hvac_actual_w

    # ------------------------------------------------------------------ step

    def step(self, state: CabinState, inputs: CabinInputs, dt_s: float = 60.0) -> StepResult:
        """Advance the cabin by `dt_s` seconds. Mutates and returns from `state`."""
        n_sub = max(1, int(round(dt_s / SUBSTEP_S)))
        h = dt_s / n_sub

        ua_vent = ventilation_ua(self.cfg, inputs.n_pax)
        ua_door = self.door_ua if inputs.door_open else 0.0
        q_pax = inputs.n_pax * self.pax_sensible_w
        q_latent = inputs.n_pax * self.pax_latent_w
        q_solar = inputs.ghi_w_m2 * self.solar_aperture
        q_fan = self.supply_fan_w if inputs.fan_on else 0.0

        # Moisture balance (M9) -- deliberately simplified, see the note below
        # the substep loop for exactly what this does and does not account for.
        # Mass flow rates from the SAME UA figures the temperature balance
        # already computed (UA = m_dot * c_p, so m_dot = UA / c_p) rather than
        # a second, separately-derived ventilation/door model.
        w_out_g_kg = humidity_ratio(inputs.t_out_c, inputs.rh_out_pct)
        m_dot_vent_kg_s = ua_vent / self.air_cp
        m_dot_door_kg_s = ua_door / self.air_cp
        moisture_gen_g_s = inputs.n_pax * self.moisture_gen_g_s_per_pax

        q_env = q_vent = q_door = 0.0
        q_hvac = 0.0

        for _ in range(n_sub):
            q_hvac = self._actuate(state, inputs.q_hvac_cmd_w, h)

            d_env = self.ua_env * (inputs.t_out_c - state.t_air_c)
            d_vent = ua_vent * (inputs.t_out_c - state.t_air_c)
            d_door = ua_door * (inputs.t_out_c - state.t_air_c)
            d_coupling = self.ha * (state.t_mass_c - state.t_air_c)

            q_air = (d_env + d_vent + d_door + d_coupling
                     + q_pax + q_solar + q_fan + q_hvac)
            q_mass = -d_coupling

            state.t_air_c += h * q_air / self.c_air
            state.t_mass_c += h * q_mass / self.c_mass

            d_moisture = (
                m_dot_vent_kg_s * (w_out_g_kg - state.w_air_g_kg)
                + m_dot_door_kg_s * (w_out_g_kg - state.w_air_g_kg)
                + moisture_gen_g_s
            )
            state.w_air_g_kg = max(0.0, state.w_air_g_kg + h * d_moisture / self.air_mass_kg)

            q_env += d_env * h
            q_vent += d_vent * h
            q_door += d_door * h

        # Latent load is carried by the coil (it costs energy) but does not
        # drive air temperature -- unchanged from before M9. Only cooling
        # dehumidifies, so latent load never applies to heating.
        cop = self.cop(inputs.t_out_c, heating=q_hvac > 0.0)
        coil_w = abs(q_hvac) + (q_latent if q_hvac < 0 else 0.0)
        compressor_w = coil_w / cop

        # w_air_g_kg above tracks passenger + ventilation/door moisture, but
        # -- same simplification the energy accounting already made, now
        # stated where the number itself is visible -- deliberately WITHOUT
        # a coil dehumidification term: doing that properly needs an
        # Apparatus Dew Point / Bypass Factor coil model this project
        # doesn't have the numbers or the time budget for (see ROADMAP.md's
        # M9). The energy side already charges for passenger latent load as
        # if the coil removes it (compressor_w above, unchanged); the
        # humidity STATE doesn't reflect that removal. Net effect: whenever
        # q_hvac < 0 (actively cooling), rh_air_pct below is an UPPER BOUND
        # on real cabin humidity, not a corrected true value. Disclosed, not
        # hidden -- the same honesty pattern as every other stated
        # simplification in this project.
        rh_air_pct = relative_humidity_from_ratio(state.t_air_c, state.w_air_g_kg)

        return StepResult(
            t_air_c=state.t_air_c,
            t_mass_c=state.t_mass_c,
            w_air_g_kg=state.w_air_g_kg,
            rh_air_pct=rh_air_pct,
            q_hvac_actual_w=q_hvac,
            q_envelope_w=q_env / dt_s,
            q_ventilation_w=q_vent / dt_s,
            q_door_w=q_door / dt_s,
            q_passengers_w=q_pax,
            q_solar_w=q_solar,
            q_latent_w=q_latent,
            q_fan_w=q_fan,
            compressor_w=compressor_w,
            fan_w=q_fan,
            electrical_w=compressor_w + q_fan,
            cop=cop,
        )


if __name__ == "__main__":
    from src.config import air_time_constant, thermal_time_constant

    model = CabinModel()
    cfg = model.cfg
    n_pax = cfg["occupancy"]["design_load_pax"]

    print(f"tau_fast (air)   : {air_time_constant(cfg) / 60:.1f} min")
    print(f"tau_slow (cabin) : {thermal_time_constant(cfg) / 60:.1f} min")
    print(f"door_ua          : {model.door_ua:.0f} W/K while open")
    print(f"solar_aperture   : {model.solar_aperture:.1f} m2 effective")
    print(f"COP cooling @ 45 C : {model.cop_cooling(45.0):.2f}")
    print(f"COP heating @ 5 C  : {model.cop_heating(5.0):.2f}  (resistive, no T_out dependence)")
    print(f"setpoint @ 45 C    : {model.setpoint(45.0):.1f} C")
    print()

    # Free-float for 3 h at 45 C with a full coach, HVAC off.
    state = model.initial_state(24.0)
    print("free-float, HVAC off, T_out 45 C / RH 30%, 73 pax:")
    for minute in range(1, 181):
        r = model.step(state, CabinInputs(t_out_c=45.0, n_pax=73, rh_out_pct=30.0), dt_s=60.0)
        if minute in (5, 15, 30, 60, 120, 180):
            print(f"  t={minute:3d} min  T_air {r.t_air_c:5.2f} C  T_mass {r.t_mass_c:5.2f} C  "
                  f"RH_air {r.rh_air_pct:5.1f} %  (upper bound -- HVAC off here, so no dehumidification caveat applies)")
