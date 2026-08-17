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


@dataclass
class CabinState:
    """Mutable cabin state. `q_hvac_actual_w` is delivered power, not commanded."""

    t_air_c: float
    t_mass_c: float
    q_hvac_actual_w: float = 0.0
    _pipeline: deque = field(default_factory=deque, repr=False)

    def copy(self) -> "CabinState":
        s = CabinState(self.t_air_c, self.t_mass_c, self.q_hvac_actual_w)
        s._pipeline = deque(self._pipeline)
        return s


@dataclass
class StepResult:
    """What happened over one step, for logging and evaluation."""

    t_air_c: float
    t_mass_c: float
    q_hvac_actual_w: float
    q_envelope_w: float
    q_ventilation_w: float
    q_door_w: float
    q_passengers_w: float
    q_solar_w: float
    q_latent_w: float
    electrical_w: float
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

        hv = self.cfg["hvac"]
        self.cooling_capacity_w = hv["cooling_capacity_w"]
        self.heating_capacity_w = hv["heating_capacity_w"]
        self.dead_time_s = hv["dead_time_min"] * 60.0
        self.tau_act_s = hv["tau_act_min"] * 60.0

        # Door infiltration is a bulk air exchange spread over the dwell.
        doors = self.cfg["doors"]
        tm = self.cfg["thermal_mass"]
        dwells = [s["dwell_min"] for s in self.cfg["route"]["stations"] if s["dwell_min"] > 0]
        mean_dwell_min = sum(dwells) / len(dwells) if dwells else 1.0
        m3_per_s = doors["air_exchange_m3_per_stop"] / (mean_dwell_min * 60.0)
        self.door_ua = m3_per_s * tm["air_density_kg_m3"] * tm["air_cp_j_kgk"]

    # ---------------------------------------------------------------- helpers

    def initial_state(self, t_air_c: float, t_mass_c: float | None = None) -> CabinState:
        return CabinState(t_air_c, t_air_c if t_mass_c is None else t_mass_c)

    def cop(self, t_out_c: float) -> float:
        """COP degrades as outdoor temperature rises above the reference point."""
        hv = self.cfg["hvac"]
        value = hv["cop_nominal"] - hv["cop_degradation_per_k"] * (
            t_out_c - hv["cop_reference_t_out_c"]
        )
        return max(hv["cop_min"], value)

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

    def _actuate(self, state: CabinState, q_cmd_w: float, dt_s: float) -> float:
        """Dead time, then first-order lag, then capacity clamp.

        Returns delivered thermal power [W].
        """
        q_cmd_w = min(max(q_cmd_w, -self.cooling_capacity_w), self.heating_capacity_w)

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

        q_env = q_vent = q_door = 0.0
        q_hvac = 0.0

        for _ in range(n_sub):
            q_hvac = self._actuate(state, inputs.q_hvac_cmd_w, h)

            d_env = self.ua_env * (inputs.t_out_c - state.t_air_c)
            d_vent = ua_vent * (inputs.t_out_c - state.t_air_c)
            d_door = ua_door * (inputs.t_out_c - state.t_air_c)
            d_coupling = self.ha * (state.t_mass_c - state.t_air_c)

            q_air = d_env + d_vent + d_door + d_coupling + q_pax + q_solar + q_hvac
            q_mass = -d_coupling

            state.t_air_c += h * q_air / self.c_air
            state.t_mass_c += h * q_mass / self.c_mass

            q_env += d_env * h
            q_vent += d_vent * h
            q_door += d_door * h

        # Latent load is carried by the coil (it costs energy) but does not
        # drive air temperature — we deliberately do not model a humidity state.
        cop = self.cop(inputs.t_out_c)
        coil_w = abs(q_hvac) + (q_latent if q_hvac < 0 else 0.0)
        electrical_w = coil_w / cop

        return StepResult(
            t_air_c=state.t_air_c,
            t_mass_c=state.t_mass_c,
            q_hvac_actual_w=q_hvac,
            q_envelope_w=q_env / dt_s,
            q_ventilation_w=q_vent / dt_s,
            q_door_w=q_door / dt_s,
            q_passengers_w=q_pax,
            q_solar_w=q_solar,
            q_latent_w=q_latent,
            electrical_w=electrical_w,
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
    print(f"COP @ 45 C       : {model.cop(45.0):.2f}")
    print(f"setpoint @ 45 C  : {model.setpoint(45.0):.1f} C")
    print()

    # Free-float for 3 h at 45 C with a full coach, HVAC off.
    state = model.initial_state(24.0)
    print("free-float, HVAC off, T_out 45 C, 73 pax:")
    for minute in range(1, 181):
        r = model.step(state, CabinInputs(t_out_c=45.0, n_pax=73), dt_s=60.0)
        if minute in (5, 15, 30, 60, 120, 180):
            print(f"  t={minute:3d} min  T_air {r.t_air_c:5.2f} C  T_mass {r.t_mass_c:5.2f} C")
