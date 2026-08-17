"""Load cabin_params.yaml and expose derived physical quantities."""

import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "cabin_params.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def air_heat_capacity(cfg: dict) -> float:
    """C_air [J/K] of the saloon air alone — the fast node."""
    tm = cfg["thermal_mass"]
    return cfg["geometry"]["saloon_volume_m3"] * tm["air_density_kg_m3"] * tm["air_cp_j_kgk"]


def interior_mass_capacity(cfg: dict) -> float:
    """C_mass [J/K] of seats, panels, floor — the slow node."""
    return cfg["thermal_mass"]["interior_mass_capacity_j_k"]


def internal_coupling_ua(cfg: dict) -> float:
    """hA [W/K] between the air node and the interior-mass node."""
    tm = cfg["thermal_mass"]
    return tm["interior_surface_area_m2"] * tm["internal_h_w_m2k"]


def effective_heat_capacity(cfg: dict) -> float:
    """C_eff [J/K] = air + interior mass. Sets the slow mode of the 2-node system."""
    return air_heat_capacity(cfg) + interior_mass_capacity(cfg)


def envelope_ua(cfg: dict) -> float:
    """UA [W/K] of the car body."""
    return cfg["geometry"]["envelope_area_m2"] * cfg["envelope"]["u_value_w_m2k"]


def fresh_air_mass_flow(cfg: dict, n_pax: float) -> float:
    """Fresh air mass flow [kg/s] for a given passenger count."""
    v = cfg["ventilation"]
    m3_h = max(n_pax * v["fresh_air_m3_h_per_passenger"], v["min_fresh_air_m3_h"])
    return m3_h * cfg["thermal_mass"]["air_density_kg_m3"] / 3600.0


def ventilation_ua(cfg: dict, n_pax: float) -> float:
    """Ventilation heat loss coefficient [W/K]: m_dot * c_p."""
    return fresh_air_mass_flow(cfg, n_pax) * cfg["thermal_mass"]["air_cp_j_kgk"]


def system_time_constants(cfg: dict, n_pax: float | None = None) -> tuple[float, float]:
    """Exact (tau_fast, tau_slow) [s] of the two-node cabin, DERIVED.

    The state matrix of the 2R2C network is

        dTa/dt = -(UA + hA)/C_air  * Ta + hA/C_air  * Tm
        dTm/dt =        hA/C_mass  * Ta - hA/C_mass * Tm

    and its two time constants are -1/lambda for the eigenvalues of that 2x2.
    Both are consequences of C and UA values declared in the config; neither is
    asserted anywhere.

    Note on a discarded shortcut: the familiar single-node approximation
    C_eff / UA is only valid when hA >> UA. Here hA is ~1500 W/K against a UA of
    ~740-1040 W/K, so that shortcut materially under-reports the slow constant.
    tests/test_physics.py checks the simulator against these exact values,
    which is how the discrepancy was found.
    """
    if n_pax is None:
        n_pax = cfg["occupancy"]["design_load_pax"]

    ua = envelope_ua(cfg) + ventilation_ua(cfg, n_pax)
    ha = internal_coupling_ua(cfg)
    c_air = air_heat_capacity(cfg)
    c_mass = interior_mass_capacity(cfg)

    a11 = -(ua + ha) / c_air
    a12 = ha / c_air
    a21 = ha / c_mass
    a22 = -ha / c_mass

    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = math.sqrt(trace * trace - 4.0 * det)

    lambda_slow = (trace + disc) / 2.0
    lambda_fast = (trace - disc) / 2.0
    return -1.0 / lambda_fast, -1.0 / lambda_slow


def thermal_time_constant(cfg: dict, n_pax: float | None = None) -> float:
    """Slow-mode cabin time constant [s] — whole-cabin settling."""
    return system_time_constants(cfg, n_pax)[1]


def air_time_constant(cfg: dict, n_pax: float | None = None) -> float:
    """Fast-mode air time constant [s] — sharpness of the boarding response."""
    return system_time_constants(cfg, n_pax)[0]


if __name__ == "__main__":
    cfg = load_config()
    n_pax = cfg["occupancy"]["design_load_pax"]
    print(f"C_air            : {air_heat_capacity(cfg) / 1e6:.3f} MJ/K")
    print(f"C_mass           : {interior_mass_capacity(cfg) / 1e6:.3f} MJ/K")
    print(f"C_eff            : {effective_heat_capacity(cfg) / 1e6:.3f} MJ/K")
    print(f"UA (envelope)    : {envelope_ua(cfg):.0f} W/K")
    print(f"UA (ventilation) : {ventilation_ua(cfg, n_pax):.0f} W/K  @ {n_pax} pax")
    print(f"hA (internal)    : {internal_coupling_ua(cfg):.0f} W/K")
    print(f"tau_fast (air)   : {air_time_constant(cfg) / 60:.1f} min  (derived)")
    print(f"tau_slow (cabin) : {thermal_time_constant(cfg) / 60:.1f} min  (derived)")
