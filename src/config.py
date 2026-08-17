"""Load cabin_params.yaml and expose derived physical quantities."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "cabin_params.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def effective_heat_capacity(cfg: dict) -> float:
    """C_eff [J/K] = cabin air + interior mass coupled to it."""
    tm = cfg["thermal_mass"]
    air = cfg["geometry"]["saloon_volume_m3"] * tm["air_density_kg_m3"] * tm["air_cp_j_kgk"]
    return air + tm["interior_mass_capacity_j_k"]


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


def thermal_time_constant(cfg: dict, n_pax: float | None = None) -> float:
    """Cabin thermal time constant [s], DERIVED — never read from config.

    tau = C_eff / (UA_envelope + UA_ventilation)
    """
    if n_pax is None:
        n_pax = cfg["occupancy"]["design_load_pax"]
    return effective_heat_capacity(cfg) / (envelope_ua(cfg) + ventilation_ua(cfg, n_pax))


if __name__ == "__main__":
    cfg = load_config()
    n_pax = cfg["occupancy"]["design_load_pax"]
    tau_s = thermal_time_constant(cfg)
    print(f"C_eff            : {effective_heat_capacity(cfg) / 1e6:.2f} MJ/K")
    print(f"UA (envelope)    : {envelope_ua(cfg):.0f} W/K")
    print(f"UA (ventilation) : {ventilation_ua(cfg, n_pax):.0f} W/K  @ {n_pax} pax")
    print(f"tau_thermal      : {tau_s / 60:.1f} min  (derived, not asserted)")
