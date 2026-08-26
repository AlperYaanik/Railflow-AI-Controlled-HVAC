"""Tests for M14's disturbance layer (src/scenarios.py).

The load-bearing one is the FIRST test below. Every number this project has
on record -- M5's comparison, M6's sweep, M10-M13's results -- was measured
with no scenario applied. If `scenario=None` or `scenario=STABLE` ever
stopped reproducing that exactly, every one of those numbers would silently
shift, the same class of cross-milestone contamination M11's tunnel-masking
bug caused once already (see src/evaluate.py's note on it). So that
invariant is pinned bit-for-bit, not approximately.
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import pandas as pd
import pytest

from src import scenarios as sm
from src.config import load_config
from src.controllers import ThermostatController
from src.evaluate import run_controller, score
from src.occupancy import Service, simulate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_stable_scenario_is_bit_identical_to_no_scenario(cfg, require_weather):
    """The backward-compatibility guarantee every pre-M14 number depends on."""
    baseline = run_controller(lambda m: ThermostatController(m), cfg)
    with_stable = run_controller(lambda m: ThermostatController(m), cfg, scenario=sm.STABLE)

    for col in ("t_air_c", "electrical_w", "advised_setpoint_c", "n_pax", "q_solar_w"):
        assert baseline[col].equals(with_stable[col]), (
            f"{col} changed under the 'stable' scenario -- it must be a no-op"
        )


def test_stable_is_recognised_as_stable():
    assert sm.STABLE.is_stable()
    for scenario in (sm.RAPID_WARMING, sm.SOLAR_SURGE, sm.CROWD_SURGE, sm.COMBINED):
        assert not scenario.is_stable(), f"{scenario.name} should not report as stable"


def test_every_disturbance_actually_changes_the_outcome(cfg, require_weather):
    """A scenario that doesn't move the physics is a scenario that isn't
    testing anything -- this is exactly how the first `solar_surge` was
    caught (additive irradiance clipping against the observed maximum and
    delivering a 0.5% effect, see that field's docstring)."""
    baseline = score(run_controller(lambda m: ThermostatController(m), cfg), cfg)

    for scenario in (sm.RAPID_WARMING, sm.SOLAR_SURGE, sm.CROWD_SURGE, sm.COMBINED):
        disturbed = score(
            run_controller(lambda m: ThermostatController(m), cfg, scenario=scenario), cfg)
        delta_pct = abs(100.0 * (1.0 - disturbed.energy_kwh / baseline.energy_kwh))
        assert delta_pct > 1.0, (
            f"scenario {scenario.name!r} moved journey energy by only {delta_pct:.2f}% "
            "-- too small to be testing the controller against anything"
        )


def test_warming_heats_and_cloud_cools(cfg, require_weather):
    """Direction check, not just magnitude: a temperature ramp must INCREASE
    cooling demand, and cloud cover must REDUCE it (less solar gain). Getting
    a sign backwards here would produce a plausible-looking table that means
    the opposite of what it says."""
    baseline = score(run_controller(lambda m: ThermostatController(m), cfg), cfg)
    warmed = score(run_controller(lambda m: ThermostatController(m), cfg,
                                   scenario=sm.RAPID_WARMING), cfg)
    clouded = score(run_controller(lambda m: ThermostatController(m), cfg,
                                    scenario=sm.SOLAR_SURGE), cfg)

    assert warmed.energy_kwh > baseline.energy_kwh, "a +6 K outdoor ramp must cost more energy"
    assert clouded.energy_kwh < baseline.energy_kwh, "cloud cover must reduce solar load"


def test_cloud_cover_never_invents_sunlight(cfg):
    """The invariant that keeps these scenarios defensible: no scenario may
    produce more irradiance than 2024 actually delivered, and night must stay
    night."""
    w = pd.DataFrame({
        "t_out_c": [30.0] * 100,
        "ghi_w_m2": [0.0] * 20 + [900.0] * 60 + [0.0] * 20,
        "rh_out_pct": [40.0] * 100,
    })
    out = sm._apply_weather(w, sm.SOLAR_SURGE, ghi_cap_w_m2=1016.0)

    assert out["ghi_w_m2"].max() <= 1016.0
    assert (out["ghi_w_m2"].to_numpy()[:20] == 0.0).all(), "night must stay night"
    assert (out["ghi_w_m2"].to_numpy() <= w["ghi_w_m2"].to_numpy() + 1e-9).all(), (
        "cloud cover may only subtract irradiance, never add it"
    )


def test_crowd_surge_respects_the_crush_cap_and_only_advertises_its_own_station(cfg):
    """Two things at once, both of which were wrong in a first draft: the cap
    must be crush load (not design load, which made the surge nearly inert on
    busy services), and `expected_boarding` must only be raised for the
    minutes where the SURGE station is genuinely the next one."""
    service = Service(direction="down", pattern="semi_express", load_factor=1.0)
    profile = simulate(service, cfg)
    w = pd.DataFrame({"t_out_c": [30.0] * len(profile),
                      "ghi_w_m2": [500.0] * len(profile),
                      "rh_out_pct": [40.0] * len(profile)})

    _, overrides = sm.apply(w, profile, sm.CROWD_SURGE, cfg)
    assert overrides is not None

    crush_cap = int(cfg["occupancy"]["design_load_pax"] * sm.CRUSH_LOAD_FACTOR)
    assert max(overrides["n_pax"]) <= crush_cap

    idx = sm.CROWD_SURGE.crowd_surge_station_index
    surge_arrive = profile.stations[idx].arrive_min
    prev_arrive = profile.stations[idx - 1].arrive_min if idx > 0 else 0
    original = list(profile.expected_boarding)
    boosted = overrides["expected_boarding"]

    # Raised strictly inside the window where the surge station is next...
    assert boosted[prev_arrive] > original[prev_arrive]
    # ...and untouched once it has been passed.
    assert boosted[surge_arrive] == original[surge_arrive]


def test_both_arms_can_be_given_the_identical_scenario(cfg, require_weather):
    """The symmetry rule this whole comparison rests on: the disturbance is
    the weather both controllers fly through, not something applied to one of
    them. Two runs of the SAME controller under the same scenario must agree
    exactly -- if the scenario layer carried any per-run state, they wouldn't."""
    first = run_controller(lambda m: ThermostatController(m), cfg, scenario=sm.COMBINED)
    second = run_controller(lambda m: ThermostatController(m), cfg, scenario=sm.COMBINED)
    assert first["t_air_c"].equals(second["t_air_c"])
    assert first["electrical_w"].equals(second["electrical_w"])


def test_power_metrics_are_populated_and_sane(cfg, require_weather):
    """M14 added four power-shape fields to score(); a metric that silently
    returns zero is worse than no metric."""
    result = score(run_controller(lambda m: ThermostatController(m), cfg), cfg)

    assert result.peak_power_kw > 0.0
    assert result.mean_power_kw > 0.0
    assert result.peak_power_kw >= result.mean_power_kw
    assert result.power_ramp_w_per_min > 0.0
    # Energy and mean power must describe the same trajectory.
    expected_kwh = result.mean_power_kw * result.minutes / 60.0
    assert result.energy_kwh == pytest.approx(expected_kwh, rel=1e-6)


def test_unknown_scenario_name_fails_loudly():
    with pytest.raises(KeyError):
        sm.get("no_such_scenario")
