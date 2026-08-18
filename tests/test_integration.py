"""Cross-layer compatibility between M1 (config/weather), M2 (cabin) and M3 (occupancy).

The unit tests check each layer in isolation. These check the seams — the places
where a change in one layer silently invalidates another. Most bugs found so far
have lived exactly here.
"""

import pandas as pd
import pytest

from src.cabin_model import CabinInputs, CabinModel
from src.config import (
    DATA_DIR,
    air_time_constant,
    envelope_ua,
    load_config,
    thermal_time_constant,
    ventilation_ua,
)
from src.occupancy import Service, service_catalogue, simulate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def model(cfg):
    return CabinModel(cfg)


@pytest.fixture(scope="module")
def weather():
    """Cached weather, or skip.

    data/ is gitignored because the CSVs are regenerable, so a fresh clone has
    none. Skipping with an actionable message beats failing with a KeyError —
    this is the first thing anyone cloning the repo will hit.
    """
    frames = {}
    for city in ("cairo", "aswan"):
        for season in ("summer", "winter"):
            path = DATA_DIR / f"weather_{city}_{season}.csv"
            if path.exists():
                frames[(city, season)] = pd.read_csv(path, parse_dates=["timestamp"])

    required = [("cairo", "summer"), ("aswan", "summer")]
    missing = [k for k in required if k not in frames]
    if missing:
        pytest.skip(
            f"weather cache missing {missing}. Run:  python -m src.weather"
        )
    return frames


# --------------------------------------------------- M1 <-> M2: config vs plant


def test_supply_air_exceeds_fresh_air_at_design_load(cfg):
    """Total supply must exceed the fresh-air component it contains."""
    fresh = (cfg["ventilation"]["fresh_air_m3_h_per_passenger"]
             * cfg["occupancy"]["design_load_pax"])
    assert cfg["hvac"]["supply_air_m3_h"] > fresh


def test_supply_temperature_is_below_the_coldest_setpoint(cfg, model):
    """A cabin can only be cooled toward supply temperature, never past it."""
    assert model.supply_min_c < cfg["comfort"]["sliding_setpoint_low_c"]


def _equilibrium_temperature(cfg, model, t_out, ghi, n_pax):
    """Steady cabin temperature where load and supply-air authority balance.

    Includes fan heat. The supply fan's motor sits in the air stream, so its
    power is a permanent sensible load — worth about +1.0 K on equilibrium, and
    enough on its own to push the hottest hours outside the comfort band.
    """
    ua = envelope_ua(cfg) + ventilation_ua(cfg, n_pax)
    q_int = (n_pax * cfg["occupancy"]["sensible_heat_w_per_pax"]
             + ghi * model.solar_aperture
             + model.supply_fan_w)
    return (ua * t_out + q_int + model.supply_ua * model.supply_min_c) / (ua + model.supply_ua)


def test_unit_holds_the_band_through_typical_summer_conditions(cfg, model, weather):
    """M1's sizing must survive M2's state-dependent authority plus fan heat.

    Checked at the 95th percentile rather than the absolute peak, because the
    unit genuinely cannot hold the band at the very hottest hours — see the
    test below, which pins how far it misses.
    """
    band = cfg["comfort"]["band_k"]
    n_pax = simulate(Service(load_factor=1.0), cfg).peak_pax

    for (city, season), df in weather.items():
        if season != "summer":
            continue
        t_eq = df.apply(
            lambda r: _equilibrium_temperature(
                cfg, model, float(r["t_out_c"]), float(r["ghi_w_m2"]), n_pax),
            axis=1,
        )
        limit = df["t_out_c"].apply(lambda t: model.setpoint(t) + band)
        exceeded = (t_eq > limit).mean()

        assert exceeded < 0.05, (
            f"{city}: equilibrium outside the comfort band in {exceeded:.1%} of "
            "summer hours — the unit is undersized for this climate"
        )


def test_extreme_hours_miss_the_band_only_slightly(cfg, model, weather):
    """Pins a known and accepted limitation rather than hiding it.

    At the hottest measured hours the modelled unit cannot hold the comfort
    band: supply-air authority shrinks as the cabin approaches setpoint, and the
    fan contributes a further ~1 K. Real trains behave this way in extreme heat.

    The bound matters. If it grows, something has regressed — and it is also the
    strongest argument for anticipatory control, because at peak there is no
    spare authority left to react with.
    """
    band = cfg["comfort"]["band_k"]
    n_pax = simulate(Service(load_factor=1.0), cfg).peak_pax

    for (city, season), df in weather.items():
        if season != "summer":
            continue
        overshoot = df.apply(
            lambda r: _equilibrium_temperature(
                cfg, model, float(r["t_out_c"]), float(r["ghi_w_m2"]), n_pax)
            - (model.setpoint(float(r["t_out_c"])) + band),
            axis=1,
        ).max()

        assert overshoot < 1.5, (
            f"{city}: worst-case equilibrium is {overshoot:.2f} K outside the band, "
            "beyond the documented tolerance"
        )


def test_control_horizon_covers_the_actuator_delay(cfg):
    """Predicting less far ahead than the plant takes to respond is pointless."""
    hv = cfg["hvac"]
    needed = hv["dead_time_min"] + 3 * hv["tau_act_min"]
    assert cfg["simulation"]["control_horizon_min"] >= needed


def test_logging_interval_resolves_the_fast_mode(cfg):
    """A 1-minute log must not alias the fast air dynamics away."""
    assert air_time_constant(cfg) >= 60.0
    assert thermal_time_constant(cfg) > air_time_constant(cfg)


# ------------------------------------------------- M2 <-> M3: plant vs schedule


def test_occupancy_drives_ventilation_load(cfg):
    """Passenger count must actually reach the plant's ventilation term."""
    profile = simulate(Service(load_factor=1.0), cfg)
    assert ventilation_ua(cfg, profile.peak_pax) > ventilation_ua(cfg, 0)


def test_profile_arrays_are_all_the_same_length(cfg):
    p = simulate(Service(), cfg)
    lengths = {len(p.n_pax), len(p.door_open), len(p.at_station),
               len(p.time_to_next_station_min), len(p.expected_boarding)}
    assert len(lengths) == 1


def test_journey_fits_inside_one_day_of_weather(cfg, weather):
    """The service must not run off the end of the weather series."""
    longest = max(len(simulate(s, cfg)) for s in service_catalogue())
    assert longest / 60.0 < 24.0
    for df in weather.values():
        assert len(df) >= 24


# ------------------------------------------------------- full-stack robustness


def test_every_service_and_city_runs_without_going_unphysical(cfg, model, weather):
    """The combination that will generate the dataset must not blow up anywhere.

    Runs a modulating thermostat over both directions, three load factors and
    both cities. Cheap, and it covers the cross-product the unit tests never do.
    """
    for (city, season), df in weather.items():
        if season != "summer":
            continue
        day = df.tail(24).reset_index(drop=True)

        for svc in service_catalogue():
            profile = simulate(svc, cfg)
            state = model.initial_state(model.setpoint(float(day["t_out_c"].iloc[8])))

            for t in range(len(profile)):
                i = min(8 + t // 60, len(day) - 1)
                t_out = float(day["t_out_c"].iloc[i])
                setpoint = model.setpoint(t_out)
                frac = min(1.0, max(0.0, (state.t_air_c - setpoint) / 1.5))

                result = model.step(state, CabinInputs(
                    t_out_c=t_out,
                    ghi_w_m2=float(day["ghi_w_m2"].iloc[i]),
                    n_pax=profile.n_pax[t],
                    door_open=profile.door_open[t],
                    q_hvac_cmd_w=-model.cooling_capacity_w * frac,
                ), dt_s=60.0)

                assert model.supply_min_c - 1.0 <= result.t_air_c <= 60.0, (
                    f"{city}/{svc.label} went unphysical at minute {t}: "
                    f"{result.t_air_c:.1f} C"
                )
                assert result.electrical_w >= 0.0


def test_cooling_authority_is_never_exceeded_across_the_catalogue(cfg, model, weather):
    """The supply-air clamp must hold under every scenario, not just in isolation."""
    df = weather[("aswan", "summer")]
    day = df.tail(24).reset_index(drop=True)

    for svc in service_catalogue():
        profile = simulate(svc, cfg)
        state = model.initial_state(30.0)
        for t in range(len(profile)):
            i = min(8 + t // 60, len(day) - 1)
            t_before = state.t_air_c
            result = model.step(state, CabinInputs(
                t_out_c=float(day["t_out_c"].iloc[i]),
                ghi_w_m2=float(day["ghi_w_m2"].iloc[i]),
                n_pax=profile.n_pax[t],
                door_open=profile.door_open[t],
                q_hvac_cmd_w=-1e6,
            ), dt_s=60.0)

            # Each substep clamps against its own temperature, so the bound is
            # the limit at the warmest point of the step. Temperature does not
            # always fall while cooling — with passengers boarding and a door
            # open, the cabin can warm despite full cooling, which raises the
            # limit as the step proceeds.
            limit = max(
                abs(model.deliverable_cooling_w(t_before)),
                abs(model.deliverable_cooling_w(result.t_air_c)),
            )
            assert abs(result.q_hvac_actual_w) <= limit + 1.0


def test_passengers_dominate_the_disturbance_not_doors(cfg, model, weather):
    """Guards the finding the project's premise rests on.

    Anticipation works because the timetable predicts *boarding*. If door
    infiltration were the dominant term instead, the story would be much weaker
    — so this asserts the balance rather than assuming it.
    """
    day = weather[("cairo", "summer")].tail(24).reset_index(drop=True)
    profile = simulate(Service(load_factor=1.0), cfg)

    def energy(doors: bool, pax: bool) -> float:
        state = model.initial_state(model.setpoint(float(day["t_out_c"].iloc[8])))
        total = 0.0
        for t in range(len(profile)):
            i = min(8 + t // 60, len(day) - 1)
            t_out = float(day["t_out_c"].iloc[i])
            frac = min(1.0, max(0.0, (state.t_air_c - model.setpoint(t_out)) / 1.5))
            r = model.step(state, CabinInputs(
                t_out_c=t_out,
                ghi_w_m2=float(day["ghi_w_m2"].iloc[i]),
                n_pax=profile.n_pax[t] if pax else 0,
                door_open=profile.door_open[t] and doors,
                q_hvac_cmd_w=-model.cooling_capacity_w * frac,
            ), dt_s=60.0)
            total += r.electrical_w / 60000.0
        return total

    both = energy(True, True)
    passenger_share = both - energy(True, False)
    door_share = both - energy(False, True)

    assert passenger_share > 0 and door_share > 0, "both terms must add heat"
    assert passenger_share > 3 * door_share, (
        f"passengers {passenger_share:.2f} kWh vs doors {door_share:.2f} kWh — "
        "the premise assumes the predictable term dominates"
    )


def test_energy_decomposition_is_monotonic(cfg, model, weather):
    """Adding a heat source must never reduce energy use.

    Asserted on ENERGY, not on peak error. Peak excursion under closed-loop
    control is not guaranteed monotonic in the disturbances: adding a load
    shifts when the controller acts, and that phase change can move the peak
    either way. Energy has no such escape — more heat in means more work out.

    The earlier version of this test used peak error and passed only by
    coincidence; it broke as soon as fan heat was added, which is what exposed
    that it was asserting the wrong invariant.
    """
    day = weather[("cairo", "summer")].tail(24).reset_index(drop=True)
    profile = simulate(Service(load_factor=1.0), cfg)

    def energy_kwh(doors: bool, pax: bool) -> float:
        state = model.initial_state(model.setpoint(float(day["t_out_c"].iloc[8])))
        total = 0.0
        for t in range(len(profile)):
            i = min(8 + t // 60, len(day) - 1)
            t_out = float(day["t_out_c"].iloc[i])
            frac = min(1.0, max(0.0, (state.t_air_c - model.setpoint(t_out)) / 1.5))
            r = model.step(state, CabinInputs(
                t_out_c=t_out,
                ghi_w_m2=float(day["ghi_w_m2"].iloc[i]),
                n_pax=profile.n_pax[t] if pax else 0,
                door_open=profile.door_open[t] and doors,
                q_hvac_cmd_w=-model.cooling_capacity_w * frac,
            ), dt_s=60.0)
            total += r.electrical_w / 60000.0
        return total

    both = energy_kwh(True, True)
    assert both >= energy_kwh(True, False) - 1e-6
    assert both >= energy_kwh(False, True) - 1e-6
    assert energy_kwh(False, False) <= min(energy_kwh(True, False), energy_kwh(False, True))
