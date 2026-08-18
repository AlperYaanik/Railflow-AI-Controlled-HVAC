"""Tests for the timetable and occupancy layer.

The invariants here are cheap to check and expensive to get wrong: an
unbalanced route silently leaves passengers aboard forever, which corrupts
every heat load downstream and would not be obvious in a plot.
"""

import pytest

from src.config import load_config
from src.occupancy import (
    Service,
    build_route,
    service_catalogue,
    simulate,
)

LOAD_FACTORS = [0.0, 0.25, 0.4, 0.7, 1.0, 1.3]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------------------------ invariants


@pytest.mark.parametrize("direction", ["down", "up"])
@pytest.mark.parametrize("load_factor", LOAD_FACTORS)
def test_route_empties_at_the_terminus(cfg, direction, load_factor):
    """Nobody may still be aboard after the last stop, at any load factor."""
    p = simulate(Service(direction=direction, load_factor=load_factor), cfg)
    assert p.n_pax[-1] == 0


@pytest.mark.parametrize("direction", ["down", "up"])
@pytest.mark.parametrize("load_factor", LOAD_FACTORS)
def test_occupancy_is_never_negative(cfg, direction, load_factor):
    p = simulate(Service(direction=direction, load_factor=load_factor), cfg)
    assert min(p.n_pax) >= 0


def test_occupancy_never_exceeds_a_physical_bound(cfg):
    """Even overloaded, a coach cannot hold an absurd number of people."""
    p = simulate(Service(load_factor=1.3), cfg)
    standing_allowance = 2.5 * cfg["occupancy"]["seats"]
    assert p.peak_pax <= standing_allowance


def test_base_pattern_reproduces_the_config_exactly(cfg):
    """At load factor 1.0 the model must match the hand-written config counts.

    This is what justifies converting the config's absolute alighting numbers
    into fractions: the conversion has to be lossless at the point it was
    calibrated, or the config comments stop describing reality.
    """
    p = simulate(Service(direction="down", load_factor=1.0), cfg)
    expected, aboard = [], 0
    for s in cfg["route"]["stations"]:
        aboard += s["boarding"] - s["alighting"]
        expected.append(aboard)

    actual = [p.n_pax[st.arrive_min] for st in p.stations]
    assert actual == expected


# ------------------------------------------------------------------- scaling


def test_load_factor_scales_occupancy_monotonically(cfg):
    peaks = [simulate(Service(load_factor=lf), cfg).peak_pax for lf in LOAD_FACTORS]
    assert peaks == sorted(peaks)
    assert peaks[0] == 0, "a zero load factor must produce an empty train"


def test_zero_load_factor_still_produces_a_valid_timetable(cfg):
    """An empty run is a legitimate scenario and must not degenerate."""
    p = simulate(Service(load_factor=0.0), cfg)
    assert max(p.n_pax) == 0
    assert sum(p.door_open) > 0, "doors still open at stations on an empty run"


def test_alighting_cannot_exceed_those_aboard(cfg):
    """The reason alighting is a fraction rather than an absolute count.

    With absolute counts from the config, a low load factor would try to remove
    more passengers than are present and drive occupancy negative.
    """
    p = simulate(Service(load_factor=0.25), cfg)
    for i, st in enumerate(p.stations[1:], start=1):
        before = p.n_pax[st.arrive_min - 1]
        after = p.n_pax[st.arrive_min]
        assert after >= 0
        assert after - before <= round(st.boarding_base * 0.25) + 1


# --------------------------------------------------------------------- doors


def test_doors_are_open_only_while_stopped(cfg):
    p = simulate(Service(), cfg)
    for t, is_open in enumerate(p.door_open):
        if is_open:
            assert p.at_station[t] >= 0, f"doors open at t={t} with no station"
        else:
            assert p.at_station[t] == -1


def test_every_station_opens_its_doors(cfg):
    p = simulate(Service(), cfg)
    served = {p.at_station[t] for t, o in enumerate(p.door_open) if o}
    assert served == set(range(len(p.stations)))


def test_doors_are_shut_for_most_of_the_journey(cfg):
    """Sanity on the running/stopped ratio for an intercity service."""
    p = simulate(Service(), cfg)
    assert sum(p.door_open) / len(p) < 0.25


def test_dwell_scale_changes_door_time_but_not_the_route(cfg):
    """Dwell length is a low-confidence input, so it must be adjustable safely."""
    short = simulate(Service(dwell_scale=0.5), cfg)
    base = simulate(Service(dwell_scale=1.0), cfg)

    assert sum(short.door_open) < sum(base.door_open)
    assert short.peak_pax == base.peak_pax, "dwell length must not change loading"
    assert short.n_pax[-1] == 0


# ----------------------------------------------------------------- lookahead


def test_time_to_next_station_counts_down(cfg):
    """The lookahead feature the anticipatory controller depends on."""
    p = simulate(Service(), cfg)
    arrivals = {st.arrive_min for st in p.stations}
    for t in range(len(p) - 1):
        if p.time_to_next_station_min[t] > 0 and (t + 1) not in arrivals:
            assert p.time_to_next_station_min[t + 1] <= p.time_to_next_station_min[t]


def test_expected_boarding_matches_the_next_station(cfg):
    p = simulate(Service(load_factor=1.0), cfg)
    for t in range(len(p)):
        nxt = next((st for st in p.stations if st.arrive_min > t), None)
        expected = 0 if nxt is None else round(nxt.boarding_base)
        assert p.expected_boarding[t] == expected


def test_lookahead_leads_the_actual_boarding_event(cfg):
    """Before a big boarding, the lookahead must already show it.

    This is the whole mechanism of the project: if this fails, the controller
    has no advance warning and there is nothing to anticipate.
    """
    p = simulate(Service(direction="down", load_factor=1.0), cfg)
    busiest = max(p.stations[1:], key=lambda s: s.boarding_base)
    lead = busiest.arrive_min - 20
    assert lead > 0
    assert p.expected_boarding[lead] == round(busiest.boarding_base)
    assert p.time_to_next_station_min[lead] == 20


# ----------------------------------------------------------------- direction


def test_up_direction_is_a_valid_mirror(cfg):
    down = simulate(Service(direction="down"), cfg)
    up = simulate(Service(direction="up"), cfg)

    assert [s.name for s in up.stations] == [s.name for s in down.stations][::-1]
    assert up.n_pax[-1] == 0
    assert abs(len(up) - len(down)) <= 2, "mirrored journey should take similar time"
    assert up.peak_pax > 0


def test_catalogue_covers_both_directions_and_a_load_spread(cfg):
    cat = service_catalogue()
    assert {s.direction for s in cat} == {"down", "up"}
    assert len({s.load_factor for s in cat}) >= 3
    for svc in cat:
        assert simulate(svc, cfg).n_pax[-1] == 0


def test_unknown_direction_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown direction"):
        build_route(cfg, direction="sideways")
