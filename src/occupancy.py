"""Timetable and passenger occupancy for a mainline intercity service.

Produces the per-minute exogenous inputs the cabin model needs: how many
passengers are aboard, and whether the doors are open. It also exposes the
lookahead quantities the anticipatory controller depends on — time to the next
station and how many passengers are expected to board there — because the
timetable is the only thing that knows them.

Design note: this is modelled as a *service*, not as a time-of-day occupancy
curve. A metro runs continuously and its loading varies through the day; a
Cairo-Alexandria intercity train is a discrete run with a departure time, and
its loading is a property of which service it is. So "peak" here scales a whole
run rather than varying within one.

Confidence warning: no Egyptian National Railways ridership data is public.
Load factors and dwell times here are assumptions, and both are deliberately
exposed as scalable parameters so their influence can be measured rather than
trusted. See docs/PARAMETERS.md.
"""

from dataclasses import dataclass, field

from src.config import load_config


@dataclass
class Station:
    """One stop, with times resolved to minutes from service departure."""

    name: str
    arrive_min: int
    dwell_min: int
    boarding_base: int
    alight_fraction: float

    @property
    def depart_min(self) -> int:
        return self.arrive_min + self.dwell_min


@dataclass
class Service:
    """One train run."""

    direction: str = "down"          # "down" = Cairo -> Alexandria
    load_factor: float = 1.0         # scales boarding against the base pattern
    dwell_scale: float = 1.0         # scales every dwell; see the confidence note
    pattern: str = "semi_express"    # "semi_express" (all calls) or "express"
    label: str = ""


@dataclass
class OccupancyProfile:
    """Per-minute exogenous inputs for one service."""

    n_pax: list[int]
    door_open: list[bool]
    at_station: list[int]            # index of the station being served, else -1
    time_to_next_station_min: list[int]
    expected_boarding: list[int]
    stations: list[Station] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.n_pax)

    @property
    def peak_pax(self) -> int:
        return max(self.n_pax)

    @property
    def passenger_minutes(self) -> int:
        """Total occupancy-minutes — the correct denominator for per-passenger metrics."""
        return sum(self.n_pax)


def _alight_fractions(stations_cfg: list[dict]) -> list[float]:
    """Convert the config's absolute alighting counts into fractions of those aboard.

    Absolute counts break as soon as the load factor changes: scale boarding by
    0.5 and a fixed alighting count can exceed the number of people present.
    Fractions preserve the config's intent, keep occupancy non-negative for any
    load factor, and guarantee the route still empties at the terminus.
    """
    fractions = []
    aboard = 0
    for i, s in enumerate(stations_cfg):
        if i == len(stations_cfg) - 1:
            fractions.append(1.0)          # terminus: everyone leaves
        elif aboard <= 0:
            fractions.append(0.0)
        else:
            fractions.append(min(1.0, s["alighting"] / aboard))
        aboard = aboard - s["alighting"] + s["boarding"]
    return fractions


def _apply_pattern(raw: list[dict], cfg: dict, pattern: str) -> list[dict]:
    """Filter the station list to a stopping pattern.

    "express" keeps only the calls listed in `route.express_calls`, which is the
    real Talgo 2027 pattern (Cairo, Sidi Gaber, Alexandria). Skipped calls also
    return their dwell time, so the express arrives earlier — about 2 h 35
    against the semi-express 2 h 45, close to the real 2 h 30.
    """
    if pattern == "semi_express":
        return [dict(s) for s in raw]
    if pattern != "express":
        raise ValueError(f"unknown pattern {pattern!r}, expected 'semi_express' or 'express'")

    calls = cfg["route"].get("express_calls")
    if not calls:
        raise ValueError("route.express_calls is not configured")

    kept, saved = [], 0
    for station in raw:
        if station["name"] in calls:
            s = dict(station)
            s["arrive_min"] = s["arrive_min"] - saved
            kept.append(s)
        else:
            saved += station["dwell_min"]

    if len(kept) < 2:
        raise ValueError("express pattern must keep at least an origin and a terminus")
    return kept


def build_route(
    cfg: dict | None = None,
    direction: str = "down",
    pattern: str = "semi_express",
) -> list[Station]:
    """Resolve the configured route into Station objects.

    The "up" direction is a mirror of the configured one: the station order
    reverses, and each station's boarding and alighting roles swap. This is an
    [ASSUMPTION] — a real timetable would have its own asymmetric pattern — but
    it doubles scenario variety at near-zero cost and keeps the route balanced.
    """
    cfg = cfg if cfg is not None else load_config()
    raw = _apply_pattern(cfg["route"]["stations"], cfg, pattern)

    if direction == "down":
        seq = [dict(s) for s in raw]
    elif direction == "up":
        seq = [dict(s) for s in reversed(raw)]
        for s in seq:
            s["boarding"], s["alighting"] = s["alighting"], s["boarding"]
        # Rebuild the timetable from reversed segment runtimes.
        runtimes = [raw[i + 1]["arrive_min"] - raw[i]["depart_min"]
                    if "depart_min" in raw[i]
                    else raw[i + 1]["arrive_min"] - (raw[i]["arrive_min"] + raw[i]["dwell_min"])
                    for i in range(len(raw) - 1)]
        runtimes = list(reversed(runtimes))
        # Origin gets the configured origin dwell (terminus preparation); the
        # final stop gets none.
        dwells = [s["dwell_min"] for s in seq]
        dwells[0] = raw[0]["dwell_min"]
        dwells[-1] = 0
        t = 0
        for i, s in enumerate(seq):
            s["arrive_min"] = t
            s["dwell_min"] = dwells[i]
            if i < len(seq) - 1:
                t = t + dwells[i] + runtimes[i]
    else:
        raise ValueError(f"unknown direction {direction!r}, expected 'down' or 'up'")

    fractions = _alight_fractions(seq)
    return [
        Station(
            name=s["name"],
            arrive_min=int(s["arrive_min"]),
            dwell_min=int(s["dwell_min"]),
            boarding_base=int(s["boarding"]),
            alight_fraction=fractions[i],
        )
        for i, s in enumerate(seq)
    ]


def _scale_dwells(stations: list[Station], dwell_scale: float) -> list[Station]:
    """Rescale dwells and shift the timetable so runtimes between stops are preserved."""
    if dwell_scale == 1.0:
        return stations

    out, shift = [], 0
    for i, st in enumerate(stations):
        new_dwell = st.dwell_min if i == len(stations) - 1 else max(1, round(st.dwell_min * dwell_scale))
        out.append(Station(st.name, st.arrive_min + shift, new_dwell,
                           st.boarding_base, st.alight_fraction))
        shift += new_dwell - st.dwell_min
    return out


def simulate(service: Service, cfg: dict | None = None) -> OccupancyProfile:
    """Expand a service into per-minute occupancy, door state and lookahead."""
    cfg = cfg if cfg is not None else load_config()
    stations = _scale_dwells(
        build_route(cfg, service.direction, service.pattern), service.dwell_scale
    )
    total_min = stations[-1].arrive_min + stations[-1].dwell_min + 1

    n_pax = [0] * total_min
    door_open = [False] * total_min
    at_station = [-1] * total_min
    time_to_next = [0] * total_min
    expected_boarding = [0] * total_min

    # Passenger exchange is resolved at arrival: alighting first, then boarding.
    # Both then sit aboard for the whole dwell, which is the conservative choice
    # for heat load — it does not let boarding passengers appear only at
    # departure and so understate the disturbance.
    aboard = 0
    events: dict[int, int] = {}
    for st in stations:
        aboard -= round(aboard * st.alight_fraction)
        aboard += round(st.boarding_base * service.load_factor)
        events[st.arrive_min] = aboard

    aboard = 0
    for t in range(total_min):
        if t in events:
            aboard = events[t]
        n_pax[t] = aboard

        for i, st in enumerate(stations):
            if st.arrive_min <= t < st.depart_min or (st.dwell_min == 0 and t == st.arrive_min):
                at_station[t] = i
                door_open[t] = True
                break

        nxt = next((st for st in stations if st.arrive_min > t), None)
        if nxt is None:
            time_to_next[t] = 0
            expected_boarding[t] = 0
        else:
            time_to_next[t] = nxt.arrive_min - t
            expected_boarding[t] = round(nxt.boarding_base * service.load_factor)

    return OccupancyProfile(
        n_pax=n_pax,
        door_open=door_open,
        at_station=at_station,
        time_to_next_station_min=time_to_next,
        expected_boarding=expected_boarding,
        stations=stations,
    )


def service_catalogue(
    load_factors=(0.4, 0.7, 1.0, 1.1),
    dwell_scale: float = 1.0,
    patterns=("semi_express", "express"),
) -> list[Service]:
    """The set of services used to generate the dataset.

    Load factors are [ASSUMPTION] — no ENR ridership data is public. The spread
    matters more than the individual values: the model must see lightly and
    heavily loaded runs, since passenger heat is the disturbance being
    anticipated.

    1.1 is included deliberately: it is the factor at which the route peaks at
    exactly 80 passengers, the EN 13129 design load for mainline vehicles (all
    seats occupied). Without it the standard's own worst case is never simulated
    — the base pattern peaks at 73.

    Both stopping patterns are included because real ENR services vary: the
    semi-express calls at all six stations, the express follows the Talgo 2027
    pattern of three.
    """
    return [
        Service(direction=d, load_factor=lf, dwell_scale=dwell_scale, pattern=p,
                label=f"{d}-{'exp' if p == 'express' else 'semi'}-lf{lf:g}")
        for d in ("down", "up")
        for p in patterns
        for lf in load_factors
    ]


if __name__ == "__main__":
    cfg = load_config()
    for svc in service_catalogue():
        p = simulate(svc, cfg)
        doors = sum(p.door_open)
        print(f"{svc.label:12s} {len(p):4d} min  peak {p.peak_pax:3d} pax  "
              f"final {p.n_pax[-1]:2d}  doors open {doors:3d} min  "
              f"pax-minutes {p.passenger_minutes:6d}")

    print()
    print("down, load factor 1.0 — station by station:")
    p = simulate(Service(direction="down", load_factor=1.0), cfg)
    for i, st in enumerate(p.stations):
        print(f"  {st.name:18s} arr {st.arrive_min:4d}  dwell {st.dwell_min}  "
              f"board {st.boarding_base:3d}  alight_frac {st.alight_fraction:.3f}  "
              f"-> {p.n_pax[st.arrive_min]:3d} aboard")
