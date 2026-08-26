"""M14: named, seeded disturbance scenarios -- the honest alternative to
inventing results.

WHY THIS EXISTS, STATED PLAINLY. The feedback on this project was that the
AI advisor doesn't visibly beat the non-AI baseline, and the suggestion that
came with it was to inject fake data so the AI's numbers look better. That
is not what this module does and not what it is for. Nothing here touches a
result: every scenario modifies only the ENVIRONMENT (weather, occupancy),
identically for every controller being compared, before any controller sees
it. Whatever advantage appears afterwards has to come from the control
policy, or it doesn't appear at all -- including the case where the AI comes
out worse, which stays in the table (see ROADMAP.md's M14 section).

THE HYPOTHESIS BEING TESTED. Anticipatory control should pay only when a
disturbance is (a) large enough that reacting late is genuinely costly and
(b) visible in the forecast early enough to act on. This project's existing
23 TEST scenarios are drawn from real fetched 2024 weather on a fixed
timetable, and measured this session, they are mild enough that the shipped
advisor's trajectory is BIT-IDENTICAL to the static schedule in 10 of 23
(it makes only 4 decisions per journey and a +/-0.5 K deadband zeroes most
of them). If anticipation has a regime where it wins, these scenarios are
not it. That is a falsifiable claim, and M14 exists to test it either way.

WHERE THE FORECAST COMES FROM, AND WHY NOTHING IS FAKED. src/evaluate.py's
run_controller() reads the CURRENT weather as `w[...].iloc[t]` and the
forecast the advisor sees as `w[...].iloc[t + horizon]` -- the SAME array,
different index. So a disturbance written into `w` here automatically
becomes visible to the advisor exactly `control_horizon_min` minutes early,
through the normal feature path, with no separate "forecast" channel to
tamper with. The anticipation is real or it is nothing.

EVERY MAGNITUDE HERE IS [ASSUMPTION] and documented as such below. They are
chosen to be physically defensible for the Cairo-Alexandria corridor in
summer, not to produce a particular answer.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """One named operating condition. Defaults are all-zero, i.e. `stable`:
    an untouched environment, so `Scenario("stable")` reproduces this
    project's pre-M14 behaviour bit-for-bit and serves as the control arm."""

    name: str

    t_out_ramp_k: float = 0.0
    """[ASSUMPTION] Outdoor temperature rise, in K, applied as a linear ramp
    over `t_out_ramp_duration_min` and then HELD for the rest of the journey
    (not a spike that reverts -- a spike would let a purely reactive
    controller ride it out by doing nothing, which is not the question being
    asked). Physically: moving out of the Delta's coastal moderation into
    inland desert heat, or a hot-air advection event. Egyptian summer
    inland/coastal differences of this order are ordinary, not extreme."""
    t_out_ramp_start_min: int = 0
    t_out_ramp_duration_min: int = 30

    ghi_cloud_factor: float = 1.0
    """[ASSUMPTION] Fraction of clear-sky irradiance reaching the vehicle
    while cloud cover sits over it (1.0 = unchanged, 0.35 = heavy broken
    cloud). The DISTURBANCE this models is not the shading itself but the
    step back UP when the cloud clears and full sun returns.

    Modelled as attenuation-then-recovery rather than as added irradiance,
    because adding is physically wrong: the first attempt here added
    +350 W/m^2 over minutes 40-90, but on a midday departure the real
    fetched GHI is already 891-924 W/m^2, so the addition simply clipped
    against the observed 2024 maximum (Aswan 1016 W/m^2) and delivered an
    effective +100 -- a disturbance of +0.5% journey energy, i.e. nothing.
    Sun cannot exceed clear-sky; cloud can only subtract from it. Attenuating
    first leaves real headroom for the recovery step, and matches how an
    intermittent-cloud day actually behaves."""
    ghi_cloud_start_min: int = 0
    ghi_cloud_duration_min: int = 0

    crowd_surge_pax: int = 0
    """[ASSUMPTION] Extra passengers boarding at `crowd_surge_station_index`,
    on top of whatever the timetable already schedules, and staying aboard
    afterwards. Physically: a busier-than-timetabled service carrying
    standing passengers, which mainline Egyptian services routinely do.

    Capped at `occupancy.design_load_pax` x CRUSH_LOAD_FACTOR rather than at
    design load itself -- design load is EN13129's "every seat occupied,"
    and a surge on an already-full train means people standing, not people
    failing to board. Capping at design load would have made this scenario
    almost inert on exactly the busy services it is meant to represent (the
    existing TEST scenarios already reach 73 of 80 seats, so a +40 surge
    would have been clipped to +7)."""
    crowd_surge_station_index: int = 2

    seed: int = 20260826
    """Reproducibility, matching data_generator.py's existing
    np.random.default_rng(seed) convention. Unused while no scenario adds
    noise, but fixed now so any later stochastic term is deterministic from
    the start rather than retrofitted."""

    def is_stable(self) -> bool:
        return (self.t_out_ramp_k == 0.0 and self.ghi_cloud_factor == 1.0
                and self.crowd_surge_pax == 0)

    def disturbance_windows(self) -> list[tuple[int, int, str]]:
        """(start_min, end_min, label) spans for shading on demo charts --
        so a viewer can see WHEN the disturbance ran rather than inferring it
        from the shape of a curve."""
        spans = []
        if self.t_out_ramp_k != 0.0:
            spans.append((self.t_out_ramp_start_min,
                          self.t_out_ramp_start_min + self.t_out_ramp_duration_min,
                          "outdoor temp ramp"))
        if self.ghi_cloud_factor != 1.0 and self.ghi_cloud_duration_min > 0:
            spans.append((self.ghi_cloud_start_min,
                          self.ghi_cloud_start_min + self.ghi_cloud_duration_min,
                          "cloud cover (sun returns after)"))
        return spans


def _apply_weather(w: pd.DataFrame, scenario: Scenario, ghi_cap_w_m2: float) -> pd.DataFrame:
    """Returns a MODIFIED COPY -- run_controller() reads its weather frame
    more than once (current minute and forecast minute), and mutating a
    shared frame in place would make a scenario leak between runs."""
    out = w.copy()
    n = len(out)

    if scenario.t_out_ramp_k != 0.0:
        ramp = np.zeros(n)
        start = scenario.t_out_ramp_start_min
        dur = max(1, scenario.t_out_ramp_duration_min)
        for i in range(n):
            if i < start:
                continue
            # Linear to full magnitude, then held -- see t_out_ramp_k's docstring
            # for why this is a step-and-hold rather than a spike.
            ramp[i] = scenario.t_out_ramp_k * min(1.0, (i - start) / dur)
        out["t_out_c"] = out["t_out_c"].to_numpy() + ramp

    if scenario.ghi_cloud_factor != 1.0 and scenario.ghi_cloud_duration_min > 0:
        factor = np.ones(n)
        lo = scenario.ghi_cloud_start_min
        hi = min(n, lo + scenario.ghi_cloud_duration_min)
        factor[lo:hi] = scenario.ghi_cloud_factor
        # Multiplicative, so night stays night with no special case: 0 * f = 0.
        # Clipped to the real observed maximum for safety even though
        # attenuation cannot exceed it -- cheap, and keeps the invariant
        # "no scenario invents more sun than 2024 actually delivered" true
        # regardless of what factors get configured later.
        out["ghi_w_m2"] = np.clip(out["ghi_w_m2"].to_numpy() * factor, 0.0, ghi_cap_w_m2)

    return out


CRUSH_LOAD_FACTOR = 1.5
"""[ASSUMPTION] Multiple of EN13129 design load a mainline coach can
physically carry once standing passengers are counted. 1.5x (120 on this
80-seat vehicle) is a modest crush figure -- real crush loading on
standing-permitted stock runs higher -- chosen so a crowd-surge scenario
represents a busy service rather than an implausible one."""


def apply(w: pd.DataFrame, profile, scenario: Scenario, cfg: dict):
    """Transforms (weather, occupancy) for one scenario. Returns
    `(weather, occupancy_overrides)`. The occupancy PROFILE object is not
    rebuilt -- only the two fields a crowd surge actually changes are
    overridden. Door timings, station indices and the schedule-derived
    countdowns all stay exactly as the timetable produced them.

    `occupancy_overrides` is None when the scenario doesn't touch occupancy.

    BOTH `n_pax` AND `expected_boarding` are overridden, deliberately.
    Overriding only `n_pax` would make the crowd surge invisible to the
    advisor in advance -- `expected_boarding` is the feature through which
    an upcoming passenger load reaches M4's forecaster at all, so leaving it
    at the timetabled value would have made this an unforecastable shock and
    tested robustness rather than anticipation. Updating both keeps the
    information available through the NORMAL feature path; whether the
    advisor actually uses it depends on `ff_weight`, which is precisely the
    open question M14 exists to measure, not something this module decides.
    """
    ghi_cap = float(cfg.get("scenarios", {}).get("ghi_cap_w_m2", 1016.0))
    w_out = _apply_weather(w, scenario, ghi_cap)

    overrides = None
    if scenario.crowd_surge_pax != 0:
        crush_cap = int(cfg["occupancy"]["design_load_pax"] * CRUSH_LOAD_FACTOR)
        stations = profile.stations
        idx = min(scenario.crowd_surge_station_index, len(stations) - 1)
        boarding_min = stations[idx].arrive_min

        n_pax = list(profile.n_pax)
        for t in range(boarding_min, len(n_pax)):
            n_pax[t] = min(crush_cap, n_pax[t] + scenario.crowd_surge_pax)

        # expected_boarding[t] is what boards at the station that is NEXT at
        # minute t (occupancy.py: first station with arrive_min > t). So only
        # the window where the SURGE station is the next one may advertise the
        # bigger number -- boosting every earlier minute would wrongly inflate
        # the boarding expected at the intervening stations too.
        prev_arrive = stations[idx - 1].arrive_min if idx > 0 else 0
        expected_boarding = list(profile.expected_boarding)
        for t in range(prev_arrive, min(boarding_min, len(expected_boarding))):
            expected_boarding[t] += scenario.crowd_surge_pax

        overrides = {"n_pax": n_pax, "expected_boarding": expected_boarding}

    return w_out, overrides


# --- The named set. Deliberately small: five conditions that each isolate one
# --- mechanism, plus one that combines them, rather than a large sweep whose
# --- individual effects can't be attributed.

STABLE = Scenario("stable")

RAPID_WARMING = Scenario(
    "rapid_warming",
    t_out_ramp_k=6.0, t_out_ramp_start_min=30, t_out_ramp_duration_min=30,
)

SOLAR_SURGE = Scenario(
    "solar_surge",
    ghi_cloud_factor=0.35, ghi_cloud_start_min=40, ghi_cloud_duration_min=45,
)

CROWD_SURGE = Scenario(
    "crowd_surge",
    crowd_surge_pax=40, crowd_surge_station_index=2,
)

COMBINED = Scenario(
    "combined",
    t_out_ramp_k=6.0, t_out_ramp_start_min=30, t_out_ramp_duration_min=30,
    ghi_cloud_factor=0.35, ghi_cloud_start_min=40, ghi_cloud_duration_min=45,
    crowd_surge_pax=40, crowd_surge_station_index=2,
)

ALL = (STABLE, RAPID_WARMING, SOLAR_SURGE, CROWD_SURGE, COMBINED)
BY_NAME = {s.name: s for s in ALL}


def get(name: str) -> Scenario:
    if name not in BY_NAME:
        raise KeyError(f"unknown scenario {name!r}. Known: {sorted(BY_NAME)}")
    return BY_NAME[name]
