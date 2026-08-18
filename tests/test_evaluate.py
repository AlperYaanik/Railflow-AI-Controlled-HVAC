"""Tests for the M5 scoring metrics.

degree_hours_outside_band is the one that matters most: it replaces an
earlier, uncited "degree-minutes" unit this project invented rather than
sourced. Real HVAC comfort literature uses degree-hours ("Exceedance
Degree-Hours", Salimi et al., Indoor Air, 2021). These tests check the
formula against hand-computed cases, not just that it runs.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.controllers import ThermostatController
from src.evaluate import degree_hours_outside_band, run_controller, score


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ------------------------------------------------- degree_hours_outside_band


def test_zero_when_always_inside_the_band():
    t_air = pd.Series([25.0, 25.5, 24.5, 26.0])
    setpoint = pd.Series([25.0] * 4)
    assert degree_hours_outside_band(t_air, setpoint, band_k=2.0) == pytest.approx(0.0)


def test_matches_hand_computed_single_excursion():
    """One minute at 4 K above a 2 K band -> 2 K of exceedance for 1 minute
    -> 2/60 K*h."""
    t_air = pd.Series([25.0, 29.0, 25.0])  # setpoint 25, band 2 -> excess at index 1 is 2 K
    setpoint = pd.Series([25.0] * 3)
    expected = 2.0 * (1.0 / 60.0)
    assert degree_hours_outside_band(t_air, setpoint, band_k=2.0) == pytest.approx(expected)


def test_scales_with_duration_not_just_magnitude():
    """The same 1 K exceedance sustained for twice as long must score twice
    as high -- this is what distinguishes degree-hours from a minute-count,
    which the earlier degree-minutes framing risked being confused with.
    """
    setpoint = pd.Series([25.0] * 4)
    short = pd.Series([25.0, 28.0, 25.0, 25.0])   # 1 minute at +3 (1 K over a 2 K band)
    long = pd.Series([25.0, 28.0, 28.0, 25.0])    # 2 minutes at +3
    dh_short = degree_hours_outside_band(short, setpoint, band_k=2.0)
    dh_long = degree_hours_outside_band(long, setpoint, band_k=2.0)
    assert dh_long == pytest.approx(2.0 * dh_short)


def test_scales_with_magnitude_not_just_duration():
    """Twice as far outside the band for the same duration must score twice
    as high."""
    setpoint = pd.Series([25.0] * 2)
    mild = pd.Series([25.0, 28.0])    # excess over a 2 K band = 1 K
    severe = pd.Series([25.0, 30.0])  # excess over a 2 K band = 3 K
    dh_mild = degree_hours_outside_band(mild, setpoint, band_k=2.0)
    dh_severe = degree_hours_outside_band(severe, setpoint, band_k=2.0)
    assert dh_severe == pytest.approx(3.0 * dh_mild)


def test_symmetric_for_overcooling_and_overheating():
    setpoint = pd.Series([25.0] * 2)
    hot = pd.Series([25.0, 30.0])
    cold = pd.Series([25.0, 20.0])
    assert degree_hours_outside_band(hot, setpoint, band_k=2.0) == pytest.approx(
        degree_hours_outside_band(cold, setpoint, band_k=2.0)
    )


def test_a_wider_band_never_increases_the_score():
    t_air = pd.Series([25.0, 31.0, 27.0, 25.0])
    setpoint = pd.Series([25.0] * 4)
    narrow = degree_hours_outside_band(t_air, setpoint, band_k=1.0)
    wide = degree_hours_outside_band(t_air, setpoint, band_k=3.0)
    assert wide <= narrow


def test_dt_min_scales_the_result_linearly():
    """Native resolution can be anything; the K*h conversion must track it."""
    t_air = pd.Series([25.0, 29.0])
    setpoint = pd.Series([25.0] * 2)
    per_minute = degree_hours_outside_band(t_air, setpoint, band_k=2.0, dt_min=1.0)
    per_five_minutes = degree_hours_outside_band(t_air, setpoint, band_k=2.0, dt_min=5.0)
    assert per_five_minutes == pytest.approx(5.0 * per_minute)


def test_never_negative():
    rng = np.random.default_rng(0)
    t_air = pd.Series(rng.uniform(15, 35, 200))
    setpoint = pd.Series([25.0] * 200)
    assert degree_hours_outside_band(t_air, setpoint, band_k=2.0) >= 0.0


# --------------------------------------------------------------- run_controller


def test_run_controller_matches_the_manually_traced_reference(cfg, require_weather):
    """Locks in the numbers this file's own __main__ block reports, on the
    same fixed scenario used throughout M5's development. A silent
    regression in run_controller or score() should move these.

    Updated to 30.86 kWh / +0.47 K / 5.06 K after
    ventilation.fresh_air_m3_h_per_passenger was corrected 15->20 (the real
    EN 13129 standard rate, not the reduced one -- see cabin_params.yaml).
    More fresh air is a real physical change (more ventilation heat load in
    a hot climate), so this test moving is the test doing its job, not a
    bug -- the original 26.75/0.58/4.79 reference is in git history if ever
    needed for comparison.
    """
    traj = run_controller(lambda model: ThermostatController(model), cfg)
    result = score(traj, cfg)
    assert result.energy_kwh == pytest.approx(30.86, abs=0.1)
    assert result.mean_err_k == pytest.approx(0.47, abs=0.05)
    assert result.worst_excursion_k == pytest.approx(5.06, abs=0.1)
    assert result.minutes == len(traj)


def test_score_energy_matches_trajectory_sum(cfg, require_weather):
    traj = run_controller(lambda model: ThermostatController(model), cfg)
    result = score(traj, cfg)
    assert result.energy_kwh == pytest.approx(traj["electrical_w"].sum() / 60000.0)


def test_score_uses_the_configured_band(cfg, require_weather):
    """A wider configured band must never increase degree-hours."""
    import copy

    narrow_cfg = copy.deepcopy(cfg)
    narrow_cfg["comfort"]["band_k"] = 0.5
    wide_cfg = copy.deepcopy(cfg)
    wide_cfg["comfort"]["band_k"] = 5.0

    traj = run_controller(lambda model: ThermostatController(model), cfg)
    narrow_result = score(traj, narrow_cfg)
    wide_result = score(traj, wide_cfg)
    assert wide_result.degree_hours <= narrow_result.degree_hours
