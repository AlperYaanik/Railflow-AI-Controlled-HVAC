"""Tests for the host side of the demo board (src/board_link.py,
docs/board_extension.md).

Everything runs over pyserial's in-memory "loop://" port, like
tests/test_serial_bridge.py: whatever the link writes it can read back, which
also means every poll sees the host's own outgoing frames mixed in with the
telemetry the test injects, a useful stand-in for line noise.
"""

import math

import pandas as pd
import pytest

from src.board_link import (
    FAN_AUTO,
    FAN_MANUAL,
    BoardLink,
    commands_from_trajectory,
    decode_fan_frame,
    decode_telemetry,
    encode_fan_frame,
    replay,
)
from src.config import DATA_DIR, load_config
from src.serial_bridge import checksum, decode_frame


def _telemetry_frame(payload: str) -> bytes:
    return f"T,{payload}*{checksum(payload)}\n".encode("ascii")


def test_fan_frame_matches_the_documented_worked_example():
    """docs/board_extension.md §2 states this frame byte for byte."""
    assert encode_fan_frame(seq=3, mode=FAN_MANUAL, duty_pct=60) == b"F,3,1,60*04\n"


def test_fan_frame_round_trips_through_the_reference_decoder():
    frame = encode_fan_frame(seq=255, mode=FAN_AUTO, duty_pct=0)
    assert decode_fan_frame(frame) == {"seq": 255, "mode": FAN_AUTO, "duty_pct": 0}


@pytest.mark.parametrize("kwargs", [
    dict(seq=-1, mode=0, duty_pct=0), dict(seq=256, mode=0, duty_pct=0),
    dict(seq=0, mode=2, duty_pct=0), dict(seq=0, mode=1, duty_pct=-1),
    dict(seq=0, mode=1, duty_pct=101),
])
def test_fan_frame_rejects_out_of_range_fields(kwargs):
    with pytest.raises(ValueError):
        encode_fan_frame(**kwargs)


def test_decode_telemetry_reads_every_field():
    t = decode_telemetry(_telemetry_frame("9,26.4,38.0,1,0,1,60,24.5,1"))
    assert (t.seq, t.t_in_c, t.t_out_c) == (9, 26.4, 38.0)
    assert t.motion is True and t.dark is False
    assert (t.fan_mode, t.fan_mode_name, t.fan_duty_pct) == (FAN_MANUAL, "manual", 60)
    assert t.setpoint_c == 24.5 and t.link_ok is True


def test_decode_telemetry_reads_an_invalid_sensor_as_nan():
    t = decode_telemetry(_telemetry_frame("0,nan,31.0,0,1,0,50,24.0,0"))
    assert math.isnan(t.t_in_c) and t.t_out_c == 31.0


def test_decode_telemetry_rejects_a_corrupted_frame():
    good = _telemetry_frame("1,25.0,35.0,0,0,0,0,24.0,1")
    with pytest.raises(ValueError, match="checksum"):
        decode_telemetry(good.replace(b"25.0", b"29.0"))


def test_decode_telemetry_rejects_the_wrong_marker_and_field_count():
    with pytest.raises(ValueError, match="marker"):
        decode_telemetry(b"R,1,2*00\n")
    with pytest.raises(ValueError, match="fields"):
        decode_telemetry(_telemetry_frame("1,25.0,35.0"))


def test_board_link_sends_setpoint_and_fan_frames_with_independent_sequences():
    with BoardLink.open("loop://") as link:
        link.send_setpoint(24.3, 38.7, 52, 25.1, -0.67)
        link.set_fan_manual(40)
        link.send_setpoint(24.4, 38.7, 52, 25.0, -0.60)
        link.set_fan_auto()
        raw = link._bridge._ser.read(link._bridge._ser.in_waiting)

    lines = raw.split(b"\n")[:-1]
    r = [decode_frame(line + b"\n") for line in lines if line.startswith(b"R")]
    f = [decode_fan_frame(line + b"\n") for line in lines if line.startswith(b"F")]
    assert [x["seq"] for x in r] == [0, 1]
    assert [(x["seq"], x["mode"], x["duty_pct"]) for x in f] == [(0, FAN_MANUAL, 40), (1, FAN_AUTO, 0)]


def test_poll_telemetry_returns_only_valid_telemetry_and_buffers_partial_lines():
    with BoardLink.open("loop://") as link:
        link.set_fan_auto()  # echoed back by the loopback: valid frame, wrong type
        link._bridge.write_raw(b"ets Jun  8 2016 boot noise\n")
        link._bridge.write_raw(_telemetry_frame("0,25.0,35.0,0,0,0,0,24.0,1"))
        link._bridge.write_raw(b"T,1,26.0,3")  # cut off mid-line
        first = link.poll_telemetry()
        link._bridge.write_raw(b"5.0,0,0,0,0,24.0,1*00\n")  # rest of it, wrong checksum
        second = link.poll_telemetry()

    assert [t.seq for t in first] == [0]
    assert second == []  # the completed line fails its checksum and is skipped, not raised


def _trajectory(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["t_air_c", "t_out_c", "n_pax", "advised_setpoint_c", "cmd_w"])


def test_commands_clip_the_advisors_setpoint_to_the_wire_range_and_flag_it():
    """The advisor may recommend 18-30 C but RFP1 carries 22-26 C. The clip has
    to be explicit: every clipped minute is flagged, none is silently altered."""
    cfg = load_config()
    traj = _trajectory([(25.0, 38.0, 40, 24.0, 0.0), (25.0, 38.0, 40, 19.0, -1000.0),
                        (25.0, 38.0, 40, 29.0, 500.0)])
    cmds = list(commands_from_trajectory(traj, cfg))
    assert [c.setpoint_was_clipped for c in cmds] == [False, True, True]
    assert [c.setpoint_c for c in cmds] == [24.0, 22.0, 26.0]


def test_commands_normalise_plant_power_to_a_capacity_fraction():
    cfg = load_config()
    cool, heat = cfg["hvac"]["cooling_capacity_w"], cfg["hvac"]["heating_capacity_w"]
    traj = _trajectory([(25, 38, 40, 24, -cool / 2), (25, 38, 40, 24, heat / 4),
                        (25, 38, 40, 24, -cool * 3)])  # beyond capacity, clipped to -1
    q = [c.q_cmd for c in commands_from_trajectory(traj, cfg)]
    assert q == pytest.approx([-0.5, 0.25, -1.0])


def test_commands_round_and_clamp_passenger_count():
    cfg = load_config()
    traj = _trajectory([(25, 38, 51.6, 24, 0), (25, 38, 130.0, 24, 0)])
    assert [c.n_pass for c in commands_from_trajectory(traj, cfg)] == [52, 100]


def test_replay_streams_one_valid_frame_per_simulated_minute():
    cfg = load_config()
    traj = _trajectory([(25.0 + i * 0.1, 38.0, 40 + i, 23.0 + i * 0.5, -2000.0 * i) for i in range(5)])
    seen, sent = [], []
    with BoardLink.open("loop://") as link:
        real_send = link.send_setpoint
        # replay() drains the loopback while polling telemetry, so record what it
        # sent at the call instead of reading the port afterwards.
        link.send_setpoint = lambda *a: sent.append(real_send(*a)) or sent[-1]
        report = replay(link, traj, cfg, seconds_per_minute=0,
                        on_step=lambda minute, cmd, got: seen.append(minute))

    frames = [decode_frame(frame) for frame in sent]
    assert report.frames_sent == 5 and seen == [0, 1, 2, 3, 4]
    assert [f["seq"] for f in frames] == [0, 1, 2, 3, 4]
    assert [f["setpoint_c"] for f in frames] == [23.0, 23.5, 24.0, 24.5, 25.0]
    assert report.setpoints_clipped == 0


def test_replay_reports_how_many_setpoints_it_had_to_clip():
    cfg = load_config()
    traj = _trajectory([(25.0, 38.0, 40, 19.0, 0.0), (25.0, 38.0, 40, 24.0, 0.0)])
    with BoardLink.open("loop://") as link:
        report = replay(link, traj, cfg, seconds_per_minute=0)
    assert report.setpoints_clipped == 1


def test_run_controller_trajectory_carries_the_columns_replay_needs():
    """replay() relies on run_controller() exposing t_out_c (added for the board
    link); guard that against the trajectory schema changing under it."""
    if not (DATA_DIR / "weather_cairo_summer.csv").exists():
        pytest.skip("Run: python -m src.weather")
    from src.board_link import build_trajectory
    traj = build_trajectory("thermostat")
    for col in ("t_air_c", "t_out_c", "n_pax", "advised_setpoint_c", "cmd_w"):
        assert col in traj.columns
    assert len(list(commands_from_trajectory(traj, load_config()))) == len(traj)
