"""Cross-checks the board firmware's protocol and fan code against the Python
reference implementations.

firmware/railflow-board/include/rfp1.h and fan_logic.h are pure C++ with no
Arduino dependency. This compiles test/rfp1_harness.cpp, which wraps those exact
headers, with the system C++ compiler, feeds it frames and compares its answers
with src/serial_bridge.py and src/board_link.py. If the C++ parser and the
Python encoder ever disagree about a frame, this is where it shows.

Compiler lookup: g++/c++/clang++ on PATH, else g++ inside WSL (so it runs on a
Windows machine with WSL), else the module is skipped with a message.
"""

import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from src.board_link import FAN_AUTO, FAN_MANUAL, decode_telemetry, encode_fan_frame
from src.serial_bridge import checksum, encode_frame

FIRMWARE = Path(__file__).resolve().parent.parent / "firmware" / "railflow-board"
HARNESS = FIRMWARE / "test" / "rfp1_harness.cpp"
INCLUDE = FIRMWARE / "include"
FLAGS = ["-std=c++11", "-Wall", "-Wextra", "-Werror"]


def _wsl_path(p: Path) -> str:
    # --exec skips the shell, which would otherwise eat Windows backslashes
    return subprocess.run(["wsl", "--exec", "wslpath", "-a", p.as_posix()], capture_output=True,
                          text=True, check=True).stdout.strip()


def _build(tmp_path_factory) -> list[str] | None:
    """Returns the command that runs the compiled harness, or None if there is no compiler."""
    out = tmp_path_factory.mktemp("rfp1") / "harness"
    for cxx in ("g++", "c++", "clang++"):
        if shutil.which(cxx):
            subprocess.run([cxx, *FLAGS, f"-I{INCLUDE}", str(HARNESS), "-o", str(out)], check=True)
            return [str(out)]
    if shutil.which("wsl"):
        try:
            exe = "/tmp/railflow_rfp1_harness"
            subprocess.run(["wsl", "--exec", "g++", *FLAGS, f"-I{_wsl_path(INCLUDE)}", _wsl_path(HARNESS),
                            "-o", exe], capture_output=True, check=True)
            return ["wsl", "--exec", exe]
        except (subprocess.CalledProcessError, OSError):
            return None
    return None


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    cmd = _build(tmp_path_factory)
    if cmd is None:
        pytest.skip("no C++ compiler (g++/c++/clang++ or WSL g++) available for the firmware harness")

    def run(lines: list[str]) -> list[str]:
        proc = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True,
                              check=True, timeout=60)
        return proc.stdout.splitlines()

    return run


def _frame(marker: str, payload: str) -> str:
    return f"{marker},{payload}*{checksum(payload)}"


def test_firmware_parses_the_protocol_documents_worked_example(harness):
    assert harness(["R,142,24.3,38.7,52,25.1,-0.67*3F"]) == ["OK 142 24.3 38.7 52 25.1 -0.67"]


def test_firmware_parser_agrees_with_the_python_encoder_on_random_valid_frames(harness):
    rng = random.Random(1234)
    sent, expected = [], []
    for _ in range(300):
        kw = dict(seq=rng.randint(0, 255), t_in_c=round(rng.uniform(-10, 60), 1),
                  t_out_c=round(rng.uniform(-10, 55), 1), n_pass=rng.randint(0, 100),
                  setpoint_c=round(rng.uniform(22, 26), 1), q_cmd=round(rng.uniform(-1, 1), 2))
        sent.append(encode_frame(**kw).decode().rstrip("\n"))
        expected.append("OK {seq} {t_in_c:.1f} {t_out_c:.1f} {n_pass} {setpoint_c:.1f} {q_cmd:.2f}".format(**kw))
    assert harness(sent) == expected


def test_firmware_accepts_every_range_boundary(harness):
    lo = encode_frame(0, -10.0, -10.0, 0, 22.0, -1.0).decode().rstrip("\n")
    hi = encode_frame(255, 60.0, 55.0, 100, 26.0, 1.0).decode().rstrip("\n")
    out = harness([lo, hi])
    assert [line.split()[0] for line in out] == ["OK", "OK"]


@pytest.mark.parametrize("line, status", [
    ("R,1,25.0,35.0,40,24.0,0.00*00", "ERR_CHECKSUM"),
    ("R,1,25.0,35.0,40,24.0,0.00", "ERR_NO_CHECKSUM"),
    ("R,1,25.0,35.0,40,24.0,0.00*3f", "ERR_NO_CHECKSUM"),   # checksum is uppercase hex only
    ("R,1,25.0,35.0,40,24.0,0.00*3", "ERR_NO_CHECKSUM"),
    ("R1,25.0,35.0,40,24.0,0.00*3F", "ERR_MARKER"),
    (_frame("R", "1,25.0,35.0,40,24.0"), "ERR_FORMAT"),          # a field missing
    (_frame("R", "1,25.0,35.0,40,24.0,0.00,9"), "ERR_FORMAT"),   # a field too many
    (_frame("R", "1,warm,35.0,40,24.0,0.00"), "ERR_FORMAT"),
    (_frame("R", "1,25.0,35.0,40,,0.00"), "ERR_FORMAT"),
    (_frame("R", "256,25.0,35.0,40,24.0,0.00"), "ERR_RANGE"),
    (_frame("R", "1,60.1,35.0,40,24.0,0.00"), "ERR_RANGE"),
    (_frame("R", "1,25.0,55.1,40,24.0,0.00"), "ERR_RANGE"),
    (_frame("R", "1,25.0,35.0,101,24.0,0.00"), "ERR_RANGE"),
    (_frame("R", "1,25.0,35.0,40,26.1,0.00"), "ERR_RANGE"),      # advisor-range value, not wire-range
    (_frame("R", "1,25.0,35.0,40,21.9,0.00"), "ERR_RANGE"),
    (_frame("R", "1,25.0,35.0,40,24.0,1.01"), "ERR_RANGE"),
    (_frame("R", "1,nan,35.0,40,24.0,0.00"), "ERR_FORMAT"),
])
def test_firmware_rejects_a_bad_setpoint_frame_without_acting_on_it(harness, line, status):
    assert harness([line]) == [status]


def test_firmware_parses_fan_frames_like_the_python_encoder(harness):
    frames = [encode_fan_frame(0, FAN_AUTO, 0), encode_fan_frame(3, FAN_MANUAL, 60),
              encode_fan_frame(255, FAN_MANUAL, 100), encode_fan_frame(7, FAN_MANUAL, 0)]
    out = harness([f.decode().rstrip("\n") for f in frames])
    assert out == ["OK 0 0 0", "OK 3 1 60", "OK 255 1 100", "OK 7 1 0"]


@pytest.mark.parametrize("line, status", [
    ("F,3,1,60*00", "ERR_CHECKSUM"),
    (_frame("F", "3,2,60"), "ERR_RANGE"),    # unknown mode
    (_frame("F", "3,1,101"), "ERR_RANGE"),
    (_frame("F", "3,1,-1"), "ERR_RANGE"),
    (_frame("F", "256,1,60"), "ERR_RANGE"),
    (_frame("F", "3,1"), "ERR_FORMAT"),
    (_frame("F", "3,fast,60"), "ERR_FORMAT"),
])
def test_firmware_rejects_a_bad_fan_frame(harness, line, status):
    assert harness([line]) == [status]


def test_firmware_sequence_rule_matches_the_documented_wraparound_window(harness):
    """seq is newer when the forward distance is 1..127 (mod 256), see rfp1.h."""
    pairs = [(0, 1), (0, 0), (5, 4), (255, 0), (255, 1), (0, 255), (10, 138), (10, 137), (128, 0)]
    out = harness([f"SEQ {last} {seq}" for last, seq in pairs])
    expected = [str(int(1 <= (seq - last) % 256 < 128)) for last, seq in pairs]
    assert out == expected


@pytest.mark.parametrize("args, fields", [
    ((9, 26.4, 38.0, 1, 0, FAN_MANUAL, 60, 24.5, 1),
     dict(seq=9, t_in_c=26.4, t_out_c=38.0, motion=True, dark=False, fan_mode=FAN_MANUAL,
          fan_duty_pct=60, setpoint_c=24.5, link_ok=True)),
    ((255, -3.0, 55.0, 0, 1, FAN_AUTO, 0, 22.0, 0),
     dict(seq=255, t_in_c=-3.0, t_out_c=55.0, motion=False, dark=True, fan_mode=FAN_AUTO,
          fan_duty_pct=0, setpoint_c=22.0, link_ok=False)),
])
def test_firmware_telemetry_decodes_with_the_python_reference(harness, args, fields):
    (line,) = harness(["BUILD_T " + " ".join(str(a) for a in args)])
    t = decode_telemetry(line.encode())
    for name, value in fields.items():
        assert getattr(t, name) == pytest.approx(value) if isinstance(value, float) else getattr(t, name) == value


def test_firmware_writes_an_invalid_sensor_as_nan(harness):
    (line,) = harness(["BUILD_T 1 nan 31.0 0 0 0 50 24.0 0"])
    t = decode_telemetry(line.encode())
    assert math.isnan(t.t_in_c) and t.t_out_c == pytest.approx(31.0)


def test_auto_fan_follows_the_setpoint_with_hysteresis_and_a_failsafe(harness):
    """Tuning in the harness matches include/config.h: 0.5 K deadband, 20 %/K,
    30 % minimum duty, 50 % failsafe. Setpoint 24.0 throughout."""
    steps = [
        ("24.3", 1, 0),    # 0.3 K above: inside the deadband, off
        ("24.6", 1, 32),   # past the deadband: starts at minimum duty plus a little
        ("26.0", 1, 60),   # 2.0 K above: proportional
        ("40.0", 1, 100),  # far above: capped at 100 %
        ("24.4", 1, 30),   # back to 0.4 K: still on (off threshold is 0.25 K), floor at min duty
        ("24.2", 1, 0),    # below the off threshold: off
        ("22.0", 1, 0),    # colder than the setpoint: no heating path, fan stays off
        ("nan", 1, 50),    # reading is NaN: failsafe
        ("25.0", 0, 50),   # sensor flagged bad even though a number is present: failsafe
        ("24.4", 1, 0),    # recovery restarts from "off", so 0.4 K is back inside the deadband
    ]
    out = harness([f"FAN {t} 24.0 {ok}" for t, ok, _ in steps])
    assert out == [str(duty) for _, _, duty in steps]
