"""Host side of the demo board: the bridge between this project's simulator and
controllers and the ESP32 firmware in firmware/railflow-board/.

Three things cross the wire (docs/serial_protocol.md and docs/board_extension.md):

  setpoint  host -> board  RFP1 "R" frame, the advisor's recommended setpoint
  fan       host -> board  "F" frame, AUTO or a fixed manual duty
  telemetry board -> host  "T" frame, what the board's own sensors and fans are doing

`BoardLink` wraps all three over one `SerialBridge`. `replay()` is the
integration with the rest of the project: it takes a trajectory produced by
`run_controller()` (src/evaluate.py), the same function the Streamlit demo and
every comparison use, and streams the advisor's setpoint recommendations to the
board minute by minute. There is no second control path to keep in sync: the
board sees exactly what the live demo's chart shows.

    python -m src.board_link --port COM5 monitor
    python -m src.board_link --port COM5 fan manual 60
    python -m src.board_link --port COM5 fan auto
    python -m src.board_link --port COM5 replay --advisor anticipatory
"""

import lightgbm as lgb  # noqa: I001 -- must import before pandas (see src/train.py)

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import pandas as pd

from src.config import load_config
from src.serial_bridge import (
    BAUD_RATE,
    N_PASS_RANGE,
    Q_CMD_RANGE,
    SETPOINT_RANGE_C,
    SerialBridge,
    checksum,
)

FAN_MARKER = "F"
TELEMETRY_MARKER = "T"
FAN_AUTO = 0
FAN_MANUAL = 1
DUTY_RANGE_PCT = (0, 100)


def encode_fan_frame(seq: int, mode: int, duty_pct: int = 0) -> bytes:
    """`F,<seq>,<mode>,<duty_pct>*<CS>\\n`, docs/board_extension.md §2. Raises
    ValueError on an out-of-range field, same rule as `encode_frame`: this is an
    outbound boundary to real hardware, so it fails loudly instead of clipping."""
    if not 0 <= seq <= 255:
        raise ValueError(f"seq={seq} out of range [0, 255]")
    if mode not in (FAN_AUTO, FAN_MANUAL):
        raise ValueError(f"mode={mode} must be {FAN_AUTO} (auto) or {FAN_MANUAL} (manual)")
    if not DUTY_RANGE_PCT[0] <= duty_pct <= DUTY_RANGE_PCT[1]:
        raise ValueError(f"duty_pct={duty_pct} out of range {list(DUTY_RANGE_PCT)}")
    payload = f"{seq},{mode},{duty_pct}"
    return f"{FAN_MARKER},{payload}*{checksum(payload)}\n".encode("ascii")


def decode_fan_frame(frame: bytes) -> dict:
    """Reference decoder for F frames. The host never receives one; this exists so
    the firmware's parser can be checked against an independent implementation."""
    payload = _checked_payload(frame, FAN_MARKER)
    seq, mode, duty = payload.split(",")
    return {"seq": int(seq), "mode": int(mode), "duty_pct": int(duty)}


@dataclass(frozen=True)
class Telemetry:
    """One decoded "T" frame. Temperatures are NaN when the board's sensor is
    invalid (the board writes `nan`)."""

    seq: int
    t_in_c: float
    t_out_c: float
    motion: bool
    dark: bool
    fan_mode: int
    fan_duty_pct: int
    setpoint_c: float
    link_ok: bool

    @property
    def fan_mode_name(self) -> str:
        return "auto" if self.fan_mode == FAN_AUTO else "manual"


def decode_telemetry(frame: bytes) -> Telemetry:
    """Raises ValueError on a bad marker, bad checksum or wrong field count."""
    payload = _checked_payload(frame, TELEMETRY_MARKER)
    fields = payload.split(",")
    if len(fields) != 9:
        raise ValueError(f"telemetry frame has {len(fields)} fields, expected 9")
    seq, t_in, t_out, motion, dark, mode, duty, setpoint, link = fields
    return Telemetry(
        seq=int(seq), t_in_c=float(t_in), t_out_c=float(t_out),
        motion=motion == "1", dark=dark == "1", fan_mode=int(mode),
        fan_duty_pct=int(duty), setpoint_c=float(setpoint), link_ok=link == "1",
    )


def _checked_payload(frame: bytes, marker: str) -> str:
    line = frame.decode("ascii").rstrip("\r\n")
    head, _, rest = line.partition(",")
    if head != marker:
        raise ValueError(f"bad frame marker {head!r}, expected {marker!r}")
    payload, star, chk = rest.rpartition("*")
    if not star:
        raise ValueError("no checksum delimiter '*' found in frame")
    if chk != checksum(payload):
        raise ValueError(f"checksum mismatch: frame says {chk}, computed {checksum(payload)}")
    return payload


class BoardLink:
    """One serial connection to the board. Owns separate sequence counters for the
    setpoint frames (inside `SerialBridge`) and the fan frames."""

    def __init__(self, bridge: SerialBridge):
        self._bridge = bridge
        self._fan_seq = 0

    @classmethod
    def open(cls, port: str, baudrate: int = BAUD_RATE) -> "BoardLink":
        return cls(SerialBridge(port, baudrate=baudrate))

    def send_setpoint(self, t_in_c: float, t_out_c: float, n_pass: int,
                      setpoint_c: float, q_cmd: float) -> bytes:
        return self._bridge.send(t_in_c, t_out_c, n_pass, setpoint_c, q_cmd)

    def set_fan(self, mode: int, duty_pct: int = 0) -> bytes:
        frame = encode_fan_frame(self._fan_seq, mode, duty_pct)
        self._bridge.write_raw(frame)
        self._fan_seq = (self._fan_seq + 1) % 256
        return frame

    def set_fan_auto(self) -> bytes:
        return self.set_fan(FAN_AUTO)

    def set_fan_manual(self, duty_pct: int) -> bytes:
        return self.set_fan(FAN_MANUAL, duty_pct)

    def poll_telemetry(self) -> list[Telemetry]:
        """Everything the board has reported since the last call. Lines that are not
        valid telemetry (boot-loader noise, a truncated line) are skipped."""
        out = []
        for line in self._bridge.read_lines():
            try:
                out.append(decode_telemetry(line))
            except (ValueError, UnicodeDecodeError):
                continue
        return out

    def close(self) -> None:
        self._bridge.close()

    def __enter__(self) -> "BoardLink":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ---- Integration with the simulator and controllers -------------------------

@dataclass(frozen=True)
class SetpointCommand:
    t_in_c: float
    t_out_c: float
    n_pass: int
    setpoint_c: float
    q_cmd: float
    setpoint_was_clipped: bool


def commands_from_trajectory(trajectory: pd.DataFrame, cfg: dict) -> Iterator[SetpointCommand]:
    """Turns `run_controller()`'s per-minute trajectory into the frames to send.

    `advised_setpoint_c` is what the advisor recommended that minute. RFP1 only
    carries 22.0 to 26.0 C (docs/serial_protocol.md §4) while the advisor's own
    authority is wider (`hvac.advisor_setpoint_min_c/max_c`, and station precool
    deliberately pulls below 22), so a recommendation outside the wire range is
    clipped to it here and flagged on the command rather than silently dropped.
    `q_cmd` is the simulated plant's capacity fraction, a diagnostic only.
    """
    lo, hi = SETPOINT_RANGE_C
    cool_w = cfg["hvac"]["cooling_capacity_w"]
    heat_w = cfg["hvac"]["heating_capacity_w"]
    for row in trajectory.itertuples():
        advised = float(row.advised_setpoint_c)
        clipped = min(max(advised, lo), hi)
        cmd_w = float(row.cmd_w)
        q = cmd_w / (heat_w if cmd_w > 0 else cool_w)
        yield SetpointCommand(
            t_in_c=float(row.t_air_c), t_out_c=float(row.t_out_c),
            n_pass=int(min(max(round(row.n_pax), N_PASS_RANGE[0]), N_PASS_RANGE[1])),
            setpoint_c=clipped,
            q_cmd=min(max(q, Q_CMD_RANGE[0]), Q_CMD_RANGE[1]),
            setpoint_was_clipped=clipped != advised,
        )


@dataclass
class ReplayReport:
    frames_sent: int = 0
    setpoints_clipped: int = 0
    telemetry: list[Telemetry] = field(default_factory=list)


def replay(link: BoardLink, trajectory: pd.DataFrame, cfg: dict,
           seconds_per_minute: float = 2.0,
           on_step: Callable[[int, SetpointCommand, list[Telemetry]], None] | None = None,
           ) -> ReplayReport:
    """Streams one frame per simulated minute to the board, collecting whatever
    telemetry comes back. `seconds_per_minute` is how long one simulated minute
    takes in real time (2.0 plays a three-hour journey in about six minutes; 0
    sends as fast as the loop runs, which is what the tests use). The board's
    link watchdog is 90 s, so keep this well under that."""
    report = ReplayReport()
    for minute, cmd in enumerate(commands_from_trajectory(trajectory, cfg)):
        link.send_setpoint(cmd.t_in_c, cmd.t_out_c, cmd.n_pass, cmd.setpoint_c, cmd.q_cmd)
        report.frames_sent += 1
        report.setpoints_clipped += cmd.setpoint_was_clipped
        got = link.poll_telemetry()
        report.telemetry.extend(got)
        if on_step is not None:
            on_step(minute, cmd, got)
        if seconds_per_minute > 0:
            time.sleep(seconds_per_minute)
    return report


def build_trajectory(advisor: str = "anticipatory", cfg: dict | None = None, **scenario) -> pd.DataFrame:
    """Runs one journey through the same `run_controller()` the live demo uses and
    returns its trajectory. `scenario` is forwarded unchanged (city, date,
    depart_hour, direction, pattern, load_factor); defaults are run_controller's own.

    `advisor` is "thermostat" (the static baseline) or "anticipatory" (the shipped
    forecast-driven advisor; needs the trained model, `python -m src.train`)."""
    from src.controllers import (
        SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN, AnticipatorySetpointAdvisor, ThermostatController,
    )
    from src.evaluate import run_controller
    from src.features import LiveFeatureBuilder
    from src.train import MODEL_PATH

    cfg = cfg if cfg is not None else load_config()
    kwargs = dict(cfg=cfg, advisor_update_interval_min=SHIPPED_ADVISOR_UPDATE_INTERVAL_MIN, **scenario)
    if advisor == "thermostat":
        return run_controller(lambda m: ThermostatController(m), **kwargs)
    if advisor != "anticipatory":
        raise ValueError(f"advisor must be 'thermostat' or 'anticipatory', got {advisor!r}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"no saved model at {MODEL_PATH}. Run: python -m src.train")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    def factory(model):
        builder = LiveFeatureBuilder(
            cfg, city=kwargs.get("city", "cairo"), direction=kwargs.get("direction", "down"),
            pattern=kwargs.get("pattern", "semi_express"),
            load_factor=kwargs.get("load_factor", 1.0), depart_hour=kwargs.get("depart_hour", 8.0))
        return AnticipatorySetpointAdvisor(model, booster, builder)

    return run_controller(factory, **kwargs)


# ---- Command line -----------------------------------------------------------

def _fmt(x: float, unit: str = "") -> str:
    return "--" if math.isnan(x) else f"{x:.1f}{unit}"


def _print_telemetry(t: Telemetry) -> None:
    print(f"board  in {_fmt(t.t_in_c, 'C'):>7}  out {_fmt(t.t_out_c, 'C'):>7}  "
          f"motion {'YES' if t.motion else 'no ':3}  fan {t.fan_mode_name:6} {t.fan_duty_pct:3d}%  "
          f"setpoint {t.setpoint_c:.1f}C  link {'ok' if t.link_ok else 'DOWN'}", flush=True)


def _cmd_monitor(link: BoardLink, args) -> None:
    print("listening for board telemetry, Ctrl+C to stop")
    while True:
        for t in link.poll_telemetry():
            _print_telemetry(t)
        time.sleep(0.2)


def _cmd_fan(link: BoardLink, args) -> None:
    if args.mode == "auto":
        frame = link.set_fan_auto()
    else:
        if args.duty is None:
            raise SystemExit("fan manual needs a duty, e.g.: fan manual 60")
        frame = link.set_fan_manual(args.duty)
    print(f"sent {frame!r}")


def _cmd_replay(link: BoardLink, args) -> None:
    cfg = load_config()
    print(f"simulating one {args.advisor} journey ({args.city}, {args.date}, depart {args.depart_hour:g}h)...")
    traj = build_trajectory(args.advisor, cfg, city=args.city, date=args.date,
                            depart_hour=args.depart_hour, direction=args.direction,
                            pattern=args.pattern, load_factor=args.load_factor)

    def on_step(minute: int, cmd: SetpointCommand, got: list[Telemetry]) -> None:
        clip = "  (clipped to the wire range)" if cmd.setpoint_was_clipped else ""
        print(f"min {minute:3d}  cabin {cmd.t_in_c:5.1f}C  outside {cmd.t_out_c:5.1f}C  "
              f"pax {cmd.n_pass:3d}  setpoint -> {cmd.setpoint_c:.1f}C{clip}", flush=True)
        for t in got:
            _print_telemetry(t)

    report = replay(link, traj, cfg, args.seconds_per_minute, on_step)
    print(f"done: {report.frames_sent} frames sent, {report.setpoints_clipped} clipped, "
          f"{len(report.telemetry)} telemetry frames received")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m src.board_link", description=__doc__.split("\n\n")[0])
    p.add_argument("--port", required=True, help='serial port, e.g. COM5 or /dev/ttyUSB0 ("loop://" for a dry run)')
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("monitor", help="print the board's telemetry")

    fan = sub.add_parser("fan", help="set the fan mode")
    fan.add_argument("mode", choices=["auto", "manual"])
    fan.add_argument("duty", nargs="?", type=int, help="0-100, for manual")

    rp = sub.add_parser("replay", help="stream a simulated journey's setpoints to the board")
    rp.add_argument("--advisor", choices=["thermostat", "anticipatory"], default="anticipatory")
    rp.add_argument("--seconds-per-minute", type=float, default=2.0,
                    help="real seconds per simulated minute (default 2; keep well under 90)")
    rp.add_argument("--city", default="cairo")
    rp.add_argument("--date", default="2024-07-20")
    rp.add_argument("--depart-hour", type=float, default=8.0)
    rp.add_argument("--direction", default="down")
    rp.add_argument("--pattern", default="semi_express")
    rp.add_argument("--load-factor", type=float, default=1.0)

    args = p.parse_args(argv)
    handler = {"monitor": _cmd_monitor, "fan": _cmd_fan, "replay": _cmd_replay}[args.command]
    with BoardLink.open(args.port) as link:
        try:
            handler(link, args)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
