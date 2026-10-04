"""M8: the sender side of the Railflow Serial Protocol v1 (RFP1) -- see
docs/serial_protocol.md for the full frame spec, every design decision's
rationale, and what the receiver (board side) must do with what this sends.
This file is deliberately just the sender: the protocol is one-way (§1 of
that document), and the board can be implemented from the document alone,
in any language, without reading this file at all.

`encode_frame()`/`checksum()` are the two functions that MUST match the
document exactly -- tests/test_serial_bridge.py locks in the document's own
worked example (§6) as a regression check specifically so this file and
that document can never silently drift apart.
"""

import serial

FRAME_MARKER = "R"
BAUD_RATE = 9600

T_IN_RANGE_C = (-10.0, 60.0)
T_OUT_RANGE_C = (-10.0, 55.0)
N_PASS_RANGE = (0, 100)
SETPOINT_RANGE_C = (22.0, 26.0)
Q_CMD_RANGE = (-1.0, 1.0)
SEQ_RANGE = (0, 255)


def checksum(payload: str) -> str:
    """XOR of every byte in `payload`, as 2 uppercase hex digits -- the same
    convention NMEA-0183 sentences use, and for the same reason: a human can
    recompute it by hand from a serial-terminal capture with no tooling.
    See docs/serial_protocol.md's §4 table and §6 worked example.
    """
    x = 0
    for b in payload.encode("ascii"):
        x ^= b
    return f"{x:02X}"


def encode_frame(
    seq: int, t_in_c: float, t_out_c: float, n_pass: int, setpoint_c: float, q_cmd: float,
) -> bytes:
    """Builds one RFP1 frame. Raises ValueError if any field is outside its
    documented range (docs/serial_protocol.md §4) -- this is an outbound
    boundary to real hardware, so an out-of-range value here is a bug worth
    failing loudly on, not silently clipping into range and sending anyway.

    M10: `setpoint_c` is the primary field -- the decision this protocol
    exists to deliver (see docs/serial_protocol.md §4.1), sourced from
    AnticipatorySetpointAdvisor.recommend_setpoint() (src/controllers.py),
    already clipped to SETPOINT_RANGE_C before it ever reaches this
    function. `q_cmd` is now an optional, simulation-only diagnostic --
    not a value the board should act on -- kept in the frame for telemetry/
    debugging, not because a real receiver needs it to decide anything.
    """
    def _check(name: str, value: float, lo: float, hi: float) -> None:
        if not (lo <= value <= hi):
            raise ValueError(f"{name}={value} out of range [{lo}, {hi}]")

    _check("seq", seq, *SEQ_RANGE)
    _check("t_in_c", t_in_c, *T_IN_RANGE_C)
    _check("t_out_c", t_out_c, *T_OUT_RANGE_C)
    _check("n_pass", n_pass, *N_PASS_RANGE)
    _check("setpoint_c", setpoint_c, *SETPOINT_RANGE_C)
    _check("q_cmd", q_cmd, *Q_CMD_RANGE)

    payload = f"{seq},{t_in_c:.1f},{t_out_c:.1f},{n_pass},{setpoint_c:.1f},{q_cmd:.2f}"
    return f"{FRAME_MARKER},{payload}*{checksum(payload)}\n".encode("ascii")


def decode_frame(frame: bytes) -> dict:
    """Reference decoder. Railflow itself never calls this -- it only sends
    (§1) -- but tests/test_serial_bridge.py uses it to verify round-trip
    integrity over the loopback, and it doubles as an unambiguous reference
    for whoever implements the board-side receiver in a different language.
    Raises ValueError on a bad marker or a checksum mismatch, matching what
    docs/serial_protocol.md §7 requires the receiver to do: discard, don't
    act on any field from a frame that fails validation.
    """
    line = frame.decode("ascii").rstrip("\n")
    marker, _, rest = line.partition(",")
    if marker != FRAME_MARKER:
        raise ValueError(f"bad frame marker {marker!r}, expected {FRAME_MARKER!r}")

    payload, star, chk = rest.rpartition("*")
    if not star:
        raise ValueError("no checksum delimiter '*' found in frame")
    expected = checksum(payload)
    if chk != expected:
        raise ValueError(f"checksum mismatch: frame says {chk}, computed {expected}")

    seq, t_in_c, t_out_c, n_pass, setpoint_c, q_cmd = payload.split(",")
    return {
        "seq": int(seq), "t_in_c": float(t_in_c), "t_out_c": float(t_out_c),
        "n_pass": int(n_pass), "setpoint_c": float(setpoint_c), "q_cmd": float(q_cmd),
    }


class SerialBridge:
    """Owns one outbound RFP1 link: the pyserial connection and the
    wrapping sequence counter (§4's `seq`, wraps at 256).

    `port` is any pyserial URL -- a real port name ("COM3", "/dev/ttyUSB0")
    for actual hardware, or "loop://" for pyserial's built-in in-memory
    loopback, which is what tests/test_serial_bridge.py validates this
    class against. Both forms need nothing beyond pyserial itself: no
    virtual-COM-port driver install, no admin rights, confirmed by running
    it, not assumed from pyserial's docs.
    """

    def __init__(self, port: str, baudrate: int = BAUD_RATE, timeout: float = 1.0):
        self._ser = serial.serial_for_url(port, baudrate=baudrate, timeout=timeout)
        self._seq = 0
        self._rx = b""

    def send(self, t_in_c: float, t_out_c: float, n_pass: int, setpoint_c: float, q_cmd: float) -> bytes:
        """Encodes and writes one frame, then advances the sequence counter.
        Returns the exact bytes written, mainly so callers/tests can inspect
        or log the wire-level frame without a second encode."""
        frame = encode_frame(self._seq, t_in_c, t_out_c, n_pass, setpoint_c, q_cmd)
        self._ser.write(frame)
        self._seq = (self._seq + 1) % 256
        return frame

    def write_raw(self, data: bytes) -> None:
        """Writes bytes that are not an RFP1 "R" frame, e.g. the board-extension
        fan command (docs/board_extension.md). Does not touch the R sequence
        counter: each frame type keeps its own."""
        self._ser.write(data)

    def read_lines(self) -> list[bytes]:
        """Returns every complete '\\n'-terminated line received since the last
        call, without blocking. A trailing partial line stays buffered until its
        '\\n' arrives. Used for the board's telemetry (docs/board_extension.md §3);
        the one-way R-frame sender above never reads."""
        waiting = self._ser.in_waiting
        if waiting:
            self._rx += self._ser.read(waiting)
        *lines, self._rx = self._rx.split(b"\n")
        return lines

    def close(self) -> None:
        self._ser.close()

    def __enter__(self) -> "SerialBridge":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


if __name__ == "__main__":
    # Demonstrates one send over the built-in loopback -- no hardware, no
    # virtual-COM driver, just pyserial's own "loop://" URL. Real usage
    # would pass a real port name instead ("COM3", "/dev/ttyUSB0").
    with SerialBridge("loop://") as bridge:
        frame = bridge.send(t_in_c=24.3, t_out_c=38.7, n_pass=52, setpoint_c=25.1, q_cmd=-0.67)
        print(f"sent {len(frame)} bytes: {frame!r}")
        received = bridge._ser.read(len(frame))
        print(f"read back over the loopback: {received!r}")
        print(f"decoded: {decode_frame(received)}")
