"""Tests for M8's Railflow Serial Protocol v1 sender (src/serial_bridge.py,
docs/serial_protocol.md).

The loopback tests use pyserial's own built-in "loop://" URL -- an
in-memory virtual port pyserial ships specifically for testing without
real or virtual-COM-driver hardware. Confirmed working in this environment
by running it before writing any of this file, not assumed from pyserial's
docs: no admin rights or driver install needed either way.
"""

import pytest

from src.serial_bridge import SerialBridge, checksum, decode_frame, encode_frame


def test_checksum_matches_the_protocol_documents_worked_example():
    """docs/serial_protocol.md §6 walks through this exact payload by hand
    and states the result is 0x3F. That document and this code must never
    silently drift apart -- this is the tripwire.
    """
    assert checksum("142,24.3,38.7,52,25.1,-0.67") == "3F"


def test_encode_frame_matches_the_protocol_documents_worked_example():
    """docs/serial_protocol.md §4's example frame, byte for byte."""
    frame = encode_frame(seq=142, t_in_c=24.3, t_out_c=38.7, n_pass=52,
                          setpoint_c=25.1, q_cmd=-0.67)
    assert frame == b"R,142,24.3,38.7,52,25.1,-0.67*3F\n"


def test_decode_frame_is_the_exact_inverse_of_encode_frame():
    frame = encode_frame(seq=7, t_in_c=26.5, t_out_c=41.2, n_pass=63,
                          setpoint_c=22.8, q_cmd=0.34)
    decoded = decode_frame(frame)
    assert decoded == {"seq": 7, "t_in_c": 26.5, "t_out_c": 41.2,
                        "n_pass": 63, "setpoint_c": 22.8, "q_cmd": 0.34}


@pytest.mark.parametrize("field,value", [
    ("seq", -1), ("seq", 256),
    ("t_in_c", -10.1), ("t_in_c", 60.1),
    ("t_out_c", -10.1), ("t_out_c", 55.1),
    ("n_pass", -1), ("n_pass", 101),
    ("setpoint_c", 21.9), ("setpoint_c", 26.1),
    ("q_cmd", -1.01), ("q_cmd", 1.01),
])
def test_encode_frame_rejects_every_out_of_range_field(field, value):
    """This is an outbound boundary to real hardware (docs/serial_protocol.md
    §4.1) -- an out-of-range value must fail loudly, not get silently
    clipped and sent anyway. One case per field, at both edges."""
    kwargs = dict(seq=0, t_in_c=25.0, t_out_c=35.0, n_pass=40,
                  setpoint_c=24.0, q_cmd=0.0)
    kwargs[field] = value
    with pytest.raises(ValueError):
        encode_frame(**kwargs)


def test_encode_frame_accepts_every_range_boundary_value():
    """The edges themselves (not one past them) must be valid -- ranges in
    docs/serial_protocol.md §4 are inclusive on both ends."""
    encode_frame(seq=0, t_in_c=-10.0, t_out_c=-10.0, n_pass=0,
                 setpoint_c=22.0, q_cmd=-1.0)
    encode_frame(seq=255, t_in_c=60.0, t_out_c=55.0, n_pass=100,
                 setpoint_c=26.0, q_cmd=1.0)


def test_decode_frame_rejects_a_bad_marker():
    with pytest.raises(ValueError, match="marker"):
        decode_frame(b"X,0,25.0,35.0,40,24.0,0.00*00\n")


def test_decode_frame_rejects_a_corrupted_checksum():
    """docs/serial_protocol.md §7 requires the receiver to discard a frame
    that fails its checksum rather than act on any field from it -- a
    single flipped digit must be caught, not silently accepted."""
    good = encode_frame(seq=1, t_in_c=25.0, t_out_c=35.0, n_pass=40,
                         setpoint_c=24.0, q_cmd=0.0)
    corrupted = good.replace(b"25.0", b"29.0")
    with pytest.raises(ValueError, match="checksum"):
        decode_frame(corrupted)


def test_serial_bridge_round_trips_over_the_builtin_loopback():
    """The actual "validated against a virtual COM loopback" check
    ROADMAP.md's M8 section requires -- real pyserial writes and reads,
    not a mock of either.
    """
    with SerialBridge("loop://") as bridge:
        sent = bridge.send(t_in_c=24.3, t_out_c=38.7, n_pass=52,
                            setpoint_c=25.1, q_cmd=-0.67)
        received = bridge._ser.read(len(sent))
        assert received == sent
        assert decode_frame(received) == {
            "seq": 0, "t_in_c": 24.3, "t_out_c": 38.7,
            "n_pass": 52, "setpoint_c": 25.1, "q_cmd": -0.67,
        }


def test_serial_bridge_sequence_number_increments_then_wraps():
    """§4's seq field wraps at 256 (a fresh counter each time, uint8-sized
    even though frames are ASCII, not binary) -- checked across the wrap
    boundary specifically, not just the first few values.

    Reads each frame back immediately after sending it. Found the hard way:
    pyserial's "loop://" backend has a bounded internal buffer, simulating
    real UART hardware rather than an unbounded queue -- writing all 258
    frames first with no interleaved reads fills that buffer and blocks
    write() forever, since nothing ever drains it. Confirmed by watching an
    earlier version of this test hang indefinitely before adding the read
    below, not assumed from pyserial's docs.
    """
    with SerialBridge("loop://") as bridge:
        for expected_seq in range(258):
            sent = bridge.send(t_in_c=25.0, t_out_c=35.0, n_pass=40,
                                setpoint_c=24.0, q_cmd=0.0)
            received = bridge._ser.read(len(sent))
            assert decode_frame(received)["seq"] == expected_seq % 256
