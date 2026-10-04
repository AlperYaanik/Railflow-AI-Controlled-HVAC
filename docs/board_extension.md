# Board extension to the Railflow protocol

`docs/serial_protocol.md` (RFP1) defines one thing: the `R` frame, a one-way
setpoint recommendation from Railflow to the board. It says a board-to-host
channel, if one is ever needed, must be a separate protocol rather than reuse
the R frame. This document is that separate protocol, plus one extra
host-to-board frame so the demo board's fans can be set from the host. The R
frame is unchanged: a board that implements only RFP1 keeps working, and it
simply ignores frames whose marker it does not know (RFP1 §7.2).

| Marker | Direction | Purpose | Defined in |
|---|---|---|---|
| `R` | host → board | cabin setpoint recommendation | `serial_protocol.md` §4 |
| `F` | host → board | fan mode and manual duty | §2 below |
| `T` | board → host | telemetry | §3 below |

All three share the physical layer (9600 8N1), the framing (one ASCII line,
`\n`-terminated, never CRLF) and the checksum (XOR of the payload bytes between
the first comma and `*`, two uppercase hex digits). The firmware is in
`firmware/railflow-board/`; the host side is `src/board_link.py`.

## 1. Why these two frames and not more

The board is a demo model of a cabin: two temperature sensors, a motion sensor,
a light sensor, an LCD and two fans. It is not the HVAC unit. The fans stand in
for the unit's output so a viewer can see a recommended setpoint turn into
something physical, and RFP1's rule still holds on the board: **Railflow
recommends a setpoint, the board decides its own actuator output from it**. No
frame here carries a power or duty command from the advisor. The `F` frame is a
*human's* override (the equivalent of someone turning the thermostat to
"manual"), not part of the control loop.

## 2. `F` frame: fan mode (host → board)

```
F,<seq>,<mode>,<duty_pct>*<checksum>\n
```

Example: `F,3,1,60*04`

| Field | Type | Range | Meaning |
|---|---|---|---|
| `seq` | uint | 0–255, wraps | Own counter, independent of the R frame's. Same acceptance rule as R (§5). |
| `mode` | uint | `0` or `1` | `0` = AUTO: the board runs its own fan control on the setpoint. `1` = MANUAL: hold `duty_pct`. |
| `duty_pct` | uint | 0–100 | Fan duty in percent. Used only when `mode` is `1`; send `0` with AUTO. |

A frame that fails its checksum, has the wrong field count, or has any field out
of range is discarded whole, like an R frame (RFP1 §7.3, §7.5).

## 3. `T` frame: telemetry (board → host)

Sent every 5 seconds.

```
T,<seq>,<t_in_c>,<t_out_c>,<motion>,<dark>,<fan_mode>,<fan_duty>,<setpoint_c>,<link_ok>*<checksum>\n
```

Examples:

```
T,17,26.4,38.0,1,0,1,60,24.5,1*17
T,0,nan,31.0,0,1,0,50,24.0,0*51
```

| Field | Format | Meaning |
|---|---|---|
| `seq` | uint 0–255, wraps | Telemetry's own counter. |
| `t_in_c` | 1 decimal, or `nan` | Cabin temperature from the board's own sensor. `nan` once the sensor has failed 3 reads in a row. |
| `t_out_c` | 1 decimal, or `nan` | Outside temperature, same rule. |
| `motion` | `0`/`1` | Motion seen in the last 10 s. |
| `dark` | `0`/`1` | The light module reports darkness. |
| `fan_mode` | `0`/`1` | Current mode (see `F`). |
| `fan_duty` | 0–100 | Duty the fans are actually running at, in either mode. |
| `setpoint_c` | 1 decimal | The setpoint the board is using right now: the last valid R frame's, or the board's own default while the link is down. |
| `link_ok` | `0`/`1` | `1` if a valid R frame arrived within the last 90 s. |

The host skips any line that is not a valid `T` frame, so boot-loader messages
and the echo of the host's own frames (a loopback port) do no harm.

## 4. What the board does

**Link loss.** RFP1 §7.6 requires a watchdog. After 90 s with no valid R frame
the board stops using the last received setpoint and falls back to its own
default (24.0 °C, `DEFAULT_SETPOINT_C` in `firmware/railflow-board/include/config.h`).
The status LED is lit only while the link is healthy.

**AUTO fan.** A cooling-only proportional rule on `t_in_c` (the board's own
reading) against the active setpoint: off while the cabin is within 0.5 K of the
setpoint, then 30 % duty plus 20 % per kelvin beyond that deadband, capped at
100 %. It turns off again below 0.25 K, so sensor noise at the edge does not
make it chatter. A cabin colder than the setpoint gets no fan; there is no
heating path on this board. If the indoor sensor is invalid the fans run at a
50 % failsafe duty, because a ventilation fan that stops when its sensor dies is
the worse failure. The numbers live in `FAN_TUNING` in `config.h`.

**MANUAL fan.** Holds the requested duty until another `F` frame or a button
press changes it. Unlike a setpoint recommendation this is deliberately *not*
subject to the link watchdog: it is a person's standing choice, the same as a
manual dial on a real unit.

**Local button.** The BOOT button steps through AUTO, then manual 0, 25, 50, 75
and 100 %, then back to AUTO, so the demo works with no host attached.

**LCD.** Two pages, switching every 5 s. Page 1: `In:31°C Out:31°C` over
`Motion: YES`. Page 2: fan mode and duty over the active setpoint and link state.

## 5. Sequence numbers across the wrap

RFP1 §7.4 says a repeated or backward `seq` must be discarded without
saying how "backward" works when the counter wraps from 255 to 0. The firmware
(`seq_is_newer` in `include/rfp1.h`) takes the forward distance
`(seq - last) mod 256`: 1 to 127 is newer (a gap of up to 126 dropped frames is
tolerated), 0 is a duplicate, and 128 to 255 is treated as having gone
backward. The first frame after power-on is always accepted. The same rule
applies to `F` frames, with their own counter.

## 6. Known gap: setpoint range

RFP1 carries `setpoint_c` in 22.0–26.0 °C. The advisor's own authority is
wider (`hvac.advisor_setpoint_min_c/max_c`, 18–30 °C) and station precool
deliberately recommends below 22. `src/board_link.py` therefore clips each
recommendation to the wire range and counts the clipped minutes
(`ReplayReport.setpoints_clipped`, also printed per minute by the CLI), instead
of sending an out-of-range frame the board would correctly reject. If the demo
should show the full advisor range, widen RFP1's `setpoint_c` range on both
sides together (`SETPOINT_RANGE_C` in `src/serial_bridge.py` and the range
check in `include/rfp1.h`); the cross-check test
`tests/test_firmware_protocol.py` fails until both agree.
