# Railflow Serial Protocol v1 (RFP1)

The handover document for whoever implements the board side. Everything
Railflow's own sender (`src/serial_bridge.py`) does is specified here; the
board only needs this document, not the Python source, to build a
compatible receiver in any language.

## 1. Scope and direction

**One-way: Railflow (the compute unit running the controller) → the board
(whatever drives the physical HVAC actuator).** Railflow is the sender
only; this document does not define a return channel. If the board needs to
report its own status back (fault codes, measured compressor current, etc.),
that is a second, separate protocol — out of scope here, and shouldn't
reuse this frame format, since mixing directions on one wire format
tends to make both harder to reason about.

Each frame carries **both** the sensor readings Railflow's controller used
for its decision (`t_in_c`, `t_out_c`, `n_pass`) **and** the decision itself
(`setpoint_c`, `q_cmd`). The board is not expected to trust Railflow's
copy of the sensor readings over its own local sensors if it has them —
they're included so the board can log/display/cross-check what Railflow
saw, not because the board has no other way to get them.

## 2. Physical layer

| | |
|---|---|
| Interface | UART |
| Baud rate | **9600**, 8 data bits, no parity, 1 stop bit (8N1) |
| Flow control | None |

9600 8N1 is the one setting every microcontroller UART supports without
special configuration. A frame is at most ~40 bytes (see §4); at 9600 baud
that's under 40 ms to transmit, against a nominal 30-second send interval
(§5) — there is no bandwidth pressure here that would justify a faster,
less universally-supported baud rate.

## 3. Framing

ASCII text, one frame per line, terminated by a single `\n` (0x0A, LF —
not CRLF). A receiver reads until `\n` and parses the buffered line; this
is the simplest framing an embedded UART loop can implement (a byte-at-a-
time state machine appending to a buffer until it sees `\n`, then handing
the buffer off to the parser and clearing it), and it fails safe: a
receiver that comes online mid-frame just discards the partial first line
and is correctly synced from the second `\n` onward.

## 4. Frame format

```
R,<seq>,<t_in_c>,<t_out_c>,<n_pass>,<setpoint_c>,<q_cmd>*<checksum>\n
```

Example:
```
R,142,24.3,38.7,52,25.1,-0.67*3F
```

| Field | Type | Format | Range | Meaning |
|---|---|---|---|---|
| `R` | marker | fixed literal `R` | — | Frame-start anchor. Lets a receiver that's lost sync (e.g. just powered on mid-transmission) recognize the start of the next real frame rather than trying to parse garbage. |
| `seq` | uint | decimal, no leading zeros | 0–255, wraps to 0 | Frame counter. Lets the receiver detect a dropped frame (a gap in the sequence) or a stale/duplicate one — see §7. Not used for retransmission; there is no request-resend in this protocol. |
| `t_in_c` | float | 1 decimal place | −10.0 to 60.0 | Cabin air temperature, °C. Range matches the bound this project's own controller sensitivity tests already assert on simulated cabin air temperature (`tests/test_controllers.py`). |
| `t_out_c` | float | 1 decimal place | −10.0 to 55.0 | Outside air temperature, °C. Upper bound has headroom above the corridor's real measured 2024 peak (48.1 °C, Aswan — see `docs/PARAMETERS.md`). |
| `n_pass` | uint | decimal | 0–100 | Passenger count aboard. Upper bound has headroom above the highest scenario in this project's catalogue (80 seats × 1.1 load factor ≈ 88 — see `src/occupancy.py`). |
| `setpoint_c` | float | 1 decimal place | 22.0–26.0 | Target cabin temperature, °C — this project's sliding EN 13129-style setpoint (`config/cabin_params.yaml`'s `comfort.sliding_setpoint_low_c`/`sliding_setpoint_high_c`). |
| `q_cmd` | float | 2 decimal places, signed | −1.00 to 1.00 | **Commanded capacity fraction, not watts** — see §4.1 for why. Negative = cooling, positive = heating, 0.00 = off. |
| `checksum` | hex | 2 uppercase hex digits | 00–FF | XOR of every byte in the payload between the first comma after `R` and the `*` (i.e. `seq` through `q_cmd`, commas included, marker and `*` excluded). Same convention NMEA-0183 uses for exactly the same reason: a human can recompute it by hand from a terminal capture with no tooling, which matters when debugging a link with a scope or a bare serial monitor and nothing else. |

### 4.1 Why `q_cmd` is a fraction, not watts

The obvious alternative was to send a commanded power in watts, matching
`CabinModel`'s internal `q_hvac_cmd_w`. That was deliberately rejected:
watts would only be meaningful to the board if the board's real HVAC unit
happens to share Railflow's *modelled* capacity (`cooling_capacity_w`/
`heating_capacity_w` in `config/cabin_params.yaml`) — two numbers that have
no reason to match exactly, and no way for either side to detect it if they
don't. A normalized fraction of rated capacity needs the board to know only
its *own* actuator's real capacity to turn this into a duty cycle or
compressor stage, which is a number the board side already has to know
regardless. This also happens to be a value Railflow already computes for
its own diagnostics — `AnticipatoryController.last_frac` in
`src/controllers.py` — so sending it is reusing an existing number, not
deriving a new one.

## 5. Cadence

**Nominal: one frame every 30 seconds.** Not a hard physical requirement —
a recommendation the board owner can change — but not arbitrary either:
the cabin's fastest simulated mode (`tau_fast`, the air node alone) is
1.10 minutes (66 s) at the project's default internal convection
coefficient, and even at the fastest end of that coefficient's sensitivity
sweep in `docs/PARAMETERS.md` (`h = 15`, well above the textbook range used
there) it is still 0.59 minutes (35 s) — so 30 s stays under it, though with
little margin at that extreme. The actuator itself adds 2 minutes of dead
time plus a further ~5 minute lag on top of whichever of those applies
(`hvac.dead_time_min`, `hvac.tau_act_min` in `config/cabin_params.yaml`, the
latter characterised, not asserted — see ROADMAP.md's M6), which is what
actually dominates: the actuator's own lag, not the link's update rate, is
the real bottleneck on how quickly a command can take effect. Sending
faster than 30 s would not make the system more responsive, only busier.

## 6. Checksum worked example

For the example frame above, the payload between the marker and `*` is:

```
142,24.3,38.7,52,25.1,-0.67
```

XOR every byte of that ASCII string together (treat each character as its
byte value, XOR all of them, including the commas and the minus sign):
the result is `0x3F`, printed as the 2-character uppercase hex string `3F`
— matching the `*3F` in the example. `src/serial_bridge.py`'s `checksum()`
function is the reference implementation; `tests/test_serial_bridge.py`
locks in this exact example frame as a regression check, so it will never
silently drift from this document.

## 7. What the receiver (board side) must do

This is the part a document has to be explicit about, since "someone else
can implement the board side from this document alone, without asking
questions" is the actual bar (see `ROADMAP.md`'s M8 section):

1. **Read until `\n`, then parse.** Discard anything before the first `\n`
   after power-on (§3) — it may be a partial frame.
2. **Verify the marker is `R`.** If not, discard the line and wait for the
   next `\n`.
3. **Recompute the checksum over the payload and compare.** If it doesn't
   match, discard the frame entirely — do not act on any field from a
   frame that fails its checksum, including fields that look individually
   plausible.
4. **Track `seq`.** If it isn't exactly one more than the last accepted
   frame's `seq` (mod 256), a frame was dropped — note it (e.g. a counter
   for diagnostics) but there is nothing more to do about it: this protocol
   does not request retransmission, and the next frame 30 seconds later
   supersedes it anyway. If `seq` instead repeats or goes backward relative
   to the last accepted frame, treat the frame as stale/duplicate and
   discard it without acting on it — a valid checksum on an out-of-order
   frame is not, by itself, a reason to overwrite a more recent command
   with an older one.
5. **Range-check every field against §4's table before acting on it.**
   Railflow's own sender validates these before transmitting (raises
   rather than sends an out-of-range frame — see `src/serial_bridge.py`),
   so a frame that passes its checksum should never fail a range check.
   If one ever does, that is a bug upstream of the wire, not a value the
   board should try to clamp into range and act on anyway.
6. **Implement a link-loss watchdog.** If no *valid* frame (passed
   checksum) arrives within 3 nominal intervals (90 s at the §5 cadence),
   assume the link or the sender is down and fall back to a safe default —
   HVAC off (`q_cmd = 0`) is the recommended default, not whatever the last
   received command was. Silently continuing to act on a stale command
   forever is the one failure mode this document will not leave to
   judgement: don't do it.

## 8. What this protocol deliberately does not do

- **No acknowledgement / no retransmission.** A dropped frame is simply
  superseded by the next one 30 seconds later (§5, §7.4). Adding a
  request-resend layer would add real complexity for a link where the next
  update is already seconds away and a stale command already has a defined
  safe fallback (§7.6) — not worth it at this update rate.
- **No encryption or authentication.** This is a point-to-point wired link
  inside one vehicle, not a network link. Out of scope, not overlooked.
- **No support for multiple cabins/units on one link.** One sender, one
  receiver, one wire. A multi-drop bus (e.g. RS-485 with addressing) would
  need a different frame format with a destination address field — a
  real extension, not assumed here.
