# Railflow demo board firmware

ESP32 firmware for the scale-model demonstrator shown in the main README: two
temperature sensors (cabin and outside), a motion sensor, a light sensor, a
16x2 LCD and two fans. It listens for the setpoint Railflow recommends, drives
the fans from it, and reports telemetry back. The wire protocol is in
[`docs/serial_protocol.md`](../../docs/serial_protocol.md) and
[`docs/board_extension.md`](../../docs/board_extension.md).

## Wiring

All pins are in [`include/config.h`](include/config.h); change them there if
your model is wired differently. Defaults for a standard ESP32 DevKit:

| Part | Pin | Notes |
|---|---|---|
| DHT22, cabin | GPIO 4 | `RF_DHT_TYPE` in `config.h` if you use a DHT11 |
| DHT22, outside | GPIO 18 | |
| Motion sensor (PIR) | GPIO 27 | digital out, `PIR_ACTIVE_HIGH` |
| Light module (LM393) | GPIO 34 | digital out, input-only pin |
| LCD 16x2 with I2C backpack | SDA 21, SCL 22 | address `0x27`, or `0x3F` on some backpacks |
| Fan 1, Fan 2 | GPIO 25, 26 | **through a MOSFET or transistor each**, never straight from a pin |
| Status LED | GPIO 2 | the on-board LED: lit while the Railflow link is healthy |
| Button | GPIO 0 | the BOOT button, no wiring needed |

Power the fans from their own supply (common ground with the ESP32), not from
the 3.3 V pin. A flyback diode across each fan is advisable.

## Build and flash

Needs [PlatformIO](https://platformio.org/) (CLI or the VS Code extension).

```bash
pio run -d firmware/railflow-board -t upload
pio device monitor -d firmware/railflow-board
```

If the LCD stays blank, the backpack address is probably `0x3F`: change
`LCD_I2C_ADDR`. The serial link is the DevKit's USB port at 9600 baud, fixed by
the protocol.

## Using it

With no host attached the board is a standalone demo: the LCD shows the live
readings, the fans run AUTO against a default 24 °C setpoint, and the BOOT
button steps through AUTO and five manual fan speeds.

With the host attached (`pip install -r requirements.txt` in the repo root):

```bash
python -m src.board_link --port COM5 monitor              # print the board's telemetry
python -m src.board_link --port COM5 fan manual 60        # fans to 60 %
python -m src.board_link --port COM5 fan auto             # back to automatic
python -m src.board_link --port COM5 replay               # stream a simulated journey
```

`replay` runs one journey through the same simulator and advisor the Streamlit
demo uses and sends the advisor's setpoint to the board every simulated minute
(2 s of real time per minute by default; `--seconds-per-minute` changes that,
and `--advisor thermostat` replays the static baseline instead). Watch the LCD's
second page: the setpoint changes as the advisor reacts to the journey, and the
fans follow the board's own reading against it. From Python:

```python
from src.board_link import BoardLink, build_trajectory, replay
from src.config import load_config

with BoardLink.open("COM5") as link:
    link.set_fan_auto()
    replay(link, build_trajectory("anticipatory"), load_config(), seconds_per_minute=2.0)
```

## Tests

The protocol parser and the fan logic are plain C++ with no Arduino
dependency (`include/rfp1.h`, `include/fan_logic.h`), so they are tested on the
host: `tests/test_firmware_protocol.py` compiles `test/rfp1_harness.cpp` and
checks the C++ against the Python reference on random frames, malformed frames,
range edges, the sequence-wrap rule and the fan behaviour.

```bash
python -m pytest tests/test_firmware_protocol.py tests/test_board_link.py -q
```

The test needs a C++ compiler on PATH (or g++ inside WSL) and is skipped
without one. The sensor, LCD and PWM code in `src/main.cpp` can only be
exercised on the board itself; CI at least compiles it for the ESP32 on every
push (`.github/workflows/firmware.yml`).
