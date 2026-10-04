// Everything board-specific lives here: pins, sensor types, timing, fan tuning.
// If the wiring on your demo model differs from this, change it here only.
//
// Pin numbers are for a standard ESP32 DevKit (30/38-pin, "esp32dev").
#pragma once

#include "fan_logic.h"

// ---- Serial link to the Railflow host -------------------------------------
// 9600 8N1 is fixed by docs/serial_protocol.md §2. On a DevKit this is the USB
// port, so the host just opens the board's COM port.
#define RF_SERIAL Serial
constexpr unsigned long RF_BAUD = 9600;

// ---- Sensors ---------------------------------------------------------------
constexpr int PIN_DHT_INDOOR = 4;    // DHT22 / AM2302 inside the cabin
constexpr int PIN_DHT_OUTDOOR = 18;  // DHT22 / AM2302 on the outside (the nose in the photos)
#define RF_DHT_TYPE DHT22            // use DHT11 here if that is what you wired

constexpr int PIN_PIR = 27;          // motion sensor, digital output
constexpr bool PIR_ACTIVE_HIGH = true;
constexpr unsigned long MOTION_HOLD_MS = 10000;  // PIR pulses are short; keep "YES" readable

constexpr int PIN_LIGHT_DO = 34;     // LM393 light-sensor module, digital output (input-only pin)
constexpr bool LIGHT_DO_HIGH_MEANS_DARK = true;  // most LM393 modules: DO goes HIGH in the dark

constexpr unsigned long SENSOR_PERIOD_MS = 2500;  // DHT22 cannot be read faster than every 2 s
constexpr int SENSOR_FAIL_LIMIT = 3;  // consecutive failed reads before a value is declared invalid

// ---- 16x2 I2C LCD ----------------------------------------------------------
constexpr int PIN_I2C_SDA = 21;
constexpr int PIN_I2C_SCL = 22;
constexpr uint8_t LCD_I2C_ADDR = 0x27;  // 0x3F on some backpacks; run an I2C scanner if blank
constexpr unsigned long LCD_PERIOD_MS = 500;
constexpr unsigned long LCD_PAGE_MS = 5000;  // page 1: temperatures + motion, page 2: fan + setpoint

// ---- Fans (through a MOSFET/transistor per fan, never straight from a GPIO) --
constexpr int PIN_FAN1 = 25;
constexpr int PIN_FAN2 = 26;
constexpr int FAN1_LEDC_CHANNEL = 0;
constexpr int FAN2_LEDC_CHANNEL = 1;
constexpr int FAN_PWM_HZ = 25000;  // above hearing range; fine for MOSFET-switched 5/12 V fans
constexpr int FAN_PWM_BITS = 8;

constexpr rf::FanTuning FAN_TUNING = {
    /*deadband_k=*/0.5f,
    /*gain_pct_per_k=*/20.0f,
    /*min_duty_pct=*/30,
    /*failsafe_duty_pct=*/50,
};

// ---- Status / local controls -------------------------------------------------
constexpr int PIN_STATUS_LED = 2;  // on-board LED on most DevKits: lit while the link is healthy
constexpr int PIN_BUTTON = 0;      // the BOOT button; active-low. Cycles AUTO, 0/25/50/75/100 %

// ---- Link behaviour ------------------------------------------------------------
// docs/serial_protocol.md §7.6: fall back after 3 nominal 30 s intervals with no
// valid frame, to the board's own default rather than the last stale setpoint.
constexpr unsigned long LINK_TIMEOUT_MS = 90000;
constexpr float DEFAULT_SETPOINT_C = 24.0f;  // what the board uses with no Railflow connected
constexpr unsigned long TELEMETRY_PERIOD_MS = 5000;
