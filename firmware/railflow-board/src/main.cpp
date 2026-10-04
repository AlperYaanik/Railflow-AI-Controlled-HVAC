// Railflow demo board firmware (ESP32, Arduino framework).
//
// Reads two DHT22 sensors (cabin and outside), a PIR motion sensor and a light
// module, shows them on a 16x2 LCD, drives two fans, and talks to the Railflow
// host over the USB serial port:
//
//   host -> board   R frames  cabin setpoint recommendation   (docs/serial_protocol.md)
//   host -> board   F frames  fan mode / manual duty          (docs/board_extension.md)
//   board -> host   T frames  telemetry every 5 s             (docs/board_extension.md)
//
// Fan modes: AUTO runs a small proportional controller on the setpoint Railflow
// sent (or the board's own default when the link is down); MANUAL holds a fixed
// duty. The BOOT button steps through AUTO and five manual presets so the demo
// works with no host attached.

#include <Arduino.h>
#include <DHT.h>
#include <LiquidCrystal_I2C.h>
#include <Wire.h>
#include <math.h>

#include "config.h"
#include "fan_logic.h"
#include "rfp1.h"

#define DEG "\xDF"  // degree sign in the HD44780 character ROM

static DHT dhtIn(PIN_DHT_INDOOR, RF_DHT_TYPE);
static DHT dhtOut(PIN_DHT_OUTDOOR, RF_DHT_TYPE);
static LiquidCrystal_I2C lcd(LCD_I2C_ADDR, 16, 2);
static rf::AutoFan autoFan(FAN_TUNING);

// ---- Sensor state ----------------------------------------------------------
struct Reading {
  float value = NAN;
  int failures = 0;
  bool valid() const { return !isnan(value); }
};
static Reading tIn, tOut;
static bool motionNow = false;
static bool dark = false;
static unsigned long lastMotionMs = 0;
static bool everMoved = false;

// ---- Link / control state --------------------------------------------------
static bool haveRx = false;
static unsigned long lastRxMs = 0;
static float rxSetpointC = DEFAULT_SETPOINT_C;
static uint8_t lastRSeq = 0;
static bool haveRSeq = false;
static uint8_t lastFSeq = 0;
static bool haveFSeq = false;

static uint8_t fanMode = rf::FAN_AUTO;
static uint8_t manualDuty = 0;
static uint8_t fanDuty = 0;

// ---- Serial line reader ----------------------------------------------------
static char lineBuf[rf::MAX_LINE];
static size_t lineLen = 0;
static bool lineOverflow = false;

static bool linkOk(unsigned long now) { return haveRx && (now - lastRxMs) < LINK_TIMEOUT_MS; }
static float activeSetpoint(unsigned long now) { return linkOk(now) ? rxSetpointC : DEFAULT_SETPOINT_C; }

static void handleLine(const char* line, unsigned long now) {
  if (line[0] == 'R') {
    rf::SetpointFrame f;
    // Bad checksum/range/format: act on nothing in the frame.
    if (rf::parse_setpoint_frame(line, &f) != rf::PARSE_OK) return;
    // Duplicate or stale: discard. A gap (dropped frame) is fine; the next frame supersedes it.
    if (haveRSeq && !rf::seq_is_newer(lastRSeq, f.seq)) return;
    haveRSeq = true;
    lastRSeq = f.seq;
    rxSetpointC = f.setpoint_c;
    lastRxMs = now;
    haveRx = true;
  } else if (line[0] == 'F') {
    rf::FanFrame f;
    if (rf::parse_fan_frame(line, &f) != rf::PARSE_OK) return;
    if (haveFSeq && !rf::seq_is_newer(lastFSeq, f.seq)) return;
    haveFSeq = true;
    lastFSeq = f.seq;
    fanMode = f.mode;
    manualDuty = f.duty_pct;
  }
  // Anything else (boot noise, partial lines, a future frame type) is ignored.
}

static void pollSerial(unsigned long now) {
  while (RF_SERIAL.available() > 0) {
    char c = static_cast<char>(RF_SERIAL.read());
    if (c == '\n') {
      if (!lineOverflow && lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf, now);
      }
      lineLen = 0;
      lineOverflow = false;  // resync on the next line
    } else if (c != '\r') {
      if (lineLen < sizeof lineBuf - 1) lineBuf[lineLen++] = c;
      else lineOverflow = true;
    }
  }
}

// ---- Sensors ---------------------------------------------------------------
static void updateReading(Reading& r, float v) {
  if (isnan(v)) {
    if (++r.failures >= SENSOR_FAIL_LIMIT) r.value = NAN;  // keep the last good value until then
  } else {
    r.value = v;
    r.failures = 0;
  }
}

static void readSensors() {
  updateReading(tIn, dhtIn.readTemperature());
  updateReading(tOut, dhtOut.readTemperature());
  dark = (digitalRead(PIN_LIGHT_DO) == HIGH) == LIGHT_DO_HIGH_MEANS_DARK;
}

static void readMotion(unsigned long now) {
  bool raw = (digitalRead(PIN_PIR) == HIGH) == PIR_ACTIVE_HIGH;
  if (raw) {
    lastMotionMs = now;
    everMoved = true;
  }
  motionNow = everMoved && (now - lastMotionMs) < MOTION_HOLD_MS;
}

// ---- Fans ------------------------------------------------------------------
static void writeFans(uint8_t duty_pct) {
  const uint32_t max = (1u << FAN_PWM_BITS) - 1;
  uint32_t v = (static_cast<uint32_t>(duty_pct) * max + 50) / 100;
  ledcWrite(FAN1_LEDC_CHANNEL, v);
  ledcWrite(FAN2_LEDC_CHANNEL, v);
}

static void updateFans(unsigned long now) {
  if (fanMode == rf::FAN_MANUAL) {
    fanDuty = manualDuty;
  } else {
    fanDuty = autoFan.update(tIn.value, activeSetpoint(now), tIn.valid());
  }
  writeFans(fanDuty);
}

// ---- Local button: AUTO, then manual 0/25/50/75/100 % ----------------------
static void pollButton(unsigned long now) {
  static bool lastLevel = HIGH;
  static unsigned long lastChangeMs = 0;
  static const uint8_t presets[] = {0, 25, 50, 75, 100};
  bool level = digitalRead(PIN_BUTTON);
  if (level != lastLevel && now - lastChangeMs > 40) {  // 40 ms debounce
    lastChangeMs = now;
    lastLevel = level;
    if (level == LOW) {
      if (fanMode == rf::FAN_AUTO) {
        fanMode = rf::FAN_MANUAL;
        manualDuty = presets[0];
      } else {
        size_t i = 0;
        while (i < sizeof presets && presets[i] != manualDuty) i++;
        if (i + 1 >= sizeof presets) fanMode = rf::FAN_AUTO;
        else manualDuty = presets[i + 1];
      }
    }
  }
}

// ---- LCD -------------------------------------------------------------------
static void printRow(uint8_t row, const char* text) {
  char buf[17];
  snprintf(buf, sizeof buf, "%-16.16s", text);
  lcd.setCursor(0, row);
  lcd.print(buf);
}

static void updateLcd(unsigned long now) {
  char row0[24], row1[24];
  bool pageTwo = (now / LCD_PAGE_MS) % 2 == 1;
  if (!pageTwo) {
    // Same layout as the demo photos: "In:31°C Out:31°C" / "Motion: YES"
    char in[8], out[8];
    if (tIn.valid()) snprintf(in, sizeof in, "%2.0f", tIn.value); else snprintf(in, sizeof in, "--");
    if (tOut.valid()) snprintf(out, sizeof out, "%2.0f", tOut.value); else snprintf(out, sizeof out, "--");
    snprintf(row0, sizeof row0, "In:%s" DEG "C Out:%s" DEG "C", in, out);
    snprintf(row1, sizeof row1, "Motion: %s", motionNow ? "YES" : "NO");
  } else {
    snprintf(row0, sizeof row0, "Fan:%s %3u%%", fanMode == rf::FAN_AUTO ? "AUTO" : "MAN ",
             static_cast<unsigned>(fanDuty));
    snprintf(row1, sizeof row1, "Set:%4.1f Link:%s", activeSetpoint(now), linkOk(now) ? "OK" : "--");
  }
  printRow(0, row0);
  printRow(1, row1);
}

// ---- Telemetry -------------------------------------------------------------
static void sendTelemetry(unsigned long now) {
  static uint8_t seq = 0;
  char out[rf::MAX_LINE];
  size_t n = rf::build_telemetry(out, sizeof out, seq++, tIn.value, tOut.value, motionNow, dark,
                                 fanMode, fanDuty, activeSetpoint(now), linkOk(now));
  if (n > 0) RF_SERIAL.write(reinterpret_cast<const uint8_t*>(out), n);
}

// ---- Arduino entry points --------------------------------------------------
void setup() {
  RF_SERIAL.begin(RF_BAUD);
  pinMode(PIN_PIR, INPUT);
  pinMode(PIN_LIGHT_DO, INPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_STATUS_LED, OUTPUT);

  ledcSetup(FAN1_LEDC_CHANNEL, FAN_PWM_HZ, FAN_PWM_BITS);
  ledcSetup(FAN2_LEDC_CHANNEL, FAN_PWM_HZ, FAN_PWM_BITS);
  ledcAttachPin(PIN_FAN1, FAN1_LEDC_CHANNEL);
  ledcAttachPin(PIN_FAN2, FAN2_LEDC_CHANNEL);
  writeFans(0);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  lcd.init();
  lcd.backlight();
  printRow(0, "Railflow demo");
  printRow(1, "starting...");

  dhtIn.begin();
  dhtOut.begin();
}

void loop() {
  static unsigned long lastSensorMs = 0, lastLcdMs = 0, lastTelemetryMs = 0;
  unsigned long now = millis();

  pollSerial(now);
  pollButton(now);
  readMotion(now);

  if (now - lastSensorMs >= SENSOR_PERIOD_MS || lastSensorMs == 0) {
    lastSensorMs = now;
    readSensors();
  }

  updateFans(now);
  digitalWrite(PIN_STATUS_LED, linkOk(now) ? HIGH : LOW);

  if (now - lastLcdMs >= LCD_PERIOD_MS) {
    lastLcdMs = now;
    updateLcd(now);
  }
  if (now - lastTelemetryMs >= TELEMETRY_PERIOD_MS) {
    lastTelemetryMs = now;
    sendTelemetry(now);
  }
}
