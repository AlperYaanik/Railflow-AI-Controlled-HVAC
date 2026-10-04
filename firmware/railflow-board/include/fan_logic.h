// Fan duty selection. Pure C++11 (no Arduino calls) so it is unit-tested on the
// host together with the protocol code.
//
// The fans stand in for the HVAC unit in this demo. The board does what a
// thermostat does with the setpoint it is given: it never receives a power
// command from Railflow (docs/serial_protocol.md §4.1), it works out its own
// duty from the setpoint and its own sensors.
#pragma once

#include <math.h>
#include <stdint.h>

namespace rf {

struct FanTuning {
  float deadband_k;          // fan stays off until T_in exceeds the setpoint by this much
  float gain_pct_per_k;      // extra duty per kelvin above the deadband
  uint8_t min_duty_pct;      // duty at the moment the fan turns on (below this a small fan stalls)
  uint8_t failsafe_duty_pct; // duty when the indoor sensor has failed
};

// Cooling-only: a cabin colder than the setpoint just gets no fan, there is no
// heating path on this board. Turn-on and turn-off thresholds differ (the off
// threshold is half the deadband) so sensor noise near the boundary does not
// make the fan chatter.
class AutoFan {
 public:
  explicit AutoFan(const FanTuning& tuning) : tuning_(tuning) {}

  uint8_t update(float t_in_c, float setpoint_c, bool sensor_ok) {
    if (!sensor_ok || isnan(t_in_c)) {
      on_ = false;  // so recovery restarts through the normal turn-on threshold
      return tuning_.failsafe_duty_pct;
    }
    float error = t_in_c - setpoint_c;
    float threshold = on_ ? tuning_.deadband_k * 0.5f : tuning_.deadband_k;
    if (error <= threshold) {
      on_ = false;
      return 0;
    }
    on_ = true;
    float duty = tuning_.min_duty_pct + (error - tuning_.deadband_k) * tuning_.gain_pct_per_k;
    if (duty < tuning_.min_duty_pct) duty = tuning_.min_duty_pct;
    if (duty > 100.0f) duty = 100.0f;
    return static_cast<uint8_t>(duty + 0.5f);
  }

 private:
  FanTuning tuning_;
  bool on_ = false;
};

}  // namespace rf
