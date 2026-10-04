// Host-side harness around the firmware's real protocol and fan code.
//
// tests/test_firmware_protocol.py compiles this with the system C++ compiler,
// feeds it lines on stdin and compares the answers with the Python reference
// implementation. That is what keeps include/rfp1.h and src/serial_bridge.py /
// src/board_link.py from drifting apart: both sides are checked against each
// other, not against a copy of each other.
//
// One command per stdin line, one answer per stdout line:
//   R,...*CS / F,...*CS            parse a frame the way the board would
//   SEQ <last> <seq>               seq_is_newer(last, seq) -> 1 or 0
//   BUILD_T <seq> <t_in> <t_out> <motion> <dark> <mode> <duty> <setpoint> <link>
//   FAN <t_in> <setpoint> <sensor_ok>   stateful AutoFan.update -> duty
#include <stdio.h>
#include <string.h>

#include "fan_logic.h"
#include "rfp1.h"

int main() {
  rf::FanTuning tuning = {0.5f, 20.0f, 30, 50};  // same numbers as include/config.h
  rf::AutoFan fan(tuning);
  char line[256];

  while (fgets(line, sizeof line, stdin)) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
    if (n == 0) continue;

    // Anything that is not one of the three word commands is treated as a frame,
    // including malformed ones: those must reach the parser to be rejected.
    bool is_frame = strncmp(line, "SEQ ", 4) != 0 && strncmp(line, "FAN ", 4) != 0 &&
                    strncmp(line, "BUILD_T ", 8) != 0;
    if (is_frame && line[0] == 'R') {
      rf::SetpointFrame f;
      rf::ParseStatus st = rf::parse_setpoint_frame(line, &f);
      if (st == rf::PARSE_OK)
        printf("OK %u %.1f %.1f %d %.1f %.2f\n", f.seq, f.t_in_c, f.t_out_c, f.n_pass,
               f.setpoint_c, f.q_cmd);
      else
        printf("%s\n", rf::status_name(st));
    } else if (is_frame && line[0] == 'F') {
      rf::FanFrame f;
      rf::ParseStatus st = rf::parse_fan_frame(line, &f);
      if (st == rf::PARSE_OK) printf("OK %u %u %u\n", f.seq, f.mode, f.duty_pct);
      else printf("%s\n", rf::status_name(st));
    } else if (strncmp(line, "SEQ ", 4) == 0) {
      unsigned last, seq;
      if (sscanf(line + 4, "%u %u", &last, &seq) == 2)
        printf("%d\n", rf::seq_is_newer(static_cast<uint8_t>(last), static_cast<uint8_t>(seq)) ? 1 : 0);
    } else if (strncmp(line, "BUILD_T ", 8) == 0) {
      unsigned seq, motion, dark, mode, duty, link;
      float t_in, t_out, setpoint;
      char tin_s[16], tout_s[16];
      if (sscanf(line + 8, "%u %15s %15s %u %u %u %u %f %u", &seq, tin_s, tout_s, &motion, &dark,
                 &mode, &duty, &setpoint, &link) == 9) {
        t_in = strcmp(tin_s, "nan") == 0 ? NAN : static_cast<float>(atof(tin_s));
        t_out = strcmp(tout_s, "nan") == 0 ? NAN : static_cast<float>(atof(tout_s));
        char out[rf::MAX_LINE];
        size_t w = rf::build_telemetry(out, sizeof out, static_cast<uint8_t>(seq), t_in, t_out,
                                       motion != 0, dark != 0, static_cast<uint8_t>(mode),
                                       static_cast<uint8_t>(duty), setpoint, link != 0);
        if (w > 0) fputs(out, stdout);  // already '\n'-terminated
        else puts("BUILD_FAILED");
      }
    } else if (strncmp(line, "FAN ", 4) == 0) {
      float t_in, setpoint;
      unsigned ok;
      char tin_s[16];
      if (sscanf(line + 4, "%15s %f %u", tin_s, &setpoint, &ok) == 3) {
        t_in = strcmp(tin_s, "nan") == 0 ? NAN : static_cast<float>(atof(tin_s));
        printf("%u\n", fan.update(t_in, setpoint, ok != 0));
      }
    }
    fflush(stdout);
  }
  return 0;
}
