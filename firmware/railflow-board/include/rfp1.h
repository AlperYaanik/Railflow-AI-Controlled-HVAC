// Railflow wire protocol, board side: parsing and building frames.
//
// Pure C++11, no Arduino dependency, so the exact same code runs on the ESP32
// and in the host-side test harness (test/rfp1_harness.cpp) that checks it
// against the Python reference (src/serial_bridge.py, src/board_link.py).
//
// Frame formats are specified in docs/serial_protocol.md (R frames, from the
// Railflow host) and docs/board_extension.md (F frames from the host, T frames
// from the board). All three are one ASCII line, '\n'-terminated, with the
// NMEA-style XOR checksum after '*'.
#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace rf {

constexpr size_t MAX_LINE = 96;  // longest valid frame is ~40 bytes; headroom for noise

enum ParseStatus {
  PARSE_OK = 0,
  ERR_MARKER,       // first char is not the expected marker, or no comma after it
  ERR_NO_CHECKSUM,  // no '*' followed by exactly two hex digits at the end
  ERR_CHECKSUM,     // checksum present but does not match the payload
  ERR_FORMAT,       // wrong field count, or a field is not a number
  ERR_RANGE,        // a field parses but is outside its documented range
};

inline const char* status_name(ParseStatus s) {
  switch (s) {
    case PARSE_OK: return "OK";
    case ERR_MARKER: return "ERR_MARKER";
    case ERR_NO_CHECKSUM: return "ERR_NO_CHECKSUM";
    case ERR_CHECKSUM: return "ERR_CHECKSUM";
    case ERR_FORMAT: return "ERR_FORMAT";
    case ERR_RANGE: return "ERR_RANGE";
  }
  return "?";
}

// XOR of every byte, same convention as NMEA-0183 and src/serial_bridge.py.
inline uint8_t xor_checksum(const char* s, size_t n) {
  uint8_t x = 0;
  for (size_t i = 0; i < n; i++) x ^= static_cast<uint8_t>(s[i]);
  return x;
}

// Frame sequence numbers are uint8 and wrap. "Newer" means the forward distance
// from the last accepted frame is 1..127; 0 is a duplicate and 128..255 reads as
// "went backwards". This is how docs/serial_protocol.md §7.4 ("repeats or goes
// backward -> discard") is made precise across the wrap at 255.
inline bool seq_is_newer(uint8_t last, uint8_t seq) {
  uint8_t d = static_cast<uint8_t>(seq - last);
  return d >= 1 && d < 128;
}

struct SetpointFrame {  // "R" frame, docs/serial_protocol.md §4
  uint8_t seq;
  float t_in_c, t_out_c;
  int n_pass;
  float setpoint_c, q_cmd;
};

struct FanFrame {  // "F" frame, docs/board_extension.md §2
  uint8_t seq;
  uint8_t mode;      // FAN_AUTO or FAN_MANUAL
  uint8_t duty_pct;  // 0..100, meaningful only in manual mode
};

constexpr uint8_t FAN_AUTO = 0;
constexpr uint8_t FAN_MANUAL = 1;

namespace detail {

inline int hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;  // uppercase only, as specified
  return -1;
}

// Splits a validated line into its payload (between the first comma and '*').
// `line` is NUL-terminated with no trailing '\n' or '\r'.
inline ParseStatus split(const char* line, char marker, char* payload, size_t cap) {
  if (line[0] != marker || line[1] != ',') return ERR_MARKER;
  const char* star = strrchr(line, '*');
  if (star == nullptr || strlen(star) != 3) return ERR_NO_CHECKSUM;
  int hi = hex_nibble(star[1]), lo = hex_nibble(star[2]);
  if (hi < 0 || lo < 0) return ERR_NO_CHECKSUM;

  const char* start = line + 2;
  size_t n = static_cast<size_t>(star - start);
  if (n == 0 || n >= cap) return ERR_FORMAT;
  if (xor_checksum(start, n) != static_cast<uint8_t>((hi << 4) | lo)) return ERR_CHECKSUM;

  memcpy(payload, start, n);
  payload[n] = '\0';
  return PARSE_OK;
}

// Splits `payload` in place on ',' into at most `max` fields. Returns the count.
inline int tokenize(char* payload, char** fields, int max) {
  int n = 0;
  char* p = payload;
  while (n < max) {
    fields[n++] = p;
    char* comma = strchr(p, ',');
    if (comma == nullptr) return n;
    *comma = '\0';
    p = comma + 1;
  }
  return n + 1;  // more fields than `max`: caller sees count > max and rejects
}

inline bool to_float(const char* s, float* out) {
  if (*s == '\0') return false;
  char* end = nullptr;
  double v = strtod(s, &end);
  if (*end != '\0' || !isfinite(v)) return false;
  *out = static_cast<float>(v);
  return true;
}

inline bool to_int(const char* s, long* out) {
  if (*s == '\0') return false;
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  if (*end != '\0') return false;
  *out = v;
  return true;
}

inline bool in_range(float v, float lo, float hi) { return v >= lo && v <= hi; }

}  // namespace detail

// "R,<seq>,<t_in_c>,<t_out_c>,<n_pass>,<setpoint_c>,<q_cmd>*<CS>"
inline ParseStatus parse_setpoint_frame(const char* line, SetpointFrame* out) {
  char payload[MAX_LINE];
  ParseStatus st = detail::split(line, 'R', payload, sizeof payload);
  if (st != PARSE_OK) return st;

  char* f[8];
  if (detail::tokenize(payload, f, 8) != 6) return ERR_FORMAT;

  long seq, n_pass;
  float t_in, t_out, setpoint, q;
  if (!detail::to_int(f[0], &seq) || !detail::to_float(f[1], &t_in) ||
      !detail::to_float(f[2], &t_out) || !detail::to_int(f[3], &n_pass) ||
      !detail::to_float(f[4], &setpoint) || !detail::to_float(f[5], &q))
    return ERR_FORMAT;

  // Ranges from docs/serial_protocol.md §4. Out of range means a bug upstream of
  // the wire; the board rejects the frame rather than clamping and acting on it.
  if (seq < 0 || seq > 255 || n_pass < 0 || n_pass > 100 ||
      !detail::in_range(t_in, -10.0f, 60.0f) || !detail::in_range(t_out, -10.0f, 55.0f) ||
      !detail::in_range(setpoint, 22.0f, 26.0f) || !detail::in_range(q, -1.0f, 1.0f))
    return ERR_RANGE;

  out->seq = static_cast<uint8_t>(seq);
  out->t_in_c = t_in;
  out->t_out_c = t_out;
  out->n_pass = static_cast<int>(n_pass);
  out->setpoint_c = setpoint;
  out->q_cmd = q;
  return PARSE_OK;
}

// "F,<seq>,<mode>,<duty_pct>*<CS>"
inline ParseStatus parse_fan_frame(const char* line, FanFrame* out) {
  char payload[MAX_LINE];
  ParseStatus st = detail::split(line, 'F', payload, sizeof payload);
  if (st != PARSE_OK) return st;

  char* f[4];
  if (detail::tokenize(payload, f, 4) != 3) return ERR_FORMAT;

  long seq, mode, duty;
  if (!detail::to_int(f[0], &seq) || !detail::to_int(f[1], &mode) || !detail::to_int(f[2], &duty))
    return ERR_FORMAT;
  if (seq < 0 || seq > 255 || (mode != FAN_AUTO && mode != FAN_MANUAL) || duty < 0 || duty > 100)
    return ERR_RANGE;

  out->seq = static_cast<uint8_t>(seq);
  out->mode = static_cast<uint8_t>(mode);
  out->duty_pct = static_cast<uint8_t>(duty);
  return PARSE_OK;
}

// Telemetry the board reports back, docs/board_extension.md §3:
// "T,<seq>,<t_in_c>,<t_out_c>,<motion>,<dark>,<fan_mode>,<fan_duty>,<setpoint_c>,<link_ok>*<CS>\n"
// An invalid sensor reading is written as "nan". Returns the length written
// (excluding the NUL), or 0 if `cap` was too small.
inline size_t build_telemetry(char* out, size_t cap, uint8_t seq, float t_in_c, float t_out_c,
                              bool motion, bool dark, uint8_t fan_mode, uint8_t fan_duty,
                              float setpoint_c, bool link_ok) {
  char payload[MAX_LINE];
  int n = snprintf(payload, sizeof payload, "%u,%.1f,%.1f,%d,%d,%u,%u,%.1f,%d",
                   static_cast<unsigned>(seq), t_in_c, t_out_c, motion ? 1 : 0, dark ? 1 : 0,
                   static_cast<unsigned>(fan_mode), static_cast<unsigned>(fan_duty), setpoint_c,
                   link_ok ? 1 : 0);
  if (n <= 0 || static_cast<size_t>(n) >= sizeof payload) return 0;
  int w = snprintf(out, cap, "T,%s*%02X\n", payload,
                   static_cast<unsigned>(xor_checksum(payload, static_cast<size_t>(n))));
  return (w > 0 && static_cast<size_t>(w) < cap) ? static_cast<size_t>(w) : 0;
}

}  // namespace rf
