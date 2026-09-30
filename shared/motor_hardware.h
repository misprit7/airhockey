#pragma once

#include <cmath>
#include <cstring>

// Installed 2026-09-29: nodes 0/2 replaced with the same part as 1/3.
// Encoder counts and step/dir input pulses are independent drive settings.
namespace MotorHardware {
constexpr char REVISION[] = "four-2331S-RLNA-20260929";
constexpr char MODEL[] = "CPM-SCSK-2331S-RLNA";
constexpr unsigned ENCODER_COUNTS_PER_REV = 800;
constexpr unsigned STEP_INPUT_COUNTS_PER_REV = 800;

inline bool matches(const char *model, unsigned encoder, double step_input) {
  // Drives append identification suffixes, e.g. RLNA-1-7-D or RLNA-1-8-D.
  // Match the complete base part number at a delimiter, not an arbitrary
  // prefix that could also accept a different part such as RLNAX.
  constexpr unsigned base_length = sizeof(MODEL) - 1;
  const bool model_matches = model &&
      std::strncmp(model, MODEL, base_length) == 0 &&
      (model[base_length] == '\0' ||
       (model[base_length] == '-' && model[base_length + 1] != '\0'));
  return model_matches &&
         encoder == ENCODER_COUNTS_PER_REV && std::isfinite(step_input) &&
         std::fabs(step_input - STEP_INPUT_COUNTS_PER_REV) < 0.01;
}
} // namespace MotorHardware
