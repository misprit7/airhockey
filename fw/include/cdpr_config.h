#pragma once

#include "cdpr_geometry.h"

// Teensy stepper-control configuration.
//
// Table geometry, motor anchors, spool dimensions, paddle attachment and
// the cable kinematics all live in shared/cdpr_geometry.h, which this
// includes and sw/lib/cdpr_config.h includes too. Only things specific to
// driving the steppers from the Teensy belong below — if you find yourself
// adding a physical dimension here, it belongs in the shared header.

// ── Motor rotation direction ────────────────────────────────────────
//
// RETRACT_CW (which way each spool must turn to shorten its cable) is a
// physical fact and lives in the shared header. What belongs here is the
// electrical half: which DIR pin level produces clockwise rotation.
//
// Verify on hardware: if EVERY motor moves opposite to expected, flip
// DIR_CW_LEVEL; if only some do, fix RETRACT_CW in the shared header.

constexpr int DIR_CW_LEVEL = 1; // 1 = HIGH, 0 = LOW (TODO: verify on hardware)

// DIR pin level that retracts / extends the cable for a given motor.
inline int dirLevelRetract(int motor) {
  return RETRACT_CW[motor] ? DIR_CW_LEVEL : 1 - DIR_CW_LEVEL;
}
inline int dirLevelExtend(int motor) { return 1 - dirLevelRetract(motor); }

// ── Stepper ─────────────────────────────────────────────────────────
//
// Uniform across all four motors on this path (the host-side ClearPath
// path in sw/ has mixed 800/6400 encoder resolutions instead — that
// difference is precisely why the two configs stay separate).

constexpr int COUNTS_PER_REV = 800;

// mm of cable ↔ stepper counts
constexpr float MM_PER_COUNT = SPOOL_CIRCUMFERENCE_MM / COUNTS_PER_REV;
constexpr float COUNTS_PER_MM = (float)COUNTS_PER_REV / SPOOL_CIRCUMFERENCE_MM;

inline float mmToCounts(float mm) { return mm * COUNTS_PER_MM; }
inline float countsToMm(float counts) { return counts * MM_PER_COUNT; }

// ── Motion limits ───────────────────────────────────────────────────
//
// Actuator-specific: these are what the steppers can do, not what the
// table is. The workspace bounds are a property of the anchor geometry, so
// they live in the shared header.

constexpr float MAX_VELOCITY_MM_S = 7200.0f;

// Acceleration: a DEFAULT that motion starts at, and a CEILING the runtime
// setter will not exceed. They were one constant, which meant changing how
// hard the rig accelerates required a reflash — and reflashing to try a
// number is how you end up not trying it.
constexpr float DEFAULT_ACCEL_MM_S2 = 400.0f;
constexpr float MAX_ACCEL_MM_S2 = 5000.0f;

// ── Control loop ────────────────────────────────────────────────────

constexpr uint32_t DEFAULT_TICK_RATE_HZ = 50000;
