#pragma once

#include <math.h>

// Cable-Driven Parallel Robot (CDPR) configuration for Teensy stepper control.
//
// Motor layout (top view, looking down at table):
//
//   Motor 0 (top-left)  -------- Motor 1 (top-right)
//        |                            |
//        |         [paddle]           |
//        |                            |
//   Motor 3 (bot-left)  -------- Motor 2 (bot-right)
//
// Coordinate system: origin at bottom-left (motor 3), x-right, y-up.
// All units in millimeters unless otherwise noted.

// ---- Table geometry ----

// Distance between left and right motor columns (mm).
constexpr float TABLE_WIDTH = 606.0f;

// Distance between top and bottom motor rows (mm).
constexpr float TABLE_HEIGHT = 730.0f;

// Motor positions [x, y] in table frame (mm).
// Index: 0=top-left, 1=top-right, 2=bot-right, 3=bot-left
constexpr float MOTOR_X[4] = {0.0f, TABLE_WIDTH, TABLE_WIDTH, 0.0f};
constexpr float MOTOR_Y[4] = {TABLE_HEIGHT, TABLE_HEIGHT, 0.0f, 0.0f};

// ---- Spool parameters ----

constexpr float SPOOL_DIAMETER_MM = 41.0f;
constexpr float SPOOL_CIRCUMFERENCE_MM = SPOOL_DIAMETER_MM * (float)M_PI;  // ~128.8 mm

// ---- Stepper parameters ----

constexpr int COUNTS_PER_REV = 800;  // All 4 motors identical (stepper, not ClearPath)

// Conversion: mm of cable to stepper counts
inline float mmToCounts(float mm) {
  return mm / SPOOL_CIRCUMFERENCE_MM * COUNTS_PER_REV;
}

// Conversion: stepper counts to mm of cable
inline float countsToMm(float counts) {
  return counts / COUNTS_PER_REV * SPOOL_CIRCUMFERENCE_MM;
}

// ---- Motion limits ----

constexpr float MAX_VELOCITY_MM_S = 2000.0f;
constexpr float MAX_ACCEL_MM_S2 = 50000.0f;
constexpr float EDGE_MARGIN_MM = 30.0f;

// Workspace bounds (mm)
constexpr float WS_MIN_X = EDGE_MARGIN_MM;
constexpr float WS_MAX_X = TABLE_WIDTH - EDGE_MARGIN_MM;
constexpr float WS_MIN_Y = EDGE_MARGIN_MM;
constexpr float WS_MAX_Y = TABLE_HEIGHT - EDGE_MARGIN_MM;

// ---- Pin assignments ----

constexpr int STEP_PINS[4] = {6, 7, 8, 9};
constexpr int DIR_PINS[4]  = {34, 35, 36, 37};

// ---- Control loop ----

// Default tick rate for CDPR instances (Hz). Can be overridden per-instance.
constexpr uint32_t DEFAULT_TICK_RATE_HZ = 50000;

// Step pulse width (us). Must be >1us for most stepper drivers.
constexpr uint32_t PULSE_WIDTH_US = 2;

// ---- Kinematics ----

// Compute cable length from motor i to cart position (x, y).
inline float cableLength(int motor, float x, float y) {
  float dx = x - MOTOR_X[motor];
  float dy = y - MOTOR_Y[motor];
  return sqrtf(dx * dx + dy * dy);
}
