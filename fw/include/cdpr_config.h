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

constexpr int NUM_MOTORS = 4;

// ── Table geometry ──────────────────────────────────────────────────

constexpr float TABLE_WIDTH = 606.0f;  // mm, left–right motor distance
constexpr float TABLE_HEIGHT = 737.0f; // mm, top–bottom motor distance

// Motor positions in table frame (mm).
// Index: 0=top-left, 1=top-right, 2=bot-right, 3=bot-left
constexpr float MOTOR_X[NUM_MOTORS] = {0.0f, TABLE_WIDTH, TABLE_WIDTH, 0.0f};
constexpr float MOTOR_Y[NUM_MOTORS] = {TABLE_HEIGHT, TABLE_HEIGHT, 0.0f, 0.0f};

// ── Paddle cable attachment points ────────────────────────────────────
//
// Each cable attaches to a different point on the paddle, not the center.
// Offsets are relative to paddle center, in mm. The attachment points form
// a square with side length 21.21mm (2.121cm), axis-aligned.
//
//   attach 0 (TL)  ---- attach 1 (TR)       Motor 0 → attach 0
//        |                    |              Motor 1 → attach 1
//        |     [center]       |              Motor 2 → attach 2
//        |                    |              Motor 3 → attach 3
//   attach 3 (BL)  ---- attach 2 (BR)

constexpr float PADDLE_ATTACH_HALF =
    21.21f / 2.0f; // half side length = 10.605mm

// Offset from paddle center to cable attachment point for each motor (mm).
constexpr float ATTACH_DX[NUM_MOTORS] = {-PADDLE_ATTACH_HALF,
                                         PADDLE_ATTACH_HALF, PADDLE_ATTACH_HALF,
                                         -PADDLE_ATTACH_HALF};
constexpr float ATTACH_DY[NUM_MOTORS] = {PADDLE_ATTACH_HALF, PADDLE_ATTACH_HALF,
                                         -PADDLE_ATTACH_HALF,
                                         -PADDLE_ATTACH_HALF};

// ── Spool ───────────────────────────────────────────────────────────

constexpr float SPOOL_DIAMETER_MM = 41.0f;
constexpr float SPOOL_CIRCUMFERENCE_MM = SPOOL_DIAMETER_MM * (float)M_PI;

// ── Stepper ─────────────────────────────────────────────────────────

constexpr int COUNTS_PER_REV = 800;

// mm of cable ↔ stepper counts
constexpr float MM_PER_COUNT = SPOOL_CIRCUMFERENCE_MM / COUNTS_PER_REV;
constexpr float COUNTS_PER_MM = (float)COUNTS_PER_REV / SPOOL_CIRCUMFERENCE_MM;

inline float mmToCounts(float mm) { return mm * COUNTS_PER_MM; }
inline float countsToMm(float counts) { return counts * MM_PER_COUNT; }

// ── Motion limits ───────────────────────────────────────────────────

constexpr float MAX_VELOCITY_MM_S = 7200.0f;
constexpr float MAX_ACCEL_MM_S2 = 172800.0f;
constexpr float EDGE_MARGIN_MM = 30.0f;

// ── Workspace bounds (mm) ───────────────────────────────────────────

constexpr float WS_MIN_X = EDGE_MARGIN_MM;
constexpr float WS_MAX_X = TABLE_WIDTH - EDGE_MARGIN_MM;
constexpr float WS_MIN_Y = EDGE_MARGIN_MM;
constexpr float WS_MAX_Y = TABLE_HEIGHT - EDGE_MARGIN_MM;

// ── Control loop ────────────────────────────────────────────────────

constexpr uint32_t DEFAULT_TICK_RATE_HZ = 50000;

// ── Kinematics ──────────────────────────────────────────────────────

// Cable length from motor pulley to attachment point on paddle.
// (x, y) is the paddle CENTER; the attachment offset is added internally.
inline float cableLength(int motor, float x, float y) {
  float dx = (x + ATTACH_DX[motor]) - MOTOR_X[motor];
  float dy = (y + ATTACH_DY[motor]) - MOTOR_Y[motor];
  return sqrtf(dx * dx + dy * dy);
}
