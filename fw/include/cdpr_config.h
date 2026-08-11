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

// Ceilings, set just under what the hardware can actually do so that the
// cap is never the reason something is slow — the DEFAULTS below and in
// main.cpp are what keep it tame.
//
// Speed: the motors bind at 2580 rpm x 48 mm spool = 12968 mm/s of cable,
// and paddle speed cannot exceed cable rate (|J_i . u| <= 1). The Teensy's
// own step rate would allow 18850, so the motor is the limit, not us.
constexpr float MAX_VELOCITY_MM_S = 12000.0f;

// Acceleration: a DEFAULT that motion starts at, and a CEILING the runtime
// setter will not exceed. They were one constant, which meant changing how
// hard the rig accelerates required a reflash — and reflashing to try a
// number is how you end up not trying it.
// Accel: 20000 is a JUDGEMENT, not a derived number, and it is worth being
// honest about that. The only physical bound is torque — 46 N per cable
// into ~349 g of effective mass, roughly 261000 mm/s^2 — and that assumes
// peak torque with no opposing tension, so it is an upper bound rather than
// something to design against. 20000 sits an order of magnitude under it,
// which is a bring-up posture, not a hardware fact. Raise it once motion is
// trusted.
//
// The drives ALSO carry Motion.VelLimit = 1000 RPM and Motion.AccLimit =
// 5000 RPM/s (see sw/build/check_limits). Those are SOFTWARE settings
// stored in the drive, not physical limits, and they are almost certainly
// irrelevant here: they govern moves the DRIVE generates over sFoundation,
// and all motion on this rig is step/dir pulses the drive simply follows.
// Almost certainly is not certainly. If VelLimit did gate step/dir, the
// cable ceiling would be 1000 rpm x 48 mm = 5027 mm/s, not 12968 — a factor
// of 2.6. The test costs nothing: command a fast move and watch 'stepped'
// against 'measured' in the state view. A drive that is being limited falls
// behind the step count and never catches up.
//
// Worth knowing: within a 500 mm workspace the SPEED cap is unreachable
// anyway. From 250 mm of run-up, 20000 mm/s^2 tops out at 3162 mm/s, and
// reaching 12968 would need 336000 mm/s^2. Acceleration is what actually
// governs how fast this thing moves, not the speed limit.
constexpr float DEFAULT_ACCEL_MM_S2 = 400.0f;
constexpr float MAX_ACCEL_MM_S2 = 20000.0f;

// ── Control loop ────────────────────────────────────────────────────

constexpr uint32_t DEFAULT_TICK_RATE_HZ = 50000;
