#pragma once

#include <Arduino.h>
#include "cdpr_config.h"

// ============================================================================
// CDPR — Cable-Driven Parallel Robot motion controller
//
// Plans a smooth trajectory in cartesian (x, y) space using trapezoidal
// velocity profiles, then converts to per-motor step counts via inverse
// kinematics. A high-frequency ISR (default 50 kHz) advances the
// theoretical cart position each tick and emits step pulses when the
// physical motor counts diverge from the IK solution.
//
// Usage:
//   CDPR cdpr(stepPins, dirPins);
//   cdpr.begin(startX, startY);
//   cdpr.startTimer();
//   cdpr.setTarget(newX, newY);  // from main loop / serial
// ============================================================================

class CDPR {
public:
  CDPR(const int stepPins[NUM_MOTORS], const int dirPins[NUM_MOTORS],
       uint32_t tickRateHz = DEFAULT_TICK_RATE_HZ);

  // Initialize pins and set the calibration position (mm).
  // Motors are assumed to be at rest with cables tensioned to this point.
  void begin(float calX, float calY);

  // Set a new target cart position (mm). Thread-safe (called from main loop).
  void setTarget(float x, float y);

  // Read current state. All thread-safe (briefly disables interrupts).
  void getTarget(float &x, float &y) const;
  void getCartPosition(float &x, float &y) const;   // theoretical position
  void getCartVelocity(float &vx, float &vy) const;
  void getMotorCounts(int32_t counts[NUM_MOTORS]) const;

  void startTimer();
  void stopTimer();

  // ISR entry point — public for trampoline. Do not call from user code.
  void tick();

  // ISR dispatch table
  static constexpr int MAX_INSTANCES = 4;
  static CDPR* instances_[MAX_INSTANCES];
  static int   instanceCount_;

private:
  // ── Pin config ──
  int stepPins_[NUM_MOTORS];
  int dirPins_[NUM_MOTORS];

  // ── Timing ──
  IntervalTimer timer_;
  uint32_t tickRateHz_;
  float dt_;                // seconds per tick
  int instanceIdx_ = -1;

  // ── Calibration ──
  // At calibration we record the cable lengths for the known cart position.
  // All subsequent motor counts are relative deltas from this reference.
  float refLengths_[NUM_MOTORS];

  // ── Cart trajectory (updated every tick in ISR) ──
  volatile float cartX_, cartY_;     // theoretical position (mm)
  volatile float velX_,  velY_;      // current velocity (mm/s)
  float speed_;                      // |vel| (mm/s), scalar for trapezoidal profile

  // ── Target (set from main loop, read by ISR) ──
  volatile float targetX_, targetY_;

  // ── Motor physical state ──
  volatile int32_t motorCounts_[NUM_MOTORS];  // actual step count (ground truth)

  // ── Helpers ──
  int32_t cableLengthToCounts(int motor, float x, float y) const;
  static float clampf(float v, float lo, float hi);
};
