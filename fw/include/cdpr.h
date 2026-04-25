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

  // True if cart is within `tol` mm of target.
  bool atTarget(float tol = 0.5f) const;

  // Retract all cables by `mm` to add tension. Call before motion.
  // Blocking — steps all motors slowly at the given speed (mm/s).
  // Must be called BEFORE startTimer().
  void tension(float mm, float speed_mm_s = 5.0f);

  // Release tension (extend cables by the previously tensioned amount).
  // Blocking — call AFTER stopTimer().
  void releaseTension(float speed_mm_s = 5.0f);

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

  // ── Target (set from main loop, read by ISR) ──
  volatile float targetX_, targetY_;

  // ── Motor physical state ──
  volatile int32_t motorCounts_[NUM_MOTORS];  // actual step count (ground truth)

  // ── Tension ──
  float tensionMm_ = 0;  // how much tension was applied (for release)

  // ── Precomputed GPIO bitmasks for atomic register writes ──
  // Step pins (GPIO7) and dir pins (GPIO8) are written via DR_SET/DR_CLEAR
  // registers so all motors transition in the same bus cycle.
  uint32_t stepBitmask_[NUM_MOTORS];  // per-motor step pin bitmask
  uint32_t dirBitmask_[NUM_MOTORS];   // per-motor dir pin bitmask
  volatile uint32_t *stepSetReg_;     // GPIO DR_SET register for step pins
  volatile uint32_t *stepClrReg_;     // GPIO DR_CLEAR register for step pins
  volatile uint32_t *dirSetReg_;      // GPIO DR_SET register for dir pins
  volatile uint32_t *dirClrReg_;      // GPIO DR_CLEAR register for dir pins

private:
  // ── Helpers ──
  int32_t cableLengthToCounts(int motor, float x, float y) const;
  static float clampf(float v, float lo, float hi);
};
