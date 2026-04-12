#pragma once

#include <Arduino.h>
#include "cdpr_config.h"

// A single CDPR unit: 4 motors driving cables to a paddle.
// Multiple instances can coexist, each with its own pins and timer.

struct MotorState {
  volatile int32_t  position;       // Current position (stepper counts)
  volatile float    velocity;       // Velocity estimate (counts/s), signed
  volatile uint32_t ticksSinceStep; // Ticks since last step pulse
};

class CDPR {
public:
  // Construct with pin arrays and tick rate.
  CDPR(const int stepPins[4], const int dirPins[4], uint32_t tickRateHz = DEFAULT_TICK_RATE_HZ);

  // Initialize pins and calibrate at a given paddle position (mm).
  void begin(float calX, float calY);

  // Set the target paddle position (mm). Called from main loop.
  void setTarget(float x, float y);

  // Read back the current target.
  void getTarget(float &x, float &y) const;

  // Snapshot motor state (safe to call from main loop; briefly disables interrupts).
  void getMotorState(int32_t pos[4], float vel[4]) const;

  // Start the 50 kHz control timer. Only call once after begin().
  void startTimer();

  // Stop the control timer.
  void stopTimer();

private:
  // Pin assignments (copied in constructor)
  int stepPins_[4];
  int dirPins_[4];

  // Motor state
  MotorState motors_[4];

  // Target position (written by main loop, read by ISR)
  volatile float targetX_;
  volatile float targetY_;

  // Reference calibration
  float   refLengths_[4];
  int32_t refCounts_[4];

  // Timer
  IntervalTimer timer_;
  uint32_t tickRateHz_;
  float tickDt_;  // 1.0 / tickRateHz_

  // Compute motor count target from cart position
  int32_t cartToMotorTarget(int motor, float x, float y) const;

public:
  // ISR internals — public for trampoline access. Do not call from user code.
  static constexpr int MAX_INSTANCES = 4;
  static CDPR* instances_[MAX_INSTANCES];
  static int   instanceCount_;
  int          instanceIdx_;
  void tick();
};
