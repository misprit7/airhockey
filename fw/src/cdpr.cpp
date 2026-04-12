#include "cdpr.h"
#include <math.h>

// ============================================================================
// ISR trampoline dispatch
// ============================================================================

CDPR* CDPR::instances_[MAX_INSTANCES] = {};
int   CDPR::instanceCount_ = 0;

template <int N> static void trampoline() {
  if (CDPR::instances_[N]) CDPR::instances_[N]->tick();
}

using Fn = void (*)();
static constexpr Fn trampolines[CDPR::MAX_INSTANCES] = {
  trampoline<0>, trampoline<1>, trampoline<2>, trampoline<3>,
};

// ============================================================================
// Helpers
// ============================================================================

float CDPR::clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// Convert a cart (x,y) to the motor count delta from calibration reference.
int32_t CDPR::cableLengthToCounts(int motor, float x, float y) const {
  float len = cableLength(motor, x, y);
  float delta = len - refLengths_[motor];
  return (int32_t)roundf(mmToCounts(delta));
}

// ============================================================================
// Construction / initialization
// ============================================================================

CDPR::CDPR(const int stepPins[NUM_MOTORS], const int dirPins[NUM_MOTORS],
           uint32_t tickRateHz)
    : tickRateHz_(tickRateHz), dt_(1.0f / tickRateHz),
      cartX_(0), cartY_(0), velX_(0), velY_(0), speed_(0),
      targetX_(0), targetY_(0) {
  for (int i = 0; i < NUM_MOTORS; i++) {
    stepPins_[i]    = stepPins[i];
    dirPins_[i]     = dirPins[i];
    motorCounts_[i] = 0;
    refLengths_[i]  = 0;
  }
}

void CDPR::begin(float calX, float calY) {
  // Validate: max velocity must not cause >1 step per tick on any motor.
  // Worst case is cable changing at full cart velocity (motor directly in line).
  // cable_vel ≈ cart_vel (geometric max). Counts/s = cable_vel * COUNTS_PER_MM.
  // Need: counts/s < tickRateHz  →  vel < tickRateHz / COUNTS_PER_MM.
  float maxSafeVelocity = (float)tickRateHz_ / COUNTS_PER_MM;
  if (MAX_VELOCITY_MM_S > maxSafeVelocity * 0.9f) {
    Serial.printf("WARNING: MAX_VELOCITY_MM_S (%.0f) may exceed safe limit (%.0f) "
                  "for tick rate %lu Hz. Risk of >1 step/tick.\n",
                  MAX_VELOCITY_MM_S, maxSafeVelocity * 0.9f,
                  (unsigned long)tickRateHz_);
  }

  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(stepPins_[i], OUTPUT);
    pinMode(dirPins_[i], OUTPUT);
    digitalWriteFast(stepPins_[i], LOW);
    digitalWriteFast(dirPins_[i], LOW);

    refLengths_[i]  = cableLength(i, calX, calY);
    motorCounts_[i] = 0;
  }

  cartX_ = calX;  cartY_ = calY;
  velX_  = 0;     velY_  = 0;
  speed_ = 0;
  targetX_ = calX;
  targetY_ = calY;
}

// ============================================================================
// Target position (main loop ↔ ISR)
// ============================================================================

void CDPR::setTarget(float x, float y) {
  noInterrupts();
  targetX_ = clampf(x, WS_MIN_X, WS_MAX_X);
  targetY_ = clampf(y, WS_MIN_Y, WS_MAX_Y);
  interrupts();
}

void CDPR::getTarget(float &x, float &y) const {
  noInterrupts();
  x = targetX_;
  y = targetY_;
  interrupts();
}

void CDPR::getCartPosition(float &x, float &y) const {
  noInterrupts();
  x = cartX_;
  y = cartY_;
  interrupts();
}

void CDPR::getCartVelocity(float &vx, float &vy) const {
  noInterrupts();
  vx = velX_;
  vy = velY_;
  interrupts();
}

void CDPR::getMotorCounts(int32_t counts[NUM_MOTORS]) const {
  noInterrupts();
  for (int i = 0; i < NUM_MOTORS; i++)
    counts[i] = motorCounts_[i];
  interrupts();
}

// ============================================================================
// Timer
// ============================================================================

void CDPR::startTimer() {
  if (instanceIdx_ >= 0) return;
  instanceIdx_ = instanceCount_++;
  instances_[instanceIdx_] = this;
  timer_.begin(trampolines[instanceIdx_], 1000000.0f / tickRateHz_);
}

void CDPR::stopTimer() {
  timer_.end();
  if (instanceIdx_ >= 0) {
    instances_[instanceIdx_] = nullptr;
    instanceIdx_ = -1;
  }
}

// ============================================================================
// ISR tick — trapezoidal trajectory in cartesian space
// ============================================================================

void CDPR::tick() {
  // ── 1. Read target ──
  float tx = targetX_;
  float ty = targetY_;

  // ── 2. Compute error vector from theoretical cart to target ──
  float ex = tx - cartX_;
  float ey = ty - cartY_;
  float dist = sqrtf(ex * ex + ey * ey);

  // ── 3. Trapezoidal velocity profile ──
  //
  // We maintain a scalar speed along the direction toward the target.
  // Each tick we accelerate or decelerate within limits.
  //
  // Decel distance: the distance needed to stop from current speed.
  //   d_stop = v² / (2 * a_max)
  //
  // If dist <= d_stop, we must decelerate. Otherwise accelerate up to max.

  float desiredSpeed;

  if (dist < 0.001f) {
    // Already at target — stop.
    desiredSpeed = 0.0f;
  } else {
    float decelDist = (speed_ * speed_) / (2.0f * MAX_ACCEL_MM_S2);

    if (dist <= decelDist) {
      // Must decelerate to stop at target.
      // Target speed to stop in exactly 'dist': v = sqrt(2 * a * dist)
      desiredSpeed = sqrtf(2.0f * MAX_ACCEL_MM_S2 * dist);
    } else {
      // Accelerate toward max velocity.
      desiredSpeed = MAX_VELOCITY_MM_S;
    }
  }

  // Apply acceleration limit to scalar speed.
  float maxDeltaSpeed = MAX_ACCEL_MM_S2 * dt_;
  if (desiredSpeed > speed_) {
    speed_ = fminf(desiredSpeed, speed_ + maxDeltaSpeed);
  } else {
    speed_ = fmaxf(desiredSpeed, speed_ - maxDeltaSpeed);
  }

  // ── 4. Convert scalar speed to velocity vector toward target ──
  if (dist > 0.001f) {
    float invDist = 1.0f / dist;
    velX_ = speed_ * ex * invDist;
    velY_ = speed_ * ey * invDist;
  } else {
    velX_ = 0;
    velY_ = 0;
    speed_ = 0;
  }

  // ── 5. Advance theoretical cart position ──
  cartX_ += velX_ * dt_;
  cartY_ += velY_ * dt_;

  // ── 6. Convert theoretical cart position to motor counts via IK ──
  //       Step each motor if physical count != theoretical count ──
  for (int i = 0; i < NUM_MOTORS; i++) {
    int32_t target = cableLengthToCounts(i, cartX_, cartY_);
    int32_t pos    = motorCounts_[i];
    int32_t error  = target - pos;

    if (error != 0) {
      int dir = (error > 0) ? HIGH : LOW;
      digitalWriteFast(dirPins_[i], dir);
      digitalWriteFast(stepPins_[i], HIGH);
      motorCounts_[i] += (error > 0) ? 1 : -1;
    } else {
      digitalWriteFast(stepPins_[i], LOW);
    }
  }
}
