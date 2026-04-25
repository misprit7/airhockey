#include "cdpr.h"
#include <math.h>

// ============================================================================
// ISR trampoline dispatch
// ============================================================================

CDPR *CDPR::instances_[MAX_INSTANCES] = {};
int CDPR::instanceCount_ = 0;

template <int N> static void trampoline() {
  if (CDPR::instances_[N])
    CDPR::instances_[N]->tick();
}

using Fn = void (*)();
static constexpr Fn trampolines[CDPR::MAX_INSTANCES] = {
    trampoline<0>,
    trampoline<1>,
    trampoline<2>,
    trampoline<3>,
};

// ============================================================================
// Helpers
// ============================================================================

float CDPR::clampf(float v, float lo, float hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

int32_t CDPR::cableLengthToCounts(int motor, float x, float y) const {
  float len = cableLength(motor, x, y);
  float delta = len - refLengths_[motor];
  return (int32_t)roundf(mmToCounts(delta));
}

// ============================================================================
// GPIO helpers — resolve Arduino pin to bitmask and register
// ============================================================================

// On Teensy 4.1, digitalWriteFast uses the "fast" GPIO ports (GPIO6-9).
// We need the set/clear registers for atomic multi-pin writes.
// The core provides macros but they're per-pin. We resolve at runtime in
// begin().

struct GpioInfo {
  volatile uint32_t *setReg;
  volatile uint32_t *clrReg;
  uint32_t bitmask;
};

static GpioInfo resolvePin(int pin) {
  // digitalWriteFast(pin, HIGH) expands to: portSetRegister(pin) = bitmask
  // We extract the same info the core uses.
  GpioInfo info;
  info.bitmask = digitalPinToBitMask(pin);
  volatile uint32_t *portSet = portSetRegister(pin);
  volatile uint32_t *portClr = portClearRegister(pin);
  info.setReg = portSet;
  info.clrReg = portClr;
  return info;
}

// ============================================================================
// Construction / initialization
// ============================================================================

CDPR::CDPR(const int stepPins[NUM_MOTORS], const int dirPins[NUM_MOTORS],
           uint32_t tickRateHz)
    : tickRateHz_(tickRateHz), dt_(1.0f / tickRateHz), cartX_(0), cartY_(0),
      velX_(0), velY_(0), targetX_(0), targetY_(0), stepSetReg_(nullptr),
      stepClrReg_(nullptr), dirSetReg_(nullptr), dirClrReg_(nullptr) {
  for (int i = 0; i < NUM_MOTORS; i++) {
    stepPins_[i] = stepPins[i];
    dirPins_[i] = dirPins[i];
    motorCounts_[i] = 0;
    refLengths_[i] = 0;
    stepBitmask_[i] = 0;
    dirBitmask_[i] = 0;
  }
}

void CDPR::begin(float calX, float calY) {
  // ── Safety check: max velocity must not require >1 step per tick ──
  float maxSafeVel = (float)tickRateHz_ / COUNTS_PER_MM;
  if (MAX_VELOCITY_MM_S >= maxSafeVel) {
    Serial.printf("ERROR: MAX_VELOCITY_MM_S (%.0f) >= safe limit (%.0f) "
                  "for %lu Hz tick rate. Would need >1 step/tick.\n",
                  MAX_VELOCITY_MM_S, maxSafeVel, (unsigned long)tickRateHz_);
    while (1) {
      digitalToggle(LED_BUILTIN);
      delay(100);
    }
  }

  // ── Resolve GPIO registers and bitmasks ──
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(stepPins_[i], OUTPUT);
    pinMode(dirPins_[i], OUTPUT);
    digitalWriteFast(stepPins_[i], LOW);
    digitalWriteFast(dirPins_[i], LOW);

    GpioInfo step = resolvePin(stepPins_[i]);
    GpioInfo dir = resolvePin(dirPins_[i]);

    stepBitmask_[i] = step.bitmask;
    dirBitmask_[i] = dir.bitmask;

    // All step pins must share a GPIO port, same for dir pins.
    if (i == 0) {
      stepSetReg_ = step.setReg;
      stepClrReg_ = step.clrReg;
      dirSetReg_ = dir.setReg;
      dirClrReg_ = dir.clrReg;
    } else {
      if (step.setReg != stepSetReg_ || dir.setReg != dirSetReg_) {
        Serial.printf("ERROR: All step pins must be on the same GPIO port, "
                      "and all dir pins on the same GPIO port.\n");
        while (1) {
          digitalToggle(LED_BUILTIN);
          delay(100);
        }
      }
    }

    refLengths_[i] = cableLength(i, calX, calY);
    motorCounts_[i] = 0;
  }

  cartX_ = calX;
  cartY_ = calY;
  velX_ = 0;
  velY_ = 0;
  targetX_ = calX;
  targetY_ = calY;
}

// ============================================================================
// Thread-safe accessors
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

bool CDPR::atTarget(float tol) const {
  noInterrupts();
  float dx = cartX_ - targetX_;
  float dy = cartY_ - targetY_;
  interrupts();
  return (dx * dx + dy * dy) < (tol * tol);
}

// ============================================================================
// Tension — blocking, call before/after timer
// ============================================================================

void CDPR::tension(float mm, float speed_mm_s) {
  tensionMm_ = mm;
  int32_t counts = (int32_t)roundf(mmToCounts(mm));
  if (counts <= 0) return;

  // Interval between steps to achieve desired speed.
  // speed_mm_s → counts/s = speed_mm_s * COUNTS_PER_MM
  // interval_us = 1e6 / counts_per_sec
  float countsPerSec = speed_mm_s * COUNTS_PER_MM;
  uint32_t intervalUs = (uint32_t)(1e6f / countsPerSec);

  Serial.printf("Tensioning: retract %.1fmm (%ld counts) at %.1f mm/s\n",
                mm, (long)counts, speed_mm_s);

  for (int32_t step = 0; step < counts; step++) {
    // Retract = negative direction (cable shorter)
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(dirPins_[i], HIGH);  // HIGH = retract (inverted convention)
    }
    delayMicroseconds(2);
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(stepPins_[i], HIGH);
    }
    delayMicroseconds(2);
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(stepPins_[i], LOW);
    }
    delayMicroseconds(intervalUs);
  }

  Serial.println("Tension applied");
}

void CDPR::releaseTension(float speed_mm_s) {
  if (tensionMm_ <= 0) return;

  int32_t counts = (int32_t)roundf(mmToCounts(tensionMm_));
  float countsPerSec = speed_mm_s * COUNTS_PER_MM;
  uint32_t intervalUs = (uint32_t)(1e6f / countsPerSec);

  Serial.printf("Releasing tension: extend %.1fmm (%ld counts)\n",
                tensionMm_, (long)counts);

  for (int32_t step = 0; step < counts; step++) {
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(dirPins_[i], LOW);  // LOW = extend
    }
    delayMicroseconds(2);
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(stepPins_[i], HIGH);
    }
    delayMicroseconds(2);
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(stepPins_[i], LOW);
    }
    delayMicroseconds(intervalUs);
  }

  tensionMm_ = 0;
  Serial.println("Tension released");
}

// ============================================================================
// Timer
// ============================================================================

void CDPR::startTimer() {
  if (instanceIdx_ >= 0)
    return;
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
// 1D trapezoidal profile helper
//
// Given current position, velocity, and target position along one axis,
// compute the new velocity after one tick. Accelerates toward target,
// decelerates to stop exactly at target.
// ============================================================================

static float trapezoidalStep(float pos, float vel, float target, float maxVel,
                             float maxAccel, float dt) {
  float err = target - pos;
  float absErr = fabsf(err);

  if (absErr < 0.001f && fabsf(vel) < 0.1f) {
    return 0.0f;
  }

  float sign = (err > 0) ? 1.0f : -1.0f;
  bool movingToward = (vel * sign > 0);
  float brakeDist = (vel * vel) / (2.0f * maxAccel);

  float desiredVel;
  if (movingToward && brakeDist >= absErr) {
    desiredVel = 0.0f;
  } else {
    desiredVel = sign * maxVel;
  }

  float dv = desiredVel - vel;
  float maxDv = maxAccel * dt;
  if (dv > maxDv)
    dv = maxDv;
  if (dv < -maxDv)
    dv = -maxDv;

  return vel + dv;
}

// ============================================================================
// ISR tick
//
// Runs at tickRateHz_ (default 50 kHz). Each tick:
//   1. Run independent trapezoidal profiles for X and Y
//   2. Advance theoretical cart position
//   3. Convert to motor counts via IK, emit steps atomically
// ============================================================================

void CDPR::tick() {
  float tx = targetX_;
  float ty = targetY_;

  // ── Independent trapezoidal profiles for each axis ──
  velX_ = trapezoidalStep(cartX_, velX_, tx, MAX_VELOCITY_MM_S, MAX_ACCEL_MM_S2,
                          dt_);
  velY_ = trapezoidalStep(cartY_, velY_, ty, MAX_VELOCITY_MM_S, MAX_ACCEL_MM_S2,
                          dt_);

  // ── Advance theoretical cart position ──
  if (fabsf(tx - cartX_) < 0.01f) {
    cartX_ = tx;
    velX_ = 0;
  } else {
    cartX_ += velX_ * dt_;
  }

  if (fabsf(ty - cartY_) < 0.01f) {
    cartY_ = ty;
    velY_ = 0;
  } else {
    cartY_ += velY_ * dt_;
  }

  // ── Convert to motor counts and step atomically ──
  //
  // Build bitmasks for which step pins need to go HIGH and which LOW,
  // and which dir pins need to be set/cleared. Then write each GPIO
  // port once for simultaneous transitions.

  for (int i = 0; i < NUM_MOTORS; i++) {
    int32_t target = cableLengthToCounts(i, cartX_, cartY_);
    int32_t error = target - motorCounts_[i];

    if (error > 0) {
      digitalWriteFast(dirPins_[i], LOW);
    } else if (error < 0) {
      digitalWriteFast(dirPins_[i], HIGH);
    }
  }

  delayMicroseconds(2);  // direction setup time

  for (int i = 0; i < NUM_MOTORS; i++) {
    int32_t target = cableLengthToCounts(i, cartX_, cartY_);
    int32_t error = target - motorCounts_[i];

    if (error != 0) {
      digitalWriteFast(stepPins_[i], HIGH);
      motorCounts_[i] += (error > 0) ? 1 : -1;
    }
  }

  delayMicroseconds(2);  // step pulse width

  for (int i = 0; i < NUM_MOTORS; i++) {
    digitalWriteFast(stepPins_[i], LOW);
  }
}
