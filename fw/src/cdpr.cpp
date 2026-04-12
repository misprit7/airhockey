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
// The core provides macros but they're per-pin. We resolve at runtime in begin().

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
    : tickRateHz_(tickRateHz), dt_(1.0f / tickRateHz),
      cartX_(0), cartY_(0), velX_(0), velY_(0),
      targetX_(0), targetY_(0),
      stepSetReg_(nullptr), stepClrReg_(nullptr),
      dirSetReg_(nullptr), dirClrReg_(nullptr) {
  for (int i = 0; i < NUM_MOTORS; i++) {
    stepPins_[i]    = stepPins[i];
    dirPins_[i]     = dirPins[i];
    motorCounts_[i] = 0;
    refLengths_[i]  = 0;
    stepBitmask_[i] = 0;
    dirBitmask_[i]  = 0;
  }
}

void CDPR::begin(float calX, float calY) {
  // ── Safety check: max velocity must not require >1 step per tick ──
  float maxSafeVel = (float)tickRateHz_ / COUNTS_PER_MM;
  if (MAX_VELOCITY_MM_S >= maxSafeVel) {
    Serial.printf("ERROR: MAX_VELOCITY_MM_S (%.0f) >= safe limit (%.0f) "
                  "for %lu Hz tick rate. Would need >1 step/tick.\n",
                  MAX_VELOCITY_MM_S, maxSafeVel, (unsigned long)tickRateHz_);
    while (1) { digitalToggle(LED_BUILTIN); delay(100); }
  }

  // ── Resolve GPIO registers and bitmasks ──
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(stepPins_[i], OUTPUT);
    pinMode(dirPins_[i], OUTPUT);
    digitalWriteFast(stepPins_[i], LOW);
    digitalWriteFast(dirPins_[i], LOW);

    GpioInfo step = resolvePin(stepPins_[i]);
    GpioInfo dir  = resolvePin(dirPins_[i]);

    stepBitmask_[i] = step.bitmask;
    dirBitmask_[i]  = dir.bitmask;

    // All step pins must share a GPIO port, same for dir pins.
    if (i == 0) {
      stepSetReg_ = step.setReg;  stepClrReg_ = step.clrReg;
      dirSetReg_  = dir.setReg;   dirClrReg_  = dir.clrReg;
    } else {
      if (step.setReg != stepSetReg_ || dir.setReg != dirSetReg_) {
        Serial.printf("ERROR: All step pins must be on the same GPIO port, "
                      "and all dir pins on the same GPIO port.\n");
        while (1) { digitalToggle(LED_BUILTIN); delay(100); }
      }
    }

    refLengths_[i]  = cableLength(i, calX, calY);
    motorCounts_[i] = 0;
  }

  cartX_ = calX;  cartY_ = calY;
  velX_  = 0;     velY_  = 0;
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
  noInterrupts(); x = targetX_; y = targetY_; interrupts();
}

void CDPR::getCartPosition(float &x, float &y) const {
  noInterrupts(); x = cartX_; y = cartY_; interrupts();
}

void CDPR::getCartVelocity(float &vx, float &vy) const {
  noInterrupts(); vx = velX_; vy = velY_; interrupts();
}

void CDPR::getMotorCounts(int32_t counts[NUM_MOTORS]) const {
  noInterrupts();
  for (int i = 0; i < NUM_MOTORS; i++) counts[i] = motorCounts_[i];
  interrupts();
}

bool CDPR::atTarget(float tol) const {
  noInterrupts();
  float dx = cartX_ - targetX_;
  float dy = cartY_ - targetY_;
  float vx = velX_, vy = velY_;
  interrupts();
  float dist = sqrtf(dx * dx + dy * dy);
  float speed = sqrtf(vx * vx + vy * vy);
  return dist < tol && speed < 1.0f;  // within tol mm and < 1 mm/s
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
// ISR tick
//
// Runs at tickRateHz_ (default 50 kHz). Each tick:
//   1. Compute error vector from cart to target
//   2. Plan speed using trapezoidal profile (accel/decel/cruise)
//   3. Advance theoretical cart position
//   4. Convert to motor counts via IK, emit steps atomically
// ============================================================================

void CDPR::tick() {
  float tx = targetX_;
  float ty = targetY_;

  // ── Error vector: theoretical cart → target ──
  float ex = tx - cartX_;
  float ey = ty - cartY_;
  float dist = sqrtf(ex * ex + ey * ey);

  // ── Compute desired velocity vector ──
  //
  // 1. Pick a desired speed (scalar) using trapezoidal logic:
  //    - If far from target: accelerate up to max velocity
  //    - If close enough that we need to brake: decelerate
  //    - If at target: stop
  //
  // 2. Multiply by the unit direction toward target to get desired velocity.
  //
  // 3. Clamp the *change* from current velocity to desired velocity by the
  //    acceleration limit. This ensures smooth transitions in ALL directions,
  //    including perpendicular target changes.

  float desVx, desVy;

  if (dist < 0.001f) {
    // At target — desired velocity is zero.
    desVx = 0;
    desVy = 0;
  } else {
    // Unit vector toward target.
    float ux = ex / dist;
    float uy = ey / dist;

    // Current speed along the target direction (can be negative if
    // moving away). Used for braking distance calculation.
    float speedAlongTarget = velX_ * ux + velY_ * uy;

    float desiredSpeed;
    if (speedAlongTarget > 0) {
      // Moving toward target. Check braking distance.
      float decelDist = (speedAlongTarget * speedAlongTarget) / (2.0f * MAX_ACCEL_MM_S2);
      if (dist <= decelDist) {
        // Must brake: target speed to stop in exactly 'dist'.
        desiredSpeed = sqrtf(2.0f * MAX_ACCEL_MM_S2 * dist);
      } else {
        desiredSpeed = MAX_VELOCITY_MM_S;
      }
    } else {
      // Stopped or moving away — accelerate toward target.
      desiredSpeed = MAX_VELOCITY_MM_S;
    }

    desVx = desiredSpeed * ux;
    desVy = desiredSpeed * uy;
  }

  // ── Clamp velocity change by acceleration limit ──
  //
  // delta = desired_vel - current_vel
  // if |delta| > max_accel * dt, scale it down.
  // This is the key: velocity changes smoothly in 2D, never jumps.

  float dvx = desVx - velX_;
  float dvy = desVy - velY_;
  float dvMag = sqrtf(dvx * dvx + dvy * dvy);
  float maxDv = MAX_ACCEL_MM_S2 * dt_;

  if (dvMag > maxDv) {
    float scale = maxDv / dvMag;
    dvx *= scale;
    dvy *= scale;
  }

  velX_ += dvx;
  velY_ += dvy;

  // ── Advance theoretical cart position ──
  cartX_ += velX_ * dt_;
  cartY_ += velY_ * dt_;

  // ── Convert to motor counts and step atomically ──
  //
  // Build bitmasks for which step pins need to go HIGH and which LOW,
  // and which dir pins need to be set/cleared. Then write each GPIO
  // port once for simultaneous transitions.

  uint32_t stepSet = 0;
  uint32_t dirSet = 0, dirClr = 0;

  for (int i = 0; i < NUM_MOTORS; i++) {
    int32_t target = cableLengthToCounts(i, cartX_, cartY_);
    int32_t error  = target - motorCounts_[i];

    if (error > 0) {
      dirSet  |= dirBitmask_[i];
      stepSet |= stepBitmask_[i];
      motorCounts_[i]++;
    } else if (error < 0) {
      dirClr  |= dirBitmask_[i];
      stepSet |= stepBitmask_[i];
      motorCounts_[i]--;
    }
  }

  // Write direction first (must settle before step rising edge).
  if (dirSet) *dirSetReg_ = dirSet;
  if (dirClr) *dirClrReg_ = dirClr;

  // Emit step pulse: HIGH → 2µs → LOW, all motors simultaneously.
  // NOTE: The 2µs busy-wait blocks all other interrupts. Fine for now since
  // nothing else is latency-sensitive on the MCU. If that changes, replace
  // with a one-shot hardware timer to clear the pins asynchronously.
  if (stepSet) {
    *stepSetReg_ = stepSet;
    delayMicroseconds(2);
    *stepClrReg_ = stepSet;
  }
}
