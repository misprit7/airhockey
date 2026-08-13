#include "cdpr.h"
#include <math.h>

// ============================================================================
// ISR trampoline dispatch
// ============================================================================

CDPR *CDPR::instances_[MAX_INSTANCES] = {};

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
  float len = cableLength(motor, x, y, theta_);
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
      velX_(0), velY_(0), accX_(0), accY_(0),
      velLimit_(MAX_VELOCITY_MM_S), accelLimit_(DEFAULT_ACCEL_MM_S2),
      accelRamp_(MOTION_ACCEL_RAMP_S),
      limitFlags_(0), speedFrac_(0), accelFrac_(0),
      peakSpeedFrac_(0), peakAccelFrac_(0), theta_(MALLET_THETA_RAD),
      targetX_(0), targetY_(0), stepSetReg_(nullptr),
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

void CDPR::begin(float calX, float calY, float theta) {
  theta_ = theta;
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

    refLengths_[i] = cableLength(i, calX, calY, theta_);
    motorCounts_[i] = 0;
  }

  cartX_ = calX;
  cartY_ = calY;
  velX_ = 0;
  velY_ = 0;
  accX_ = 0;
  accY_ = 0;
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

void CDPR::setVelocityLimit(float mm_s) {
  if (mm_s <= 0.0f) return;
  noInterrupts();
  velLimit_ = (mm_s > MAX_VELOCITY_MM_S) ? MAX_VELOCITY_MM_S : mm_s;
  interrupts();
}

void CDPR::setAccelLimit(float mm_s2) {
  if (mm_s2 <= 0.0f) return;
  noInterrupts();
  accelLimit_ = (mm_s2 > MAX_ACCEL_MM_S2) ? MAX_ACCEL_MM_S2 : mm_s2;
  interrupts();
}

void CDPR::setAccelRamp(float seconds) {
  if (seconds < MIN_ACCEL_RAMP_S) seconds = MIN_ACCEL_RAMP_S;
  if (seconds > MAX_ACCEL_RAMP_S) seconds = MAX_ACCEL_RAMP_S;
  noInterrupts();
  accelRamp_ = seconds;
  interrupts();
}

float CDPR::getAccelRamp() const { return accelRamp_; }

float CDPR::getVelocityLimit() const { return velLimit_; }
float CDPR::getAccelLimit() const { return accelLimit_; }
uint8_t CDPR::getLimitFlags() const { return limitFlags_; }
float CDPR::getSpeedFrac() const { return speedFrac_; }
float CDPR::getAccelFrac() const { return accelFrac_; }
float CDPR::getPeakSpeedFrac() const { return peakSpeedFrac_; }
float CDPR::getPeakAccelFrac() const { return peakAccelFrac_; }

void CDPR::resetPeaks() {
  noInterrupts();
  peakSpeedFrac_ = 0.0f;
  peakAccelFrac_ = 0.0f;
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
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(dirPins_[i], dirLevelRetract(i));
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
      digitalWriteFast(dirPins_[i], dirLevelExtend(i));
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
  // Claim a FREE slot, not the next one ever handed out: start/stop cycles
  // are normal (every test lap is one), and a monotonic counter walks off
  // the end of the array after MAX_INSTANCES laps.
  for (int i = 0; i < MAX_INSTANCES; i++) {
    if (instances_[i] == nullptr) {
      instanceIdx_ = i;
      break;
    }
  }
  if (instanceIdx_ < 0) {
    Serial.println("ERROR: no free timer slot");
    return;
  }
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
//   1. Advance the velocity VECTOR one step toward the target
//   2. Advance theoretical cart position
//   3. Convert to motor counts via IK, emit steps atomically
//
// Only step 1 changed when the profile went vector. Steps 2 and 3 are what
// keep the four motors synchronised and they work the same way they always
// did: every motor's target is derived from ONE shared cart position, so the
// motors cannot drift relative to each other by construction — whatever the
// trajectory law does, it moves a single point and the IK follows it. The
// profile is upstream of synchronisation, not part of it.
// ============================================================================

void CDPR::tick() {
  const float tx = targetX_;
  const float ty = targetY_;

  // ── One profile along the direction of travel ──
  const float vx0 = velX_, vy0 = velY_;
  float nvx, nvy, nax, nay;
  limitFlags_ = motionProfileStep(cartX_, cartY_, velX_, velY_,
                                  accX_, accY_, tx, ty,
                                  velLimit_, accelLimit_, accelRamp_, dt_,
                                  nvx, nvy, nax, nay);
  velX_ = nvx;
  velY_ = nvy;
  accX_ = nax;
  accY_ = nay;

  // Magnitudes, not worst-axis. The per-axis version read 100% on each axis
  // during a diagonal while the cart was really at 141% of both caps, so the
  // gauges understated exactly the move that used the most of the machine.
  const float ax = (velX_ - vx0) / dt_;
  const float ay = (velY_ - vy0) / dt_;
  const float sf = sqrtf(velX_ * velX_ + velY_ * velY_) / velLimit_;
  const float af = sqrtf(ax * ax + ay * ay) / accelLimit_;
  speedFrac_ = sf;
  accelFrac_ = af;
  if (sf > peakSpeedFrac_) peakSpeedFrac_ = sf;
  if (af > peakAccelFrac_) peakAccelFrac_ = af;

  // ── Advance theoretical cart position ──
  //
  // Park on the target only when close AND slow, and park both axes at once.
  // Zeroing one axis' velocity the moment the cart crossed the target's x —
  // which the per-axis version did, mid-flight, while y was still running —
  // is what put a kink in the end of every diagonal move.
  const float rx = tx - cartX_;
  const float ry = ty - cartY_;
  const float distSq = rx * rx + ry * ry;
  const float speedSq = velX_ * velX_ + velY_ * velY_;
  if (distSq < MOTION_POS_EPS_MM * MOTION_POS_EPS_MM &&
      speedSq < MOTION_VEL_EPS_MM_S * MOTION_VEL_EPS_MM_S) {
    cartX_ = tx;
    cartY_ = ty;
    velX_ = 0;
    velY_ = 0;
    accX_ = 0;   // parked: drop the slew state too, or the next move starts
    accY_ = 0;   // by unwinding a stale acceleration
  } else {
    cartX_ += velX_ * dt_;
    cartY_ += velY_ * dt_;
  }

  // ── Convert to motor counts and step atomically ──
  //
  // Set every dir pin, wait out the drives' setup time, pulse every step pin,
  // wait out the pulse width, drop them together. All four motors see their
  // edges within the same pair of microsecond windows — that, and the shared
  // cart position above, is the whole synchronisation story.
  //
  // The IK runs ONCE per motor and the result is reused for both the dir and
  // the step pass. It used to be computed twice per motor per tick, with
  // identical inputs both times (neither cart position nor this motor's count
  // changes in between), so this is the same numbers for half the work — and
  // at 50 kHz, ISR headroom is what keeps ticks from being dropped, which
  // WOULD desynchronise the motors.

  int32_t err[NUM_MOTORS];
  for (int i = 0; i < NUM_MOTORS; i++) {
    err[i] = cableLengthToCounts(i, cartX_, cartY_) - motorCounts_[i];

    if (err[i] > 0) {
      digitalWriteFast(dirPins_[i], dirLevelExtend(i));   // cable lengthens
    } else if (err[i] < 0) {
      digitalWriteFast(dirPins_[i], dirLevelRetract(i));  // cable shortens
    }
  }

  delayMicroseconds(2);  // direction setup time

  for (int i = 0; i < NUM_MOTORS; i++) {
    if (err[i] != 0) {
      digitalWriteFast(stepPins_[i], HIGH);
      motorCounts_[i] += (err[i] > 0) ? 1 : -1;
    }
  }

  delayMicroseconds(2);  // step pulse width

  for (int i = 0; i < NUM_MOTORS; i++) {
    digitalWriteFast(stepPins_[i], LOW);
  }
}
