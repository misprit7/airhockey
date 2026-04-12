#include "cdpr.h"

// ============================================================================
// Static instance table for ISR dispatch
// ============================================================================

CDPR* CDPR::instances_[MAX_INSTANCES] = {};
int   CDPR::instanceCount_ = 0;

// Each instance gets its own static trampoline. We generate up to 4.
// (IntervalTimer callbacks are void(void), no context pointer.)
template <int N>
static void trampolineN() {
  if (CDPR::instances_[N]) CDPR::instances_[N]->tick();
}

using TrampolineFn = void (*)();
static constexpr TrampolineFn trampolines[CDPR::MAX_INSTANCES] = {
  trampolineN<0>, trampolineN<1>, trampolineN<2>, trampolineN<3>,
};

// ============================================================================
// Construction / initialization
// ============================================================================

CDPR::CDPR(const int stepPins[4], const int dirPins[4], uint32_t tickRateHz)
    : targetX_(0), targetY_(0), tickRateHz_(tickRateHz),
      tickDt_(1.0f / tickRateHz), instanceIdx_(-1) {
  for (int i = 0; i < 4; i++) {
    stepPins_[i] = stepPins[i];
    dirPins_[i]  = dirPins[i];
    motors_[i]   = {};
  }
}

void CDPR::begin(float calX, float calY) {
  for (int i = 0; i < 4; i++) {
    pinMode(stepPins_[i], OUTPUT);
    pinMode(dirPins_[i], OUTPUT);
    digitalWriteFast(stepPins_[i], LOW);
    digitalWriteFast(dirPins_[i], LOW);

    refLengths_[i] = cableLength(i, calX, calY);
    refCounts_[i]  = 0;
    motors_[i].position      = 0;
    motors_[i].velocity       = 0;
    motors_[i].ticksSinceStep = tickRateHz_;  // large initial value
  }

  targetX_ = calX;
  targetY_ = calY;
}

// ============================================================================
// Target position
// ============================================================================

void CDPR::setTarget(float x, float y) {
  noInterrupts();
  targetX_ = x;
  targetY_ = y;
  interrupts();
}

void CDPR::getTarget(float &x, float &y) const {
  noInterrupts();
  x = targetX_;
  y = targetY_;
  interrupts();
}

void CDPR::getMotorState(int32_t pos[4], float vel[4]) const {
  noInterrupts();
  for (int i = 0; i < 4; i++) {
    pos[i] = motors_[i].position;
    vel[i] = motors_[i].velocity;
  }
  interrupts();
}

// ============================================================================
// Timer start/stop
// ============================================================================

void CDPR::startTimer() {
  if (instanceIdx_ >= 0) return;  // already running

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
// Inverse kinematics helper
// ============================================================================

int32_t CDPR::cartToMotorTarget(int motor, float x, float y) const {
  float len = cableLength(motor, x, y);
  float deltaLen = len - refLengths_[motor];
  return refCounts_[motor] + (int32_t)roundf(mmToCounts(deltaLen));
}

// ============================================================================
// ISR tick — runs at TICK_RATE_HZ (50 kHz)
// ============================================================================

void CDPR::tick() {
  float tx = targetX_;
  float ty = targetY_;

  for (int i = 0; i < 4; i++) {
    motors_[i].ticksSinceStep++;

    int32_t target = cartToMotorTarget(i, tx, ty);
    int32_t pos    = motors_[i].position;
    int32_t error  = target - pos;

    // ---------------------------------------------------------------
    // TODO: Step decision logic
    //
    // Inputs available:
    //   error                    - signed distance to target (counts)
    //   pos                      - current motor position (counts)
    //   target                   - desired motor position (counts)
    //   motors_[i].velocity      - current velocity estimate (counts/s)
    //   motors_[i].ticksSinceStep - ticks since last step
    //   tickDt_                  - time per tick (seconds)
    //
    // Should set:
    //   step = true/false  - whether to emit a step pulse this tick
    //   dir  = HIGH/LOW    - direction for this step
    //
    // Constraints:
    //   - Max step rate per motor: limited by driver (probably ~100kHz)
    //   - Must respect acceleration limits
    //   - Should converge smoothly to target, not bang-bang
    // ---------------------------------------------------------------

    bool step = false;
    int dir = LOW;

    (void)error;  // suppress unused warning until logic is implemented

    if (step) {
      digitalWriteFast(dirPins_[i], dir);
      digitalWriteFast(stepPins_[i], HIGH);
      motors_[i].position += (dir == HIGH) ? 1 : -1;
      motors_[i].velocity = (dir == HIGH ? 1.0f : -1.0f) / (motors_[i].ticksSinceStep * tickDt_);
      motors_[i].ticksSinceStep = 0;
    } else {
      digitalWriteFast(stepPins_[i], LOW);
    }
  }
}
