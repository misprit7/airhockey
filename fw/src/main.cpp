#include <Arduino.h>
#include <math.h>

// Pin assignments
constexpr int kStepPins[] = {6, 7, 8, 9};
constexpr int kDirPins[]  = {34, 35, 36, 37};
constexpr int kNumMotors = 4;

// Sine oscillation parameters
constexpr float kMaxStepsPerSec = 100.0f;  // peak step rate
constexpr float kOscFreqHz = 0.25f;        // full cycle period = 4s
constexpr uint32_t kLoopUs = 100;          // 10 kHz control loop
constexpr uint32_t kPulseWidthUs = 2;

// LED heartbeat
constexpr uint32_t kBlinkIntervalUs = 500000;
static uint32_t lastBlinkUs = 0;
static bool ledState = false;

// Per-motor step accumulator (fractional step tracking)
static float stepAccum[kNumMotors] = {};
static uint32_t loopStartUs = 0;
static int activeMotor = 0;
static float cycleStartTime = 0;

void setup() {
  for (int i = 0; i < kNumMotors; i++) {
    pinMode(kStepPins[i], OUTPUT);
    pinMode(kDirPins[i], OUTPUT);
    digitalWriteFast(kStepPins[i], LOW);
    digitalWriteFast(kDirPins[i], LOW);
  }
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWriteFast(LED_BUILTIN, LOW);

  loopStartUs = micros();
  lastBlinkUs = loopStartUs;
}

void loop() {
  uint32_t now = micros();
  float t = (now - loopStartUs) * 1e-6f;
  float dt = kLoopUs * 1e-6f;

  // One motor at a time, full sine cycle then switch
  float cycleT = t - cycleStartTime;
  float cyclePeriod = 1.0f / kOscFreqHz;
  if (cycleT >= cyclePeriod) {
    cycleStartTime += cyclePeriod;
    cycleT -= cyclePeriod;
    stepAccum[activeMotor] = 0;
    activeMotor = (activeMotor + 1) % kNumMotors;
  }

  float velocity = kMaxStepsPerSec * sinf(2.0f * M_PI * kOscFreqHz * cycleT);
  float stepsDelta = velocity * dt;

  // Set direction for active motor
  digitalWriteFast(kDirPins[activeMotor], stepsDelta >= 0 ? HIGH : LOW);

  // Accumulate fractional steps
  bool needPulse = false;
  stepAccum[activeMotor] += fabsf(stepsDelta);
  if (stepAccum[activeMotor] >= 1.0f) {
    stepAccum[activeMotor] -= 1.0f;
    needPulse = true;
  }

  // Emit step pulse
  if (needPulse) {
    digitalWriteFast(kStepPins[activeMotor], HIGH);
    delayMicroseconds(kPulseWidthUs);
    digitalWriteFast(kStepPins[activeMotor], LOW);
  }

  // LED heartbeat
  if (now - lastBlinkUs >= kBlinkIntervalUs) {
    ledState = !ledState;
    digitalWriteFast(LED_BUILTIN, ledState ? HIGH : LOW);
    lastBlinkUs = now;
  }

  // Wait for next loop iteration
  while (micros() - now < kLoopUs) {}
}
