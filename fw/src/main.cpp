#include <Arduino.h>

// Step 4 motors one at a time, 800 steps each at 100 steps/s cruise,
// with a trapezoidal accel/decel ramp. Pins 2-5, one motor per pin.

constexpr int kMotorPins[] = {2, 3, 4, 5};
constexpr int kNumMotors = 4;
constexpr uint32_t kTotalSteps = 800;
constexpr float kCruiseRate = 100.0f;   // steps/sec
constexpr float kAccel = 200.0f;        // steps/sec^2
constexpr uint32_t kPulseWidthUs = 2;

// Trapezoidal velocity profile: accelerate at kAccel up to kCruiseRate,
// cruise, then decelerate symmetrically to stop at exactly kTotalSteps.
// Returns cumulative target step count at time t (seconds).
static float trapezoidal_target(float t, float *out_duration) {
  // Time to accelerate to cruise speed.
  const float t_accel = kCruiseRate / kAccel;
  // Steps consumed during accel (and decel, symmetric).
  const float steps_accel = 0.5f * kAccel * t_accel * t_accel;

  float t_cruise, total_duration;
  float steps_cruise_phase;

  if (2.0f * steps_accel >= kTotalSteps) {
    // Triangle profile — can't reach cruise speed.
    // Peak rate = sqrt(kAccel * kTotalSteps)
    // Each half: 0.5 * a * t_half^2 = kTotalSteps/2
    //   t_half = sqrt(kTotalSteps / kAccel)
    const float t_half = sqrtf((float)kTotalSteps / kAccel);
    total_duration = 2.0f * t_half;
    if (out_duration) *out_duration = total_duration;

    if (t <= 0) return 0;
    if (t >= total_duration) return kTotalSteps;
    if (t <= t_half) {
      return 0.5f * kAccel * t * t;
    }
    const float rem = total_duration - t;
    return kTotalSteps - 0.5f * kAccel * rem * rem;
  }

  // Full trapezoidal profile.
  steps_cruise_phase = kTotalSteps - 2.0f * steps_accel;
  t_cruise = steps_cruise_phase / kCruiseRate;
  total_duration = 2.0f * t_accel + t_cruise;
  if (out_duration) *out_duration = total_duration;

  if (t <= 0) return 0;
  if (t >= total_duration) return kTotalSteps;

  if (t <= t_accel) {
    // Accelerating.
    return 0.5f * kAccel * t * t;
  }
  if (t <= t_accel + t_cruise) {
    // Cruising.
    return steps_accel + kCruiseRate * (t - t_accel);
  }
  // Decelerating.
  const float rem = total_duration - t;
  return kTotalSteps - 0.5f * kAccel * rem * rem;
}

static void run_motor(int pin) {
  pinMode(pin, OUTPUT);
  digitalWriteFast(pin, LOW);

  float duration;
  trapezoidal_target(0, &duration);
  const uint32_t duration_us = (uint32_t)(duration * 1e6f);

  uint32_t steps_emitted = 0;
  const uint32_t t0 = micros();
  for (;;) {
    const uint32_t elapsed = micros() - t0;
    const float t = elapsed * 1e-6f;
    const uint32_t target = (uint32_t)trapezoidal_target(t, nullptr);

    if (elapsed >= duration_us && steps_emitted >= kTotalSteps) {
      break;
    }
    if (target > steps_emitted) {
      digitalWriteFast(pin, HIGH);
      delayMicroseconds(kPulseWidthUs);
      digitalWriteFast(pin, LOW);
      steps_emitted++;
      delayMicroseconds(kPulseWidthUs);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=== Motor step test ===");
  Serial.printf("  %d steps/motor, %d steps/s cruise, %d steps/s^2 accel\n",
                (int)kTotalSteps, (int)kCruiseRate, (int)kAccel);

  for (int i = 0; i < kNumMotors; i++) {
    Serial.printf("Motor %d (pin %d)... ", i, kMotorPins[i]);
    run_motor(kMotorPins[i]);
    Serial.println("done");
    delay(500);  // pause between motors
  }

  Serial.println("All done.");
}

void loop() {
  // Hang.
}
