#include <Arduino.h>
#include "cdpr.h"

// ============================================================================
// Instance
// ============================================================================

static const int stepPins[NUM_MOTORS] = {6, 7, 8, 9};
static const int dirPins[NUM_MOTORS]  = {34, 35, 36, 37};

static CDPR cdpr(stepPins, dirPins);

// ============================================================================
// Reset pin — short pin 33 to GND to reboot
// ============================================================================

constexpr int RESET_PIN = 33;

static void checkReset() {
  if (digitalReadFast(RESET_PIN) == LOW) {
    delay(50);
    if (digitalReadFast(RESET_PIN) == LOW) {
      SCB_AIRCR = 0x05FA0004;
    }
  }
}

// ============================================================================
// Line test: back and forth along motor0–motor2 diagonal
// ============================================================================

static constexpr float LINE_LENGTH = 100.0f;  // 10cm each side of center
static float centerX, centerY;
static float endAX, endAY, endBX, endBY;
static bool goingToB = true;
static int tripsRemaining = 10;  // each trip = one direction

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(RESET_PIN, INPUT_PULLUP);

  centerX = TABLE_WIDTH  / 2.0f;
  centerY = TABLE_HEIGHT / 2.0f;

  float dx = MOTOR_X[2] - MOTOR_X[0];
  float dy = MOTOR_Y[2] - MOTOR_Y[0];
  float len = sqrtf(dx * dx + dy * dy);
  float ux = dx / len;
  float uy = dy / len;

  endAX = centerX - LINE_LENGTH * ux;
  endAY = centerY - LINE_LENGTH * uy;
  endBX = centerX + LINE_LENGTH * ux;
  endBY = centerY + LINE_LENGTH * uy;

  cdpr.begin(centerX, centerY);
  cdpr.startTimer();

  Serial.println("CDPR motion controller ready");
  Serial.printf("Line: (%.1f,%.1f) <-> (%.1f,%.1f)\n", endAX, endAY, endBX, endBY);

  cdpr.setTarget(endBX, endBY);
}

// ============================================================================
// Main loop
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 10;

void loop() {
  checkReset();

  if (tripsRemaining > 0 && cdpr.atTarget()) {
    if (goingToB) {
      cdpr.setTarget(endAX, endAY);
    } else {
      cdpr.setTarget(endBX, endBY);
    }
    goingToB = !goingToB;
    tripsRemaining--;
  } else if (tripsRemaining == 0 && cdpr.atTarget()) {
    cdpr.setTarget(centerX, centerY);
    tripsRemaining = -1;  // done
  }

  uint32_t now = millis();
  if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
    lastStatusMs = now;
    int32_t counts[NUM_MOTORS];
    float cx, cy, tx, ty, vx, vy;
    cdpr.getMotorCounts(counts);
    cdpr.getCartPosition(cx, cy);
    cdpr.getTarget(tx, ty);
    cdpr.getCartVelocity(vx, vy);
    float speed = sqrtf(vx * vx + vy * vy);
    Serial.printf("c2=%ld pos=(%.1f,%.1f) tgt=(%.1f,%.1f) spd=%.0f\n",
                  (long)counts[2], cx, cy, tx, ty, speed);
  }
}
