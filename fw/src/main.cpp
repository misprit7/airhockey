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
// Square test pattern (once, then return to center)
// ============================================================================

static constexpr float SQUARE_SIZE = 100.0f;
static float centerX, centerY;
static float squareX[4], squareY[4];
static int squareIdx = 0;
static bool testDone = false;

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
  float half = SQUARE_SIZE / 2.0f;

  squareX[0] = centerX - half;  squareY[0] = centerY - half;
  squareX[1] = centerX + half;  squareY[1] = centerY - half;
  squareX[2] = centerX + half;  squareY[2] = centerY + half;
  squareX[3] = centerX - half;  squareY[3] = centerY + half;

  cdpr.begin(centerX, centerY);
  cdpr.startTimer();

  Serial.println("CDPR motion controller ready");
  Serial.printf("Square %.0fmm at center (%.0f, %.0f)\n", SQUARE_SIZE, centerX, centerY);

  cdpr.setTarget(squareX[0], squareY[0]);
}

// ============================================================================
// Main loop
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 100;

void loop() {
  checkReset();

  if (!testDone && cdpr.atTarget()) {
    squareIdx++;
    if (squareIdx < 4) {
      Serial.printf("Corner %d: (%.1f, %.1f)\n", squareIdx,
                    squareX[squareIdx], squareY[squareIdx]);
      cdpr.setTarget(squareX[squareIdx], squareY[squareIdx]);
    } else if (squareIdx == 4) {
      Serial.println("Returning to center");
      cdpr.setTarget(centerX, centerY);
    } else {
      Serial.println("Done");
      testDone = true;
    }
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
    Serial.printf("pos=(%.1f,%.1f) tgt=(%.1f,%.1f) spd=%.0f c=[%ld,%ld,%ld,%ld]%s\n",
                  cx, cy, tx, ty, speed,
                  (long)counts[0], (long)counts[1],
                  (long)counts[2], (long)counts[3],
                  testDone ? " IDLE" : "");
  }
}
