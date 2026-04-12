#include <Arduino.h>
#include "cdpr.h"

// ============================================================================
// Instance
// ============================================================================

static const int stepPins[NUM_MOTORS] = {6, 7, 8, 9};
static const int dirPins[NUM_MOTORS]  = {34, 35, 36, 37};

static CDPR cdpr(stepPins, dirPins);

// ============================================================================
// Square test pattern
// ============================================================================

static constexpr float SQUARE_SIZE = 50.0f;  // 5cm square
static float centerX, centerY;

// Corners: center ± 25mm
static float squareX[4], squareY[4];

static int  squareIdx = 0;
static bool squareDone = false;

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  centerX = TABLE_WIDTH  / 2.0f;
  centerY = TABLE_HEIGHT / 2.0f;
  float half = SQUARE_SIZE / 2.0f;

  // CCW square: bottom-left, bottom-right, top-right, top-left
  squareX[0] = centerX - half;  squareY[0] = centerY - half;
  squareX[1] = centerX + half;  squareY[1] = centerY - half;
  squareX[2] = centerX + half;  squareY[2] = centerY + half;
  squareX[3] = centerX - half;  squareY[3] = centerY + half;

  cdpr.begin(centerX, centerY);
  cdpr.startTimer();

  Serial.println("CDPR motion controller ready");
  Serial.printf("Table: %.0f x %.0f mm\n", TABLE_WIDTH, TABLE_HEIGHT);
  Serial.printf("Tracing 50mm square at center (%.0f, %.0f)\n", centerX, centerY);

  // Send first corner
  cdpr.setTarget(squareX[0], squareY[0]);
}

// ============================================================================
// Main loop
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 200;

void loop() {
  // Advance square pattern
  if (!squareDone && cdpr.atTarget()) {
    squareIdx++;
    if (squareIdx < 4) {
      Serial.printf("Corner %d: (%.1f, %.1f)\n", squareIdx, squareX[squareIdx], squareY[squareIdx]);
      cdpr.setTarget(squareX[squareIdx], squareY[squareIdx]);
    } else {
      // Return to center
      Serial.println("Returning to center");
      cdpr.setTarget(centerX, centerY);
      squareDone = true;
    }
  }

  // Status reporting
  uint32_t now = millis();
  if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
    lastStatusMs = now;

    float tx, ty, cx, cy, vx, vy;
    int32_t counts[NUM_MOTORS];
    cdpr.getTarget(tx, ty);
    cdpr.getCartPosition(cx, cy);
    cdpr.getCartVelocity(vx, vy);
    cdpr.getMotorCounts(counts);

    float speed = sqrtf(vx * vx + vy * vy);
    Serial.printf("tgt=(%.1f,%.1f) pos=(%.1f,%.1f) spd=%.0f "
                  "counts=[%ld,%ld,%ld,%ld]%s\n",
                  tx, ty, cx, cy, speed,
                  (long)counts[0], (long)counts[1],
                  (long)counts[2], (long)counts[3],
                  squareDone ? " IDLE" : "");

    digitalToggle(LED_BUILTIN);
  }
}
