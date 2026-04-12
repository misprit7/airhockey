#include <Arduino.h>
#include "cdpr.h"

// ============================================================================
// Instance
// ============================================================================

static const int stepPins[NUM_MOTORS] = {6, 7, 8, 9};
static const int dirPins[NUM_MOTORS]  = {34, 35, 36, 37};

static CDPR cdpr(stepPins, dirPins);

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  float cx = TABLE_WIDTH  / 2.0f;
  float cy = TABLE_HEIGHT / 2.0f;
  cdpr.begin(cx, cy);
  cdpr.startTimer();

  Serial.println("CDPR motion controller ready");
  Serial.printf("Table: %.0f x %.0f mm\n", TABLE_WIDTH, TABLE_HEIGHT);
  Serial.printf("Counts/rev: %d  (%.3f mm/count)\n", COUNTS_PER_REV, MM_PER_COUNT);
  Serial.printf("Max vel: %.0f mm/s  Max accel: %.0f mm/s^2\n",
                MAX_VELOCITY_MM_S, MAX_ACCEL_MM_S2);
}

// ============================================================================
// Main loop — status reporting (serial commands TODO)
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 500;

void loop() {
  // TODO: serial command parsing → cdpr.setTarget(x, y)

  uint32_t now = millis();
  if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
    lastStatusMs = now;

    float tx, ty, cx, cy, vx, vy;
    int32_t counts[NUM_MOTORS];
    cdpr.getTarget(tx, ty);
    cdpr.getCartPosition(cx, cy);
    cdpr.getCartVelocity(vx, vy);
    cdpr.getMotorCounts(counts);

    Serial.printf("tgt=(%.1f,%.1f) pos=(%.1f,%.1f) vel=(%.0f,%.0f) "
                  "counts=[%ld,%ld,%ld,%ld]\n",
                  tx, ty, cx, cy, vx, vy,
                  (long)counts[0], (long)counts[1],
                  (long)counts[2], (long)counts[3]);

    digitalToggle(LED_BUILTIN);
  }
}
