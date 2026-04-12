#include <Arduino.h>
#include "cdpr.h"

// ============================================================================
// Instance configuration
// ============================================================================

static const int stepPins[4] = {6, 7, 8, 9};
static const int dirPins[4]  = {34, 35, 36, 37};

static CDPR cdpr(stepPins, dirPins);

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  float cx = TABLE_WIDTH / 2.0f;
  float cy = TABLE_HEIGHT / 2.0f;
  cdpr.begin(cx, cy);
  cdpr.startTimer();

  Serial.println("CDPR motion controller ready");
  Serial.printf("Tick rate: %lu Hz (%.1f us)\n", (unsigned long)DEFAULT_TICK_RATE_HZ, 1e6f / DEFAULT_TICK_RATE_HZ);
  Serial.printf("Table: %.0f x %.0f mm\n", TABLE_WIDTH, TABLE_HEIGHT);
  Serial.printf("Spool circumference: %.1f mm\n", SPOOL_CIRCUMFERENCE_MM);
  Serial.printf("Counts/rev: %d\n", COUNTS_PER_REV);
}

// ============================================================================
// Main loop — serial commands + status reporting
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 500;

void loop() {
  // TODO: Parse serial commands to call cdpr.setTarget(x, y)

  uint32_t now = millis();
  if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
    lastStatusMs = now;

    int32_t pos[4];
    float vel[4];
    cdpr.getMotorState(pos, vel);

    float tx, ty;
    cdpr.getTarget(tx, ty);

    Serial.printf("target=(%.1f, %.1f) motors=[%ld, %ld, %ld, %ld] vel=[%.0f, %.0f, %.0f, %.0f]\n",
                  tx, ty,
                  (long)pos[0], (long)pos[1], (long)pos[2], (long)pos[3],
                  vel[0], vel[1], vel[2], vel[3]);

    digitalToggle(LED_BUILTIN);
  }
}
