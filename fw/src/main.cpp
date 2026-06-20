#include <Arduino.h>
#include "cdpr.h"
#include "serial_protocol.h"

// ============================================================================
// Motor control pins
// ============================================================================
//
// Each pin array is a contiguous block of 4 pins (one per motor). Side A
// drives the active CDPR controller on this half of the table. Side B is
// reserved for the other half of the table and is not wired into a
// controller yet — kept here so the second side can reuse this firmware.
//
//   STEPA  6..9     DIRA  34..37   (active)
//   STEPB  14..17   DIRB  18..21   (reserved for the other side)

static const int stepPinsA[NUM_MOTORS] = {6, 7, 8, 9};
static const int dirPinsA[NUM_MOTORS]  = {34, 35, 36, 37};

[[maybe_unused]] static const int stepPinsB[NUM_MOTORS] = {14, 15, 16, 17};
[[maybe_unused]] static const int dirPinsB[NUM_MOTORS]  = {18, 19, 20, 21};

static CDPR cdpr(stepPinsA, dirPinsA);

// ============================================================================
// Mode select
// ============================================================================
//
// TEST_MODE true  → autonomous 3 cm square test on boot (no host needed).
// TEST_MODE false → normal serial-command controller (see serial_protocol.h).

constexpr bool TEST_MODE = true;

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
// State
// ============================================================================

static bool timerRunning = false;
static bool calibrated   = false;

// ============================================================================
// Serial command parsing
// ============================================================================

static char cmdBuf[128];
static int  cmdLen = 0;

static void sendStatus() {
  float x, y, vx, vy;
  int32_t counts[NUM_MOTORS];
  cdpr.getCartPosition(x, y);
  cdpr.getCartVelocity(vx, vy);
  cdpr.getMotorCounts(counts);
  Serial.printf("S %.2f %.2f %.2f %.2f %ld %ld %ld %ld\n",
                x, y, vx, vy,
                (long)counts[0], (long)counts[1],
                (long)counts[2], (long)counts[3]);
}

static void processCommand(char *line) {
  // Skip leading whitespace
  while (*line == ' ' || *line == '\t') line++;
  if (*line == '\0') return;

  // Parse first token
  char *cmd = line;
  char *args = line;
  while (*args && *args != ' ' && *args != '\t') args++;
  if (*args) {
    *args = '\0';
    args++;
    while (*args == ' ' || *args == '\t') args++;
  }

  if (strcasecmp(cmd, "CMD") == 0) {
    if (!timerRunning) {
      Serial.println("ERR timer not running");
      return;
    }
    float x, y;
    if (sscanf(args, "%f %f", &x, &y) != 2) {
      Serial.println("ERR CMD requires x y");
      return;
    }
    cdpr.setTarget(x, y);
    Serial.println("OK CMD");

  } else if (strcasecmp(cmd, "TENSION") == 0) {
    if (timerRunning) {
      Serial.println("ERR stop timer before tensioning");
      return;
    }
    float mm;
    if (sscanf(args, "%f", &mm) != 1) {
      Serial.println("ERR TENSION requires mm");
      return;
    }
    cdpr.tension(mm);
    Serial.println("OK TENSION");

  } else if (strcasecmp(cmd, "RELEASE") == 0) {
    if (timerRunning) {
      Serial.println("ERR stop timer before releasing");
      return;
    }
    cdpr.releaseTension();
    Serial.println("OK RELEASE");

  } else if (strcasecmp(cmd, "START") == 0) {
    if (timerRunning) {
      Serial.println("ERR already running");
      return;
    }
    if (!calibrated) {
      Serial.println("ERR not calibrated");
      return;
    }
    cdpr.startTimer();
    timerRunning = true;
    Serial.println("OK START");

  } else if (strcasecmp(cmd, "STOP") == 0) {
    if (!timerRunning) {
      Serial.println("ERR not running");
      return;
    }
    cdpr.stopTimer();
    timerRunning = false;
    Serial.println("OK STOP");

  } else if (strcasecmp(cmd, "CAL") == 0) {
    if (timerRunning) {
      Serial.println("ERR stop timer before calibrating");
      return;
    }
    float x = TABLE_WIDTH / 2.0f;
    float y = TABLE_HEIGHT / 2.0f;
    // Optional x y arguments; default to table center
    sscanf(args, "%f %f", &x, &y);
    cdpr.begin(x, y);
    calibrated = true;
    Serial.println("OK CAL");

  } else if (strcasecmp(cmd, "STATUS") == 0) {
    sendStatus();

  } else {
    Serial.print("ERR unknown command: ");
    Serial.println(cmd);
  }
}

// ============================================================================
// 3 cm square test
//
// Autonomous test: calibrate at table center, then trace a 30 mm square
// indefinitely, pausing briefly at each corner. No host commands needed.
// ============================================================================

constexpr float SQUARE_SIZE_MM = 30.0f; // 3 cm side length
constexpr float TENSION_MM     = 0.0f;  // pretension before motion (0 = none)
constexpr uint32_t DWELL_MS    = 500;   // pause at each corner
constexpr float TARGET_TOL_MM  = 0.5f;  // "arrived" tolerance

static float cornerX[4];
static float cornerY[4];
static int   cornerIdx = 0;
static uint32_t dwellUntil = 0; // 0 = not currently dwelling

static void startSquareTest() {
  const float cx = TABLE_WIDTH / 2.0f;
  const float cy = TABLE_HEIGHT / 2.0f;
  const float h  = SQUARE_SIZE_MM / 2.0f;

  // Square corners centered on the calibration point, traversed CCW.
  cornerX[0] = cx - h; cornerY[0] = cy - h;
  cornerX[1] = cx + h; cornerY[1] = cy - h;
  cornerX[2] = cx + h; cornerY[2] = cy + h;
  cornerX[3] = cx - h; cornerY[3] = cy + h;

  // Paddle is assumed to physically start at table center.
  cdpr.begin(cx, cy);
  if (TENSION_MM > 0.0f) cdpr.tension(TENSION_MM);
  cdpr.startTimer();
  timerRunning = true;
  calibrated   = true;

  cdpr.setTarget(cornerX[cornerIdx], cornerY[cornerIdx]);
  Serial.printf("Square test: %.0fmm square centered at (%.1f, %.1f)\n",
                SQUARE_SIZE_MM, cx, cy);
}

static void squareTestLoop() {
  if (cdpr.atTarget(TARGET_TOL_MM)) {
    if (dwellUntil == 0) {
      dwellUntil = millis() + DWELL_MS;
    } else if (millis() >= dwellUntil) {
      dwellUntil = 0;
      cornerIdx = (cornerIdx + 1) % 4;
      cdpr.setTarget(cornerX[cornerIdx], cornerY[cornerIdx]);
      digitalToggle(LED_BUILTIN);
    }
  }
}

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(RESET_PIN, INPUT_PULLUP);

  if (TEST_MODE) {
    startSquareTest();
  } else {
    Serial.println("CDPR ready");
  }
}

// ============================================================================
// Main loop
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 20;  // ~50Hz

void loop() {
  checkReset();

  if (TEST_MODE) {
    squareTestLoop();
    return;
  }

  // Read serial commands
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        processCommand(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < (int)sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }

  // Periodic status at ~50Hz when timer is running
  if (timerRunning) {
    uint32_t now = millis();
    if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
      lastStatusMs = now;
      sendStatus();
    }
  }
}
