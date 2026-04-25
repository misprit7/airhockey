#include <Arduino.h>
#include "cdpr.h"
#include "serial_protocol.h"

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
// Setup
// ============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(RESET_PIN, INPUT_PULLUP);

  Serial.println("CDPR ready");
}

// ============================================================================
// Main loop
// ============================================================================

static uint32_t lastStatusMs = 0;
constexpr uint32_t STATUS_INTERVAL_MS = 20;  // ~50Hz

void loop() {
  checkReset();

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
