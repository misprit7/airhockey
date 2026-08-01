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
// MODE_SERIAL  → normal serial-command controller (see serial_protocol.h).
// MODE_SQUARE  → 5 cm square, ONE lap, armed on boot and started by
//                typing GO on serial. No host needed.
// MODE_RETRACT → step/dir direction check, mirroring sw/bin/retract_test.cpp:
//                eight moves (each motor extends 10 mm, then retracts it
//                back), ONE move per "GO" over serial. Never moves
//                unattended. Verifies RETRACT_CW / DIR_CW_LEVEL.
// MODE_DIRTOGGLE → hardware probe: all four DIR pins toggle together at
//                0.5 Hz (LED in sync) for multimeter tracing. STEP pins held
//                LOW — cannot cause motion.

enum Mode { MODE_SERIAL, MODE_SQUARE, MODE_RETRACT, MODE_DIRTOGGLE };
constexpr Mode ACTIVE_MODE = MODE_SQUARE;

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
    float x = HOME_X;
    float y = HOME_Y;
    // Optional x y arguments; default to robot-half center
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

constexpr float SQUARE_DEFAULT_MM = 50.0f;  // 5 cm side length
constexpr float SQUARE_MAX_MM     = 800.0f; // sanity cap, workspace is ~945
constexpr float SQUARE_SPEED_MM_S = 25.0f;  // slow: first closed-loop motion
constexpr float TENSION_MM        = 0.0f;   // pretension before motion
constexpr uint32_t DWELL_MS       = 700;    // pause at each corner
constexpr float TARGET_TOL_MM     = 0.5f;   // "arrived" tolerance

// One lap: centre -> SW -> SE -> NE -> NW -> SW (closes the square) -> centre.
// Deliberately not a repeating loop — each lap is started by hand.
constexpr int SQUARE_WAYPOINTS = 6;
static float wpX[SQUARE_WAYPOINTS];
static float wpY[SQUARE_WAYPOINTS];
static int   wpIdx = 0;
static uint32_t dwellUntil = 0;
static bool  squareRunning = false;
static bool  squareInited = false;   // begin() called at least once
static float squareSize = SQUARE_DEFAULT_MM;
static int   lapCount = 0;

static void printSquareHelp() {
  Serial.println();
  Serial.printf("SQUARE TEST  |  size %.0f mm  |  %.0f mm/s  |  centre (%.1f, %.1f)\n",
                squareSize, SQUARE_SPEED_MM_S, HOME_X, HOME_Y);
  Serial.println("  GO        run one lap");
  Serial.println("  SIZE <mm> change the square size");
  Serial.println("  CAL       re-zero here (use after repositioning by hand)");
  Serial.println("  STOP      abort a lap in progress");
  Serial.println("Nothing moves until you type GO.");
}

static void armSquareTest() {
  printSquareHelp();
  Serial.println("Place the paddle near the centre and take up cable slack.");
  Serial.println("Placement need not be exact: it offsets the square but");
  Serial.println("barely changes its SHAPE, which is what you measure.");
}

// Zero the controller's reference at the nominal centre. Only do this when
// the paddle has actually been put there — every call throws away whatever
// drift previous laps accumulated, which is usually the thing you wanted to
// see.
static void calibrateHere() {
  cdpr.begin(HOME_X, HOME_Y);
  squareInited = true;
  calibrated = true;
  Serial.printf("Reference zeroed at (%.1f, %.1f)\n", HOME_X, HOME_Y);
}

static void startSquareLap() {
  const float cx = HOME_X, cy = HOME_Y, h = squareSize / 2.0f;
  wpX[0] = cx - h; wpY[0] = cy - h;   // SW
  wpX[1] = cx + h; wpY[1] = cy - h;   // SE
  wpX[2] = cx + h; wpY[2] = cy + h;   // NE
  wpX[3] = cx - h; wpY[3] = cy + h;   // NW
  wpX[4] = cx - h; wpY[4] = cy - h;   // SW again, closing the square
  wpX[5] = cx;     wpY[5] = cy;       // back to the start point

  if (!squareInited) calibrateHere();
  cdpr.setVelocityLimit(SQUARE_SPEED_MM_S);
  if (TENSION_MM > 0.0f) cdpr.tension(TENSION_MM);
  cdpr.startTimer();
  timerRunning = true;
  squareRunning = true;
  wpIdx = 0;
  dwellUntil = 0;
  cdpr.setTarget(wpX[0], wpY[0]);
  Serial.printf("LAP %d RUNNING (%.0f mm) -> 1/%d (%.1f, %.1f)\n",
                lapCount + 1, squareSize, SQUARE_WAYPOINTS, wpX[0], wpY[0]);
}

static void endLap(const char *why) {
  cdpr.stopTimer();
  timerRunning = false;
  squareRunning = false;
  float x, y;
  cdpr.getCartPosition(x, y);
  int32_t counts[NUM_MOTORS];
  cdpr.getMotorCounts(counts);
  Serial.printf("%s - controller position (%.1f, %.1f), counts [%ld %ld %ld %ld]\n",
                why, x, y, (long)counts[0], (long)counts[1], (long)counts[2],
                (long)counts[3]);
  // After a full lap those counts are zero by construction, so a nonzero
  // reading means the lap was cut short, not that the machine drifted.
  // Physical drift shows up only by measuring where the paddle really is.
  Serial.println("Measure where the paddle ACTUALLY is, then GO again.");
}

// Reads one line; returns true when a complete line is ready in buf.
static bool readSquareLine(char *buf, uint8_t cap, uint8_t &n) {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (n == 0) continue;  // swallow the second half of CRLF / blank lines
      buf[n] = 0;
      n = 0;
      return true;
    }
    if (n < cap - 1) buf[n++] = c;
  }
  return false;
}

static void handleSquareCommand(char *line) {
  if (strcasecmp(line, "GO") == 0) {
    if (squareRunning) {
      Serial.println("already running - STOP first");
      return;
    }
    startSquareLap();
  } else if (strcasecmp(line, "STOP") == 0) {
    if (!squareRunning) {
      Serial.println("not running");
      return;
    }
    endLap("ABORTED");
  } else if (strcasecmp(line, "CAL") == 0) {
    if (squareRunning) {
      Serial.println("cannot re-zero mid-lap - STOP first");
      return;
    }
    calibrateHere();
  } else if (strncasecmp(line, "SIZE", 4) == 0) {
    if (squareRunning) {
      Serial.println("cannot resize mid-lap - STOP first");
      return;
    }
    float v = atof(line + 4);
    if (v < 5.0f || v > SQUARE_MAX_MM) {
      Serial.printf("SIZE must be 5..%.0f mm\n", SQUARE_MAX_MM);
      return;
    }
    // Corners must sit inside the workspace or setTarget silently clamps
    // them and the "square" comes out a rectangle.
    const float h = v / 2.0f;
    if (!inWorkspace(HOME_X - h, HOME_Y - h) ||
        !inWorkspace(HOME_X + h, HOME_Y + h)) {
      Serial.printf("%.0f mm square does not fit in the workspace\n", v);
      return;
    }
    squareSize = v;
    Serial.printf("size now %.0f mm\n", squareSize);
  } else {
    printSquareHelp();
  }
}

static void squareTestLoop() {
  static char buf[24];
  static uint8_t n = 0;
  if (readSquareLine(buf, sizeof(buf), n)) handleSquareCommand(buf);

  if (!squareRunning) return;
  if (!cdpr.atTarget(TARGET_TOL_MM)) return;

  if (dwellUntil == 0) {
    dwellUntil = millis() + DWELL_MS;
    return;
  }
  if (millis() < dwellUntil) return;
  dwellUntil = 0;
  digitalToggle(LED_BUILTIN);

  wpIdx++;
  if (wpIdx >= SQUARE_WAYPOINTS) {
    lapCount++;
    endLap("LAP COMPLETE");
    return;
  }
  cdpr.setTarget(wpX[wpIdx], wpY[wpIdx]);
  Serial.printf("-> %d/%d (%.1f, %.1f)\n", wpIdx + 1, SQUARE_WAYPOINTS,
                wpX[wpIdx], wpY[wpIdx]);
}

// ============================================================================
// Per-motor retract test (MODE_RETRACT)
//
// Direction check for the step/dir path. Eight moves per pass: each motor
// extends RETRACT_TEST_MM of cable, then retracts it back. One move per
// explicit "GO" over serial — nothing moves unattended. Extend always comes
// first: with the other three motors holding, the mallet is force-closed and
// a lone retraction against taut strings just spikes tension. Keep a few cm
// of slack in every string.
// ============================================================================

constexpr float RETRACT_TEST_MM   = 10.0f; // cable travel per move (~28° of spool)
constexpr float RETRACT_TEST_MM_S = 5.0f;  // slow: ~2 s per move

static uint32_t lastPromptMs = 0;

// Step one motor `steps` counts in the given DIR level at speed_mm_s.
// Blocking; drives the A-side pins directly (no CDPR ISR involved).
static void pulseMotor(int motor, int dirLevel, int32_t steps,
                       float speed_mm_s) {
  uint32_t intervalUs = (uint32_t)(1e6f / (speed_mm_s * COUNTS_PER_MM));
  digitalWriteFast(dirPinsA[motor], dirLevel);
  delayMicroseconds(2);
  for (int32_t s = 0; s < steps; s++) {
    digitalWriteFast(stepPinsA[motor], HIGH);
    delayMicroseconds(2);
    digitalWriteFast(stepPinsA[motor], LOW);
    delayMicroseconds(intervalUs);
  }
}

// Next move: motor retractPhase/2, extend if even, retract if odd.
static int retractPhase = 0;

static void printNextMove() {
  int m = retractPhase / 2;
  bool extend = (retractPhase % 2) == 0;
  Serial.printf("Next: motor %d %s %.0f mm (spool should %s, %s). Send GO to run it.\n",
                m, extend ? "EXTEND" : "RETRACT", RETRACT_TEST_MM,
                extend ? "pay out" : "wind in",
                (RETRACT_CW[m] != extend) ? "CW" : "CCW");
}

static void runRetractPhase() {
  const int32_t steps = (int32_t)roundf(mmToCounts(RETRACT_TEST_MM));
  int m = retractPhase / 2;
  bool extend = (retractPhase % 2) == 0;
  Serial.printf("Motor %d: %s %.0f mm... ", m,
                extend ? "extending" : "retracting", RETRACT_TEST_MM);
  pulseMotor(m, extend ? dirLevelExtend(m) : dirLevelRetract(m), steps,
             RETRACT_TEST_MM_S);
  Serial.println("done");
  retractPhase = (retractPhase + 1) % (2 * NUM_MOTORS);
  if (retractPhase == 0)
    Serial.println("Pass complete - all motors exercised. GO starts a new pass.");
  printNextMove();
}

static void retractTestLoop() {
  if (millis() - lastPromptMs > 5000) {
    lastPromptMs = millis();
    printNextMove();
  }

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen > 0) {
        cmdBuf[cmdLen] = '\0';
        if (strcasecmp(cmdBuf, "GO") == 0) {
          runRetractPhase();
          while (Serial.available()) Serial.read(); // drop input typed mid-move
          lastPromptMs = millis();
        } else {
          printNextMove();
        }
        cmdLen = 0;
      }
    } else if (cmdLen < (int)sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }
}

// ============================================================================
// DIR toggle probe (MODE_DIRTOGGLE)
//
// Toggles every A-side DIR pin HIGH/LOW at 0.5 Hz (1 s per level) so the
// lines can be traced with a multimeter. The LED mirrors the level and the
// state is printed over serial. STEP pins are driven constantly LOW.
// ============================================================================

constexpr uint32_t DIR_TOGGLE_HALF_PERIOD_MS = 1000; // 0.5 Hz

static bool dirToggleLevel   = false;
static uint32_t dirToggleLastMs = 0;

static void dirToggleLoop() {
  uint32_t now = millis();
  if (now - dirToggleLastMs >= DIR_TOGGLE_HALF_PERIOD_MS) {
    dirToggleLastMs = now;
    dirToggleLevel = !dirToggleLevel;
    for (int i = 0; i < NUM_MOTORS; i++) {
      digitalWriteFast(dirPinsA[i], dirToggleLevel ? HIGH : LOW);
    }
    digitalWriteFast(LED_BUILTIN, dirToggleLevel ? HIGH : LOW);
    Serial.printf("DIR pins %d %d %d %d: %s\n",
                  dirPinsA[0], dirPinsA[1], dirPinsA[2], dirPinsA[3],
                  dirToggleLevel ? "HIGH (3.3V)" : "LOW (0V)");
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

  if (ACTIVE_MODE == MODE_SQUARE) {
    armSquareTest();
  } else if (ACTIVE_MODE == MODE_RETRACT) {
    // Init step/dir pins directly — the CDPR controller is not used here.
    for (int i = 0; i < NUM_MOTORS; i++) {
      pinMode(stepPinsA[i], OUTPUT);
      pinMode(dirPinsA[i], OUTPUT);
      digitalWriteFast(stepPinsA[i], LOW);
      digitalWriteFast(dirPinsA[i], LOW);
    }
    Serial.println("RETRACT TEST armed - one move per GO");
    printNextMove();
  } else if (ACTIVE_MODE == MODE_DIRTOGGLE) {
    for (int i = 0; i < NUM_MOTORS; i++) {
      pinMode(stepPinsA[i], OUTPUT);
      pinMode(dirPinsA[i], OUTPUT);
      digitalWriteFast(stepPinsA[i], LOW);
      digitalWriteFast(dirPinsA[i], LOW);
    }
    Serial.println("DIR TOGGLE probe: DIR pins toggle at 0.5 Hz, LED in sync");
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

  if (ACTIVE_MODE == MODE_SQUARE) {
    squareTestLoop();
    return;
  }
  if (ACTIVE_MODE == MODE_RETRACT) {
    retractTestLoop();
    return;
  }
  if (ACTIVE_MODE == MODE_DIRTOGGLE) {
    dirToggleLoop();
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
