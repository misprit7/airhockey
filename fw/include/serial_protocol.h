#pragma once

// ============================================================================
// Serial protocol between Teensy (fw/) and desktop master (sw/)
//
// USB Serial, 115200 baud, text-based, newline-terminated.
// All coordinates in millimeters, velocities in mm/s.
//
// ── Desktop → Teensy (commands) ──────────────────────────────────────
//
//   CMD x y          Set target position (mm). Teensy motion controller
//                    handles trajectory planning and stepping.
//
//   TENSION mm       Retract all cables by mm to add pretension.
//                    Blocking on Teensy side. Send before enabling motion.
//
//   RELEASE          Release previously applied tension.
//                    Blocking on Teensy side. Send after stopping motion.
//
//   START            Start the motion controller timer (50kHz ISR).
//                    Must be called after TENSION and before CMD.
//
//   STOP             Stop the motion controller timer.
//                    Motors hold position but no new steps are generated.
//
//   CAL x y          Set calibration position (mm). Tells the Teensy
//                    where the paddle currently is. Resets all motor counts.
//                    Must be called before START.
//
//   STATUS           Request an immediate status line (in addition to
//                    the periodic ones).
//
// ── Teensy → Desktop (responses) ─────────────────────────────────────
//
//   OK cmd           Acknowledgment of a command.
//
//   ERR msg          Error message.
//
//   S x y vx vy c0 c1 c2 c3
//                    Status line. Sent periodically (~50Hz) and on STATUS
//                    request. Fields:
//                      x y      - theoretical cart position (mm)
//                      vx vy    - cart velocity (mm/s)
//                      c0..c3   - motor step counts (from calibration ref)
//
// ============================================================================
