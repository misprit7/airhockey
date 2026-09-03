// Host test for the vector velocity profile.
//
// Build/run:  make -C fw/test
//
// The profile decides where the machine goes, so it is worth checking without
// a Teensy in the loop. This drives the REAL law (fw/include/motion_profile.h)
// against the REAL cable kinematics (shared/cdpr_geometry.h) and asserts the
// properties the firmware depends on — above all the one that keeps the four
// motors in step with each other.
//
// The old per-axis law is reimplemented here purely as a baseline, so the
// claims about what changed are measured rather than asserted.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cdpr_geometry.h"
#include "motion_profile.h"

// Mirrors fw/include/cdpr_config.h. Not included directly: that header is
// Teensy-facing and pulls in Arduino types.
static const int COUNTS_PER_REV = 800;
static const float COUNTS_PER_MM = (float)COUNTS_PER_REV / SPOOL_CIRCUMFERENCE_MM;
static const float MAX_VELOCITY_MM_S = 12000.0f;
static const uint32_t TICK_RATE_HZ = 50000;
static const float DT = 1.0f / TICK_RATE_HZ;

static int g_failures = 0;

static void check(bool ok, const char *what, const char *detail = "") {
  if (!ok) {
    printf("  FAIL: %s %s\n", what, detail);
    g_failures++;
  }
}

// ── The old law, for comparison only ────────────────────────────────────────
static float trapezoidalStep(float pos, float vel, float target, float maxVel,
                             float maxAccel, float dt) {
  float err = target - pos;
  float absErr = fabsf(err);
  if (absErr < 0.001f && fabsf(vel) < 0.1f) return 0.0f;
  float sign = (err > 0) ? 1.0f : -1.0f;
  bool movingToward = (vel * sign > 0);
  float brakeDist = (vel * vel) / (2.0f * maxAccel);
  float desiredVel = (movingToward && brakeDist >= absErr) ? 0.0f : sign * maxVel;
  float dv = desiredVel - vel;
  float maxDv = maxAccel * dt;
  if (dv > maxDv) dv = maxDv;
  if (dv < -maxDv) dv = -maxDv;
  return vel + dv;
}

// ── One simulated run through the full tick, IK and stepping included ───────
struct Result {
  float maxSpeed;        // peak |v|
  float maxAccel;        // peak |dv|/dt
  float maxJerk;         // peak |da|/dt
  float maxCrossTrack;   // furthest the path strayed from the straight line
  int   maxStepErr;      // worst |counts owed| to any motor in any tick  <-- sync
  float finalDist;       // how close it parked
  long  ticks;
};

static Result run(float x0, float y0, float vx0, float vy0, float tx, float ty,
                  float vMax, float aMax, bool vectorLaw, long maxTicks = 4000000,
                  float ramp = MOTION_ACCEL_RAMP_S) {
  Result r = {0, 0, 0, 0, 0, 0, 0};

  float x = x0, y = y0, vx = vx0, vy = vy0;
  float ax_ = 0.0f, ay_ = 0.0f;

  // Straight line from start to target, for the cross-track measurement.
  const float lx = tx - x0, ly = ty - y0;
  const float lLen = sqrtf(lx * lx + ly * ly);

  // Motors start exactly on the IK of the start position, as begin() arranges.
  int32_t counts[NUM_MOTORS];
  float ref[NUM_MOTORS];
  for (int i = 0; i < NUM_MOTORS; i++) {
    ref[i] = cableLength(i, x0, y0, MALLET_THETA_RAD);
    counts[i] = 0;
  }

  for (long t = 0; t < maxTicks; t++) {
    const float pvx = vx, pvy = vy;

    if (vectorLaw) {
      float nvx, nvy, nax, nay;
      const float pax = ax_, pay = ay_;
      motionProfileStep(x, y, vx, vy, ax_, ay_, tx, ty, vMax, aMax, ramp, DT,
                        nvx, nvy, nax, nay);
      vx = nvx; vy = nvy; ax_ = nax; ay_ = nay;
      const float jk = sqrtf((nax-pax)*(nax-pax) + (nay-pay)*(nay-pay)) / DT;
      if (jk > r.maxJerk) r.maxJerk = jk;
    } else {
      vx = trapezoidalStep(x, vx, tx, vMax, aMax, DT);
      vy = trapezoidalStep(y, vy, ty, vMax, aMax, DT);
    }

    const float sp = sqrtf(vx * vx + vy * vy);
    if (sp > r.maxSpeed) r.maxSpeed = sp;
    const float ac = sqrtf((vx - pvx) * (vx - pvx) + (vy - pvy) * (vy - pvy)) / DT;
    if (ac > r.maxAccel) r.maxAccel = ac;

    // Advance the cart exactly as tick() does.
    const float rx = tx - x, ry = ty - y;
    const float distSq = rx * rx + ry * ry;
    const float speedSq = vx * vx + vy * vy;
    if (vectorLaw) {
      if (distSq < MOTION_POS_EPS_MM * MOTION_POS_EPS_MM &&
          speedSq < MOTION_VEL_EPS_MM_S * MOTION_VEL_EPS_MM_S) {
        x = tx; y = ty; vx = 0; vy = 0; ax_ = 0; ay_ = 0;
      } else {
        x += vx * DT; y += vy * DT;
      }
    } else {
      if (fabsf(tx - x) < 0.01f) { x = tx; vx = 0; } else { x += vx * DT; }
      if (fabsf(ty - y) < 0.01f) { y = ty; vy = 0; } else { y += vy * DT; }
    }

    if (lLen > 1e-3f) {
      const float cross = fabsf((x - x0) * ly - (y - y0) * lx) / lLen;
      if (cross > r.maxCrossTrack) r.maxCrossTrack = cross;
    }

    // IK + bang-bang stepping, exactly as tick() does it. The number that
    // matters is how many counts a motor is owed at the top of a tick: the
    // hardware can emit at most ONE. If this ever exceeds 1, that motor is
    // falling behind the cart while the others keep up, and the mallet leaves
    // the commanded path — that is what desynchronised motors look like.
    for (int i = 0; i < NUM_MOTORS; i++) {
      const float len = cableLength(i, x, y, MALLET_THETA_RAD);
      const int32_t want = (int32_t)lroundf((len - ref[i]) * COUNTS_PER_MM);
      const int32_t err = want - counts[i];
      const int a = abs((int)err);
      if (a > r.maxStepErr) r.maxStepErr = a;
      if (err != 0) counts[i] += (err > 0) ? 1 : -1;
    }

    r.ticks = t + 1;
    const float dx = tx - x, dy = ty - y;
    if (dx * dx + dy * dy < 1e-6f && vx == 0.0f && vy == 0.0f) break;
  }

  const float dx = tx - x, dy = ty - y;
  r.finalDist = sqrtf(dx * dx + dy * dy);
  return r;
}

int main() {
  const float cx = (WS_MIN_X + WS_MAX_X) * 0.5f;
  const float cy = (WS_MIN_Y + WS_MAX_Y) * 0.5f;

  printf("Vector motion profile — host tests\n");
  printf("workspace x[%.0f %.0f] y[%.0f %.0f], %.3f counts/mm, dt=%.1f us\n\n",
         WS_MIN_X, WS_MAX_X, WS_MIN_Y, WS_MAX_Y, COUNTS_PER_MM, DT * 1e6f);

  // ── 1. The diagonal, which is what the change is about ──────────────────
  //
  // Two shapes, because they fail differently. A 45-degree move is the worst
  // case for the CAPS (both axes run flat out, so the vector is sqrt(2) over)
  // but the best case for the PATH — equal distances means the axes finish
  // together and the line comes out straight anyway. An asymmetric move is the
  // reverse: the short axis arrives first and the cart crabs the remainder.
  // Testing only the symmetric one would have missed half the bug.
  //
  // Straightness is judged against one motor count of cable (~0.38 mm), since
  // deviation the machine cannot express is not deviation.
  {
    const float v = 3000.0f, a = 40000.0f;
    const float countMm = 1.0f / COUNTS_PER_MM;
    // minOldOverspeed differs by shape: at 45 degrees both axes run flat out
    // together so the vector hits the full sqrt(2); when one axis is shorter
    // it tops out earlier and the overlap is smaller. The overshoot is real in
    // both, just not the same size — asserting 141% everywhere would be
    // asserting something untrue.
    struct { const char *name; float x0, y0, x1, y1;
             float minOldOverspeed, minOldCrossTrackMm; } moves[] = {
      {"symmetric 45deg (500 x 500)", WS_MIN_X, WS_MIN_Y, WS_MAX_X, WS_MAX_Y,
       1.35f, 0.0f},
      {"asymmetric     (500 x 150)", WS_MIN_X, cy - 75.0f, WS_MAX_X, cy + 75.0f,
       1.10f, 10.0f},
    };
    printf("1. Diagonal moves, caps %.0f mm/s / %.0f mm/s^2"
           " (1 motor count = %.3f mm)\n", v, a, countMm);
    for (unsigned m = 0; m < 2; m++) {
      Result nw = run(moves[m].x0, moves[m].y0, 0, 0, moves[m].x1, moves[m].y1,
                      v, a, true);
      Result od = run(moves[m].x0, moves[m].y0, 0, 0, moves[m].x1, moves[m].y1,
                      v, a, false);
      printf("   %s\n", moves[m].name);
      printf("     old per-axis: |v|max %5.0f (%3.0f%%)  |a|max %6.0f (%3.0f%%)"
             "  cross-track %7.3f mm (%.1f counts)\n",
             od.maxSpeed, 100 * od.maxSpeed / v, od.maxAccel,
             100 * od.maxAccel / a, od.maxCrossTrack, od.maxCrossTrack / countMm);
      printf("     new vector:   |v|max %5.0f (%3.0f%%)  |a|max %6.0f (%3.0f%%)"
             "  cross-track %7.3f mm (%.1f counts)\n",
             nw.maxSpeed, 100 * nw.maxSpeed / v, nw.maxAccel,
             100 * nw.maxAccel / a, nw.maxCrossTrack, nw.maxCrossTrack / countMm);
      check(nw.maxSpeed <= v * 1.001f, "vector law exceeded the speed cap");
      check(nw.maxAccel <= a * 1.001f, "vector law exceeded the accel cap");
      check(nw.maxCrossTrack < countMm, "vector path strayed over a full count");
      check(od.maxAccel > a * 1.35f, "baseline should overshoot the accel cap");
      check(od.maxSpeed > v * moves[m].minOldOverspeed,
            "baseline should overshoot the speed cap");
      check(od.maxCrossTrack >= moves[m].minOldCrossTrackMm,
            "baseline should bend the path when the axes are unequal");
    }
    printf("\n");
  }

  // ── 2. Caps hold everywhere, including from a moving start ──────────────
  {
    printf("2. Cap compliance over the workspace, incl. adversarial entry velocity\n");
    const float v = MAX_VELOCITY_MM_S, a = 120000.0f;
    float worstV = 0, worstA = 0;
    int cases = 0;
    const float xs[] = {WS_MIN_X, cx, WS_MAX_X};
    const float ys[] = {WS_MIN_Y, cy, WS_MAX_Y};
    // Entry velocities including ones pointing away from and across the target.
    const float evs[][2] = {{0, 0}, {v, 0}, {-v, 0}, {0, v}, {0, -v},
                            {0.7f * v, 0.7f * v}, {-0.7f * v, 0.7f * v}};
    for (int sx = 0; sx < 3; sx++)
      for (int sy = 0; sy < 3; sy++)
        for (int ex = 0; ex < 3; ex++)
          for (int ey = 0; ey < 3; ey++)
            for (int e = 0; e < 7; e++) {
              if (sx == ex && sy == ey) continue;
              Result r = run(xs[sx], ys[sy], evs[e][0], evs[e][1],
                             xs[ex], ys[ey], v, a, true);
              if (r.maxSpeed > worstV) worstV = r.maxSpeed;
              if (r.maxAccel > worstA) worstA = r.maxAccel;
              check(r.maxSpeed <= v * 1.001f, "speed cap exceeded");
              check(r.maxAccel <= a * 1.001f, "accel cap exceeded");
              check(r.finalDist < 0.05f, "did not converge to target");
              cases++;
            }
    printf("   %d runs: worst |v| %.0f / %.0f cap, worst |a| %.0f / %.0f cap,"
           " all converged\n\n", cases, worstV, v, worstA, a);
  }

  // ── 3. THE SYNC INVARIANT ───────────────────────────────────────────────
  {
    printf("3. Step synchronisation: counts owed per motor per tick must be <= 1\n");
    const float v = MAX_VELOCITY_MM_S, a = 120000.0f;
    int worstNew = 0, worstOld = 0;
    const float xs[] = {WS_MIN_X, cx, WS_MAX_X};
    const float ys[] = {WS_MIN_Y, cy, WS_MAX_Y};
    for (int sx = 0; sx < 3; sx++)
      for (int sy = 0; sy < 3; sy++)
        for (int ex = 0; ex < 3; ex++)
          for (int ey = 0; ey < 3; ey++) {
            if (sx == ex && sy == ey) continue;
            Result n = run(xs[sx], ys[sy], 0, 0, xs[ex], ys[ey], v, a, true);
            Result o = run(xs[sx], ys[sy], 0, 0, xs[ex], ys[ey], v, a, false);
            if (n.maxStepErr > worstNew) worstNew = n.maxStepErr;
            if (o.maxStepErr > worstOld) worstOld = o.maxStepErr;
          }
    printf("   old per-axis: worst counts owed in one tick = %d\n", worstOld);
    printf("   new vector:   worst counts owed in one tick = %d\n", worstNew);
    check(worstNew <= 1, "MOTORS WOULD DESYNCHRONISE: a motor was owed >1 step");
    // Headroom: the ceiling is a cart speed of tickRate/countsPerMm.
    printf("   cart speed ceiling for 1 step/tick = %.0f mm/s; cap is %.0f"
           " (%.0f%% used)\n\n",
           TICK_RATE_HZ / COUNTS_PER_MM, MAX_VELOCITY_MM_S,
           100 * MAX_VELOCITY_MM_S / (TICK_RATE_HZ / COUNTS_PER_MM));
  }

  // ── 3b. PATH CONTAINMENT ────────────────────────────────────────────────
  //
  // A learned policy's commands flip between the box's two back corners
  // every few ticks. The target is always inside the box; the cart, turning
  // at speed, was not: at 12000 / 60000 the cart model swung 105 mm past the
  // end rail on the rig (2026-09-02). Replays that stream through the law
  // with and without containment.
  {
    printf("3b. Path containment: corner-flipping targets at 12000 / 60000\n");
    const float v = MAX_VELOCITY_MM_S, a = 60000.0f;
    float worstOut[2] = {0, 0};
    int worstStep[2] = {0, 0};
    float worstDv[2] = {0, 0};        // largest single-tick |dv|, mm/s
    for (int bounded = 0; bounded < 2; bounded++) {
      float x = cx, y = cy, vx = 0, vy = 0, ax_ = 0, ay_ = 0;
      int32_t counts[NUM_MOTORS];
      float ref[NUM_MOTORS];
      for (int i = 0; i < NUM_MOTORS; i++) {
        ref[i] = cableLength(i, x, y, MALLET_THETA_RAD);
        counts[i] = 0;
      }
      const long ticks = (long)(5.0f * TICK_RATE_HZ);     // 5 s
      for (long t = 0; t < ticks; t++) {
        // Flip every 15 ms between the two back corners, like the log.
        const bool low = ((t / (long)(0.015f * TICK_RATE_HZ)) % 2) == 0;
        const float tx = WS_MAX_X, ty = low ? WS_MIN_Y : WS_MAX_Y;
        const float pvx = vx, pvy = vy;
        if (bounded)
          motionProfileAdvanceBounded(x, y, vx, vy, ax_, ay_, tx, ty, v, a,
                                      MOTION_ACCEL_RAMP_S, DT,
                                      WS_MIN_X, WS_MAX_X, WS_MIN_Y, WS_MAX_Y);
        else
          motionProfileAdvance(x, y, vx, vy, ax_, ay_, tx, ty, v, a,
                               MOTION_ACCEL_RAMP_S, DT);
        const float dv = sqrtf((vx - pvx) * (vx - pvx) + (vy - pvy) * (vy - pvy));
        if (dv > worstDv[bounded]) worstDv[bounded] = dv;
        float out = 0;
        if (x > WS_MAX_X) out = x - WS_MAX_X;
        if (x < WS_MIN_X) out = fmaxf(out, WS_MIN_X - x);
        if (y > WS_MAX_Y) out = fmaxf(out, y - WS_MAX_Y);
        if (y < WS_MIN_Y) out = fmaxf(out, WS_MIN_Y - y);
        if (out > worstOut[bounded]) worstOut[bounded] = out;
        for (int i = 0; i < NUM_MOTORS; i++) {
          const float len = cableLength(i, x, y, MALLET_THETA_RAD);
          const int32_t want = (int32_t)lroundf((len - ref[i]) * COUNTS_PER_MM);
          const int32_t err = want - counts[i];
          const int ae = abs((int)err);
          if (ae > worstStep[bounded]) worstStep[bounded] = ae;
          if (err != 0) counts[i] += (err > 0) ? 1 : -1;
        }
      }
    }
    // The accel cap per tick, and the residual the backstop may eat: the
    // same aMax*ramp/2 bound the speed backstop carries.
    const float dvCap = a * DT;
    const float dvBackstop = a * MOTION_ACCEL_RAMP_S * 0.5f;
    printf("   unbounded law: cart left the box by up to %.0f mm, worst tick"
           " |dv| %.1f mm/s (cap %.1f)\n", worstOut[0], worstDv[0], dvCap);
    printf("   bounded law:   cart left the box by up to %.2f mm, worst tick"
           " |dv| %.1f mm/s (cap %.1f, backstop allowance %.0f), worst counts"
           " owed %d\n\n", worstOut[1], worstDv[1], dvCap, dvBackstop,
           worstStep[1]);
    check(worstOut[0] > 20.0f, "the unbounded law should leave the box here "
          "(else this test no longer exercises the case)");
    check(worstOut[1] <= 0.001f, "CART LEFT THE BOX with containment on");
    check(worstStep[1] <= 1, "MOTORS WOULD DESYNC under containment");
    // What the backstop may take in one tick: the jerk-lag residual. The
    // acceleration slews over `ramp`, so the accel vector can keep pointing
    // into the wall for up to that long after the demand reversed --
    // aMax*ramp of velocity at the very worst, half that typically. Anything
    // beyond it is a hard stop dressed as containment.
    check(worstDv[1] <= a * MOTION_ACCEL_RAMP_S,
          "WALL STOP EXCEEDED THE JERK-LAG RESIDUAL: containment is a hard "
          "stop, not a brake");
  }

  // ── 4. Short moves, where the braking curve does the work ───────────────
  {
    printf("4. Short moves (braking curve, never reaches the speed cap)\n");
    const float v = MAX_VELOCITY_MM_S, a = 20000.0f;
    const float dists[] = {0.05f, 0.5f, 5.0f, 50.0f};
    for (unsigned k = 0; k < sizeof(dists) / sizeof(dists[0]); k++) {
      const float d = dists[k];
      Result r = run(cx, cy, 0, 0, cx + d, cy, v, a, true);
      printf("   %6.2f mm -> settled %.4f mm from target in %.2f ms,"
             " |a|max %.0f (%.0f%% of cap)\n",
             d, r.finalDist, r.ticks * DT * 1e3f, r.maxAccel, 100 * r.maxAccel / a);
      check(r.finalDist < 0.02f, "short move did not converge");
      check(r.maxAccel <= a * 1.001f, "short move exceeded the accel cap");
    }
    printf("\n");
  }

  // ── 5. Jerk limiting ────────────────────────────────────────────────────
  //
  // The regression that matters here is the limit cycle. An earlier attempt
  // set the desired acceleration to (vDes - v)/dt, which at dt = 20 us
  // saturates at the cap for any velocity error above ~0.1 mm/s — a relay.
  // Slew-limiting a relay hunts instead of settling, and it did: every ramp
  // beyond ~1 ms failed to converge, worse at higher accel caps. Nothing else
  // in this file catches that, because at the default ramp the old code
  // happened to converge. So sweep cap AGAINST ramp and require settling
  // everywhere.
  {
    printf("5. Jerk limiting: acceleration slews rather than steps\n");
    const float v = MAX_VELOCITY_MM_S;
    const long LIM = 200000;              // 4 s of simulated time
    const float ramps[] = {0.001f, 0.003f, 0.008f, 0.030f};
    const float accels[] = {8000.0f, 40000.0f, 120000.0f};
    printf("   settling (must converge for every cap x ramp pair):\n");
    printf("        cap  ");
    for (unsigned k = 0; k < 4; k++) printf("%8.0fms", ramps[k] * 1000.0f);
    printf("\n");
    for (unsigned ai = 0; ai < 3; ai++) {
      printf("   %9.0f  ", accels[ai]);
      for (unsigned k = 0; k < 4; k++) {
        bool all_ok = true;
        const float sweep_d[] = {40.0f, 300.0f};
        for (unsigned di = 0; di < 2; di++) {
          const float d = sweep_d[di];
          Result r = run(cx - d / 2, cy, 0, 0, cx + d / 2, cy, v, accels[ai],
                         true, LIM, ramps[k]);
          if (r.ticks >= LIM || r.finalDist > 0.05f) all_ok = false;
          // the jerk cap itself, and the guarantee the backstop carries
          check(r.maxJerk <= (accels[ai] / ramps[k]) * 1.02f + 1.0f,
                "jerk cap exceeded");
          check(r.maxSpeed <= v * 1.001f, "speed cap exceeded under jerk lag");
          check(r.maxStepErr <= 1, "MOTORS WOULD DESYNC under jerk limiting");
        }
        printf("%10s", all_ok ? "ok" : "HUNT");
        check(all_ok, "profile did not settle — limit cycle regression");
      }
      printf("\n");
    }
    // What it costs, so the number in the header stays honest.
    printf("\n   move-time cost vs a 0.2 ms ramp, cap 8000 mm/s^2:\n");
    const float cost_d[] = {25.0f, 100.0f, 500.0f};
    for (unsigned di = 0; di < 3; di++) {
      const float d = cost_d[di];
      Result base = run(cx - d / 2, cy, 0, 0, cx + d / 2, cy, v, 8000.0f,
                        true, LIM, 0.0002f);
      printf("     %5.0f mm:", d);
      for (unsigned k = 0; k < 4; k++) {
        Result r = run(cx - d / 2, cy, 0, 0, cx + d / 2, cy, v, 8000.0f,
                       true, LIM, ramps[k]);
        printf("  %4.0fms %+6.1f%%", ramps[k] * 1000.0f,
               100.0f * ((float)r.ticks / (float)base.ticks - 1.0f));
      }
      printf("\n");
    }
    printf("\n");
  }

  if (g_failures) {
    printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
  }
  printf("All checks passed.\n");
  return 0;
}
