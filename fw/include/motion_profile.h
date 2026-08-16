#pragma once

#include <math.h>
#include <stdint.h>

// ============================================================================
// Vector velocity profile
//
// One profile along the direction of travel, replacing the two independent
// per-axis trapezoids this used to run.
//
// The per-axis version capped |vx| and |vy| separately, so a 45-degree move
// ran at sqrt(2) x the speed cap and sqrt(2) x the accel cap — 41% over both,
// in the one direction the rig is asked to move most. It also bent the path:
// whichever axis had less distance to cover finished first and the cart
// crabbed the rest of the way, so a "straight" move was two sides of a
// triangle. Neither is a tuning problem; they are what decomposing a vector
// into axes and limiting the pieces does.
//
// Here the cap applies to the MAGNITUDE of the velocity and of its change, so
// both hold in every direction, and the desired velocity always points
// straight at the target.
//
// Deliberately kept free of Arduino.h so the law can be compiled and tested on
// the host — see fw/test/test_motion_profile.cpp. It is the piece that decides
// where the machine goes, so it should be checkable without a Teensy.
// ============================================================================

// Which cap bound the last step. One profile, not one per axis.
constexpr uint8_t MOTION_LIMIT_ACCEL = 1u << 0;
constexpr uint8_t MOTION_LIMIT_SPEED = 1u << 1;
constexpr uint8_t MOTION_LIMIT_JERK = 1u << 2;

// ── Jerk limiting ─────────────────────────────────────────────────────────
//
// Acceleration is slewed rather than stepped. Without this the profile puts
// full acceleration on in ONE 20 us tick, which does two things to a
// cable-driven rig: it applies the tipping moment impulsively, and it steps
// an elastic system, which overshoots the steady-state tension by up to 2x.
// The paddle is pulled 32.7 mm above the surface over a 50.4 mm radius, so
// it tips at about g*r/h ~ 1.5 g — the 2x overshoot is the difference between
// tipping and not. (It was 0.8 g when the attachment sat at 49 mm and the
// radius was assumed to be 40; lowering h is the only lever that moves this,
// and it bought a factor of 1.5.)
//
// Parameterised as a RAMP TIME rather than an absolute jerk, so the shape of
// the move does not change when the accel cap does: jerk = aMax / ramp. The
// velocity overshoot past vMax then grows as aMax*ramp/2 (linear) instead of
// aMax^2/2J (quadratic) — 180 mm/s at the 120000 ceiling and a 3 ms ramp,
// against 6850 mm/s of headroom below the step-rate ceiling.
constexpr float MOTION_ACCEL_RAMP_S = 0.003f;

// The velocity loop needs a time constant of its own, and this is the whole
// reason an earlier attempt at this did not work. The desired acceleration
// used to be (vDes - v)/dt, which with dt = 20 us saturates at +-aMax for any
// velocity error above ~0.1 mm/s — i.e. the acceleration command was
// bang-bang. Rate-limiting a relay is the textbook way to build a limit
// cycle, and it duly hunted around the target instead of settling, at every
// ramp longer than about 1 ms. Making the demand PROPORTIONAL to velocity
// error gives the slew something smooth to track, and every combination of
// cap and ramp then converges.
constexpr float MOTION_VEL_TAU_MULT = 2.0f;

// sqrt(2*a*d) is the fastest approach that can still stop within d. Riding it
// exactly is marginally stable — any quantisation error puts the cart on the
// wrong side of a curve it can no longer brake off, i.e. overshoot. Backing
// off by this factor means the stop only ever demands GAIN^2 (0.64) of the
// accel cap, leaving 36% for the tick discretisation to eat.
constexpr float MOTION_APPROACH_GAIN = 0.8f;

// Below this distance the target counts as reached and the profile commands
// zero. One motor count is ~0.38 mm of cable, so this is ~40x finer than the
// machine can physically resolve; it exists to stop the sqrt chattering near
// zero, not to bound accuracy.
constexpr float MOTION_POS_EPS_MM = 0.01f;

// Speed below which the cart may be parked exactly on the target.
constexpr float MOTION_VEL_EPS_MM_S = 0.5f;

// Advance the velocity and acceleration vectors one tick toward `t`, and
// report which cap bound.
//
// Acceleration is now STATE — pass the previous value back in. It has to be,
// because bounding its rate of change is the whole point; a stateless version
// has nothing to slew from.
//
// GUARANTEE (relied on for step synchronisation, see CDPR::tick):
//   |vOut| <= vMax, unconditionally.
// This used to follow from convexity — old and desired velocity both lay in
// the disk of radius vMax and the output was on the segment between them.
// That argument DIES with jerk limiting, because the acceleration now lags
// and can still be pointing outward when the demand has already reversed, so
// the output is no longer on that segment. The final clamp below is what
// carries the guarantee instead. It is a backstop, not a working part: the
// overshoot it catches is bounded by aMax*ramp/2, ~180 mm/s at the ceiling,
// and the step-rate headroom is 6850. If it ever fires hard, the ramp is too
// long for the accel cap and the profile shape is not what you think.
inline uint8_t motionProfileStep(float px, float py, float vx, float vy,
                                 float ax, float ay, float tx, float ty,
                                 float vMax, float aMax, float rampS, float dt,
                                 float &vxOut, float &vyOut,
                                 float &axOut, float &ayOut) {
  uint8_t flags = 0;

  const float ex = tx - px;
  const float ey = ty - py;
  const float dist = sqrtf(ex * ex + ey * ey);

  // Desired velocity: straight at the target, at the speed cap until close
  // enough that stopping needs the whole accel budget, then down the braking
  // curve.
  float vdx = 0.0f, vdy = 0.0f;
  if (dist > MOTION_POS_EPS_MM) {
    float vDes = MOTION_APPROACH_GAIN * sqrtf(2.0f * aMax * dist);
    if (vDes > vMax) {
      vDes = vMax;
      flags |= MOTION_LIMIT_SPEED;
    }
    const float scale = vDes / dist;   // (e/|e|) * vDes, one divide
    vdx = ex * scale;
    vdy = ey * scale;
  }

  // Desired acceleration: proportional to the velocity error, NOT the error
  // divided by dt. See MOTION_VEL_TAU_MULT — dividing by dt saturates and
  // makes this a relay, which the jerk slew below then turns into a limit
  // cycle. Magnitude-clipped, so the cap holds in every direction.
  const float tau = MOTION_VEL_TAU_MULT * rampS;
  float adx = (vdx - vx) / tau;
  float ady = (vdy - vy) / tau;
  const float adMag = sqrtf(adx * adx + ady * ady);
  if (adMag > aMax) {
    const float s = aMax / adMag;
    adx *= s;
    ady *= s;
    flags |= MOTION_LIMIT_ACCEL;
  }

  // Slew the acceleration VECTOR toward that demand. Bounding the magnitude
  // of the change rather than each component is what makes the jerk cap hold
  // through a turn as well as along a straight line.
  float dax = adx - ax;
  float day = ady - ay;
  const float daMag = sqrtf(dax * dax + day * day);
  const float daMax = (aMax / rampS) * dt;
  if (daMag > daMax) {
    const float s = daMax / daMag;
    dax *= s;
    day *= s;
    flags |= MOTION_LIMIT_JERK;
  }
  axOut = ax + dax;
  ayOut = ay + day;

  vxOut = vx + axOut * dt;
  vyOut = vy + ayOut * dt;

  // Backstop — see the guarantee above.
  const float vMag = sqrtf(vxOut * vxOut + vyOut * vyOut);
  if (vMag > vMax) {
    const float s = vMax / vMag;
    vxOut *= s;
    vyOut *= s;
    flags |= MOTION_LIMIT_SPEED;
  }
  return flags;
}
