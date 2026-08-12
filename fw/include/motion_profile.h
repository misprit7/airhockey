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

// Which cap bound the last step. Two bits now, not four: there is one profile,
// not one per axis.
constexpr uint8_t MOTION_LIMIT_ACCEL = 1u << 0;
constexpr uint8_t MOTION_LIMIT_SPEED = 1u << 1;

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

// Advance the velocity vector one tick toward `t`, and report which cap bound.
//
// GUARANTEE (relied on for step synchronisation, see CDPR::tick):
//   if |v| <= vMax on entry then |vOut| <= vMax on exit.
// Both the current and the desired velocity lie in the disk of radius vMax,
// and the output is a point on the segment between them; a disk is convex, so
// the segment cannot leave it. Starting from rest, |v| <= vMax holds forever
// by induction. If vMax is LOWERED mid-flight the cart is briefly outside the
// new disk, and the same argument makes it converge back monotonically without
// ever exceeding the old cap.
inline uint8_t motionProfileStep(float px, float py, float vx, float vy,
                                 float tx, float ty, float vMax, float aMax,
                                 float dt, float &vxOut, float &vyOut) {
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

  // Bound the change in the velocity VECTOR rather than each component. This
  // is both what holds |a| <= aMax in every direction and what lets the law
  // work from a moving start: dv corrects magnitude and heading together, so
  // there is no assumption anywhere that the cart began at rest.
  float dvx = vdx - vx;
  float dvy = vdy - vy;
  const float dvMag = sqrtf(dvx * dvx + dvy * dvy);
  const float dvMax = aMax * dt;
  if (dvMag > dvMax) {
    const float s = dvMax / dvMag;
    dvx *= s;
    dvy *= s;
    flags |= MOTION_LIMIT_ACCEL;
  }

  vxOut = vx + dvx;
  vyOut = vy + dvy;
  return flags;
}
