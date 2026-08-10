#pragma once

#include <math.h>

// ════════════════════════════════════════════════════════════════════════
// CANONICAL CDPR geometry and cable kinematics.
//
// This is the ONE definition of where the machine's parts are and how
// cable length relates to paddle position. Its consumers:
//
//   fw/include/cdpr_config.h   Teensy step/dir control — ALL motion runs
//                              here, and this is the only code that turns
//                              paddle position into motor counts
//   sw/bin/cdpr_master.cpp     host bridge: energizes the ClearPath servos
//                              and forwards commands to the Teensy
//
// Anything that is a physical fact about the table, the spools, the
// anchors, or the paddle belongs in this file. Anything about how a
// particular controller drives a motor does not.
//
// (A second, host-side motion controller used to exist in sw/lib/cdpr.*
// with its own copy of this geometry. It was deleted on 2026-08-01: it
// duplicated the firmware's job, had already drifted — wrong frame, wrong
// attachment radius, no tangency or wrap term — and having two answers to
// "where is the paddle" was worse than having one.)
//
// Python consumers (ai/bin/calibrate_fit.py) cannot include this header;
// they mirror these constants and the forward model below. If you change a
// number here, change it there too — calibrate_fit.py's selftest and
// shared/check_geometry.py both exist to catch drift.
//
// ── Table coordinate system: the GRID frame ─────────────────────────
//
// All units millimetres. The playing field's CNC air-hole grid defines the
// frame (the rails are less straight than the grid): the origin is the
// corner HOLE nearest the human player's right corner. +x runs from the
// human toward the robot end; +y from the human's right toward their left.
// Holes sit at exact multiples of 25.4 mm; rails are approximate.
//
// Canonical orientation — human on the left, robot on the right, looking
// down at the playing surface:
//
//   far rail  ┌──────────────────────M0───────────────M1
//             │                      ┊                 │
//             │     human half       ┊   robot half    │
//             │                      ┊    [paddle]     │
//   near rail └──────────────────────M3───────────────M2
//         x=0 (origin hole)     CENTERLINE_X       x=GRID_X (robot-end hole)
// ════════════════════════════════════════════════════════════════════════

constexpr int NUM_MOTORS = 4;

// ── Table geometry ──────────────────────────────────────────────────
//
// Grid is 78 columns x 38 rows (verified empirically via calibration
// residuals — see vision/bin/table_grid.py).

constexpr float GRID_PITCH_MM = 25.4f;
constexpr float GRID_X_MM = 77.0f * GRID_PITCH_MM;    // 1955.8, robot-end hole
constexpr float GRID_Y_MM = 37.0f * GRID_PITCH_MM;    // 939.8, far-rail hole
constexpr float CENTERLINE_X = 38.5f * GRID_PITCH_MM; // 977.9, between columns

// Rail positions are APPROXIMATE in this frame (the grid is the truth):
constexpr float RAIL_MIN_X = -32.1f;
constexpr float RAIL_MAX_X = 1987.9f;
constexpr float RAIL_MIN_Y = -32.6f;
constexpr float RAIL_MAX_Y = 972.4f;

// ── Motor anchor positions ──────────────────────────────────────────
//
// MEASURED 2026-08-02, camera AND calipers together — neither alone is
// enough, and the reason is worth understanding before touching these.
//
// vision/bin/measure_anchors.py finds a retroreflective marker on each
// spool axis and back-projects its centroid through the calibrated camera
// pose onto a plane. That fixes the RAY each anchor lies on very well
// (repeatable to 0.1 mm over 9 bursts) but leaves the position ALONG the
// ray set entirely by the assumed plane height — and every anchor sits
// outside the quadrilateral the extrinsics were fitted on (x 38..1918,
// y 38..902), with M1 and M2 landing 30-43 px from the frame edge.
//
// 2026-08-09: M2 REPLACED by direct trilateration from caliper distances to
// three air holes — 131.0 mm from the corner hole (col 77, row 0), 265.0 mm
// from (77, 6), 259.0 mm from (71, 0). Residuals +0.10 / -0.06 / -0.05 mm,
// rms 0.07: the three measurements agree to a tenth of a millimetre, which
// nothing optical here has come close to. It moved M2 by 3.5 mm.
//
// That is now the preferred method for all four — shared/fit_anchors.py.
// Distances need no squareness and no axis to measure along, and the third
// distance turns a bare solution into a check. M0, M1 and M3 below are still
// the older camera-plus-caliper numbers and should be redone the same way.
//
// The older method, for the three that still use it: the height came from
// calipers against the air-hole grid, because
//     M0   144.5 mm past the row-37 hole line
//     M3   139.5 mm past the row-0 hole line
//     M1   132.5 mm from the corner hole at (1955.8, 939.8)
//     M2   133.0 mm from the corner hole at (1955.8, 0)
// all +/-0.5 mm. Sliding all four anchors along their rays until those four
// independent constraints are best satisfied gives an effective plane of
// z = 24.3 mm and an rms residual of 1.38 mm (M0 -2.3, M1 -0.2, M2 +1.4,
// M3 -0.3). At the calipered marker height of 33.5 mm the rms is 5.97 mm,
// so this is not a small correction.
//
// 24.3 mm is therefore an EFFECTIVE height, not a physical one — it is
// absorbing whatever the residual peripheral distortion is, along with any
// error in where the marker's optical centroid sits. That is fine because
// the correction is only ever applied at these four points, but do not
// treat it as a measurement of the markers, and pass --height 24.3 if you
// re-run measure_anchors.py without re-doing the caliper fit.
//
// This replaced an ellipse fit to the spool TOP FACES at an assumed 36 mm
// (vision/bin/measure_motors.py), which inferred the axis from a dim,
// partly-occluded outline. The anchors moved 8.7 to 17.6 mm in total.
//
// These are NOT a rectangle — the mid-table pair spans 1222 mm in y while
// the robot-end pair spans 1138 mm, because the two pairs use different
// mounting brackets. Any code that derives anchors from a width/height pair
// is wrong by construction. Re-measure if the spools are re-mounted.

constexpr float MOTOR_X[NUM_MOTORS] = {
    1067.6f, // 0: mid-table, far side
    2043.3f, // 1: robot corner, far side
    2043.2f, // 2: robot corner, near side  CALIPERED
    1063.0f, // 3: mid-table, near side
};
constexpr float MOTOR_Y[NUM_MOTORS] = {
    1082.0f, // 0
    1041.2f, // 1
    -97.7f,  // 2  CALIPERED
    -139.7f, // 3
};

// ── Spool ───────────────────────────────────────────────────────────
//
// Motor shaft vertical. This is the WINDING radius — where the cable
// actually sits — which is the single largest scale factor in the machine:
// every commanded millimetre of paddle motion is converted through it, so a
// 1% error here is a 1% error on every move, everywhere.
//
// History, because none of it carries over and it is easy to compare the
// wrong pair of numbers:
//   original   3D-printed, r = 35 mm     — SLIPPED on the shafts
//   2026-08-02 McMaster 6245K418 iron pulley, r = 41.275 mm — too much
//              rotational inertia for the drives' tune, vibrated at rest
//   2026-08-05 goBILDA 3400 Series hub-mount round-belt pulley, 96 mm PD
//
// The 96 is a DIAMETER — the part is named for its 96 mm pitch diameter, so
// r = 48 mm. Confirmed with the builder 2026-08-05, after it was first
// reported as a 9.6 cm radius. Worth stating plainly because the series
// also comes in 32 and 64 mm PD, and picking the wrong one puts a 50% error
// straight into the one constant that scales every commanded millimetre.
//
// !! STILL VERIFY BY MEASUREMENT, NOT FROM THE PART NAME. !!
// Pitch diameter is defined by where a round BELT of the matching section
// rides. A cable is thinner than that belt, so it sits deeper in the groove
// and winds at a smaller radius. The definitive check needs no motion of
// the robot — with the drives down, rotate one spool by hand through N
// whole turns and measure the cable paid out. That length is N * 2 * pi * r
// directly; 10 turns is ~3016 mm at this radius, so a tape good to 5 mm
// pins r to about 0.15%.
//
// The groove is single, and the workspace needs ~2.4 turns, so the cable
// cannot lie in one lane: turns sit at different lateral positions and
// therefore at slightly different depths in a curved groove. That is a
// radius variation of order 1%, unmodelled, and it varies with how much
// cable is out. Same for the wire stacking on itself.

constexpr float SPOOL_RADIUS_MM = 48.0f; // 96 mm pitch DIAMETER / 2
constexpr float SPOOL_CIRCUMFERENCE_MM = 2.0f * (float)M_PI * SPOOL_RADIUS_MM;

// Which side of each spool the wire leaves, as a sign in the wrap term.
//
// !! UNVERIFIED — the single biggest correctness risk in this file. !!
// The wrap term below is worth tens of millimetres of cable; applied with
// the wrong sign it is WORSE THAN OMITTING IT, because the error doubles
// instead of cancelling. Settle it by commanding a large move and watching
// the vision-measured paddle position: a wrong sign shows up as error that
// grows with how far the cable bearing has swept, and it is a one-line flip
// here (plus SIDES in ai/bin/calibrate_fit.py).
constexpr float WINDING_SIDE[NUM_MOTORS] = {-1.0f, 1.0f, -1.0f, 1.0f};

// Retraction sense per motor, viewed facing the motor: motors 0 and 2
// retract (shorten the cable) rotating CLOCKWISE, motors 1 and 3
// counter-clockwise. This is a physical fact about the build; how each
// controller expresses it (DIR pin level, API count sign) is not, and
// stays in the per-controller headers. Confirmed on hardware.
constexpr bool RETRACT_CW[NUM_MOTORS] = {true, false, true, false};

// ── Paddle cable attachment ─────────────────────────────────────────
//
// Each cable attaches to its own arm of a cross on the paddle, at radius
// ATTACH_R_MM from the paddle centre. Arms 0-3 run CLOCKWISE viewed from
// above, hence the negative chirality:
//
//     attach_m = centre + ATTACH_R * [cos(phi_m), sin(phi_m)]
//     phi_m    = theta + ATTACH_CHIRALITY * 90deg * m
//
// where theta is the paddle's orientation. The paddle is held by four
// cables constraining three planar DOF, so theta is determined rather than
// free — but it is NOT constant, and cable length genuinely depends on it.
//
// 31.1 mm is CONFIRMED against the current paddle (2026-08-01). The old
// firmware encoded an axis-aligned square of side 21.21 mm — an effective
// radius of 15.0 mm, less than half — left over from the prototype cart.
// If you see that square anywhere, it is stale.

constexpr float ATTACH_R_MM = 31.1f;
constexpr float ATTACH_CHIRALITY = -1.0f; // arms 0-3 clockwise from above

// The orientation the paddle actually sits at, and therefore the one every
// cable length is computed for.
//
// This is NOT a free choice of zero. At 135 deg each arm points at its own
// motor: bearings from the workspace centre to M0..M3 are 125/46/-45/-125
// deg, and arm m points at theta - 90*m, so all four agree on theta ~= 135.
// Across the whole workspace the best-aligned value only moves between 124
// and 146 deg, so a constant is fine — and a constant is self-consistent,
// because four cables cut to a given (x, y, theta) admit exactly that pose,
// so the paddle holds the orientation rather than fighting it.
//
// Computing lengths at theta = 0 instead — as this did until 2026-08-01 —
// asks for a paddle rotated 135 deg from where it physically is. The
// resulting per-motor length offsets (about +50, +53, +54, +56 mm) are
// unequal, so calibration absorbs only their average: what survives is a
// commanded motion that disagrees with the required one by roughly 1 mm
// over a 30 mm move and 5 mm over 200 mm, with OPPOSITE SIGNS across
// motors. That is slack on two cables and tension on the other two, and on
// an over-constrained rig it also lets the slack cable swap mid-move, which
// is a mechanism for the paddle to chatter between constraint sets.
constexpr float MALLET_THETA_RAD = 2.3561945f; // 135 deg

// ── Workspace bounds ────────────────────────────────────────────────
//
// The middle half of the robot half: same centre, half the span in each
// axis. Deliberately conservative, and deliberately NOT the rails.
//
// A cable pulls and cannot push, so the paddle is only holdable while every
// cable stays in tension — which is only true strictly inside the convex
// hull of the four anchors, and the tension needed to resist a given
// disturbance grows without bound as you approach that hull. The anchors
// span x = 1064..2031, so the region the CABLES control is much smaller
// than the robot half of the TABLE; the centreline at x = 977.9 is 86 mm
// outside the hull and is not reachable at any torque. Staying near the
// middle keeps the cables well spread and the tensions modest.
//
// These are hardcoded on purpose: they are a bring-up choice, not a derived
// quantity. Widen them when the winding sign and the command-then-measure
// residual are settled.

constexpr float WS_MIN_X = 1230.0f;
constexpr float WS_MAX_X = 1730.0f;
constexpr float WS_MIN_Y = 220.0f;
constexpr float WS_MAX_Y = 720.0f;

// Default calibration/home position: centre of the workspace.
constexpr float HOME_X = (WS_MIN_X + WS_MAX_X) / 2.0f;  // 1480
constexpr float HOME_Y = (WS_MIN_Y + WS_MAX_Y) / 2.0f;  // 470

// Bearing from each anchor toward HOME. This is the zero reference for the
// wrap angle: measuring psi relative to a fixed per-motor direction avoids
// an atan2 branch cut inside the workspace. The choice only adds a constant
// per-motor offset to the computed length, and every constant offset is
// absorbed when the controller is homed, so it is free.
// Values are atan2(HOME_Y - MOTOR_Y[m], HOME_X - MOTOR_X[m]).
constexpr float WRAP_REF_ANGLE[NUM_MOTORS] = {
    -0.977833f, // 0
    -2.349231f, // 1
    2.352215f,  // 2
    0.970928f,  // 3
};

// ── Kinematics ──────────────────────────────────────────────────────

// Cable attachment point for one motor, given paddle centre and rotation.
inline void attachPoint(int motor, float x, float y, float theta, float &ax,
                        float &ay) {
  const float phi = theta + ATTACH_CHIRALITY * (float)M_PI_2 * (float)motor;
  ax = x + ATTACH_R_MM * cosf(phi);
  ay = y + ATTACH_R_MM * sinf(phi);
}

// Wrap the argument into (-pi, pi].
inline float wrapPi(float a) {
  while (a > (float)M_PI) a -= 2.0f * (float)M_PI;
  while (a <= -(float)M_PI) a += 2.0f * (float)M_PI;
  return a;
}

// Cable length as the ENCODER sees it, for the paddle centred at (x, y)
// with orientation theta.
//
// The wire does not leave the motor centre — it leaves the spool at a
// tangent point, so the free span is d' = sqrt(d^2 - r^2), and as the
// paddle moves the tangent point travels around the spool, winding or
// unwinding an extra r*psi. Both terms are included:
//
//     u = d' - side * r * psi
//
// d' is a sub-millimetre correction; the r*psi wrap term sweeps tens of
// millimetres across the workspace and is the one that matters. See
// WINDING_SIDE above — its sign is not yet verified against hardware.
//
// Returns a length that is correct up to a constant per-motor offset (the
// encoder zero and the wrap reference choice), which homing absorbs.
inline float cableLength(int motor, float x, float y,
                         float theta = MALLET_THETA_RAD) {
  float ax, ay;
  attachPoint(motor, x, y, theta, ax, ay);
  const float dx = ax - MOTOR_X[motor];
  const float dy = ay - MOTOR_Y[motor];

  float d = sqrtf(dx * dx + dy * dy);
  if (d < SPOOL_RADIUS_MM + 1e-6f) d = SPOOL_RADIUS_MM + 1e-6f; // degenerate
  const float dp = sqrtf(d * d - SPOOL_RADIUS_MM * SPOOL_RADIUS_MM);

  // Unit vector from the anchor to the tangent point: rotate the
  // anchor->attach direction toward the spool by the tangent geometry.
  const float ux = dx / d, uy = dy / d;
  const float nx = -uy, ny = ux;
  const float s = WINDING_SIDE[motor];
  const float tx = (SPOOL_RADIUS_MM / d) * ux + s * (dp / d) * nx;
  const float ty = (SPOOL_RADIUS_MM / d) * uy + s * (dp / d) * ny;

  const float psi = wrapPi(atan2f(ty, tx) - WRAP_REF_ANGLE[motor]);
  return dp - s * SPOOL_RADIUS_MM * psi;
}

// Is a paddle centre inside the usable workspace? Bounds are absolute grid
// coordinates already inset by the edge margin; they are NOT derived from
// the anchors, which do not form a rectangle.
inline bool inWorkspace(float x, float y) {
  return x >= WS_MIN_X && x <= WS_MAX_X && y >= WS_MIN_Y && y <= WS_MAX_Y;
}
