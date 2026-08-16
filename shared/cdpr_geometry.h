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
// Grid is 80 columns x 39 rows — COUNTED on the table 2026-08-10 and
// confirmed optically (0.276 px pose RMS against 0.728 for the previous
// 78x38). It was 78x38 here until then, and that was wrong: see
// vision/bin/table_grid.py for how a scan that only varied one count at a
// time picked the best of three wrong answers, and what it cost.

constexpr float GRID_PITCH_MM = 25.4f;
constexpr float GRID_X_MM = 79.0f * GRID_PITCH_MM;    // 2006.6, robot-end hole
constexpr float GRID_Y_MM = 38.0f * GRID_PITCH_MM;    // 965.2, far-rail hole
constexpr float CENTERLINE_X = 39.5f * GRID_PITCH_MM; // 1003.3, between columns

// Wall positions, MEASURED 2026-08-12 with calipers via the sticker markers
// (vision/bin/table_grid.py _WALL_OFFSETS), replacing the earlier nominal
// values that assumed a 2020 x 1005 table with the grid centred on it. It is
// not centred: the gap is 4.25 mm on the human side and 11.50 mm on the
// robot side, so the grid sits 3.6 mm off centre.
//
// Each wall is slightly out of parallel with the grid, so these are the
// INSCRIBED rectangle — the closest approach of each wall along its length.
// A workspace derived from them is therefore safe everywhere, not just at
// the point that happened to be measured.
constexpr float RAIL_MIN_X = -3.8f;    // human end   (4.7 at the near rail)
constexpr float RAIL_MAX_X = 2017.9f;  // robot end   (2018.3 at the near rail)
constexpr float RAIL_MIN_Y = -19.0f;   // near rail   (19.4 at the robot end)
constexpr float RAIL_MAX_Y = 984.9f;   // far rail    (parallel to 0.1 mm)

// ── Motor anchor positions ──────────────────────────────────────────
//
// MEASURED 2026-08-10, after the 78x38 -> 80x39 grid fix and against the
// re-solved camera pose. Two methods, and the mix is deliberate:
//
// ALL FOUR CALIPERED, trilaterated from distances to three air holes each
// (shared/fit_anchors.py). Distances beat axis offsets: they need no
// squareness and no axis to measure along, 100 mm outside the rails with
// nothing to square against, and the third distance turns a bare solution
// into a check.
//   M0  145.8 mm from hole (44,38), 225.5 from (50,38), 250.5 from (41,34)
//       -> rms 0.51 mm
//   M1  131.0 mm from hole (79,38), 259.5 from (73,38), 265.0 from (79,32)
//       -> rms 0.19 mm
//   M2  131.0 mm from hole (79,0),  259.0 from (73,0),  265.0 from (79,6)
//       -> rms 0.07 mm
//   M3  141.9 mm from hole (43,0),  226.5 from (50,0),  255.5 from (40,4)
//       -> rms 0.01 mm
//
// Two symmetries fall out that nothing in the fit forces, so they are real
// checks rather than construction: the robot-end pair sits 97.4 and 97.7 mm
// outside its rails and 88.0 and 87.4 mm past col 79; the mid-table pair
// sits 143.5 and 141.9 mm outside its rails at x within 2.3 mm.
//
// WHAT THE CAMERA WAS WORTH, now that there is truth to compare it to.
// vision/bin/measure_anchors.py had measured all four optically at the
// calipered z = 33.5 mm. Its errors, against these:
//   M0  1.5 mm   400 px from the principal point
//   M3  2.2 mm   457 px
//   M1  3.4 mm   770 px
//   M2  4.2 mm   806 px
// Monotonic in image radius, with the intrinsics data ending at 728 px —
// M0 and M3 interpolating, M1 and M2 extrapolating. That is a lens-model
// limit, not noise: the optical measurement repeats to 0.1 mm. Use the
// camera for anything inside the marker quadrilateral and calipers for
// anything outside it.
//
// The optical route also needs an assumed plane height, and the camera
// cannot supply it — the lever is 0.42 mm/mm for the mid-table pair and
// 0.82 for the robot-end pair. Fitting that height to the caliper
// constraints wants 35.0 mm against the calipered 33.5. Before the
// 78x38 -> 80x39 grid fix the same fit demanded 24.3 mm: that 9 mm was the
// grid error in disguise, absorbed by the only free parameter the procedure
// had. A fitted constant that drifts far from a directly measured one is a
// symptom, not a calibration.
//
// These are NOT a rectangle — the two pairs use different mounting
// brackets, so any code deriving anchors from a width/height pair is wrong
// by construction. Re-measure if the spools are re-mounted.

constexpr float MOTOR_X[NUM_MOTORS] = {
    1095.7f, // 0: mid-table, far side    CALIPERED
    2094.6f, // 1: robot corner, far side  CALIPERED
    2094.0f, // 2: robot corner, near side CALIPERED
    1093.4f, // 3: mid-table, near side    CALIPERED
};
constexpr float MOTOR_Y[NUM_MOTORS] = {
    1108.7f, // 0  CALIPERED
    1062.6f, // 1  CALIPERED
    -97.7f,  // 2  CALIPERED
    -141.9f, // 3  CALIPERED
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
// 26.5 mm, CALIPERED 2026-08-16, replacing 31.1 which was confirmed on
// 2026-08-01 when the attachment sat at 49 mm. Lowering it to 32.7 mm pulled
// it 4.6 mm INWARD — the arms are angled about 16 degrees off vertical, so
// height and radius do not move independently. Changing one without the
// other is not a refinement, it is a broken model:
//
// Four cables constrain three DOF, so a wrong radius does not merely
// displace the paddle, it asks for a paddle that does not exist. At R off by
// 5.65 mm the four commanded lengths disagree by 4.4-7.5 mm even AFTER the
// paddle slides to its best fit — 12 to 20 motor counts that cannot be
// relieved by moving, so they become tension. Worse, which two cables go
// taut changes as it moves, and swapping constraint sets mid-move is exactly
// the chatter this file warns about further down.
//
// The optical arm-marker radius reads 25.45 mm here, about 1.05 mm INBOARD
// of the calipered attachment. So track_mallet's arm_r is a good indicator
// that this constant has drifted, but not a substitute for measuring it:
// 1 mm of radius error is ~1 mm of irreducible cable disagreement, and the
// cables have no way to express that except as tension.
//
// The old firmware encoded an axis-aligned square of side 21.21 mm — an
// effective radius of 15.0 mm — left over from the prototype cart. If you
// see that square anywhere, it is stale.

constexpr float ATTACH_R_MM = 26.5f;
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
// Three of the four bounds are the measured WALL less a clearance margin
// less the mallet radius. The fourth cannot be: a cable pulls and cannot
// push, so the paddle is only holdable while every cable stays in tension,
// which is only true strictly inside the convex hull of the four anchors,
// and the tension needed to resist a given disturbance grows without bound
// as you approach it. The anchors span x = 1093..2095, so the human-end
// wall at x = -3.8 is about 1100 mm outside the hull and is not reachable
// at any torque. WS_MIN_X is set from the hull, not the wall.
//
// Pushing WS_MIN_X toward the hull buys very little: 160 mm of clearance
// down to 60 mm adds 12% of area while heading straight at the edge where
// tension diverges. The expansion is almost all in y and in WS_MAX_X, so
// the clearance here stays generous.

// Outer radius of the paddle, for keeping it off the walls.
// MEASURED 2026-08-16: 100.8 mm diameter. (Was 40.0 here, taken from the
// simulator's paddle_radius, which was a modelling choice and 10.4 mm
// optimistic on all four sides.)
constexpr float MALLET_RADIUS_MM = 50.4f;

// Clearance from the wall to the paddle's RIM.
constexpr float WALL_MARGIN_MM = 10.0f;

// Clearance from the anchor hull to the paddle CENTRE. Not a rim clearance:
// what must stay inside the hull is where the cables attach, not the plastic.
constexpr float HULL_CLEARANCE_MM = 104.3f;

constexpr float WS_MIN_X =
    (MOTOR_X[0] > MOTOR_X[3] ? MOTOR_X[0] : MOTOR_X[3]) + HULL_CLEARANCE_MM;
constexpr float WS_MAX_X = RAIL_MAX_X - WALL_MARGIN_MM - MALLET_RADIUS_MM;
constexpr float WS_MIN_Y = RAIL_MIN_Y + WALL_MARGIN_MM + MALLET_RADIUS_MM;
constexpr float WS_MAX_Y = RAIL_MAX_Y - WALL_MARGIN_MM - MALLET_RADIUS_MM;

// Default calibration/home position: centre of the workspace.
constexpr float HOME_X = (WS_MIN_X + WS_MAX_X) / 2.0f;  // 1584.0
constexpr float HOME_Y = (WS_MIN_Y + WS_MAX_Y) / 2.0f;  // 483.0

// Bearing from each anchor toward HOME. This is the zero reference for the
// wrap angle: measuring psi relative to a fixed per-motor direction avoids
// an atan2 branch cut inside the workspace. The choice only adds a constant
// per-motor offset to the computed length, and every constant offset is
// absorbed when the controller is homed, so it is free.
// Values are atan2(HOME_Y - MOTOR_Y[m], HOME_X - MOTOR_X[m]).
constexpr float WRAP_REF_ANGLE[NUM_MOTORS] = {
    -0.988161f, // 0
    -2.362197f, // 1
    2.360737f, // 2
    0.985013f, // 3
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
