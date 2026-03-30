#pragma once

// Cable-Driven Parallel Robot (CDPR) configuration.
//
// Motor layout (top view, looking down at table):
//
//   Motor 0 (top-left)  -------- Motor 1 (top-right)
//        |                            |
//        |         [cart]             |
//        |                            |
//   Motor 3 (bot-left)  -------- Motor 2 (bot-right)
//
// Coordinate system: origin at bottom-left (motor 3), x-right, y-up.
// All units in millimeters unless otherwise noted.

struct CDPRConfig {
  // ---- Table geometry ----

  // Distance between left and right motor columns (mm).
  double width = 606.0;

  // Distance between top and bottom motor rows (mm).
  double height = 730.0;

  // ---- Motor/spool parameters ----

  // Spool diameter (mm). Cable wraps around this.
  double spool_diameter = 41.0;

  // Encoder counts per revolution for each motor.
  // Index matches SC4-Hub node number.
  // RLNA (Regular) = 800, ELNA (Enhanced) = 6400.
  int counts_per_rev[4] = {800, 6400, 800, 6400};

  // ---- Derived motor positions (computed from width/height) ----
  // Motor positions in the table coordinate frame.
  // These are computed by motorPos() below, not set directly.

  // Returns motor position for motor index 0-3.
  //   Motor 0: (0, height)      - top-left
  //   Motor 1: (width, height)  - top-right
  //   Motor 2: (width, 0)       - bottom-right
  //   Motor 3: (0, 0)           - bottom-left
  void motorPos(int motor, double &x, double &y) const {
    switch (motor) {
    case 0:
      x = 0;
      y = height;
      break;
    case 1:
      x = width;
      y = height;
      break;
    case 2:
      x = width;
      y = 0;
      break;
    case 3:
      x = 0;
      y = 0;
      break;
    }
  }

  // ---- Motion limits ----

  // Maximum velocity (mm/s). Start very conservative.
  double max_velocity = 2000.0;

  // Maximum acceleration (mm/s^2).
  double max_acceleration = 50000.0;

  // Margin from motor positions to keep the cart away from edges (mm).
  double edge_margin = 30.0;
};
