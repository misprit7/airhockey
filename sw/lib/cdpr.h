#pragma once

#include "cdpr_config.h"
#include "pubSysCls.h"
#include <cmath>

// Cable-Driven Parallel Robot controller.
//
// Usage:
//   CDPRConfig config;
//   CDPR robot(config);
//   robot.connect();
//   robot.enable();
//   robot.setPosition(303, 365);  // tell it where the cart currently is
//   robot.moveTo(310, 370, 5.0);  // move to (310, 370) at 5 mm/s
//   robot.disable();
//   robot.disconnect();

class CDPR {
public:
    explicit CDPR(const CDPRConfig &config);
    ~CDPR();

    // Connect to SC4-Hub and open port. Returns false on failure.
    bool connect();

    // Enable all motors. Must be connected first.
    bool enable();

    // Disable all motors (de-energize windings).
    void disable();

    // Close the port.
    void disconnect();

    // Tell the controller where the cart currently is (mm).
    // Must be called before any moveTo/moveBy commands.
    // This does NOT move anything — it just sets the internal state.
    void setPosition(double x, double y);

    // Move cart to absolute position (mm) at given speed (mm/s).
    // All 4 motors move simultaneously, coordinated so the cart
    // travels in a straight line at the requested speed.
    // Returns false if target is out of bounds or move fails.
    bool moveTo(double x, double y, double speed_mm_s);

    // Move cart by a relative offset (mm).
    bool moveBy(double dx, double dy, double speed_mm_s);

    // Current cart position (mm).
    double x() const { return x_; }
    double y() const { return y_; }
    bool positionKnown() const { return pos_known_; }

    // Compute cable length from motor i to position (x, y).
    double cableLength(int motor, double x, double y) const;

    // Compute all 4 cable lengths for a position.
    void cableLengths(double x, double y, double lengths[4]) const;

    // Retract all cables by the given amount (mm) simultaneously.
    // Positive = retract (shorten), negative = extend (lengthen).
    // This is a raw operation that bypasses IK — use for tensioning.
    bool retractAll(double mm, double speed_mm_s);

    // Check if a position is within the reachable workspace.
    bool inBounds(double x, double y) const;

    // Access config.
    const CDPRConfig &config() const { return cfg_; }

private:
    CDPRConfig cfg_;
    double spool_circumference_;

    // Current cart position.
    double x_, y_;
    bool pos_known_;

    // sFoundation state.
    sFnd::SysManager *mgr_;
    sFnd::IPort *port_;
    bool connected_;
    bool enabled_;
    int node_count_;

    // Convert cable length delta (mm) to encoder counts for a motor.
    int mmToCounts(double mm, int motor) const;

    // Convert mm/s linear speed to RPM for a given motor's spool.
    double mmPerSecToRPM(double mm_s) const;

    // Set velocity and acceleration limits on a node (in RPM).
    void setMotionParams(sFnd::INode &node, double vel_rpm, double accel_rpm_s);
};
