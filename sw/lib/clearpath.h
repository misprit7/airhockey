#pragma once

#include "pubSysCls.h"

// Minimal ClearPath-SC power/enable control over sFoundation.
//
// This is deliberately NOT a motion controller. All CDPR motion now goes
// through the Teensy over step/dir; the host's only remaining job on the
// sFoundation side is to bring the four servos up and put them down again.
// The former host-side motion controller (cable kinematics, coordinated
// moveTo, trajectory timing) was removed on 2026-08-01 — it duplicated the
// firmware's job and drifted out of sync with it. If you need cable
// kinematics, they live in shared/cdpr_geometry.h.
class ClearPath {
public:
  ~ClearPath();

  // Find the SC hub, open the port, verify all four nodes are present.
  bool connect();

  // Energize all four motors and wait for them to report ready.
  bool enable();

  // De-energize the windings. Safe to call when not connected.
  void disable();

  // Close the port. Safe to call when not connected.
  void disconnect();

  // Print each drive's global torque limit. A limit left over from an
  // earlier experiment lives in the DRIVE, not in this source tree, and
  // starves the motors in a way that reads as cable slack and as servo
  // hunting — both easy to misattribute to the kinematic model. Reporting
  // it on every launch means it can never be invisible again.
  void reportTorqueLimits();

  // Poll the drives while they are running. The Teensy drives step/dir
  // open-loop and never reads the servos back, so without this a drive
  // that faults or saturates mid-move is completely silent — and both look
  // like a kinematics problem from the outside.
  //
  // Returns true if anything was worth reporting. `torque_warn_pct` is the
  // level above which sustained torque is called out: on a force-closed rig
  // two motors pulling hard against each other is the signature of cables
  // fighting, which no position readout will show you.
  bool pollHealth(double torque_warn_pct = 35.0);

  bool connected() const { return connected_; }
  bool enabled() const { return enabled_; }

private:
  sFnd::SysManager *mgr_ = nullptr;
  sFnd::IPort *port_ = nullptr;
  bool connected_ = false;
  bool enabled_ = false;
  int node_count_ = 0;
};
