#include "clearpath.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace sFnd;

ClearPath::~ClearPath() {
  if (enabled_) disable();
  if (connected_) disconnect();
}

bool ClearPath::connect() {
  if (connected_) return true;

  mgr_ = SysManager::Instance();

  std::vector<std::string> hubPorts;
  SysManager::FindComHubPorts(hubPorts);
  if (hubPorts.empty()) {
    fprintf(stderr, "ClearPath: No SC Hub ports found\n");
    return false;
  }

  try {
    mgr_->ComHubPort(0, hubPorts[0].c_str());
    mgr_->PortsOpen(1);
    port_ = &mgr_->Ports(0);
    node_count_ = port_->NodeCount();

    if (node_count_ != 4) {
      fprintf(stderr, "ClearPath: Expected 4 motors, found %d\n", node_count_);
      mgr_->PortsClose();
      return false;
    }

    connected_ = true;
    return true;
  } catch (mnErr &err) {
    fprintf(stderr, "ClearPath connect error: 0x%08x %s\n", err.ErrorCode,
            err.ErrorMsg);
    return false;
  }
}

int ClearPath::clearFaults() {
  if (!connected_ || !port_) {
    printf("clearFaults: not connected to the SC-Hub.\n");
    fflush(stdout);
    return -1;
  }

  int stuck = 0, found = 0;
  for (int i = 0; i < 4; i++) {
    try {
      INode &node = port_->Nodes(i);
      char buf[512];

      node.Status.Alerts.Refresh();
      alertReg before = node.Status.Alerts.Value();
      if (before.isInAlert()) {
        before.StateStr(buf, sizeof(buf));
        printf("  motor %d fault at startup: %s\n", i, buf);
        found++;
      }

      // NodeStopClear as well as AlertsClear: a node stop is latched
      // separately from the alert register, and a drive left stopped by the
      // previous session's shutdown will take the enable and then refuse to
      // move, which looks like a dead axis rather than a cleared fault.
      node.Status.AlertsClear();
      node.Motion.NodeStopClear();
      mgr_->Delay(50);

      node.Status.Alerts.Refresh();
      alertReg after = node.Status.Alerts.Value();
      if (after.isInAlert()) {
        after.StateStr(buf, sizeof(buf));
        printf("  motor %d STILL IN ALERT after clear: %s\n", i, buf);
        stuck++;
      }
    } catch (mnErr &e) {
      printf("  motor %d fault clear failed: 0x%08x %s\n", i, e.ErrorCode,
             e.ErrorMsg);
      stuck++;
    }
  }

  if (found == 0 && stuck == 0) {
    printf("No drive faults at startup.\n");
  } else if (stuck == 0) {
    printf("Cleared faults on %d motor(s).\n", found);
  } else {
    printf("WARNING: %d motor(s) still in alert. An RMS overload will not "
           "clear until the drive's thermal model cools -- wait rather than "
           "retry.\n", stuck);
  }
  fflush(stdout);
  return stuck;
}

bool ClearPath::enable() {
  // Every failure path here says WHY, on stdout, before returning false.
  // It used to be possible for this to refuse silently -- the !connected_
  // branch printed nothing, and the caller's only reply was "ERR motor
  // enable failed" down a TCP socket to whoever asked. Watching the master's
  // console you saw "ENABLE" and then nothing at all, which is the least
  // useful thing a machine can do when it declines to energize.
  //
  // stdout rather than stderr so it lands in logs/cdpr_master.log with
  // everything else; a diagnosis that is only ever on someone's screen is
  // gone the moment the terminal scrolls.
  if (!connected_) {
    printf("ENABLE refused: not connected to the SC-Hub. The drives are "
           "found over USB at startup, so this means the link dropped "
           "since then.\n");
    fflush(stdout);
    return false;
  }
  if (enabled_) return true;

  try {
    for (int i = 0; i < 4; i++) {
      INode &node = port_->Nodes(i);
      node.EnableReq(false);
      mgr_->Delay(100);
      node.Status.AlertsClear();
      node.Motion.NodeStopClear();
      node.EnableReq(true);

      double timeout = mgr_->TimeStampMsec() + 5000;
      while (!node.Motion.IsReady()) {
        if (mgr_->TimeStampMsec() > timeout) {
          // Say what the drive itself thinks is wrong. Nearly always either
          // no bus voltage (the hub enumerates over USB and reports the node
          // happily with the 24-75 V supply off) or a latched shutdown --
          // and an RMS overload will not clear until its thermal model has
          // cooled, so retrying straight away just re-trips it.
          printf("ENABLE refused: motor %d did not become ready within 5 s.\n", i);
          try {
            node.Status.Alerts.Refresh();
            alertReg a = node.Status.Alerts.Value();
            if (a.isInAlert()) {
              char buf[512];
              a.StateStr(buf, sizeof(buf));
              printf("  motor %d alert: %s\n", i, buf);
            } else {
              printf("  motor %d reports NO alert, which points at bus "
                     "voltage rather than a fault -- check the 24-75 V "
                     "supply is on.\n", i);
            }
          } catch (mnErr &e2) {
            printf("  motor %d alert unreadable (0x%08x)\n", i, e2.ErrorCode);
          }
          fflush(stdout);
          return false;
        }
      }
    }
    enabled_ = true;
    printf("All four motors energized.\n");
    fflush(stdout);
    return true;
  } catch (mnErr &err) {
    printf("ENABLE refused: 0x%08x %s\n", err.ErrorCode, err.ErrorMsg);
    fflush(stdout);
    return false;
  }
}

void ClearPath::disable() {
  if (!connected_ || !port_) return;
  // Per-node try/catch: one node refusing to disable must not strand the
  // other three energized.
  for (int i = 0; i < 4; i++) {
    try {
      port_->Nodes(i).EnableReq(false);
    } catch (mnErr &err) {
      fprintf(stderr, "ClearPath: motor %d failed to disable: 0x%08x %s\n", i,
              err.ErrorCode, err.ErrorMsg);
    }
  }
  enabled_ = false;
}

void ClearPath::disconnect() {
  if (!connected_ || !mgr_) return;
  try {
    mgr_->PortsClose();
  } catch (...) {
  }
  connected_ = false;
  port_ = nullptr;
}


void ClearPath::reportTorqueLimits() {
  if (!connected_ || !port_) return;
  for (unsigned i = 0; i < port_->NodeCount(); i++) {
    try {
      INode &n = port_->Nodes(i);
      n.TrqUnit(INode::PCT_MAX);
      n.Limits.TrqGlobal.Refresh();
      double t = n.Limits.TrqGlobal.Value();
      printf("  motor %u torque limit %.1f %%%s\n", i, t,
             t < 95.0 ? "   <<< LIMITED - see sw/build/check_limits" : "");
    } catch (mnErr &e) {
      printf("  motor %u torque limit unreadable (0x%08x)\n", i, e.ErrorCode);
    }
  }
}


bool ClearPath::readEncoders(double posn[4], unsigned res[4], double trq[4]) {
  for (int i = 0; i < 4; i++) { posn[i] = 0; res[i] = 0; trq[i] = 0; }
  if (!connected_ || !port_) return false;
  bool any = false;
  for (unsigned i = 0; i < port_->NodeCount() && i < 4; i++) {
    try {
      INode &n = port_->Nodes(i);
      n.Motion.PosnMeasured.Refresh();
      posn[i] = n.Motion.PosnMeasured.Value();
      res[i] = n.Info.PositioningResolution.Value();
      n.TrqUnit(INode::PCT_MAX);
      n.Motion.TrqMeasured.Refresh();
      trq[i] = n.Motion.TrqMeasured.Value();
      any = true;
    } catch (mnErr &) {
      // Leave this node zeroed; a stale read is worse than an obvious gap.
    }
  }
  return any;
}


bool ClearPath::pollHealth(double torque_warn_pct) {
  if (!connected_ || !port_ || !enabled_) return false;
  bool reported = false;
  double trq[4] = {0, 0, 0, 0};
  for (unsigned i = 0; i < port_->NodeCount() && i < 4; i++) {
    try {
      INode &n = port_->Nodes(i);
      n.Status.Alerts.Refresh();
      alertReg a = n.Status.Alerts.Value();
      if (a.isInAlert()) {
        char buf[512];
        a.StateStr(buf, sizeof(buf));
        printf("!! motor %u ALERT: %s\n", i, buf);
        reported = true;
      }
      n.TrqUnit(INode::PCT_MAX);
      n.Motion.TrqMeasured.Refresh();
      trq[i] = n.Motion.TrqMeasured.Value();
    } catch (mnErr &) {
      // A read failure here is not worth aborting motion over.
    }
  }
  double worst = 0;
  for (int i = 0; i < 4; i++)
    if (fabs(trq[i]) > fabs(worst)) worst = trq[i];
  if (fabs(worst) > torque_warn_pct) {
    printf("!! high torque: [%.1f %.1f %.1f %.1f] %% — opposing signs here "
           "mean cables are fighting\n", trq[0], trq[1], trq[2], trq[3]);
    reported = true;
  }
  return reported;
}
