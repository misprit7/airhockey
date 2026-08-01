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

bool ClearPath::enable() {
  if (!connected_) return false;
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
          fprintf(stderr, "ClearPath: Motor %d timed out enabling\n", i);
          return false;
        }
      }
    }
    enabled_ = true;
    return true;
  } catch (mnErr &err) {
    fprintf(stderr, "ClearPath enable error: 0x%08x %s\n", err.ErrorCode,
            err.ErrorMsg);
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
