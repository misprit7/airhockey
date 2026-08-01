#include "pubSysCls.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace sFnd;

// ============================================================================
// check_limits — read-only report of what the drives are actually configured
// to do. Moves nothing, enables nothing.
//
// Exists because a torque limit set during an experiment is invisible in the
// source tree afterwards: it lives in the drive, not in the code. A low
// global torque limit starves the motors, which on a force-closed rig looks
// exactly like slack and can make a servo hunt because it cannot reach the
// position it was told to hold. Both symptoms are easy to misattribute to
// the kinematic model.
//
// TrqGlobal is reported as a percentage of drive maximum. 100% means no
// limiting. Anything much below that during normal running is suspect.
//
// Usage: sw/build/check_limits
// ============================================================================

int main(int argc, char *argv[]) {
    bool restore = argc > 1 && std::string(argv[1]) == "--restore";
    SysManager *mgr = SysManager::Instance();
    std::vector<std::string> ports;
    SysManager::FindComHubPorts(ports);
    if (ports.empty()) {
        printf("ERROR: no SC hub found. If cdpr_master or activate is\n"
               "running it holds the port — stop it and retry.\n");
        return 1;
    }

    try {
        mgr->ComHubPort(0, ports[0].c_str());
        mgr->PortsOpen(1);
        IPort &port = mgr->Ports(0);
        printf("port %s, %d node(s)\n\n", ports[0].c_str(), port.NodeCount());

        printf("node  model              torque limit   vel limit   acc limit"
               "   measured trq  alerts\n");
        printf("----  -----------------  ------------   ---------   ---------"
               "   ------------  ------\n");

        bool anyLimited = false, anyAlert = false;
        for (unsigned i = 0; i < port.NodeCount(); i++) {
            INode &n = port.Nodes(i);
            n.TrqUnit(INode::PCT_MAX);
            n.VelUnit(INode::RPM);
            n.AccUnit(INode::RPM_PER_SEC);

            n.Limits.TrqGlobal.Refresh();
            n.Motion.VelLimit.Refresh();
            n.Motion.AccLimit.Refresh();
            n.Motion.TrqMeasured.Refresh();
            n.Status.Alerts.Refresh();

            double trq = n.Limits.TrqGlobal.Value();
            double meas = n.Motion.TrqMeasured.Value();
            bool alert = n.Status.Alerts.Value().isInAlert();
            if (trq < 95.0) anyLimited = true;
            if (alert) anyAlert = true;

            printf("  %u   %-17s  %7.1f %%%s   %8.0f    %8.0f   %8.1f %%   %s\n",
                   i, n.Info.Model.Value(), trq,
                   trq < 95.0 ? " <<" : "  ",
                   n.Motion.VelLimit.Value(), n.Motion.AccLimit.Value(),
                   meas, alert ? "YES <<" : "-");
        }

        printf("\n");
        if (anyLimited) {
            printf("!! At least one drive is torque limited (marked <<).\n"
                   "   On a force-closed cable rig a starved motor cannot take\n"
                   "   up its cable, which reads as slack, and a drive that\n"
                   "   cannot hold position will hunt. Restore with:\n"
                   "       sw/build/check_limits --restore\n"
                   "   (that writes 100%% to every node; it does not move them)\n");
        } else {
            printf("All drives at full torque — limiting is NOT the problem.\n");
        }
        if (anyAlert)
            printf("!! A drive is in alert. Clear it before drawing any other\n"
                   "   conclusion: an alerting node ignores step/dir input.\n");

        if (restore) {
            // Deliberate, explicit, and separate from reporting: this removes
            // a limit somebody chose on purpose. It writes a parameter; it
            // does not command motion.
            printf("\n--restore: writing TrqGlobal = 100%% to every node\n");
            for (unsigned i = 0; i < port.NodeCount(); i++) {
                INode &n = port.Nodes(i);
                n.TrqUnit(INode::PCT_MAX);
                n.Limits.TrqGlobal = 100.0;
                n.Limits.TrqGlobal.Refresh();
                printf("  node %u -> %.1f %%\n", i,
                       n.Limits.TrqGlobal.Value());
            }
            printf("Torque limits are RAM values on the drive: they revert on\n"
                   "power cycle unless saved from ClearView.\n");
        }

        mgr->PortsClose();
    } catch (mnErr &e) {
        printf("sFoundation error 0x%08x: %s\n", e.ErrorCode, e.ErrorMsg);
        return 1;
    }
    return 0;
}
