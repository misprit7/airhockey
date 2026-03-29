#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cmath>
#include <string>
#include <vector>
#include "pubSysCls.h"

using namespace sFnd;

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

// Motor layout (CDPR rectangle):
//   Motor 0 (top-left)  ---- Motor 1 (top-right)
//        |                        |
//        |       [cart]           |
//        |                        |
//   Motor 3 (bot-left)  ---- Motor 2 (bot-right)
//
// Coordinate system: origin at bottom-left, x-right, y-up.

static const double SPOOL_DIAMETER_MM = 20.5;
static const double SPOOL_CIRCUMFERENCE_MM = M_PI * SPOOL_DIAMETER_MM;  // ~64.4mm

// Encoder counts per revolution by motor type.
// RLNA = Regular (800), ELNA = Enhanced (6400).
static const int COUNTS_PER_REV_REGULAR = 800;
static const int COUNTS_PER_REV_ENHANCED = 6400;

// Clockwise (positive encoder direction) = retract cable.
// IMPORTANT: This assumption must be verified before first run.
// If a motor extends instead of retracting, flip its sign below.
static const int RETRACT_SIGN = -1;  // negative counts = clockwise = retract

static int countsPerRev(INode &node) {
    std::string model = node.Info.Model.Value();
    // Check for Enhanced (E) vs Regular (R) encoder in part number.
    // Format: CPM-SCSK-2331x-xLNA where first char after dash is E or R.
    if (model.find("-EL") != std::string::npos)
        return COUNTS_PER_REV_ENHANCED;
    return COUNTS_PER_REV_REGULAR;
}

static int mmToCounts(double mm, int cpr) {
    return (int)round(mm / SPOOL_CIRCUMFERENCE_MM * cpr);
}

int main(int argc, char *argv[]) {
    // Use sigaction without SA_RESTART so getchar() is interrupted by Ctrl+C
    struct sigaction sa = {};
    sa.sa_handler = sigHandler;
    sigaction(SIGINT, &sa, NULL);

    double retract_mm = 10.0;  // 1cm default
    if (argc > 1)
        retract_mm = atof(argv[1]);

    printf("=== CDPR Retraction Test ===\n");
    printf("Retracting each cable by %.1fmm (%.1fcm)\n", retract_mm, retract_mm / 10.0);
    printf("Spool circumference: %.2fmm\n\n", SPOOL_CIRCUMFERENCE_MM);

    SysManager *mgr = SysManager::Instance();

    try {
        std::vector<std::string> hubPorts;
        SysManager::FindComHubPorts(hubPorts);
        if (hubPorts.empty()) {
            printf("ERROR: No SC Hub ports found.\n");
            return 1;
        }

        mgr->ComHubPort(0, hubPorts[0].c_str());
        mgr->PortsOpen(1);

        IPort &port = mgr->Ports(0);
        printf("Found %d motors\n\n", port.NodeCount());

        if (port.NodeCount() != 4) {
            printf("WARNING: Expected 4 motors, found %d\n", port.NodeCount());
        }

        // Print plan before doing anything
        printf("--- Plan ---\n");
        for (size_t i = 0; i < port.NodeCount(); i++) {
            INode &node = port.Nodes(i);
            int cpr = countsPerRev(node);
            int counts = mmToCounts(retract_mm, cpr) * RETRACT_SIGN;
            printf("  Motor %zu (%s): %d counts/rev, retract %+d counts\n",
                   i, node.Info.UserID.Value(), cpr, counts);
        }
        printf("\nPress Enter to proceed, Ctrl+C to abort...\n");
        if (getchar() == EOF || g_stop) {
            mgr->PortsClose();
            return 0;
        }

        // Enable all motors
        printf("\nEnabling motors...\n");
        for (size_t i = 0; i < port.NodeCount() && !g_stop; i++) {
            INode &node = port.Nodes(i);
            node.EnableReq(false);
            mgr->Delay(100);
            node.Status.AlertsClear();
            node.Motion.NodeStopClear();
            node.EnableReq(true);

            double timeout = mgr->TimeStampMsec() + 5000;
            while (!node.Motion.IsReady()) {
                if (mgr->TimeStampMsec() > timeout) {
                    printf("ERROR: Motor %zu timed out enabling\n", i);
                    mgr->PortsClose();
                    return 1;
                }
            }
            printf("  Motor %zu enabled\n", i);
        }

        // Retract each motor one at a time, confirming each
        for (size_t i = 0; i < port.NodeCount() && !g_stop; i++) {
            INode &node = port.Nodes(i);
            int cpr = countsPerRev(node);
            int counts = mmToCounts(retract_mm, cpr) * RETRACT_SIGN;

            printf("\nMotor %zu (%s): retract %+d counts (%.1fmm). Enter to go, 's' to skip, Ctrl+C to abort: ",
                   i, node.Info.UserID.Value(), counts, retract_mm);
            fflush(stdout);
            int c = getchar();
            if (c == EOF || g_stop) break;
            if (c == 's' || c == 'S') {
                if (c != '\n') while (getchar() != '\n') {}  // consume rest of line
                printf("  Skipped\n");
                continue;
            }
            if (c != '\n') while (getchar() != '\n') {}  // consume rest of line

            // Conservative motion parameters
            node.AccUnit(INode::RPM_PER_SEC);
            node.VelUnit(INode::RPM);
            node.Motion.AccLimit = 1000;   // RPM/sec - gentle ramp
            node.Motion.VelLimit = 4.7;    // RPM - ~0.5 cm/s on 20.5mm spool

            printf("  Moving... ");
            fflush(stdout);

            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart(counts);

            double timeout = mgr->TimeStampMsec() + 10000;
            while (!node.Motion.MoveIsDone() && !g_stop) {
                if (mgr->TimeStampMsec() > timeout) {
                    printf("TIMEOUT!\n");
                    break;
                }
            }

            node.Motion.PosnMeasured.Refresh();
            printf("done (pos=%.0f)\n", node.Motion.PosnMeasured.Value());
        }

        // Disable all motors
        printf("\nDisabling motors...\n");
        for (size_t i = 0; i < port.NodeCount(); i++) {
            port.Nodes(i).EnableReq(false);
        }
        mgr->PortsClose();
        printf("Done!\n");

    } catch (mnErr &err) {
        printf("ERROR: addr=%d, code=0x%08x\n  %s\n",
               err.TheAddr, err.ErrorCode, err.ErrorMsg);
        mgr->PortsClose();
        return 1;
    }

    return 0;
}
