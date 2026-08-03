#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cmath>
#include <string>
#include <vector>
#include "cdpr_geometry.h"
#include "pubSysCls.h"

using namespace sFnd;

// ============================================================================
// Per-motor direction sanity check (software API, no Teensy involved).
//
// For each motor in turn: EXTEND the cable (pays out slack — safe in either
// polarity), then RETRACT it back, pause 1 s, move to the next. One pass
// only, and nothing moves until you confirm at the prompt.
//
// IMPORTANT: put a few cm of slack in every string first (motors disabled,
// pull the strings out by hand). With all strings taut the mallet is
// force-closed by the four motors, and any lone retraction just spikes
// tension and rattles the mallet.
//
// Rotation convention (viewed facing the motor):
//   motors 0 and 2 retract CLOCKWISE, motors 1 and 3 COUNTER-CLOCKWISE.
// CW_API_SIGN maps API move sign to shaft direction (from old-rig testing:
// negative counts = clockwise). Verify while watching the spools:
//   ALL motors backwards -> flip CW_API_SIGN.
//   Only some backwards  -> fix RETRACT_CW.
// ============================================================================

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

// Spool size and retraction sense come from the shared header, NOT from a
// local copy. This file had its own SPOOL_RADIUS_MM = 35.0 and its own
// RETRACT_CW until 2026-08-02, and when the spools were replaced the shared
// header moved to 41.275 mm while this kept quietly converting millimetres
// at the old scale — an 18% error in a tool whose entire job is to tell you
// which way a motor turns.
static const int CW_API_SIGN = -1;  // negative counts = clockwise

static const double TEST_SPEED_MM_S = 5.0;   // slow: ~2 s per 10mm phase
static const double ACCEL_RPM_PER_S = 300;   // gentle ramp

// Encoder counts per revolution by motor type.
// RLNA = Regular (800), ELNA = Enhanced (6400).
static int countsPerRev(INode &node) {
    std::string model = node.Info.Model.Value();
    if (model.find("-EL") != std::string::npos)
        return 6400;
    return 800;
}

static int mmToCounts(double mm, int cpr) {
    return (int)round(mm / SPOOL_CIRCUMFERENCE_MM * cpr);
}

// Move sign that retracts (shortens) motor m's cable.
static int retractSign(int m) {
    return RETRACT_CW[m] ? CW_API_SIGN : -CW_API_SIGN;
}

// Wait for Enter; returns false on EOF/Ctrl+C.
static bool waitEnter() {
    int c;
    do {
        c = getchar();
        if (c == EOF || g_stop) return false;
    } while (c != '\n');
    return true;
}

// Blocking relative move; returns false on timeout/abort.
static bool moveBlocking(SysManager *mgr, INode &node, int counts) {
    node.Motion.MoveWentDone();
    node.Motion.MovePosnStart(counts);
    double timeout = mgr->TimeStampMsec() + 15000;
    while (!node.Motion.MoveIsDone() && !g_stop) {
        if (mgr->TimeStampMsec() > timeout) {
            printf("TIMEOUT!\n");
            return false;
        }
    }
    return !g_stop;
}

int main(int argc, char *argv[]) {
    // Use sigaction without SA_RESTART so getchar() is interrupted by Ctrl+C
    struct sigaction sa = {};
    sa.sa_handler = sigHandler;
    sigaction(SIGINT, &sa, NULL);

    double test_mm = 10.0;  // per-phase cable travel (~14 deg of spool)
    if (argc > 1)
        test_mm = atof(argv[1]);

    printf("=== CDPR Direction Test ===\n");
    printf("Each motor: extend %.1fmm (pay out), retract %.1fmm (wind in), 1s pause.\n",
           test_mm, test_mm);
    printf("Spool circumference: %.1fmm -> %.1fmm is %.0f deg of rotation.\n\n",
           SPOOL_CIRCUMFERENCE_MM, test_mm, 360.0 * test_mm / SPOOL_CIRCUMFERENCE_MM);
    printf("*** Make sure every string has a few cm of SLACK before starting. ***\n\n");

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
            int counts = mmToCounts(test_mm, cpr) * retractSign(i);
            printf("  Motor %zu (%s): %d counts/rev, extend %+d then retract %+d counts"
                   " (retract = %s)\n",
                   i, node.Info.UserID.Value(), cpr, -counts, counts,
                   RETRACT_CW[i] ? "CW" : "CCW");
        }
        printf("\nEach move waits for Enter. Press Enter to enable motors, "
               "Ctrl+C aborts at any prompt...\n");
        if (!waitEnter()) {
            mgr->PortsClose();
            return 0;
        }

        // Enable all motors
        printf("Enabling motors...\n");
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

        // One pass: extend then retract, each motor in turn
        for (size_t i = 0; i < port.NodeCount() && !g_stop; i++) {
            INode &node = port.Nodes(i);
            int cpr = countsPerRev(node);
            int counts = mmToCounts(test_mm, cpr) * retractSign(i);

            node.AccUnit(INode::RPM_PER_SEC);
            node.VelUnit(INode::RPM);
            node.Motion.AccLimit = ACCEL_RPM_PER_S;
            node.Motion.VelLimit = TEST_SPEED_MM_S * 60.0 / SPOOL_CIRCUMFERENCE_MM;

            printf("\nMotor %zu: extend %.1fmm - spool should PAY OUT (%s). Enter to go...",
                   i, test_mm, RETRACT_CW[i] ? "CCW" : "CW");
            fflush(stdout);
            if (!waitEnter()) break;
            printf("  moving... ");
            fflush(stdout);
            if (!moveBlocking(mgr, node, -counts)) break;
            printf("done\n");

            printf("Motor %zu: retract %.1fmm - spool should WIND IN (%s). Enter to go...",
                   i, test_mm, RETRACT_CW[i] ? "CW" : "CCW");
            fflush(stdout);
            if (!waitEnter()) break;
            printf("  moving... ");
            fflush(stdout);
            if (!moveBlocking(mgr, node, counts)) break;
            node.Motion.PosnMeasured.Refresh();
            printf("done (pos=%.0f)\n", node.Motion.PosnMeasured.Value());
        }

        // Disable all motors
        printf("\nDisabling motors...\n");
        for (size_t i = 0; i < port.NodeCount(); i++) {
            port.Nodes(i).EnableReq(false);
        }
        mgr->PortsClose();
        printf("Done. Any motor whose spool WOUND IN on the extend phase has\n"
               "its polarity flipped: all four -> flip CW_API_SIGN; some -> fix RETRACT_CW.\n");

    } catch (mnErr &err) {
        printf("ERROR: addr=%d, code=0x%08x\n  %s\n",
               err.TheAddr, err.ErrorCode, err.ErrorMsg);
        mgr->PortsClose();
        return 1;
    }

    return 0;
}
