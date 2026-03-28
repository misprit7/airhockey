#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <string>
#include <vector>
#include "pubSysCls.h"

using namespace sFnd;

static volatile bool g_stop = false;

void sigHandler(int) { g_stop = true; }

int main(int argc, char *argv[]) {
    signal(SIGINT, sigHandler);

    printf("=== ClearPath-SC Motor Test ===\n\n");

    SysManager *mgr = SysManager::Instance();

    try {
        // Find SC Hub ports
        std::vector<std::string> hubPorts;
        SysManager::FindComHubPorts(hubPorts);

        if (hubPorts.empty()) {
            // Allow manual port specification
            if (argc > 1) {
                hubPorts.push_back(argv[1]);
                printf("No hub ports auto-detected, using manual port: %s\n", argv[1]);
            } else {
                printf("ERROR: No SC Hub ports found.\n");
                printf("  - Is the SC4-Hub connected via USB?\n");
                printf("  - Do you have a /dev/ttyACM* or /dev/ttyXRUSB* device?\n");
                printf("  - Are you in the 'dialout' group? (run: groups)\n");
                printf("\nYou can also specify a port manually: %s /dev/ttyACM0\n", argv[0]);
                return 1;
            }
        }

        printf("Found %zu SC Hub port(s):\n", hubPorts.size());
        for (size_t i = 0; i < hubPorts.size(); i++) {
            printf("  [%zu] %s\n", i, hubPorts[i].c_str());
        }

        // Open first port
        mgr->ComHubPort(0, hubPorts[0].c_str());
        printf("\nOpening port %s...\n", hubPorts[0].c_str());
        mgr->PortsOpen(1);

        IPort &port = mgr->Ports(0);
        printf("Port opened: state=%d, nodes=%d\n", port.OpenState(), port.NodeCount());

        if (port.NodeCount() == 0) {
            printf("ERROR: No nodes (motors) found on port.\n");
            printf("  - Is the motor powered? (24-75V supply)\n");
            printf("  - Is the motor cable connected to the SC4-Hub?\n");
            mgr->PortsClose();
            return 1;
        }

        // Work with first node
        INode &node = port.Nodes(0);

        printf("\n--- Motor Info ---\n");
        printf("  Type:     %d\n", node.Info.NodeType());
        printf("  UserID:   %s\n", node.Info.UserID.Value());
        printf("  FW:       %s\n", node.Info.FirmwareVersion.Value());
        printf("  Serial:   %d\n", node.Info.SerialNumber.Value());
        printf("  Model:    %s\n", node.Info.Model.Value());

        // Enable node
        printf("\nEnabling motor...\n");
        node.EnableReq(false);
        mgr->Delay(200);
        node.Status.AlertsClear();
        node.Motion.NodeStopClear();
        node.EnableReq(true);

        // Wait for ready
        double timeout = mgr->TimeStampMsec() + 5000;
        while (!node.Motion.IsReady()) {
            if (mgr->TimeStampMsec() > timeout) {
                printf("ERROR: Timed out waiting for motor to become ready.\n");
                if (node.Status.Power.Value().fld.InBusLoss) {
                    printf("  Bus power is LOW - check your 75V supply.\n");
                }
                mgr->PortsClose();
                return 1;
            }
        }
        printf("Motor enabled and ready!\n");

        // Configure motion parameters
        node.AccUnit(INode::RPM_PER_SEC);
        node.VelUnit(INode::RPM);
        node.Motion.AccLimit = 50000;   // RPM/sec - moderate acceleration
        node.Motion.VelLimit = 500;     // RPM

        // Read initial position
        node.Motion.PosnMeasured.Refresh();
        printf("Current position: %.0f counts\n", node.Motion.PosnMeasured.Value());

        // Move back and forth
        const int MOVE_COUNTS = 6400;  // ~1 revolution with enhanced encoder
        const int NUM_CYCLES = 3;

        printf("\n--- Moving back and forth (%d cycles, %d counts each) ---\n",
               NUM_CYCLES, MOVE_COUNTS);
        printf("Press Ctrl+C to stop.\n\n");

        for (int cycle = 0; cycle < NUM_CYCLES && !g_stop; cycle++) {
            // Move forward
            printf("Cycle %d: moving +%d counts... ", cycle + 1, MOVE_COUNTS);
            fflush(stdout);
            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart(MOVE_COUNTS);

            timeout = mgr->TimeStampMsec() + 10000;
            while (!node.Motion.MoveIsDone() && !g_stop) {
                if (mgr->TimeStampMsec() > timeout) {
                    printf("TIMEOUT!\n");
                    break;
                }
            }
            node.Motion.PosnMeasured.Refresh();
            printf("done (pos=%.0f)\n", node.Motion.PosnMeasured.Value());

            if (g_stop) break;
            mgr->Delay(500);

            // Move backward
            printf("Cycle %d: moving -%d counts... ", cycle + 1, MOVE_COUNTS);
            fflush(stdout);
            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart(-MOVE_COUNTS);

            timeout = mgr->TimeStampMsec() + 10000;
            while (!node.Motion.MoveIsDone() && !g_stop) {
                if (mgr->TimeStampMsec() > timeout) {
                    printf("TIMEOUT!\n");
                    break;
                }
            }
            node.Motion.PosnMeasured.Refresh();
            printf("done (pos=%.0f)\n", node.Motion.PosnMeasured.Value());

            if (g_stop) break;
            mgr->Delay(500);
        }

        // Disable and cleanup
        printf("\nDisabling motor...\n");
        node.EnableReq(false);
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
