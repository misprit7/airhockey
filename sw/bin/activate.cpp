#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "pubSysCls.h"

using namespace sFnd;

static SysManager *mgr = nullptr;
static IPort *port = nullptr;
static int nodeCount = 0;
static bool motorEnabled[4] = {};

static bool enableOne(int i) {
    INode &node = port->Nodes(i);
    node.EnableReq(false);
    mgr->Delay(100);
    node.Status.AlertsClear();
    node.Motion.NodeStopClear();
    node.EnableReq(true);

    double timeout = mgr->TimeStampMsec() + 5000;
    while (!node.Motion.IsReady()) {
        if (mgr->TimeStampMsec() > timeout) {
            printf("ERROR: Motor %d enable timed out\n", i);
            return false;
        }
    }
    printf("  Motor %d enabled\n", i);
    motorEnabled[i] = true;
    return true;
}

static void disableOne(int i) {
    port->Nodes(i).EnableReq(false);
    printf("  Motor %d disabled\n", i);
    motorEnabled[i] = false;
}

static bool enableAll() {
    for (int i = 0; i < nodeCount; i++)
        if (!enableOne(i)) return false;
    return true;
}

static void disableAll() {
    for (int i = 0; i < nodeCount; i++)
        disableOne(i);
}

static bool anyEnabled() {
    for (int i = 0; i < nodeCount; i++)
        if (motorEnabled[i]) return true;
    return false;
}

int main(int argc, char *argv[]) {
    printf("=== ClearPath-SC Motor Activate/Deactivate ===\n\n");

    mgr = SysManager::Instance();

    try {
        std::vector<std::string> hubPorts;
        SysManager::FindComHubPorts(hubPorts);

        if (hubPorts.empty() && argc > 1)
            hubPorts.push_back(argv[1]);

        if (hubPorts.empty()) {
            printf("ERROR: No SC Hub ports found.\n");
            return 1;
        }

        printf("Using port: %s\n", hubPorts[0].c_str());
        mgr->ComHubPort(0, hubPorts[0].c_str());
        mgr->PortsOpen(1);

        port = &mgr->Ports(0);
        nodeCount = port->NodeCount();
        printf("Found %d motor(s)\n\n", nodeCount);

        if (nodeCount == 0) {
            printf("No motors found.\n");
            mgr->PortsClose();
            return 1;
        }

        printf("Commands:\n");
        printf("  ENTER     - toggle all motors\n");
        printf("  0-3       - toggle single motor\n");
        printf("  q         - quit\n\n");

        char buf[64];
        while (true) {
            // Show per-motor status
            printf("[");
            for (int i = 0; i < nodeCount; i++)
                printf("%d:%s ", i, motorEnabled[i] ? "ON" : "off");
            printf("] > ");
            fflush(stdout);

            if (!fgets(buf, sizeof(buf), stdin))
                break;

            if (buf[0] == 'q' || buf[0] == 'Q')
                break;

            if (buf[0] >= '0' && buf[0] < '0' + nodeCount) {
                int m = buf[0] - '0';
                if (motorEnabled[m]) {
                    printf("Disabling motor %d...\n", m);
                    disableOne(m);
                } else {
                    printf("Enabling motor %d...\n", m);
                    enableOne(m);
                }
            } else {
                // ENTER — toggle all
                if (anyEnabled()) {
                    printf("Disabling all motors...\n");
                    disableAll();
                } else {
                    printf("Enabling all motors...\n");
                    enableAll();
                }
            }
            printf("\n");
        }

        if (anyEnabled()) {
            printf("Disabling motors before exit...\n");
            disableAll();
        }

        mgr->PortsClose();
    } catch (mnErr &e) {
        printf("sFoundation error: %s\n", e.ErrorMsg);
        return 1;
    }

    printf("Done.\n");
    return 0;
}
