#include <cstdio>
#include <string>
#include <vector>
#include "pubSysCls.h"

using namespace sFnd;

int main(int argc, char *argv[]) {
    printf("=== ClearPath-SC Motor Scan ===\n\n");

    SysManager *mgr = SysManager::Instance();

    try {
        std::vector<std::string> hubPorts;
        SysManager::FindComHubPorts(hubPorts);

        if (hubPorts.empty() && argc > 1) {
            hubPorts.push_back(argv[1]);
        }

        if (hubPorts.empty()) {
            printf("ERROR: No SC Hub ports found.\n");
            return 1;
        }

        printf("Found %zu SC Hub port(s):\n", hubPorts.size());
        for (size_t i = 0; i < hubPorts.size(); i++) {
            printf("  [%zu] %s\n", i, hubPorts[i].c_str());
        }

        size_t portCount = hubPorts.size();
        if (portCount > NET_CONTROLLER_MAX)
            portCount = NET_CONTROLLER_MAX;

        for (size_t i = 0; i < portCount; i++) {
            mgr->ComHubPort(i, hubPorts[i].c_str());
        }

        printf("\nOpening %zu port(s)...\n", portCount);
        mgr->PortsOpen(portCount);

        for (size_t i = 0; i < portCount; i++) {
            IPort &port = mgr->Ports(i);
            printf("\nPort %zu (%s): nodes=%d\n", i, hubPorts[i].c_str(), port.NodeCount());

            for (size_t n = 0; n < port.NodeCount(); n++) {
                INode &node = port.Nodes(n);
                printf("  Node %zu:\n", n);
                printf("    UserID:   %s\n", node.Info.UserID.Value());
                printf("    Model:    %s\n", node.Info.Model.Value());
                printf("    Serial:   %d\n", node.Info.SerialNumber.Value());
                printf("    FW:       %s\n", node.Info.FirmwareVersion.Value());
            }
        }

        mgr->PortsClose();
        printf("\nAll nodes scanned successfully.\n");

    } catch (mnErr &err) {
        printf("ERROR: addr=%d, code=0x%08x\n  %s\n",
               err.TheAddr, err.ErrorCode, err.ErrorMsg);
        mgr->PortsClose();
        return 1;
    }

    return 0;
}
