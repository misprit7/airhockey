#!/bin/bash
# Fetch and build the Teknic sFoundation SDK for ClearPath-SC motors.
# Run this once before 'make'.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SF_DIR="$SCRIPT_DIR/third_party/sFoundation"

if [ -d "$SF_DIR" ]; then
    echo "sFoundation already exists at $SF_DIR"
    echo "Delete it first if you want to re-fetch: rm -rf $SF_DIR"
    exit 0
fi

echo "Cloning sFoundation SDK..."
git clone --depth 1 https://github.com/magnus-haw/clearpathAPI.git /tmp/clearpathAPI_$$
mkdir -p "$SCRIPT_DIR/third_party"
mv /tmp/clearpathAPI_$$/sFoundation "$SF_DIR"
rm -rf /tmp/clearpathAPI_$$

# Patch: fix GCC 14+ build errors (pointer vs integer comparisons)
sed -i 's/if (pfd <= 0)/if (pfd == NULL)/' "$SF_DIR/sFoundation/src-linux/lnkAccessLinux.cpp"
sed -i 's/if (fgets(buf, sizeof(buf), pfd) > 0)/if (fgets(buf, sizeof(buf), pfd) != NULL)/' "$SF_DIR/sFoundation/src-linux/lnkAccessLinux.cpp"

# Patch: support both ttyXRUSB (Exar driver) and ttyACM (cdc_acm) for SC4-Hub
cat > /tmp/port_patch_$$.py << 'PYEOF'
import re, sys
f = sys.argv[1]
src = open(f).read()
old = '''MN_EXPORT cnErrCode MN_DECL infcGetHubPorts(std::vector<std::string> &comHubPorts) {

    comHubPorts.clear();
    FILE *pfd = popen("ls /dev/ttyXRUSB*", "r");

    if (pfd == NULL) {
        throw std::runtime_error("Command or process could not be executed.");
    }

    while (!feof(pfd)) {
        char buf[ 1024 ] = {0};

        if (fgets(buf, sizeof(buf), pfd) != NULL) {
            std::string str(buf);
            // TODO: check the VID/PID of the device using udevadm/libudev to
            // verify that this is a Teknic SC4-Hub (vid=2890, pid=0213)
            comHubPorts.push_back(str.substr(0, str.length() - 1));
        }
    }
    pclose(pfd);

    return MN_OK;
}'''
new = '''MN_EXPORT cnErrCode MN_DECL infcGetHubPorts(std::vector<std::string> &comHubPorts) {

    comHubPorts.clear();

    // Search for both Exar driver (ttyXRUSB*) and standard CDC ACM (ttyACM*) devices.
    // The SC4-Hub works with either driver.
    const char *patterns[] = {
        "ls /dev/ttyXRUSB* 2>/dev/null",
        "ls /dev/ttyACM* 2>/dev/null",
        NULL
    };

    for (int p = 0; patterns[p] != NULL; p++) {
        FILE *pfd = popen(patterns[p], "r");
        if (pfd == NULL) {
            continue;
        }

        while (!feof(pfd)) {
            char buf[ 1024 ] = {0};

            if (fgets(buf, sizeof(buf), pfd) != NULL) {
                std::string str(buf);
                comHubPorts.push_back(str.substr(0, str.length() - 1));
            }
        }
        pclose(pfd);
    }

    return MN_OK;
}'''
if old in src:
    src = src.replace(old, new)
    open(f, 'w').write(src)
    print("Applied port discovery patch")
else:
    print("Port discovery patch already applied or source changed")
PYEOF
python3 /tmp/port_patch_$$.py "$SF_DIR/sFoundation/src-linux/lnkAccessLinux.cpp"
rm /tmp/port_patch_$$.py

echo "Building sFoundation..."
make -C "$SF_DIR/sFoundation" -j$(nproc)
ldconfig -n "$SF_DIR/sFoundation"

echo ""
echo "Done! Now run 'make' in sw/ to build the control programs."
