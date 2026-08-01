#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include "pubSysCls.h"

using namespace sFnd;

// ============================================================================
// Calibration pose capture — passive encoder reader.
//
// Motors stay DISABLED the whole time; you position the mallet and manage
// string tautness entirely by hand (a back-driven ClearPath still updates
// its encoder). At each reference position: press the mallet against its
// reference, gently snug each of the four strings so none sags, hold still,
// and record. Fit the motor anchors afterwards with ai/bin/calibrate_fit.py.
//
// The tool first walks a guided sequence of four default poses (Enter to
// record each, 'k' to skip, 'q' to drop to manual mode):
//   1. bottom-right corner (pressed into the corner)
//   2. against the bottom rail, 24cm gap between mallet edge and right rail
//   3. top-right corner
//   4. against the top rail, 24cm gap to the right rail
// then drops into manual mode for extra poses:
//   c1        record the bottom-right corner pose
//   c2        record the top-right corner pose
//   r X Y     record a pose at known mallet-center X Y (mm, grid frame)
//   s         show live encoder counts
//   q         save JSON and quit
//
// Tips: re-recording the same position with the mallet rotated ~45deg is a
// free extra pose for the fit. Spread poses in x — don't take them all along
// one wall. Counts only matter at the instant of recording (5-sample
// average), so tautness only needs to hold while you record.
//
// GRID frame: origin at the corner hole nearest the human's right corner,
// +x toward the robot, +y from the human's right to their left. See
// fw/include/cdpr_config.h. Rail positions used here are approximate.
// ============================================================================

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

// GRID frame (origin = corner hole nearest the human's right corner; see
// fw/include/cdpr_config.h). This tool references the RAILS, which are only
// approximate in that frame — superseded by the vision calibration
// (vision/bin/measure_motors.py); kept as a fallback.
static const double RAIL_MAX_X  = 1987.9;
static const double RAIL_MIN_Y  = -32.6;
static const double RAIL_MAX_Y  = 972.4;
static const double MALLET_R_MM = 50.6; // radius (mallet is 101.2 mm diameter)

static const int SAMPLES_PER_POSE = 5;  // averaged per recording

// Gap between the RIGHT rail and the mallet's edge for the rail presets
// (e.g. a 24cm spacer block laid along the rail).
static const double RAIL_OFFSET_MM = 240.0;

struct Pose {
    double x, y;
    double counts[4];
};

struct Preset {
    const char *desc;
    double x, y;
};

static const Preset PRESETS[] = {
    {"bottom-right corner — press mallet into the corner",
     RAIL_MAX_X - MALLET_R_MM, RAIL_MIN_Y + MALLET_R_MM},
    {"bottom rail — mallet edge 24cm from the right rail",
     RAIL_MAX_X - RAIL_OFFSET_MM - MALLET_R_MM, RAIL_MIN_Y + MALLET_R_MM},
    {"top-right corner — press mallet into the corner",
     RAIL_MAX_X - MALLET_R_MM, RAIL_MAX_Y - MALLET_R_MM},
    {"top rail — mallet edge 24cm from the right rail",
     RAIL_MAX_X - RAIL_OFFSET_MM - MALLET_R_MM, RAIL_MAX_Y - MALLET_R_MM},
};
static const int NUM_PRESETS = sizeof(PRESETS) / sizeof(PRESETS[0]);

static int countsPerRev(INode &node) {
    std::string model = node.Info.Model.Value();
    if (model.find("-EL") != std::string::npos)
        return 6400;
    return 800;
}

static void readCounts(SysManager *mgr, IPort &port, double out[4], int samples) {
    for (int m = 0; m < 4; m++) out[m] = 0.0;
    for (int s = 0; s < samples; s++) {
        for (size_t m = 0; m < port.NodeCount(); m++) {
            port.Nodes(m).Motion.PosnMeasured.Refresh();
            out[m] += port.Nodes(m).Motion.PosnMeasured.Value();
        }
        if (samples > 1) mgr->Delay(50);
    }
    for (int m = 0; m < 4; m++) out[m] /= samples;
}

static void recordPose(SysManager *mgr, IPort &port, std::vector<Pose> &poses,
                       double x, double y) {
    Pose p;
    p.x = x;
    p.y = y;
    readCounts(mgr, port, p.counts, SAMPLES_PER_POSE);
    poses.push_back(p);
    printf("  Recorded pose %zu at (%.1f, %.1f): counts %.0f %.0f %.0f %.0f\n",
           poses.size(), x, y, p.counts[0], p.counts[1], p.counts[2], p.counts[3]);
}

static bool savePoses(const char *path, IPort &port, const std::vector<Pose> &poses) {
    FILE *f = fopen(path, "w");
    if (!f) {
        printf("ERROR: cannot write %s\n", path);
        return false;
    }
    fprintf(f, "{\n  \"cpr\": [");
    for (size_t m = 0; m < port.NodeCount(); m++)
        fprintf(f, "%s%d", m ? ", " : "", countsPerRev(port.Nodes(m)));
    fprintf(f, "],\n  \"poses\": [\n");
    for (size_t i = 0; i < poses.size(); i++) {
        const Pose &p = poses[i];
        fprintf(f, "    {\"x\": %.2f, \"y\": %.2f, \"counts\": [%.1f, %.1f, %.1f, %.1f]}%s\n",
                p.x, p.y, p.counts[0], p.counts[1], p.counts[2], p.counts[3],
                i + 1 < poses.size() ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("Saved %zu poses to %s\n", poses.size(), path);
    return true;
}

int main(int argc, char *argv[]) {
    struct sigaction sa = {};
    sa.sa_handler = sigHandler;
    sigaction(SIGINT, &sa, NULL);

    const char *outPath = (argc > 1) ? argv[1] : "calib_poses.json";

    printf("=== CDPR Calibration Pose Capture (passive) ===\n");
    printf("Output: %s\n", outPath);
    printf("Motors stay disabled — position the mallet and snug the strings "
           "by hand.\n");
    printf("Corner shortcuts: c1 = (%.1f, %.1f), c2 = (%.1f, %.1f)\n\n",
           RAIL_MAX_X - MALLET_R_MM, RAIL_MIN_Y + MALLET_R_MM,
           RAIL_MAX_X - MALLET_R_MM, RAIL_MAX_Y - MALLET_R_MM);

    SysManager *mgr = SysManager::Instance();
    std::vector<Pose> poses;

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
        printf("Found %d motors\n", port.NodeCount());
        if (port.NodeCount() != 4) {
            printf("ERROR: expected 4 motors\n");
            mgr->PortsClose();
            return 1;
        }

        // Make sure nothing is left enabled from a previous tool/run.
        for (size_t m = 0; m < port.NodeCount(); m++) {
            try { port.Nodes(m).EnableReq(false); } catch (mnErr &) {}
        }

        char line[128];

        // Guided default sequence.
        for (int i = 0; i < NUM_PRESETS && !g_stop; i++) {
            const Preset &P = PRESETS[i];
            printf("\nPose %d/%d: %s\n"
                   "  -> center (%.1f, %.1f). Snug all four strings, then\n"
                   "  Enter = record, k = skip, q = manual mode: ",
                   i + 1, NUM_PRESETS, P.desc, P.x, P.y);
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) {
                g_stop = 1;
                break;
            }
            if (line[0] == 'q' || line[0] == 'Q') break;
            if (line[0] == 'k' || line[0] == 'K') {
                printf("  skipped\n");
                continue;
            }
            recordPose(mgr, port, poses, P.x, P.y);
        }

        printf("\nManual mode — extra poses sharpen the fit (e.g. rotate the\n"
               "mallet ~45deg in place at a reference and re-record it).\n");
        printf("Commands: c1 | c2 | r X Y | s | q\n\n");
        while (!g_stop) {
            printf("> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin))
                break;
            double x, y;
            if (strncmp(line, "c1", 2) == 0) {
                recordPose(mgr, port, poses, RAIL_MAX_X - MALLET_R_MM, RAIL_MIN_Y + MALLET_R_MM);
            } else if (strncmp(line, "c2", 2) == 0) {
                recordPose(mgr, port, poses, RAIL_MAX_X - MALLET_R_MM,
                           RAIL_MAX_Y - MALLET_R_MM);
            } else if (sscanf(line, "r %lf %lf", &x, &y) == 2) {
                recordPose(mgr, port, poses, x, y);
            } else if (line[0] == 's') {
                double c[4];
                readCounts(mgr, port, c, 1);
                printf("  counts: %.0f %.0f %.0f %.0f\n", c[0], c[1], c[2], c[3]);
            } else if (line[0] == 'q') {
                break;
            } else if (line[0] != '\n') {
                printf("  ? commands: c1 | c2 | r X Y | s | q\n");
            }
        }

        if (!poses.empty())
            savePoses(outPath, port, poses);
        else
            printf("No poses recorded.\n");

        mgr->PortsClose();

    } catch (mnErr &err) {
        printf("ERROR: addr=%d, code=0x%08x\n  %s\n",
               err.TheAddr, err.ErrorCode, err.ErrorMsg);
        mgr->PortsClose();
        return 1;
    }

    return 0;
}
