#include <cstdio>
#include <csignal>
#include "cdpr.h"

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

int main() {
    struct sigaction sa = {};
    sa.sa_handler = sigHandler;
    sigaction(SIGINT, &sa, NULL);

    CDPRConfig config;
    // Physical parameters are set in cdpr_config.h defaults.
    // Override here if needed:
    //   config.width = 606.0;
    //   config.height = 730.0;
    //   config.spool_diameter = 20.5;
    //   config.counts_per_rev = {800, 6400, 800, 6400};

    CDPR robot(config);

    printf("=== CDPR Test ===\n\n");
    printf("Table: %.0f x %.0f mm\n", config.width, config.height);
    printf("Spool diameter: %.1f mm (circumference: %.1f mm)\n",
           config.spool_diameter, M_PI * config.spool_diameter);
    printf("Max velocity: %.0f mm/s\n", config.max_velocity);
    printf("Edge margin: %.0f mm\n\n", config.edge_margin);

    if (!robot.connect()) return 1;
    printf("Connected to 4 motors.\n");

    if (!robot.enable()) {
        robot.disconnect();
        return 1;
    }
    printf("All motors enabled.\n\n");

    // Assume cart starts at center of table.
    double cx = config.width / 2.0;
    double cy = config.height / 2.0;
    robot.setPosition(cx, cy);
    printf("Initial position set to center: (%.1f, %.1f) mm\n", cx, cy);

    // Print cable lengths at center.
    double lengths[4];
    robot.cableLengths(cx, cy, lengths);
    printf("Cable lengths at center:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Motor %d: %.1f mm\n", i, lengths[i]);
    }

    // Small test moves — 5mm square at 5 mm/s.
    double step = 5.0;
    double speed = 5.0;

    struct { double dx, dy; const char *desc; } moves[] = {
        { step,  0,     "right" },
        { 0,     step,  "up"    },
        {-step,  0,     "left"  },
        { 0,    -step,  "down"  },
    };

    printf("\nMoving in a %.0fmm square at %.0f mm/s.\n", step, speed);
    for (auto &m : moves) {
        if (g_stop) break;
        double tx = robot.x() + m.dx;
        double ty = robot.y() + m.dy;
        printf("  %s to (%.1f, %.1f)... ", m.desc, tx, ty);
        fflush(stdout);

        if (robot.moveTo(tx, ty, speed)) {
            printf("done\n");
        } else {
            printf("FAILED\n");
            break;
        }
    }

    printf("\nFinal position: (%.1f, %.1f)\n", robot.x(), robot.y());

    robot.disable();
    robot.disconnect();
    printf("Done!\n");
    return 0;
}
