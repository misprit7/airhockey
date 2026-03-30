#include "cdpr.h"
#include <csignal>
#include <cstdio>

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

int main() {
  struct sigaction sa = {};
  sa.sa_handler = sigHandler;
  sigaction(SIGINT, &sa, NULL);

  CDPRConfig config;

  CDPR robot(config);

  printf("=== CDPR Test ===\n\n");
  printf("Table: %.0f x %.0f mm\n", config.width, config.height);
  printf("Spool diameter: %.1f mm (circumference: %.1f mm)\n",
         config.spool_diameter, M_PI * config.spool_diameter);
  printf("Max velocity: %.0f mm/s\n", config.max_velocity);
  printf("Edge margin: %.0f mm\n\n", config.edge_margin);

  if (!robot.connect())
    return 1;
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

  double lengths[4];
  robot.cableLengths(cx, cy, lengths);
  printf("Cable lengths at center:\n");
  for (int i = 0; i < 4; i++) {
    printf("  Motor %d: %.1f mm\n", i, lengths[i]);
  }

  // --- Phase 1: Tension cables ---
  double tension_mm = 5;      // set >0 to tension cables before motion
  double tension_speed = 5.0; // mm/s

  if (tension_mm > 0) {
    printf("\nPhase 1: Retract all cables by %.1fmm to tension.\n", tension_mm);
    printf("Press Enter to tension, Ctrl+C to abort: ");
    fflush(stdout);
    if (getchar() == EOF || g_stop)
      goto cleanup;

    printf("Tensioning... ");
    fflush(stdout);
    if (!robot.retractAll(tension_mm, tension_speed)) {
      printf("FAILED\n");
      goto cleanup;
    }
    printf("done\n");
  }

  // --- Phase 2: Square motion ---
  {
    double step = 50.0;
    // double speed = 800.0;
    double speed = 800.0;

    struct {
      double dx, dy;
      const char *desc;
    } moves[] = {
        {step, 0, "right"},
        {0, step, "up"},
        {-step, 0, "left"},
        {0, -step, "down"},
    };

    printf("\nPhase 2: Moving in a %.0fmm square at %.0f mm/s.\n", step, speed);
    printf("Press Enter to start, Ctrl+C to skip to release: ");
    fflush(stdout);
    if (getchar() == EOF || g_stop)
      goto release;

    for (auto &m : moves) {
      if (g_stop)
        break;
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
    printf("Final position: (%.1f, %.1f)\n", robot.x(), robot.y());
  }

release:
  if (tension_mm > 0) {
    // --- Phase 3: Release tension ---
    printf("\nPhase 3: Releasing tension (extending %.1fmm).\n", tension_mm);
    printf("Press Enter to release, Ctrl+C to abort: ");
    fflush(stdout);
    if (getchar() == EOF)
      goto cleanup;

    printf("Releasing... ");
    fflush(stdout);
    if (!robot.retractAll(-tension_mm, tension_speed)) {
      printf("FAILED\n");
    } else {
      printf("done\n");
    }
  }

cleanup:
  robot.disable();
  robot.disconnect();
  printf("Done!\n");
  return 0;
}
