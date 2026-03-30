#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <cmath>
#include <string>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <errno.h>
#include "cdpr.h"

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

static const int DEFAULT_PORT = 8421;

// Simple line-based protocol over TCP:
//   MOVE x_mm y_mm speed_mm_s\n  →  OK x_mm y_mm\n
//   POS\n                         →  OK x_mm y_mm\n
//   RETRACT mm speed_mm_s\n       →  OK\n
//   DISABLE\n                     →  OK\n
//   ENABLE\n                      →  OK\n
//   QUIT\n                        →  (closes connection)

static bool handleCommand(const char *line, CDPR &robot, int client_fd) {
    char resp[256];
    double x, y, speed, mm;

    if (sscanf(line, "MOVE %lf %lf %lf", &x, &y, &speed) == 3) {
        // Blocking move — waits for completion. Use for scripted moves.
        printf("  MOVE (%.1f, %.1f) @ %.0f mm/s\n", x, y, speed);
        if (robot.moveTo(x, y, speed)) {
            snprintf(resp, sizeof(resp), "OK %.2f %.2f\n", robot.x(), robot.y());
            printf("  -> OK\n");
        } else {
            snprintf(resp, sizeof(resp), "ERR moveTo failed\n");
            printf("  -> ERR\n");
        }
    } else if (sscanf(line, "CMD %lf %lf %lf", &x, &y, &speed) == 3) {
        // Non-blocking command — for real-time streaming (web UI).
        printf("  CMD (%.1f, %.1f) @ %.0f mm/s\n", x, y, speed);
        if (robot.commandPosition(x, y, speed)) {
            snprintf(resp, sizeof(resp), "OK %.2f %.2f\n", robot.x(), robot.y());
            printf("  -> OK\n");
        } else {
            snprintf(resp, sizeof(resp), "ERR commandPosition failed\n");
            printf("  -> ERR commandPosition\n");
        }
    } else if (strncmp(line, "POS", 3) == 0) {
        snprintf(resp, sizeof(resp), "OK %.2f %.2f\n", robot.x(), robot.y());
        printf("  POS -> (%.1f, %.1f)\n", robot.x(), robot.y());
    } else if (sscanf(line, "RETRACT %lf %lf", &mm, &speed) == 2) {
        printf("  RETRACT %.1f mm @ %.0f mm/s\n", mm, speed);
        if (robot.retractAll(mm, speed)) {
            snprintf(resp, sizeof(resp), "OK\n");
            printf("  -> OK\n");
        } else {
            snprintf(resp, sizeof(resp), "ERR retract failed\n");
            printf("  -> ERR retract\n");
        }
    } else if (strncmp(line, "DISABLE", 7) == 0) {
        printf("  DISABLE\n");
        robot.disable();
        snprintf(resp, sizeof(resp), "OK\n");
    } else if (strncmp(line, "ENABLE", 6) == 0) {
        printf("  ENABLE\n");
        if (robot.enable()) {
            snprintf(resp, sizeof(resp), "OK\n");
            printf("  -> OK\n");
        } else {
            snprintf(resp, sizeof(resp), "ERR enable failed\n");
            printf("  -> ERR enable\n");
        }
    } else if (strncmp(line, "SETPOS", 6) == 0) {
        if (sscanf(line, "SETPOS %lf %lf", &x, &y) == 2) {
            printf("  SETPOS (%.1f, %.1f)\n", x, y);
            robot.setPosition(x, y);
            snprintf(resp, sizeof(resp), "OK %.2f %.2f\n", robot.x(), robot.y());
            printf("  -> OK\n");
        } else {
            snprintf(resp, sizeof(resp), "ERR bad SETPOS args\n");
            printf("  -> ERR bad SETPOS args\n");
        }
    } else if (strncmp(line, "QUIT", 4) == 0) {
        return false;
    } else {
        snprintf(resp, sizeof(resp), "ERR unknown command\n");
    }

    write(client_fd, resp, strlen(resp));
    return true;
}

int main(int argc, char *argv[]) {
    struct sigaction sa = {};
    sa.sa_handler = sigHandler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    int port = DEFAULT_PORT;
    if (argc > 1) port = atoi(argv[1]);

    CDPRConfig config;
    CDPR robot(config);

    printf("=== CDPR Server ===\n");
    printf("Table: %.0f x %.0f mm\n", config.width, config.height);

    if (!robot.connect()) {
        fprintf(stderr, "Failed to connect to CDPR hardware\n");
        return 1;
    }
    printf("Connected to motors.\n");

    if (!robot.enable()) {
        fprintf(stderr, "Failed to enable motors\n");
        robot.disconnect();
        return 1;
    }
    printf("Motors enabled.\n");

    // Start at center.
    double cx = config.width / 2.0;
    double cy = config.height / 2.0;
    robot.setPosition(cx, cy);
    printf("Position set to center (%.1f, %.1f)\n", cx, cy);

    // Create TCP server socket.
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        robot.disable();
        robot.disconnect();
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        robot.disable();
        robot.disconnect();
        return 1;
    }

    listen(server_fd, 1);
    fcntl(server_fd, F_SETFL, O_NONBLOCK);

    printf("Listening on localhost:%d\n", port);
    printf("Ctrl+C to stop.\n\n");

    int client_fd = -1;
    char buf[4096];
    int buf_len = 0;

    while (!g_stop) {
        // Accept new connection if none active.
        if (client_fd < 0) {
            client_fd = accept(server_fd, NULL, NULL);
            if (client_fd >= 0) {
                fcntl(client_fd, F_SETFL, O_NONBLOCK);
                // Disable Nagle's algorithm for low latency.
                int flag = 1;
                setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
                buf_len = 0;
                printf("Client connected.\n");
            }
        }

        // Read and process commands from client.
        if (client_fd >= 0) {
            int n = read(client_fd, buf + buf_len, sizeof(buf) - buf_len - 1);
            if (n > 0) {
                buf_len += n;
                buf[buf_len] = '\0';

                // Process complete lines.
                char *start = buf;
                char *nl;
                while ((nl = strchr(start, '\n')) != NULL) {
                    *nl = '\0';
                    if (!handleCommand(start, robot, client_fd)) {
                        close(client_fd);
                        client_fd = -1;
                        printf("Client disconnected (QUIT).\n");
                        break;
                    }
                    start = nl + 1;
                }

                // Move remaining partial data to front of buffer.
                if (client_fd >= 0 && start > buf) {
                    buf_len = buf + buf_len - start;
                    memmove(buf, start, buf_len);
                }
            } else if (n == 0) {
                // Client disconnected.
                close(client_fd);
                client_fd = -1;
                buf_len = 0;
                printf("Client disconnected.\n");
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                close(client_fd);
                client_fd = -1;
                buf_len = 0;
                printf("Client error: %s\n", strerror(errno));
            }
        }

        // Small sleep to avoid busy-waiting.
        usleep(100);
    }

    printf("\nShutting down...\n");
    if (client_fd >= 0) close(client_fd);
    close(server_fd);
    robot.disable();
    robot.disconnect();
    printf("Done.\n");
    return 0;
}
