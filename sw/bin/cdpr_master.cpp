#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <cmath>
#include <string>
#include <mutex>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <sys/stat.h>
#include <dirent.h>
#include "cdpr.h"

static volatile sig_atomic_t g_stop = 0;
void sigHandler(int) { g_stop = 1; }

static const int DEFAULT_PORT = 8421;
static const int TEENSY_BAUD = B115200;
static FILE *g_log = nullptr;

static void logf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    if (g_log) {
        va_start(args, fmt);
        vfprintf(g_log, fmt, args);
        va_end(args);
        fflush(g_log);
    }
}

// ── Teensy serial port ──────────────────────────────────────────────────

static int openTeensy(const char *path) {
    int fd = open(path, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) return -1;

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        close(fd);
        return -1;
    }

    cfsetispeed(&tty, TEENSY_BAUD);
    cfsetospeed(&tty, TEENSY_BAUD);

    // Raw mode
    cfmakeraw(&tty);

    // 8N1, no flow control
    tty.c_cflag &= ~(CSTOPB | CRTSCTS);
    tty.c_cflag |= CLOCAL | CREAD;

    // Non-blocking reads
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    tcsetattr(fd, TCSANOW, &tty);
    tcflush(fd, TCIOFLUSH);

    return fd;
}

// Auto-detect Teensy: look for /dev/ttyACM* devices.
static std::string findTeensy() {
    DIR *dir = opendir("/dev");
    if (!dir) return "";
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strncmp(ent->d_name, "ttyACM", 6) == 0) {
            std::string path = std::string("/dev/") + ent->d_name;
            closedir(dir);
            return path;
        }
    }
    closedir(dir);
    return "";
}

static bool sendTeensy(int fd, const char *cmd) {
    size_t len = strlen(cmd);
    ssize_t n = write(fd, cmd, len);
    if (n != (ssize_t)len) return false;
    // Ensure newline termination
    if (len == 0 || cmd[len - 1] != '\n') {
        write(fd, "\n", 1);
    }
    return true;
}

// ── Shared state from Teensy status lines ───────────────────────────────

struct TeensyStatus {
    double x, y;        // cart position (mm)
    double vx, vy;      // cart velocity (mm/s)
    int c0, c1, c2, c3; // motor step counts
    bool valid;          // at least one status received
};

static std::mutex g_status_mutex;
static TeensyStatus g_status = {};

// Parse "S x y vx vy c0 c1 c2 c3"
static bool parseStatus(const char *line, TeensyStatus &st) {
    double x, y, vx, vy;
    int c0, c1, c2, c3;
    if (sscanf(line, "S %lf %lf %lf %lf %d %d %d %d",
               &x, &y, &vx, &vy, &c0, &c1, &c2, &c3) == 8) {
        st.x = x; st.y = y; st.vx = vx; st.vy = vy;
        st.c0 = c0; st.c1 = c1; st.c2 = c2; st.c3 = c3;
        st.valid = true;
        return true;
    }
    return false;
}

// ── TCP command handling ────────────────────────────────────────────────

static bool g_motors_enabled = false;

// Wait for "OK" from Teensy (with timeout). Returns true if OK received.
static bool waitTeensyOK(int teensy_fd, int timeout_ms = 5000) {
    char buf[256];
    int pos = 0;
    int elapsed = 0;
    while (elapsed < timeout_ms) {
        int n = read(teensy_fd, buf + pos, sizeof(buf) - pos - 1);
        if (n > 0) {
            pos += n;
            buf[pos] = '\0';
            // Check for OK line
            char *start = buf;
            char *nl;
            while ((nl = strchr(start, '\n')) != NULL) {
                *nl = '\0';
                if (strncmp(start, "OK", 2) == 0) {
                    logf("  Teensy: %s\n", start);
                    return true;
                } else if (strncmp(start, "ERR", 3) == 0) {
                    logf("  Teensy error: %s\n", start);
                    return false;
                } else if (start[0] == 'S') {
                    // Status line — update global state while waiting
                    TeensyStatus st;
                    if (parseStatus(start, st)) {
                        std::lock_guard<std::mutex> lock(g_status_mutex);
                        g_status = st;
                    }
                }
                start = nl + 1;
            }
            // Move remaining to front
            if (start > buf) {
                pos = buf + pos - start;
                memmove(buf, start, pos);
            }
        }
        usleep(1000);
        elapsed++;
    }
    logf("  Teensy: timeout waiting for OK\n");
    return false;
}

// Return: 1 = continue, 0 = quit, -1 = error
static int handleCommand(const char *line, CDPR &robot, int client_fd, int teensy_fd) {
    char resp[256];
    double x, y, speed;

    if (strncmp(line, "ENABLE", 6) == 0) {
        logf("  ENABLE\n");

        // 1. Enable sFoundation motors
        if (!robot.enable()) {
            snprintf(resp, sizeof(resp), "ERR motor enable failed\n");
            write(client_fd, resp, strlen(resp));
            return 1;
        }

        // 2. Apply tensioning
        robot.tension();

        // 3. Set position at center
        double cx = robot.config().width / 2.0;
        double cy = robot.config().height / 2.0;
        robot.setPosition(cx, cy);

        // 4. Send CAL, TENSION, START to Teensy
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "CAL %.1f %.1f\n", cx, cy);
        sendTeensy(teensy_fd, cmd);
        if (!waitTeensyOK(teensy_fd)) {
            snprintf(resp, sizeof(resp), "ERR teensy CAL failed\n");
            write(client_fd, resp, strlen(resp));
            return 1;
        }

        sendTeensy(teensy_fd, "TENSION 2\n");
        if (!waitTeensyOK(teensy_fd, 10000)) {
            snprintf(resp, sizeof(resp), "ERR teensy TENSION failed\n");
            write(client_fd, resp, strlen(resp));
            return 1;
        }

        sendTeensy(teensy_fd, "START\n");
        if (!waitTeensyOK(teensy_fd)) {
            snprintf(resp, sizeof(resp), "ERR teensy START failed\n");
            write(client_fd, resp, strlen(resp));
            return 1;
        }

        g_motors_enabled = true;
        snprintf(resp, sizeof(resp), "OK\n");
        logf("  -> enabled\n");

    } else if (strncmp(line, "DISABLE", 7) == 0) {
        logf("  DISABLE\n");

        // 1. Stop Teensy motion controller
        sendTeensy(teensy_fd, "STOP\n");
        waitTeensyOK(teensy_fd);

        // 2. Release tension
        sendTeensy(teensy_fd, "RELEASE\n");
        waitTeensyOK(teensy_fd, 10000);

        // 3. Disable sFoundation motors
        robot.disable();
        g_motors_enabled = false;

        snprintf(resp, sizeof(resp), "OK\n");
        logf("  -> disabled\n");

    } else if (sscanf(line, "CMD %lf %lf %lf", &x, &y, &speed) >= 2) {
        // Accept "CMD x y speed" or "CMD x y" (speed ignored, Teensy handles trajectory)
        if (!g_motors_enabled) {
            snprintf(resp, sizeof(resp), "ERR motors not enabled\n");
            write(client_fd, resp, strlen(resp));
            return 1;
        }

        // Forward to Teensy as "CMD x y\n"
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "CMD %.2f %.2f\n", x, y);
        sendTeensy(teensy_fd, cmd);

        snprintf(resp, sizeof(resp), "OK\n");

    } else if (strncmp(line, "POS", 3) == 0) {
        std::lock_guard<std::mutex> lock(g_status_mutex);
        if (g_status.valid) {
            snprintf(resp, sizeof(resp), "OK %.2f %.2f %.2f %.2f\n",
                     g_status.x, g_status.y, g_status.vx, g_status.vy);
        } else {
            snprintf(resp, sizeof(resp), "ERR no status available\n");
        }

    } else if (strncmp(line, "STATUS", 6) == 0) {
        std::lock_guard<std::mutex> lock(g_status_mutex);
        if (g_status.valid) {
            snprintf(resp, sizeof(resp), "OK %.2f %.2f %.2f %.2f %d %d %d %d\n",
                     g_status.x, g_status.y, g_status.vx, g_status.vy,
                     g_status.c0, g_status.c1, g_status.c2, g_status.c3);
        } else {
            snprintf(resp, sizeof(resp), "ERR no status available\n");
        }

    } else if (strncmp(line, "QUIT", 4) == 0) {
        return 0;

    } else {
        snprintf(resp, sizeof(resp), "ERR unknown command\n");
    }

    write(client_fd, resp, strlen(resp));
    return 1;
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    struct sigaction sa = {};
    sa.sa_handler = sigHandler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    int port = DEFAULT_PORT;
    const char *teensy_path = nullptr;

    // Parse args: [--port N] [--teensy /dev/ttyACMx]
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--teensy") == 0 && i + 1 < argc) {
            teensy_path = argv[++i];
        } else {
            // Bare argument: try as teensy path
            teensy_path = argv[i];
        }
    }

    mkdir("logs", 0755);
    g_log = fopen("logs/cdpr_master.log", "w");
    FILE *errlog = fopen("logs/cdpr_master_debug.log", "w");
    if (errlog) {
        dup2(fileno(errlog), STDERR_FILENO);
        fclose(errlog);
        setvbuf(stderr, NULL, _IOLBF, 0);
    }

    logf("=== CDPR Master ===\n");

    // ── Connect to Teensy ──

    std::string teensy_dev;
    if (teensy_path) {
        teensy_dev = teensy_path;
    } else {
        teensy_dev = findTeensy();
        if (teensy_dev.empty()) {
            logf("ERROR: No Teensy found (no /dev/ttyACM* device)\n");
            return 1;
        }
    }

    logf("Opening Teensy on %s\n", teensy_dev.c_str());
    int teensy_fd = openTeensy(teensy_dev.c_str());
    if (teensy_fd < 0) {
        logf("ERROR: Failed to open %s: %s\n", teensy_dev.c_str(), strerror(errno));
        return 1;
    }
    logf("Teensy connected.\n");

    // ── Connect to sFoundation motors ──

    CDPRConfig config;
    CDPR robot(config);

    if (!robot.connect()) {
        logf("ERROR: Failed to connect to CDPR motors\n");
        close(teensy_fd);
        return 1;
    }
    logf("Motors connected (not yet enabled).\n");

    // ── TCP server ──

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        close(teensy_fd);
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
        close(teensy_fd);
        robot.disconnect();
        return 1;
    }

    listen(server_fd, 1);
    fcntl(server_fd, F_SETFL, O_NONBLOCK);

    logf("Listening on localhost:%d\n", port);
    logf("Ctrl+C to stop.\n\n");

    // ── Main loop ──

    int client_fd = -1;
    char tcp_buf[4096];
    int tcp_len = 0;

    // Teensy read buffer
    char ser_buf[4096];
    int ser_len = 0;

    while (!g_stop) {
        // ── Accept TCP connections ──
        if (client_fd < 0) {
            client_fd = accept(server_fd, NULL, NULL);
            if (client_fd >= 0) {
                fcntl(client_fd, F_SETFL, O_NONBLOCK);
                int flag = 1;
                setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
                tcp_len = 0;
                logf("Client connected.\n");
            }
        }

        // ── Read from Teensy ──
        {
            int n = read(teensy_fd, ser_buf + ser_len, sizeof(ser_buf) - ser_len - 1);
            if (n > 0) {
                ser_len += n;
                ser_buf[ser_len] = '\0';

                char *start = ser_buf;
                char *nl;
                while ((nl = strchr(start, '\n')) != NULL) {
                    *nl = '\0';
                    // Parse status lines
                    if (start[0] == 'S' && start[1] == ' ') {
                        TeensyStatus st;
                        if (parseStatus(start, st)) {
                            std::lock_guard<std::mutex> lock(g_status_mutex);
                            g_status = st;
                        }
                    } else if (start[0] != '\0') {
                        // Log other Teensy output (OK, ERR, etc.)
                        logf("  Teensy: %s\n", start);
                    }
                    start = nl + 1;
                }

                if (start > ser_buf) {
                    ser_len = ser_buf + ser_len - start;
                    memmove(ser_buf, start, ser_len);
                }
            }
        }

        // ── Read from TCP client ──
        if (client_fd >= 0) {
            int n = read(client_fd, tcp_buf + tcp_len, sizeof(tcp_buf) - tcp_len - 1);
            if (n > 0) {
                tcp_len += n;
                tcp_buf[tcp_len] = '\0';

                char *start = tcp_buf;
                char *nl;
                while ((nl = strchr(start, '\n')) != NULL) {
                    *nl = '\0';
                    int rc = handleCommand(start, robot, client_fd, teensy_fd);
                    if (rc == 0) {
                        close(client_fd);
                        client_fd = -1;
                        logf("Client disconnected (QUIT).\n");
                        break;
                    } else if (rc < 0) {
                        logf("ERROR: command failed, shutting down.\n");
                        close(client_fd);
                        client_fd = -1;
                        g_stop = 1;
                        break;
                    }
                    start = nl + 1;
                }

                if (client_fd >= 0 && start > tcp_buf) {
                    tcp_len = tcp_buf + tcp_len - start;
                    memmove(tcp_buf, start, tcp_len);
                }
            } else if (n == 0) {
                close(client_fd);
                client_fd = -1;
                tcp_len = 0;
                logf("Client disconnected.\n");
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                close(client_fd);
                client_fd = -1;
                tcp_len = 0;
                logf("Client error: %s\n", strerror(errno));
            }
        }

        usleep(100);
    }

    // ── Shutdown ──
    logf("\nShutting down...\n");

    if (g_motors_enabled) {
        sendTeensy(teensy_fd, "STOP\n");
        usleep(50000);
        sendTeensy(teensy_fd, "RELEASE\n");
        usleep(500000);
        robot.disable();
    }

    if (client_fd >= 0) close(client_fd);
    close(server_fd);
    close(teensy_fd);
    robot.disconnect();
    logf("Done.\n");
    if (g_log) fclose(g_log);
    return 0;
}
