// Compile the real command-wait/parser code; never call its hardware main.
#define main unused_hardware_main
#include "../bin/cdpr_master.cpp"
#undef main
#include <cassert>
#include <condition_variable>
#include <future>

static void test_watchdog_rechecks_enable_after_waiting_for_mutex() {
    g_stop = 0;
    g_fault = false;
    g_motors_enabled = true;
    g_watchdog_waiting = false;
    g_watchdog_checked = 0;
    std::unique_lock<std::mutex> lock(g_robot_mutex);
    // No hardware object: a poll queued before DISABLE must never read drives
    // after it acquires the mutex and sees the completed disable.
    std::thread worker([]() { faultWatchdog(nullptr, -1); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!g_watchdog_waiting && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(g_watchdog_waiting);
    g_motors_enabled = false;
    lock.unlock();
    while (g_watchdog_waiting && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(!g_watchdog_waiting);
    g_stop = 1;
    worker.join();
    assert(!g_fault && g_watchdog_checked == 0);
    g_stop = 0;
}

static void test_slow_health_does_not_block_commands() {
    // The real worker holds the SDK mutex while a diagnostic read stalls.
    // POS and CMD must still finish before that diagnostic is released.
    // All serial/TCP traffic below stays in socketpairs; no device is opened.
    std::mutex gate;
    std::condition_variable cv;
    bool entered = false, release = false;
    g_stop = 0;
    g_fault = false;
    g_motors_enabled = true;
    std::thread worker([&]() {
        backgroundHealth([&]() {
            std::unique_lock<std::mutex> lock(gate);
            entered = true;
            cv.notify_all();
            cv.wait(lock, [&]() { return release; });
        });
    });
    {
        std::unique_lock<std::mutex> lock(gate);
        assert(cv.wait_for(lock, std::chrono::seconds(3), [&]() { return entered; }));
    }
    int tcp[2], teensy[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tcp) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, teensy) == 0);
    assert(fcntl(teensy[0], F_SETFL, O_NONBLOCK) == 0);
    std::thread emulator([&]() {
        std::string command;
        char c;
        while (read(teensy[1], &c, 1) == 1 && c != '\n') command += c;
        assert(command == "CMD 1700.00 500.00");
        const char ok[] = "OK CMD\n";
        assert(write(teensy[1], ok, sizeof(ok)-1) == sizeof(ok)-1);
    });
    auto commands = std::async(std::launch::async, [&]() {
        ClearPath disconnected_robot;
        char reply[256] = {};
        assert(handleCommand("POS", disconnected_robot, tcp[0], teensy[0]) == 1);
        assert(read(tcp[1], reply, sizeof(reply)-1) > 0 && !strncmp(reply, "OK ", 3));
        assert(handleCommand("CMD 1700 500 0", disconnected_robot, tcp[0], teensy[0]) == 1);
        memset(reply, 0, sizeof(reply));
        assert(read(tcp[1], reply, sizeof(reply)-1) > 0 && !strncmp(reply, "OK", 2));
        // Fault-latched commands still fail without issuing another serial move.
        g_fault = true;
        assert(handleCommand("CMD 1700 500 0", disconnected_robot, tcp[0], teensy[0]) == 1);
        memset(reply, 0, sizeof(reply));
        assert(read(tcp[1], reply, sizeof(reply)-1) > 0 && !strncmp(reply, "ERR fault", 9));
    });
    const bool completed_while_health_blocked =
        commands.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    {
        std::lock_guard<std::mutex> lock(gate);
        release = true;
        g_stop = 1;
    }
    cv.notify_all();
    worker.join();
    commands.get();
    emulator.join();
    assert(completed_while_health_blocked);
    for (int fd : {tcp[0], tcp[1], teensy[0], teensy[1]}) close(fd);
    g_motors_enabled = false;
    g_fault = false;
    g_stop = 0;
}

static void test_startup_pretension_configuration_never_moves() {
    int tcp[2], serial[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tcp) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, serial) == 0);
    assert(fcntl(serial[1], F_SETFL, O_NONBLOCK) == 0);
    ClearPath disconnected_robot;
    auto command = [&](const char *text, const char *prefix) {
        assert(handleCommand(text, disconnected_robot, tcp[0], serial[0]) == 1);
        char reply[256] = {};
        assert(read(tcp[1], reply, sizeof(reply)-1) > 0);
        assert(!strncmp(reply, prefix, strlen(prefix)));
        char byte;
        assert(read(serial[1], &byte, 1) == -1 && (errno == EAGAIN || errno == EWOULDBLOCK));
    };
    g_motors_enabled = false;
    for (const char *text : {"PRETENSION 0", "PRETENSION 6", "PRETENSION 1.5"})
        command(text, "OK PRETENSION");
    assert(g_tension_mm.load() == 1.5);
    for (const char *text : {"PRETENSION", "PRETENSION nan", "PRETENSION inf",
             "PRETENSION -1", "PRETENSION 6.1", "PRETENSION 1 garbage"}) {
        command(text, "ERR PRETENSION");
        assert(g_tension_mm.load() == 1.5);
    }
    g_motors_enabled = true;
    command("PRETENSION 2", "ERR disable hardware");
    assert(g_tension_mm.load() == 1.5);
    g_motors_enabled = false;
    g_tension_mm = 0;
    for (int fd : {tcp[0], tcp[1], serial[0], serial[1]}) close(fd);
}

static void test_workspace_query_forwards_actual_firmware_bounds() {
    int tcp[2], serial[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tcp) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, serial) == 0);
    assert(fcntl(serial[0], F_SETFL, O_NONBLOCK) == 0);
    std::thread firmware([&]() {
        char cmd[64] = {};
        assert(read(serial[1], cmd, sizeof(cmd)-1) > 0);
        assert(!strcmp(cmd, "WORKSPACE\n"));
        const char reply[] = "OK WORKSPACE 1200.000 1937.500 61.400 904.500\n";
        assert(write(serial[1], reply, sizeof(reply)-1) == sizeof(reply)-1);
    });
    ClearPath disconnected_robot;
    assert(handleCommand("WORKSPACE", disconnected_robot, tcp[0], serial[0]) == 1);
    char response[256] = {};
    assert(read(tcp[1], response, sizeof(response)-1) > 0);
    assert(!strcmp(response, "OK WORKSPACE 1200.000 1937.500 61.400 904.500\n"));
    firmware.join();
    for (int fd : {tcp[0], tcp[1], serial[0], serial[1]}) close(fd);
}

// ── TAKEUP against a simulated rig ──────────────────────────────────────
// Each cable has some slack; wound in past it, that drive's holding current
// rises 0.9 A per mm (the 2026-11 measurement), signed per spool as on the
// table. The Teensy end is a thread answering STOP and SLACK on a socketpair.
struct FakeRig {
    std::mutex mutex;
    double slack[4] = {0, 0, 0, 0}, wound[4] = {0, 0, 0, 0};
    bool enabled = false, knows_slack = true;
    int enables = 0, disables = 0, alert_after_reads = -1, reads = 0;
    int tcp[2], serial[2];
    std::thread firmware;

    explicit FakeRig(std::initializer_list<double> slack_mm) {
        int i = 0;
        for (double v : slack_mm) slack[i++] = v;
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tcp) == 0);
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, serial) == 0);
        assert(fcntl(serial[0], F_SETFL, O_NONBLOCK) == 0);
        firmware = std::thread([this]() {
            std::string line;
            char c;
            while (read(serial[1], &c, 1) == 1) {
                if (c != '\n') { line += c; continue; }
                std::string reply = "ERR unknown command: " + line + "\n";
                unsigned mask; int steps;
                if (line == "STOP") reply = "ERR not running\n";
                else if (knows_slack && sscanf(line.c_str(), "SLACK %u %d", &mask, &steps) == 2) {
                    std::lock_guard<std::mutex> lock(mutex);
                    // Never steps a de-energized drive, and one pulse at a time.
                    assert(enabled && steps == 1);
                    for (int m = 0; m < 4; m++)
                        if (mask & (1u << m)) wound[m] += steps * SPOOL_CIRCUMFERENCE_MM / 800.0;
                    reply = "OK SLACK\n";
                }
                assert(write(serial[1], reply.data(), reply.size()) == (ssize_t)reply.size());
                line.clear();
            }
        });
    }
    ~FakeRig() {
        shutdown(serial[0], SHUT_RDWR);
        firmware.join();
        for (int fd : {tcp[0], tcp[1], serial[0], serial[1]}) close(fd);
    }
    TakeupDrives drives() {
        return {
            [this]() { std::lock_guard<std::mutex> l(mutex); enabled = true; enables++; return true; },
            [this]() { std::lock_guard<std::mutex> l(mutex); enabled = false; disables++; },
            [this](char *why, size_t n) {
                std::lock_guard<std::mutex> l(mutex);
                if (alert_after_reads >= 0 && reads >= alert_after_reads) {
                    snprintf(why, n, "shutdown: tracking error");
                    return 2;
                }
                return -1;
            },
            [this](unsigned m) {
                std::lock_guard<std::mutex> l(mutex);
                reads++;
                LoadValue v;
                v.valid = true;
                const double over = wound[m] - slack[m];
                const double amps = over > 0 ? 0.9 * over : 0.02;
                v.value = m % 2 == 0 ? -amps : amps;
                return v;
            },
        };
    }
    std::string run(double taut = TAKEUP_TAUT_AMPS, double max_mm = TAKEUP_MAX_MM) {
        return takeUpSlack(drives(), serial[0], tcp[0], taut, max_mm);
    }
};

static void test_takeup_winds_each_cable_until_its_own_drive_feels_it() {
    g_stop = 0; g_fault = false; g_motors_enabled = false;
    FakeRig rig({3.0, 0.0, 12.2, 0.4});
    const std::string reply = rig.run();
    assert(reply.find("OK TAKEUP ") == 0);
    double travel[4], amps[4];
    assert(sscanf(reply.c_str(), "OK TAKEUP %lf %lf %lf %lf %lf %lf %lf %lf", &travel[0], &travel[1],
                  &travel[2], &travel[3], &amps[0], &amps[1], &amps[2], &amps[3]) == 8);
    for (int m = 0; m < 4; m++) {
        // Stops within one step of the threshold: 0.3 A is 0.33 mm past taut.
        assert(rig.wound[m] > rig.slack[m] + 0.3 / 0.9 - 1e-6);
        assert(rig.wound[m] <= rig.slack[m] + 0.3 / 0.9 + TAKEUP_STEP_MM + 1e-6);
        // The reply counts real cable travel (0.38 mm per step), to 0.1 mm.
        assert(fabs(travel[m] - rig.wound[m]) < 0.06 && fabs(amps[m]) >= 0.3);
    }
    // Ends as the manual procedure did: de-energized, not "enabled" to the master.
    assert(rig.enables == 1 && rig.disables == 1 && !rig.enabled && !g_motors_enabled);
}

static void test_takeup_stops_and_disables() {
    g_stop = 0; g_fault = false; g_motors_enabled = false;
    {   // Any byte from the client is the stop button, and stays queued.
        FakeRig rig({40, 40, 40, 40});
        assert(write(rig.tcp[1], "DISABLE\n", 8) == 8);
        const std::string reply = rig.run();
        assert(reply.find("ERR takeup: stopped by the client") == 0);
        char byte;
        assert(recv(rig.tcp[0], &byte, 1, MSG_PEEK | MSG_DONTWAIT) == 1 && byte == 'D');
        assert(rig.disables == 1 && !rig.enabled && rig.wound[0] == 0);
    }
    {   // A cable that never pulls back is given up on at the travel limit.
        FakeRig rig({0.2, 50, 0.2, 0.2});
        const std::string reply = rig.run(TAKEUP_TAUT_AMPS, 5);
        assert(reply.find("ERR takeup: motor 1 wound in ") == 0 &&
               reply.find("without finding tension") != std::string::npos);
        assert(rig.wound[1] <= 5 + 1e-6 && rig.disables == 1 && !rig.enabled);
    }
    {   // A drive alert ends it at once.
        FakeRig rig({20, 20, 20, 20});
        rig.alert_after_reads = 24;
        const std::string reply = rig.run();
        assert(reply.find("ERR takeup: motor 2 shutdown: tracking error") == 0);
        assert(rig.disables == 1 && !rig.enabled && rig.wound[0] < 20);
    }
    {   // Firmware older than SLACK: says to flash, moves nothing.
        FakeRig rig({5, 5, 5, 5});
        rig.knows_slack = false;
        const std::string reply = rig.run();
        assert(reply.find("flash the current image") != std::string::npos);
        assert(rig.wound[0] == 0 && rig.disables == 1 && !rig.enabled);
    }
    {   // Already taut: nothing is stepped at all.
        FakeRig rig({-2, -2, -2, -2});
        const std::string reply = rig.run();
        assert(reply.find("OK TAKEUP 0.0 0.0 0.0 0.0") == 0 && rig.wound[0] == 0);
    }
}

static void test_takeup_command_is_refused_while_running_or_malformed() {
    int tcp[2], serial[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, tcp) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, serial) == 0);
    assert(fcntl(serial[1], F_SETFL, O_NONBLOCK) == 0);
    ClearPath disconnected_robot;
    auto command = [&](const char *text, const char *prefix) {
        assert(handleCommand(text, disconnected_robot, tcp[0], serial[0]) == 1);
        char reply[256] = {};
        assert(read(tcp[1], reply, sizeof(reply)-1) > 0);
        assert(!strncmp(reply, prefix, strlen(prefix)));
        char byte;   // nothing reached the Teensy
        assert(read(serial[1], &byte, 1) == -1 && (errno == EAGAIN || errno == EWOULDBLOCK));
    };
    g_motors_enabled = true;
    command("TAKEUP", "ERR disable hardware before taking up slack");
    g_motors_enabled = false;
    for (const char *text : {"TAKEUP 5", "TAKEUP 0.01", "TAKEUP nan", "TAKEUP 0.3 1000", "TAKEUP 0.3 50 x"})
        command(text, "ERR TAKEUP takes");
    for (int fd : {tcp[0], tcp[1], serial[0], serial[1]}) close(fd);
}

int main() {
    test_startup_pretension_configuration_never_moves();
    test_workspace_query_forwards_actual_firmware_bounds();
    int fd[2]; assert(pipe(fd) == 0);
    assert(fcntl(fd[0], F_SETFL, O_NONBLOCK) == 0);
    const char packet[] = "OK CMD\r\nS 1700 400 10 20 1 2 3 4\r\n";
    assert(write(fd[1], packet, sizeof(packet) - 1) == sizeof(packet) - 1);
    assert(waitTeensyOK(fd[0], 20));
    assert(g_status.valid && g_status.x == 1700 && g_status.y == 400);
    assert(g_status.received_monotonic > 0);
    // A partial line read by the main-loop consumer survives entry to wait.
    const char partial[] = "S 1800 500 ";
    g_teensy_lines.feed(partial, sizeof(partial) - 1, receiveTeensyLine);
    const char tail[] = "30 40 5 6 7 8\nOK ACCEL\nS 1900 600 0 0 9 8 7 6\n";
    assert(write(fd[1], tail, sizeof(tail) - 1) == sizeof(tail) - 1);
    assert(waitTeensyOK(fd[0], 20));
    assert(g_status.x == 1900 && g_status.c0 == 9);
    close(fd[0]); close(fd[1]);
    // POS preserves its original four values and appends a sample timestamp.
    int sock[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sock) == 0);
    ClearPath disconnected_robot;
    assert(handleCommand("POS", disconnected_robot, sock[0], -1) == 1);
    char response[256] = {}; assert(read(sock[1], response, sizeof(response)-1) > 0);
    double x,y,vx,vy,stamp;
    assert(sscanf(response, "OK %lf %lf %lf %lf %lf", &x,&y,&vx,&vy,&stamp) == 5);
    assert(x == 1900 && stamp > 0);
    close(sock[0]); close(sock[1]);
    test_slow_health_does_not_block_commands();
    test_watchdog_rechecks_enable_after_waiting_for_mutex();
    test_takeup_winds_each_cable_until_its_own_drive_feels_it();
    test_takeup_stops_and_disables();
    test_takeup_command_is_refused_while_running_or_malformed();
    printf("test_master_serial: ok\n");
}
