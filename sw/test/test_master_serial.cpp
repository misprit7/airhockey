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
    for (const char *text : {"PRETENSION 0", "PRETENSION 3", "PRETENSION 1.5"})
        command(text, "OK PRETENSION");
    assert(g_tension_mm.load() == 1.5);
    for (const char *text : {"PRETENSION", "PRETENSION nan", "PRETENSION inf",
             "PRETENSION -1", "PRETENSION 3.1", "PRETENSION 1 garbage"}) {
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
}
