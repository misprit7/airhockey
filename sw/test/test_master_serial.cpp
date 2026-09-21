// Compile the real command-wait/parser code; never call its hardware main.
#define main unused_hardware_main
#include "../bin/cdpr_master.cpp"
#undef main
#include <cassert>

int main() {
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
}
