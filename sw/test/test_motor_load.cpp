// No SDK linkage and no hardware access: exercise the recorder with fake reads.
#include "motor_load.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

int main(int argc, char **argv) {
    assert(argc == 2);
    int calls = 0, phase = 0;
    auto reader = [&](unsigned node, LoadField field) {
        ++calls;
        LoadValue v;
        v.start = loadMonotonic();
        v.valid = true; v.value = 42 + node;
        if (field == LoadField::RmsSlow) {
            v.valid = false; v.error = "unsupported \"slow\"\nchannel";
        }
        if (field == LoadField::Rms && node == 0 && phase == 1) {
            v.valid = false; v.error = "read timeout";
        }
        if (field == LoadField::TorqueAmps) v.value = -2.5;
        if (field == LoadField::Encoder && node == 1) v.value = std::numeric_limits<double>::quiet_NaN();
        if (field == LoadField::Status) v.bits = {{phase == 0 ? 1u : 0u, 0, 0}};
        if (field == LoadField::Alerts) v.bits = {{0, 0, 0xffffffffu}};
        v.end = loadMonotonic();
        return v;
    };
    MotorLoadLogger logger(argv[1], 10, reader, [&]() {
        return phase ? "{\"fault\":true}" : "{\"fault\":false}";
    });
    assert(logger.open());
    assert(logger.cached().find("\"sample\":null") != std::string::npos);
    for (int i = 0; i < 1030; ++i) logger.command("CMD 1500 400", loadMonotonic());
    logger.sample();
    const int initial_calls = calls;
    for (int i = 0; i < 100; ++i) assert(!logger.cached().empty());
    assert(calls == initial_calls);  // Cached reads never invoke the SDK reader.
    phase = 1;
    logger.sample();
    assert(calls < initial_calls * 2); // Configuration cached and errors backed off.
    std::cout << logger.cached() << std::endl;
    logger.close();
    MotorLoadLogger again(argv[1], 10, reader, []() { return "{}"; });
    assert(!again.open());  // Existing recordings are never overwritten.
}
