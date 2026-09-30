#pragma once

// Hardware-independent schema/recorder; the reader is injected. All numbers
// carry validity and acquisition timestamps. A failed read is never zero load.
#include <array>
#include <cstdio>
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

enum class LoadField {
    Rms, RmsSlow, TorqueAmps, Encoder, Velocity, Status, Alerts, BusVolts,
    PeakAmps, RmsLimitAmps, RmsTimeSeconds, SlowLimitAmps, SlowTimeMinutes,
    EncoderResolution, Serial, Firmware, TorqueLimitAmps, StepInputResolution, Count
};

struct LoadValue {
    bool valid = false;
    double value = 0;
    // Status: [enabled, alert_present]; Alerts: three raw 32-bit words.
    std::array<unsigned, 3> bits{{0, 0, 0}};
    std::string error;
    double start = 0, end = 0;
};

double loadMonotonic();
double loadWallTime();
const char *loadFieldName(LoadField field);
std::string loadJsonString(const std::string &s);
std::string loadNumber(double value);

class MotorLoadLogger {
public:
    using Reader = std::function<LoadValue(unsigned, LoadField)>;
    using Context = std::function<std::string()>;
    MotorLoadLogger(const std::string &path, double hz, Reader reader, Context context);
    ~MotorLoadLogger();
    bool open();  // File only, no hardware access.
    void sample();  // Called only by the telemetry worker.
    void close();
    std::string cached() const; // No drive access or disk I/O.
    std::string source() const;
    void command(const std::string &line, double when); // Receipt, not execution confirmation.
private:
    std::string path_;
    double hz_, started_, wall_, next_metadata_ = 0;
    Reader reader_;
    Context context_;
    FILE *file_ = nullptr;
    bool write_ok_ = true;
    unsigned long sequence_ = 0;
    using Row = std::array<LoadValue, static_cast<unsigned>(LoadField::Count)>;
    std::array<Row, 4> values_;
    std::array<std::array<double, static_cast<unsigned>(LoadField::Count)>, 4> retry_{};
    mutable std::mutex mutex_;
    std::string cache_;
    std::vector<std::string> events_;
    unsigned long dropped_ = 0;
};
