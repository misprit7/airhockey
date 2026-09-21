#include "motor_load.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <utility>
#include <unistd.h>

static double clockSeconds(clockid_t clock) {
    timespec t{};
    clock_gettime(clock, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
double loadMonotonic() { return clockSeconds(CLOCK_MONOTONIC); }
double loadWallTime() { return clockSeconds(CLOCK_REALTIME); }

const char *loadFieldName(LoadField f) {
    static const char *names[] = {"rms_pct", "rms_slow_pct", "torque_amps",
        "encoder_counts", "velocity_counts_s", "status", "alerts",
        "peak_current_amps", "rms_limit_amps", "rms_time_constant_s",
        "rms_slow_limit_amps", "rms_slow_time_constant_min", "encoder_counts_rev",
        "serial_number", "firmware_version", "torque_limit_amps"};
    return names[static_cast<unsigned>(f)];
}
std::string loadJsonString(const std::string &s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
        else out += c;
    }
    return out + "\"";
}
std::string loadNumber(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64]; snprintf(b, sizeof(b), "%.12g", v); return b;
}
MotorLoadLogger::MotorLoadLogger(const std::string &path, double hz, Reader reader, Context context)
    : path_(path), hz_(hz), started_(loadMonotonic()), wall_(loadWallTime()),
      reader_(std::move(reader)), context_(std::move(context)) {}
MotorLoadLogger::~MotorLoadLogger() { close(); }

std::string MotorLoadLogger::source() const {
    return "{\"schema\":1,\"path\":" + loadJsonString(path_) +
        ",\"pid\":" + std::to_string(getpid()) + ",\"started_monotonic\":" + loadNumber(started_) +
        ",\"started_unix\":" + loadNumber(wall_) + ",\"requested_hz\":" + loadNumber(hz_) +
        ",\"clock\":\"CLOCK_MONOTONIC\",\"rms_units\":\"percent_of_shutdown\","
        "\"rms_source\":\"Info.Ex.Parameter; no ValueDouble torque-unit scaling\","
        "\"status_layout\":[\"enabled\",\"alert_present\"],"
        "\"alerts_layout\":\"three raw 32-bit words\"}";
}
bool MotorLoadLogger::open() {
    file_ = fopen(path_.c_str(), "wx");  // Never overwrite another recording.
    if (!file_) return false;
    std::string meta = "{\"type\":\"meta\",\"source\":" + source() + "}\n";
    write_ok_ = fwrite(meta.data(), 1, meta.size(), file_) == meta.size() && fflush(file_) == 0;
    if (!write_ok_) { close(); return false; }
    cache_ = "{\"source\":" + source() + ",\"sample\":null,\"logging_ok\":true}";
    return true;
}
void MotorLoadLogger::sample() {
    const double begin = loadMonotonic();
    const bool metadata = begin >= next_metadata_;
    // Dynamic readings first, optional/configuration reads after them.
    for (unsigned f = 0; f < static_cast<unsigned>(LoadField::Count); ++f) {
        if (f >= static_cast<unsigned>(LoadField::PeakAmps) && !metadata) break;
        for (unsigned node = 0; node < 4; ++node) {
            if (begin < retry_[node][f]) continue;
            LoadValue v = reader_(node, static_cast<LoadField>(f));
            if (!std::isfinite(v.value)) { v.valid = false; v.error = "nonfinite"; }
            values_[node][f] = v;
            // Unsupported parameters and communication failures remain explicit,
            // with old acquisition times, and don't hammer the bus every tick.
            retry_[node][f] = v.valid ? 0 : loadMonotonic() + 1;
        }
    }
    if (metadata) next_metadata_ = begin + 60;
    const double end = loadMonotonic();
    std::string s = "{\"type\":\"motor_load\",\"sequence\":" + std::to_string(++sequence_) +
        ",\"monotonic_start\":" + loadNumber(begin) + ",\"monotonic\":" + loadNumber(end) +
        ",\"unix\":" + loadNumber(loadWallTime()) + ",\"acquisition_ms\":" + loadNumber((end-begin)*1000) +
        ",\"context\":" + context_() + ",\"motors\":[";
    for (unsigned node = 0; node < 4; ++node) {
        if (node) s += ',';
        s += "{\"node\":" + std::to_string(node);
        for (unsigned f = 0; f < static_cast<unsigned>(LoadField::Count); ++f) {
            auto field = static_cast<LoadField>(f);
            const LoadValue &v = values_[node][f];
            s += "," + loadJsonString(loadFieldName(field)) + ":{\"valid\":" + (v.valid ? "true" : "false") +
                ",\"start\":" + loadNumber(v.start) + ",\"end\":" + loadNumber(v.end) + ",\"value\":";
            if (!v.valid) s += "null";
            else if (field == LoadField::Status || field == LoadField::Alerts) {
                s += "[" + std::to_string(v.bits[0]) + "," + std::to_string(v.bits[1]);
                if (field == LoadField::Alerts) s += "," + std::to_string(v.bits[2]);
                s += "]";
            } else s += loadNumber(v.value);
            s += ",\"error\":" + loadJsonString(v.error) + "}";
        }
        s += "}";
    }
    std::vector<std::string> events;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events.swap(events_);
        s += "],\"dropped_command_events\":" + std::to_string(dropped_) + "}";
    }
    if (file_ && write_ok_) {
        for (const auto &event : events) {
            if (fwrite(event.data(), 1, event.size(), file_) != event.size()) write_ok_ = false;
        }
        const std::string line = s + "\n";
        const bool line_ok = fwrite(line.data(), 1, line.size(), file_) == line.size() && fflush(file_) == 0;
        write_ok_ = write_ok_ && line_ok;
        if (!write_ok_) fprintf(stderr, "Motor-load log write failed: %s\n", path_.c_str());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    cache_ = "{\"source\":" + source() + ",\"logging_ok\":" + (write_ok_ ? "true" : "false") + ",\"sample\":" + s + "}";
}
void MotorLoadLogger::command(const std::string &line, double when) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() >= 1024) { ++dropped_; return; }
    events_.push_back("{\"type\":\"command_received\",\"monotonic\":" + loadNumber(when) +
                      ",\"command\":" + loadJsonString(line) + "}\n");
}
std::string MotorLoadLogger::cached() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_;
}
void MotorLoadLogger::close() {
    if (file_) {
        for (const auto &event : events_) fwrite(event.data(), 1, event.size(), file_);
        events_.clear();
        fprintf(file_, "{\"type\":\"end\",\"monotonic\":%.12g}\n", loadMonotonic());
        fclose(file_); file_ = nullptr;
    }
}
