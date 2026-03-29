#include "cdpr.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

using namespace sFnd;

CDPR::CDPR(const CDPRConfig &config)
    : cfg_(config)
    , spool_circumference_(M_PI * config.spool_diameter)
    , x_(0), y_(0), pos_known_(false)
    , mgr_(nullptr), port_(nullptr)
    , connected_(false), enabled_(false), node_count_(0)
{
}

CDPR::~CDPR() {
    if (enabled_) disable();
    if (connected_) disconnect();
}

bool CDPR::connect() {
    if (connected_) return true;

    mgr_ = SysManager::Instance();

    std::vector<std::string> hubPorts;
    SysManager::FindComHubPorts(hubPorts);
    if (hubPorts.empty()) {
        fprintf(stderr, "CDPR: No SC Hub ports found\n");
        return false;
    }

    try {
        mgr_->ComHubPort(0, hubPorts[0].c_str());
        mgr_->PortsOpen(1);
        port_ = &mgr_->Ports(0);
        node_count_ = port_->NodeCount();

        if (node_count_ != 4) {
            fprintf(stderr, "CDPR: Expected 4 motors, found %d\n", node_count_);
            mgr_->PortsClose();
            return false;
        }

        connected_ = true;
        return true;
    } catch (mnErr &err) {
        fprintf(stderr, "CDPR connect error: 0x%08x %s\n", err.ErrorCode, err.ErrorMsg);
        return false;
    }
}

bool CDPR::enable() {
    if (!connected_) return false;
    if (enabled_) return true;

    try {
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            node.EnableReq(false);
            mgr_->Delay(100);
            node.Status.AlertsClear();
            node.Motion.NodeStopClear();
            node.EnableReq(true);

            double timeout = mgr_->TimeStampMsec() + 5000;
            while (!node.Motion.IsReady()) {
                if (mgr_->TimeStampMsec() > timeout) {
                    fprintf(stderr, "CDPR: Motor %d timed out enabling\n", i);
                    return false;
                }
            }
        }
        enabled_ = true;
        return true;
    } catch (mnErr &err) {
        fprintf(stderr, "CDPR enable error: 0x%08x %s\n", err.ErrorCode, err.ErrorMsg);
        return false;
    }
}

void CDPR::disable() {
    if (!connected_ || !port_) return;
    try {
        for (int i = 0; i < 4; i++) {
            port_->Nodes(i).EnableReq(false);
        }
    } catch (...) {}
    enabled_ = false;
}

void CDPR::disconnect() {
    if (!connected_ || !mgr_) return;
    try {
        mgr_->PortsClose();
    } catch (...) {}
    connected_ = false;
    port_ = nullptr;
}

void CDPR::setPosition(double x, double y) {
    x_ = x;
    y_ = y;
    pos_known_ = true;
}

double CDPR::cableLength(int motor, double x, double y) const {
    double mx = 0, my = 0;
    cfg_.motorPos(motor, mx, my);
    double dx = x - mx;
    double dy = y - my;
    return sqrt(dx * dx + dy * dy);
}

void CDPR::cableLengths(double x, double y, double lengths[4]) const {
    for (int i = 0; i < 4; i++) {
        lengths[i] = cableLength(i, x, y);
    }
}

bool CDPR::inBounds(double x, double y) const {
    double m = cfg_.edge_margin;
    return x >= m && x <= cfg_.width - m &&
           y >= m && y <= cfg_.height - m;
}

int CDPR::mmToCounts(double mm, int motor) const {
    // Positive mm = cable gets longer (extend) = positive counts (CCW).
    // Negative mm = cable gets shorter (retract) = negative counts (CW).
    return (int)round(mm / spool_circumference_ * cfg_.counts_per_rev[motor]);
}

double CDPR::mmPerSecToRPM(double mm_s) const {
    // mm/s → rev/s → RPM
    return (mm_s / spool_circumference_) * 60.0;
}

void CDPR::setMotionParams(INode &node, double vel_rpm, double accel_rpm_s) {
    node.AccUnit(INode::RPM_PER_SEC);
    node.VelUnit(INode::RPM);
    node.Motion.VelLimit = vel_rpm;
    node.Motion.AccLimit = accel_rpm_s;
}

bool CDPR::moveTo(double x, double y, double speed_mm_s) {
    if (!enabled_) {
        fprintf(stderr, "CDPR: Not enabled\n");
        return false;
    }
    if (!pos_known_) {
        fprintf(stderr, "CDPR: Position not set. Call setPosition() first.\n");
        return false;
    }
    if (!inBounds(x, y)) {
        fprintf(stderr, "CDPR: Target (%.1f, %.1f) out of bounds\n", x, y);
        return false;
    }

    // Clamp speed to configured max.
    speed_mm_s = std::min(speed_mm_s, cfg_.max_velocity);

    // Compute cable length deltas.
    double cur_lengths[4], new_lengths[4], deltas[4];
    cableLengths(x_, y_, cur_lengths);
    cableLengths(x, y, new_lengths);

    double max_delta = 0;
    for (int i = 0; i < 4; i++) {
        deltas[i] = new_lengths[i] - cur_lengths[i];
        max_delta = std::max(max_delta, fabs(deltas[i]));
    }

    if (max_delta < 0.01) {
        // Already there.
        x_ = x;
        y_ = y;
        return true;
    }

    // Compute move duration from the cart's linear speed.
    double cart_dist = sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_));
    double duration_s = cart_dist / speed_mm_s;
    if (duration_s < 0.01) duration_s = 0.01;

    // Set per-motor velocity so all motors finish at the same time.
    // Each motor moves |delta[i]| mm in duration_s seconds.
    double accel_rpm_s = mmPerSecToRPM(cfg_.max_acceleration);

    try {
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            double motor_speed = fabs(deltas[i]) / duration_s;
            double vel_rpm = mmPerSecToRPM(motor_speed);
            if (vel_rpm < 0.1) vel_rpm = 0.1;  // minimum to avoid stall
            setMotionParams(node, vel_rpm, accel_rpm_s);
        }

        // Start all moves.
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            int counts = mmToCounts(deltas[i], i);
            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart(counts);
        }

        // Wait for all moves to complete.
        double timeout = mgr_->TimeStampMsec() + (duration_s + 5.0) * 1000;
        bool all_done = false;
        while (!all_done) {
            if (mgr_->TimeStampMsec() > timeout) {
                fprintf(stderr, "CDPR: Move timed out\n");
                return false;
            }
            all_done = true;
            for (int i = 0; i < 4; i++) {
                if (!port_->Nodes(i).Motion.MoveIsDone()) {
                    all_done = false;
                    break;
                }
            }
        }

        x_ = x;
        y_ = y;
        return true;

    } catch (mnErr &err) {
        fprintf(stderr, "CDPR move error: 0x%08x %s\n", err.ErrorCode, err.ErrorMsg);
        return false;
    }
}

bool CDPR::moveBy(double dx, double dy, double speed_mm_s) {
    if (!pos_known_) {
        fprintf(stderr, "CDPR: Position not set. Call setPosition() first.\n");
        return false;
    }
    return moveTo(x_ + dx, y_ + dy, speed_mm_s);
}
