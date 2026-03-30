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
    , ref_encoder_{0}, ref_lengths_{0}
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

bool CDPR::setPosition(double x, double y) {
    if (!connected_ || !enabled_) {
        fprintf(stderr, "CDPR: Must be connected and enabled before setPosition\n");
        x_ = x;
        y_ = y;
        pos_known_ = true;
        return false;
    }

    // Read current encoder positions as the reference baseline.
    try {
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            node.Motion.PosnMeasured.Refresh();
            ref_encoder_[i] = node.Motion.PosnMeasured.Value();
        }
    } catch (mnErr &err) {
        fprintf(stderr, "CDPR: Failed to read encoder positions: 0x%08x\n", err.ErrorCode);
        return false;
    }

    // Compute cable lengths at this reference position.
    cableLengths(x, y, ref_lengths_);

    x_ = x;
    y_ = y;
    pos_known_ = true;

    printf("CDPR: Reference set at (%.1f, %.1f)\n", x, y);
    printf("  Encoder positions: [%.0f, %.0f, %.0f, %.0f]\n",
           ref_encoder_[0], ref_encoder_[1], ref_encoder_[2], ref_encoder_[3]);
    printf("  Cable lengths: [%.1f, %.1f, %.1f, %.1f] mm\n",
           ref_lengths_[0], ref_lengths_[1], ref_lengths_[2], ref_lengths_[3]);

    return true;
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

    speed_mm_s = std::min(speed_mm_s, cfg_.max_velocity);

    // Compute absolute encoder targets.
    double target_encoder[4];
    double new_lengths[4];
    cableLengths(x, y, new_lengths);

    double cur_lengths[4];
    cableLengths(x_, y_, cur_lengths);

    for (int i = 0; i < 4; i++) {
        double delta_mm = new_lengths[i] - ref_lengths_[i];
        target_encoder[i] = ref_encoder_[i] + mmToCounts(delta_mm, i);
    }

    // Compute per-motor velocity for coordinated motion.
    double cart_dist = sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_));
    if (cart_dist < 0.01) {
        x_ = x;
        y_ = y;
        return true;
    }
    double duration_s = cart_dist / speed_mm_s;
    if (duration_s < 0.01) duration_s = 0.01;

    double accel_rpm_s = mmPerSecToRPM(cfg_.max_acceleration);

    try {
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            double delta_mm = fabs(new_lengths[i] - cur_lengths[i]);
            double motor_speed = delta_mm / duration_s;
            double vel_rpm = mmPerSecToRPM(motor_speed);
            if (vel_rpm < 0.1) vel_rpm = 0.1;
            setMotionParams(node, vel_rpm, accel_rpm_s);
        }

        // Send absolute moves.
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart((int32_t)round(target_encoder[i]), true);
        }

        // Wait for completion.
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

bool CDPR::commandPosition(double x, double y, double speed_mm_s) {
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

    speed_mm_s = std::min(speed_mm_s, cfg_.max_velocity);

    // Compute absolute encoder targets from the reference baseline.
    double target_encoder[4];
    double new_lengths[4], cur_lengths[4];
    cableLengths(x, y, new_lengths);
    cableLengths(x_, y_, cur_lengths);

    for (int i = 0; i < 4; i++) {
        double delta_mm = new_lengths[i] - ref_lengths_[i];
        target_encoder[i] = ref_encoder_[i] + mmToCounts(delta_mm, i);
    }

    double cart_dist = sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_));
    if (cart_dist < 0.01) {
        x_ = x;
        y_ = y;
        return true;
    }
    double duration_s = cart_dist / speed_mm_s;
    if (duration_s < 0.01) duration_s = 0.01;

    double accel_rpm_s = mmPerSecToRPM(cfg_.max_acceleration);

    try {
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            double delta_mm = fabs(new_lengths[i] - cur_lengths[i]);
            double motor_speed = delta_mm / duration_s;
            double vel_rpm = mmPerSecToRPM(motor_speed);
            if (vel_rpm < 0.1) vel_rpm = 0.1;
            setMotionParams(node, vel_rpm, accel_rpm_s);
        }

        // Send absolute moves — safe to call repeatedly.
        // Each call overwrites the previous target, no accumulation.
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart((int32_t)round(target_encoder[i]), true);
        }

        x_ = x;
        y_ = y;
        return true;

    } catch (mnErr &err) {
        fprintf(stderr, "CDPR commandPosition error: 0x%08x %s\n", err.ErrorCode, err.ErrorMsg);
        return false;
    }
}

bool CDPR::retractAll(double mm, double speed_mm_s) {
    if (!enabled_) {
        fprintf(stderr, "CDPR: Not enabled\n");
        return false;
    }

    speed_mm_s = std::min(speed_mm_s, cfg_.max_velocity);
    double vel_rpm = mmPerSecToRPM(speed_mm_s);
    double accel_rpm_s = mmPerSecToRPM(cfg_.max_acceleration);

    // retractAll still uses relative moves since it's not IK-based.
    try {
        for (int i = 0; i < 4; i++) {
            INode &node = port_->Nodes(i);
            setMotionParams(node, vel_rpm, accel_rpm_s);
            int counts = mmToCounts(-mm, i);  // negate: positive mm = retract = shorter
            node.Motion.MoveWentDone();
            node.Motion.MovePosnStart(counts);
        }

        double duration_s = fabs(mm) / speed_mm_s;
        double timeout = mgr_->TimeStampMsec() + (duration_s + 5.0) * 1000;
        bool all_done = false;
        while (!all_done) {
            if (mgr_->TimeStampMsec() > timeout) {
                fprintf(stderr, "CDPR: retractAll timed out\n");
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
        return true;
    } catch (mnErr &err) {
        fprintf(stderr, "CDPR retractAll error: 0x%08x %s\n", err.ErrorCode, err.ErrorMsg);
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
