"""Replacement-motor contracts; pure math and synthetic records only."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from airhockey.hardware import TEENSY_COUNTS_PER_REV
from airhockey.neural_deploy import LiveMotorLoad
from airhockey.replay_log import ReplayLog
from airhockey.thermal import DEFAULT_MODEL, LEGACY_MODEL, MotorThermal

ROOT = Path(__file__).resolve().parents[2]


def test_firmware_and_host_reject_wrong_motor_or_input_scale(tmp_path):
    source = tmp_path / 'check.cpp'
    source.write_text(r'''
#include <cstdint>
#include <limits>
#include <cassert>
#include "cdpr_config.h"
int main() {
  assert(COUNTS_PER_REV == 800);
  assert(MotorHardware::matches("CPM-SCSK-2331S-RLNA", 800, 800));
  assert(MotorHardware::matches("CPM-SCSK-2331S-RLNA-1-7-D", 800, 800));
  assert(MotorHardware::matches("CPM-SCSK-2331S-RLNA-1-8-D", 800, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNAX-1-8-D", 800, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNB-1-8-D", 800, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331P-ELNA-1-8-D", 800, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA-", 800, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S", 800, 800));
  assert(!MotorHardware::matches("", 800, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA-1-8-D", 6400, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA-1-7-D", 800, 6400));
  assert(!MotorHardware::matches("CPM-SCSK-2331P-ELNA", 6400, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA", 800, 6400));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA", 6400, 800));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA", 800, 0));
  assert(!MotorHardware::matches("CPM-SCSK-2331S-RLNA", 800,
                                std::numeric_limits<double>::quiet_NaN()));
  assert(!MotorHardware::matches(nullptr, 800, 800));
  assert(MAX_VELOCITY_MM_S == 12000);
  assert(MAX_ACCEL_MM_S2 == 120000);
}
''')
    binary = tmp_path / 'check'
    subprocess.run(['g++', '-std=c++11', '-Ifw/include', '-Ishared', str(source),
                    '-o', str(binary)], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)
    assert TEENSY_COUNTS_PER_REV == 800


def test_new_thermal_limits_and_historical_model_remain_distinct():
    current = MotorThermal(1, randomize=False)
    historical = MotorThermal(1, path=LEGACY_MODEL, randomize=False)
    np.testing.assert_allclose(current.limits[0], [5.8]*4, atol=1e-6)
    np.testing.assert_allclose(current.limits[1], [4.1]*4, atol=1e-6)
    np.testing.assert_allclose(current.tau[1], [509.45259528593164]*4)
    np.testing.assert_allclose(historical.limits[0], [4, 5.8, 4, 5.8], atol=1e-6)
    # Preserve spatial current priors pending new measurements; do not invent
    # a current/torque scaling from the higher peak torque rating.
    np.testing.assert_array_equal(current.coeff, historical.coeff)
    assert 'unmeasured' in current.config['status']
    assert current.config['motor_models'] == ['CPM-SCSK-2331S-RLNA']*4
    assert LiveMotorLoad().model.config == current.config


def test_new_recording_captures_motor_profile_for_future_comparisons(tmp_path):
    path = tmp_path/'session.jsonl'
    log = ReplayLog(path, SimpleNamespace(policy='test', live=False, ramp=3))
    log.close()
    meta = json.loads(path.read_text().splitlines()[0])
    assert meta['motor_profile'] == json.loads(DEFAULT_MODEL.read_text())
    assert meta['motor_profile']['step_input_counts_rev'] == [800]*4
