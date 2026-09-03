// Batched host build of the firmware's motion profile, for the simulator.
//
// WHY THIS EXISTS
//   The simulator needs to advance thousands of carts through the SAME
//   control law the Teensy runs. The alternative was a NumPy re-implementation
//   of motion_profile.h, which would be a second copy of a control law
//   created by choice rather than necessity -- and the project already
//   carries one such mirror (shared/cable_model.py) only because Python
//   cannot include a C header for the *fitter*. Here there is no fitter and
//   no autodiff: the sim only needs to EVALUATE the law, so it can call it.
//
//   motion_profile.h is deliberately free of Arduino.h precisely so this is
//   possible.
//
// WHY BATCHED
//   A per-call FFI crossing would dominate: thousands of envs times tens of
//   substeps is millions of crossings per environment step. Everything loops
//   inside C and Python crosses once per step.
//
// LAYOUT
//   Every array is length n and is updated IN PLACE, so the caller's numpy
//   arrays are the state -- no copying, no mirrored Python-side state that
//   could disagree with this.
//
// Build:  make -C fw/host

#include <stdint.h>

#include "motion_profile.h"

extern "C" {

// Advance n independent carts by `substeps` ticks of `dt` each.
//
// Per-cart caps rather than one global pair, because domain randomisation
// varies them across the batch; passing scalars would have forced the sim to
// either group envs by cap or give that up.
void motion_advance_batch(int n, int substeps, float dt,
                          float *px, float *py,
                          float *vx, float *vy,
                          float *ax, float *ay,
                          const float *tx, const float *ty,
                          const float *vMax, const float *aMax,
                          float rampS, uint8_t *flags) {
  for (int i = 0; i < n; i++) {
    uint8_t f = 0;
    float x = px[i], y = py[i];
    float ux = vx[i], uy = vy[i];
    float axi = ax[i], ayi = ay[i];
    const float txi = tx[i], tyi = ty[i];
    const float vm = vMax[i], am = aMax[i];

    // Substeps inner so the per-cart state stays in registers across the
    // whole chain rather than being reloaded from the arrays each tick.
    for (int s = 0; s < substeps; s++) {
      f |= motionProfileAdvance(x, y, ux, uy, axi, ayi, txi, tyi,
                                vm, am, rampS, dt);
    }

    px[i] = x;
    py[i] = y;
    vx[i] = ux;
    vy[i] = uy;
    ax[i] = axi;
    ay[i] = ayi;
    if (flags) flags[i] = f;
  }
}

// As above, and keep every cart inside one box. Scalar bounds per call: the
// simulator advances each side's carts in a call of their own, and the two
// sides have different boxes (the robot's, and its mirror for a robot-bodied
// far side).
void motion_advance_batch_bounded(int n, int substeps, float dt,
                                  float *px, float *py,
                                  float *vx, float *vy,
                                  float *ax, float *ay,
                                  const float *tx, const float *ty,
                                  const float *vMax, const float *aMax,
                                  float rampS, uint8_t *flags,
                                  float xMin, float xMax, float yMin, float yMax) {
  for (int i = 0; i < n; i++) {
    uint8_t f = 0;
    float x = px[i], y = py[i];
    float ux = vx[i], uy = vy[i];
    float axi = ax[i], ayi = ay[i];
    const float txi = tx[i], tyi = ty[i];
    const float vm = vMax[i], am = aMax[i];
    for (int s = 0; s < substeps; s++) {
      f |= motionProfileAdvanceBounded(x, y, ux, uy, axi, ayi, txi, tyi,
                                       vm, am, rampS, dt, xMin, xMax, yMin, yMax);
    }
    px[i] = x;
    py[i] = y;
    vx[i] = ux;
    vy[i] = uy;
    ax[i] = axi;
    ay[i] = ayi;
    if (flags) flags[i] = f;
  }
}

// Single cart, many ticks, recording the trajectory. This is what the
// tick-divergence test drives: run the same move at the firmware's 20 us and
// at a candidate simulator tick, and compare. Writes n_out samples strided
// every `every` ticks.
void motion_trace(int ticks, int every, float dt,
                  float px, float py, float vx, float vy,
                  float ax, float ay, float tx, float ty,
                  float vMax, float aMax, float rampS,
                  float *out_x, float *out_y, float *out_vx, float *out_vy) {
  int k = 0;
  for (int s = 0; s < ticks; s++) {
    motionProfileAdvance(px, py, vx, vy, ax, ay, tx, ty,
                         vMax, aMax, rampS, dt);
    if (every > 0 && (s % every) == (every - 1)) {
      out_x[k] = px;
      out_y[k] = py;
      out_vx[k] = vx;
      out_vy[k] = vy;
      k++;
    }
  }
}

}  // extern "C"
