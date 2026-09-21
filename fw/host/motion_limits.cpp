// Simulation-only predictive limit checks using the unchanged firmware law.
// A separate DSO keeps running replay/simulator processes on their original
// library while this diagnostic/guard is built and iterated.
#include <cmath>
#include "motion_profile.h"

extern "C" void motion_limits_batch(
    int n, int steps, float dt,
    float *px, float *py, float *vx, float *vy, float *ax, float *ay,
    const float *tx, const float *ty, const float *vmax, const float *amax,
    float ramp, float xmin, float xmax, float ymin, float ymax, float *peak, float *margin) {
  for (int i = 0; i < n; ++i) {
    float x=px[i], y=py[i], ux=vx[i], uy=vy[i], a=ax[i], b=ay[i], worst=peak[i], room=margin[i];
    for (int s = 0; s < steps; ++s) {
      float oldx=ux, oldy=uy;
      motionProfileAdvanceBounded(x,y,ux,uy,a,b,tx[i],ty[i],vmax[i],amax[i],
                                  ramp,dt,xmin,xmax,ymin,ymax);
      float dx=ux-oldx, dy=uy-oldy;
      worst=fmaxf(worst, sqrtf(dx*dx+dy*dy)/dt);
      room=fminf(room, fminf(fminf(x-xmin,xmax-x), fminf(y-ymin,ymax-y)));
    }
    px[i]=x; py[i]=y; vx[i]=ux; vy[i]=uy; ax[i]=a; ay[i]=b; peak[i]=worst; margin[i]=room;
  }
}
