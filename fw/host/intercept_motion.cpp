// Host-only observation-based interception forecasts. Does not control hardware.
#include "motion_profile.h"
extern "C" void intercept_motion_batch(
    int n, int delay_steps, float dt, const int *steps,
    const float *state, const float *queued, const float *command,
    float vmax, float ramp, float *output) {
  for (int i = 0; i < n; ++i) {
    float x=state[6*i], y=state[6*i+1], vx=state[6*i+2], vy=state[6*i+3];
    float ax=state[6*i+4], ay=state[6*i+5];
    for (int j = 0; j < delay_steps; ++j)
      motionProfileAdvance(x,y,vx,vy,ax,ay,queued[3*i],queued[3*i+1],
                           vmax,queued[3*i+2],ramp,dt);
    for (int j = 0; j < steps[i]; ++j)
      motionProfileAdvance(x,y,vx,vy,ax,ay,command[3*i],command[3*i+1],
                           vmax,command[3*i+2],ramp,dt);
    output[6*i]=x; output[6*i+1]=y; output[6*i+2]=vx;
    output[6*i+3]=vy; output[6*i+4]=ax; output[6*i+5]=ay;
  }
}
