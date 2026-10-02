"""Incremental excursions beyond the existing region; offline planning only."""
import numpy as np
# Expansion experiments retain their historical inner baseline so old plans
# remain comparable after the wider region becomes the deployment default.
from airhockey.neural_setup import workspace_bounds
BOUNDS = np.array(workspace_bounds('legacy'))
LOW, HIGH = BOUNDS[[0, 2]], BOUNDS[[1, 3]]
CENTER = (LOW + HIGH) / 2
from cdpr_geometry import WS_PROBE_MIN_X, WS_PROBE_MAX_X, WS_PROBE_MIN_Y, WS_PROBE_MAX_Y

PROBE_BOUNDS = np.array([WS_PROBE_MIN_X, WS_PROBE_MAX_X, WS_PROBE_MIN_Y, WS_PROBE_MAX_Y])


def design_workspace_probe(step_mm=40., speed=.3, accel=1., hold_s=0.):
    if not np.isfinite([step_mm,speed,accel,hold_s]).all() or not (
            10<=step_mm<=50 and 0<speed<=.5 and 0<accel<=2 and 0<=hold_s<=30):
        raise ValueError('workspace probe requires step 10..50 mm, speed <=0.5 m/s, accel <=2 m/s², hold 0..30 s')
    low,high=PROBE_BOUNDS[[0,2]],PROBE_BOUNDS[[1,3]]
    trials=[dict(id=0,kind='hold',site=0,start=CENTER.tolist(),end=CENTER.tolist(),
                 speed=.2,accel=.4,duration=2.)]
    rays=[]
    # Return sites are the nine-position baseline's already-tested locations.
    # Each excursion ends back there, never with a mandatory hold at the tip.
    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)):
        base=np.array([LOW[0]+40 if dx<0 else HIGH[0]-40 if dx>0 else CENTER[0],
                       LOW[1]+40 if dy<0 else HIGH[1]-40 if dy>0 else CENTER[1]])
        tip=np.array([low[0] if dx<0 else high[0] if dx>0 else CENTER[0],
                      low[1] if dy<0 else high[1] if dy>0 else CENTER[1]])
        distance=float(np.linalg.norm(tip-base))
        rays.append((base,tip,distance))
    for shell in range(1,max(int(np.ceil(d/step_mm)) for _,_,d in rays)+1):
        for site,(base,tip,distance) in enumerate(rays):
            if (shell-1)*step_mm>=distance:continue
            point=base+(tip-base)*min(1.,shell*step_mm/distance)
            trials.append(dict(id=len(trials),kind='reach',site=site,shell=shell,
                start=base.tolist(),end=base.tolist(),tip=point.tolist(),speed=speed,
                accel=accel,hold_s=hold_s,outermost=bool(shell*step_mm>=distance)))
    return trials
