#!/usr/bin/env python3
"""Turn observed simulation stalls into reset states, never action labels."""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import numpy as np
from airhockey.neural_coordinates import action_coordinates, bounds
from airhockey.neural_setup import checkpoint_environment


def build(reports, environment, legacy=None):
    box=bounds(environment);low=box[[0,2]];high=box[[1,3]]
    rows=[];sources=[];seen=set()
    for path in reports:
        document=json.loads(path.read_text());match=document.get('selfplay',{})
        if 'final_physical_state' not in match:continue
        physical=match['final_physical_state'];puck=np.asarray(physical['puck'])
        observations=np.asarray(match['final_policy_observation'])
        actions=np.asarray(match['final_arrival_action']);n=len(puck)
        source=match.get('simulation_environment',document.get('simulation_environment',environment))
        actions=action_coordinates(actions,source,environment)
        added=0
        for side,key in enumerate(('blue_paddle','red_paddle')):
            paddle=np.asarray(physical[key]).copy();local=puck.copy()
            if side:local[:,1]=2-local[:,1];local[:,3]*=-1;paddle[:,1]=2-paddle[:,1]
            for i in range(n):
                p=local[i]
                if not (.04069<=p[0]<=.95931 and .04069<=p[1]<1):continue
                if np.linalg.norm(p[2:])>.05:continue
                # Do not turn a kinematic paddle/rail compression artifact
                # into an initially interpenetrating training fixture.
                if np.linalg.norm(p[:2]-paddle[i])<.0907-.002:continue
                if np.linalg.norm(p[:2]-np.clip(p[:2],low+.001,high-.001))>.0907:continue
                heat=np.clip(observations[side*n+i,21:29],.05,.9)
                signature=tuple(np.round(np.r_[p[:2],paddle[i],heat],3))
                if signature in seen:continue
                seen.add(signature)
                rows.append((p,paddle[i],actions[side*n+i],observations[side*n+i,-3:],heat))
                added+=1
        if added:sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=added))
    if not rows:raise ValueError('No reachable stalled states found')
    names=('puck','paddle','previous_action','request','initial_load')
    arrays={name:np.asarray([row[j] for row in rows],dtype=np.float32) for j,name in enumerate(names)}
    novel=len(rows);copies=1
    if legacy:
        with np.load(legacy) as z:
            old={k:z[k].copy() for k in names[:4]}
            old['initial_load']=z['initial_load'].copy() if 'initial_load' in z else np.full((len(old['puck']),8),np.nan)
            source=json.loads(str(z['environment_options'])) if 'environment_options' in z else dict(accel=60,workspace_bounds_mm=None)
            old['previous_action']=action_coordinates(old['previous_action'],source,environment)
        copies=max(1,int(np.ceil(len(old['puck'])/novel)))
        arrays={k:np.concatenate((old[k],np.tile(arrays[k],(copies,1)))) for k in names}
    arrays['environment_options']=json.dumps(environment)
    report=dict(new_unique_cases=novel,new_case_repetitions=copies,total_cases=len(arrays['puck']),
        sources=sources,legacy=str(legacy) if legacy else None,
        legacy_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest() if legacy else None,
        meaning='Simulation reset states only. Previous actions describe history, not targets to imitate. New cases preserve per-motor load patterns with each channel capped at 90%, before shutdown; legacy cases use ordinary training heat randomization.')
    return arrays,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-glob',action='append',required=True)
    p.add_argument('--environment-from',type=Path,required=True)
    p.add_argument('--legacy',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports=sorted({Path(f) for pattern in a.report_glob for f in glob.glob(pattern)})
    arrays,report=build(reports,checkpoint_environment(a.environment_from),a.legacy)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**arrays)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
