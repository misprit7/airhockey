#!/usr/bin/env python3
"""Simulation-only warm-start PPO ablations, periodic replays and held-out checks."""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'ai'),str(ROOT/'ai/bin')]
import numpy as np
import torch
from eval_fast_defense import evaluate as defense, BASE
from eval_slow_control import evaluate as control
from eval_neural_player import load, games, skills

OUT=Path('logs/neural-player/control-defense-20261011')
REPORT=Path('ai/airhockey/web/training-report-20261011.html')
STATE=dict(status='Starting',rows=[],heuristics=[],validation=[],ladder=[],selected=None)
SETTLE={'capture-max-speed':.15,'capture-duration':.20,'capture-depth-margin':.04,
        'capture-weight':80,'control-speed-scale':.4,'slow-cushion-weight':20,
        'unproductive-return-penalty':40}
ARMS=[('no-imitation',{}),('depth200',{'defensive-depth-weight':200}),
      ('settle',SETTLE),('settle-depth200',{**SETTLE,'defensive-depth-weight':200}),
      ('settle-depth800',{**SETTLE,'defensive-depth-weight':800})]


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def publish(message=None):
    if message:STATE['status']=message
    STATE['updated']=time.strftime('%Y-%m-%d %H:%M:%S %Z');STATE['pid']=os.getpid()
    write(OUT/'experiment.json',STATE)
    rows=''
    for r in STATE['rows']:
        d=r['defense']['summary'];c=r['control']['summary']
        rows+=f'<tr><td>{html.escape(r["name"])}</td><td>{100*d["block_rate"]:.1f}%</td><td>{d["mean_release_depth"]:.2f} m</td><td>{100*c["settle_rate"]:.1f}%</td><td>{100*c["converted_rate"]:.1f}%</td><td>{100*c["fast_aimed_rate"]:.1f}%</td><td>{100*c["weak_exit_rate"]:.1f}%</td><td><a href="{r["replay"]}">Self-play</a></td></tr>'
    heur=''.join(f'<li>{r["heuristic"]}: {100*r["summary"]["block_rate"]:.1f}% blocks; depth {r["summary"]["mean_release_depth"]:.2f} m. '+', '.join(f'{k}: {100*r["groups"][k]["block_rate"]:.1f}%' for k in ['straight','left_bank','right_bank'])+'</li>' for r in STATE['heuristics'])
    validation=''.join(f'<li>{html.escape(r["name"])}: {100*r["defense"]["summary"]["block_rate"]:.1f}% blocks, {100*r["control"]["summary"]["settle_rate"]:.1f}% settled, {100*r["control"]["summary"]["fast_aimed_rate"]:.1f}% fast aimed returns.</li>' for r in STATE['validation'])
    ladder=''.join(f'<li>{html.escape(r["name"])} vs {html.escape(r["opponent"])}: {r["goals_for"]}–{r["goals_against"]}, <a href="{r["replay"]}">replay</a></li>' for r in STATE['ladder'])
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="60"><title>Control and defense experiments</title><style>body{background:#101827;color:#e5e7eb;font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}a{color:#7dd3fc}p{line-height:1.6}td,th{padding:10px;border-bottom:1px solid #374151;text-align:left}.scroll{overflow:auto}table{border-collapse:collapse}</style></head><body><h1>Control and defense — October 11</h1>'''
    page+=f'<p><b>{html.escape(STATE["status"])}</b><br>{STATE["updated"]}. Refreshes every minute. <a href="/training/">Live run status and checkpoints</a></p>'
    page+='''<p>Warm-started from <b>rail30-100-20261001-v1</b>. All new runs have <b>zero imitation loss</b>: no old observation/action targets and no reference-policy loss. Fresh PPO rollouts, neural self-play plus skill practice. Physical limits remain 100 m/s² and 3 cm rail clearance. The existing production policy is unchanged.</p><p>Controlled arms share initialization, seed, curriculum, optimizer and budget. They compare no new shaping, defensive depth preference, stricter control, and their combination. Settling requires a touched puck below 0.15 m/s for 0.20 s, near the paddle and 4 cm inside its forward reach. Extra first-touch cushioning credit applies to slow moving pucks; it cannot reward repeated taps or an untouched stationary puck. Capture reward is paid once per possession, drills continue through a shot, and fast aimed-shot rewards remain. Slow shallow incoming starts are mixed into every arm, including the control.</p><p>Defense is measured solely by blocking the first verified on-goal 10–18 m/s attack, balanced across straight and both banks. Heuristics use the same delayed/noisy observations, arrival decoder, workspace, acceleration and thermal simulation. They cannot see future launch timing, aim or shot type. They are benchmarks, never neural policy overrides.</p><p>Control fixtures balance shallow incoming, interior incoming and drifting-away pucks. Settled, subsequent aimed conversion, ≥4 m/s aimed return and weak exit are reported separately. No credit for passive waiting. Periodic checkpoints share the exact same development fixtures. Selection includes intermediate checkpoints; fresh seeds and a past-policy ladder are reserved for final validation. A larger network is tested with function-preserving initialization after the small sweep.</p><h2>Heuristic defense references (768 paired attacks)</h2><ul>'''+heur+'</ul><h2>Development checkpoints</h2><div class="scroll"><table><tr><th>Checkpoint</th><th>Blocks</th><th>Depth at release</th><th>Settled</th><th>Then aimed shot</th><th>Fast aimed return</th><th>Weak exit</th><th>Replay</th></tr>'+rows+'</table></div><h2>Fresh-seed validation</h2><ul>'+validation+'</ul><h2>Past-policy ladder</h2><ul>'+ladder+'</ul>'
    page+=f'<p>Selection: {html.escape(str(STATE["selected"] or "Pending. Current production model stays selected."))}</p></body></html>'
    temp=REPORT.with_suffix('.tmp');temp.write_text(page);temp.replace(REPORT)


def command(name,changes,steps,initial=BASE,resume=False):
    spec=importlib.util.spec_from_file_location('previous_recipe','ai/recipes/defense-science-20261005/run.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    raw=old.argv(name,{},steps,initial,resume,seed=20261071)
    def remove(key):
        flag='--'+key
        while flag in raw:
            i=raw.index(flag);del raw[i:i+(2 if i+1<len(raw) and not raw[i+1].startswith('--') else 1)]
    # Remove the DATA sources too: weights=0 alone leaves unnecessary loading
    # and makes accidental reactivation easier to miss.
    for k in ('skill-reference','skill-reference-max-load','skill-reference-incoming-only','skill-replay'):
        remove(k)
    common={'run-name':'_neural-control-defense-'+name+'-20261011',
        'evaluation-dir':str(OUT/name),'skill-replay-weight':0,'skill-reference-weight':0,
        'lr':.00001,'minutes':210,'save-every':8000000,'practice-defense-fraction':.45,
        'game-fraction':.30,'slow-receiving-fraction':.65,'recovery-fraction':.35,
        'stationary-replay-fraction':.5,'practice-selfplay-fraction':0,
        'defensive-depth-weight':0,'width':1024}
    common.update(changes)
    for k,v in common.items():
        remove(k);raw.extend(['--'+k,str(v)])
    assert '--skill-replay' not in raw and '--skill-reference' not in raw
    return raw


def evaluate(name,path,changes=None,smoke=False):
    publish('Evaluating '+name)
    d=defense(path,count=48 if smoke else 768,seed=20261073,device='cuda')
    c=control(path,count=24 if smoke else 384,seed=20261072,device='cuda')
    net,state=load(path);net.to('cpu');torch.set_num_threads(2)
    replay=Path('ai/recordings')/(path.parent.name+'_'+path.stem+'.json')
    match=games(net,seconds=6 if smoke else 45,n=2,seed=20261074,checkpoint=path,
        step=state.get('step',0),record=replay,initial_load=.4,report_sensing=True,continuous_rallies=True)
    row=dict(name=name,checkpoint=str(path),defense=d,control=c,changes=changes or {},
        replay='/?replay='+replay.name,match=match)
    STATE['rows'].append(row);write(OUT/'evaluations'/(name+'.json'),row)
    meta=json.loads((path.parent/'run.json').read_text());directory=meta.get('evaluation_dir')
    if directory and path.stem.startswith('agent_step_'):
        selection=Path(directory)/'development-selection.json'
        entries=json.loads(selection.read_text()).get('rows',[]) if selection.exists() else []
        entries.append(dict(checkpoint=str(path),sha256=d['sha256'],
            defense=dict(saved=d['summary']['saved'],trials=d['summary']['trials'],rate=d['summary']['block_rate']),
            replay=row['replay']))
        write(selection,dict(rows=entries))
    publish();return row


def train(name,changes,steps,initial=BASE,resume=False,smoke=False):
    cmd=command(name,changes,steps,initial,resume)
    run=Path('runs')/cmd[cmd.index('--run-name')+1]
    if run.exists():raise RuntimeError('Refusing to overwrite existing experiment '+str(run))
    write(OUT/name/'recipe.json',dict(argv=cmd,initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),simulation_only=True))
    publish('Training '+name)
    with (OUT/name/'train.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='ai',OMP_NUM_THREADS='2'))
        STATE['training_pid']=proc.pid;STATE['current_run']=run.name;publish()
        seen=set();last_publish=0
        try:
            while proc.poll() is None:
                snapshots=sorted(run.glob('agent_step_*.pt'))
                pending=[p for p in snapshots if p.name not in seen]
                if pending:
                    p=pending[0];seen.add(p.name)
                    evaluate(name+'-'+p.stem,p,changes,smoke)
                if time.time()-last_publish>30:
                    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
                    publish(f'Training {name}: {status.get("step",0):,} cumulative steps');last_publish=time.time()
                time.sleep(3)
            if proc.returncode:raise RuntimeError(f'{name} failed; see {OUT/name/"train.log"}')
        except BaseException:
            if proc.poll() is None:
                (run/'STOP').touch();proc.wait(timeout=120)
            raise
    for p in sorted(run.glob('agent_step_*.pt')):
        if p.name not in seen:evaluate(name+'-'+p.stem,p,changes,smoke)
    return evaluate(name,run/'agent.pt',changes,smoke)


def rank(row):
    d=row['defense']['summary'];c=row['control']['summary'];b=STATE['rows'][0]
    bc=b['control']['summary'];bd=b['defense']['summary']
    if d['overload_seconds'] or c['overload_seconds']:return -10
    # Predeclared balanced development objective, with penalties for regression.
    return (d['block_rate']-bd['block_rate'] + .6*(c['settle_rate']-bc['settle_rate'])
        + .3*(c['fast_aimed_rate']-bc['fast_aimed_rate'])
        - max(0,bc['fast_aimed_rate']-c['fast_aimed_rate']-.10)
        - 2*max(0,bd['block_rate']-d['block_rate']-.03))


def validate(winner):
    STATE['selected']='Development choice: '+winner['checkpoint']+'; validation pending';publish()
    for name,path in [('baseline',BASE),('candidate',Path(winner['checkpoint']))]:
        publish('Held-out validation '+name)
        d=defense(path,count=3072,seed=20261981,device='cuda')
        c=control(path,count=1536,seed=20261982,device='cuda')
        immediate=defense(path,count=1536,seed=20261983,immediate=True,device='cuda')
        net,_=load(path);net.to('cpu')
        shots=skills(net,seed=20261984,per_task=192,report_sensing=True,shot_request='random',initial_load=.4)
        r=dict(name=name,defense=d,control=c,immediate=immediate,shots=shots)
        STATE['validation'].append(r);write(OUT/(name+'-heldout.json'),r);publish()
        rivals=[BASE,Path('runs/possession-20260926-v2/agent.pt'),Path('runs/_neural-player-requests-defense30/agent_step_515850240.pt')]
        for rival in rivals:
            opponent,_=load(rival);opponent.to('cpu')
            for swap in (False,True):
                replay=Path('ai/recordings')/f'control-defense-{name}-vs-{rival.parent.name}-{int(swap)}.json'
                m=games(net,seconds=90,n=8,seed=20261985,checkpoint=path,opponent_net=opponent,
                    opponent_checkpoint=rival,swap_sides=swap,record=replay,initial_load=.4,
                    report_sensing=True,continuous_rallies=True)
                own=1 if swap else 0;scores=np.array(m['score'])
                row=dict(name=name,opponent=rival.parent.name,swap=swap,
                    goals_for=int(scores[own].sum()),goals_against=int(scores[1-own].sum()),
                    replay='/?replay='+replay.name,details=m)
                STATE['ladder'].append(row);write(OUT/f'{name}-ladder-{rival.parent.name}-{int(swap)}.json',row);publish()
    b,c=STATE['validation']
    delta=np.array([x['saved']-y['saved'] for x,y in zip(c['defense']['details'],b['defense']['details']) if x['valid'] and y['valid']],dtype=float)
    ci=[float(delta.mean()-1.96*delta.std(ddof=1)/np.sqrt(len(delta))),float(delta.mean()+1.96*delta.std(ddof=1)/np.sqrt(len(delta)))]
    def ladder_share(name):
        rows=[r for r in STATE['ladder'] if r['name']==name]
        gf=sum(r['goals_for'] for r in rows);ga=sum(r['goals_against'] for r in rows)
        return gf/max(1,gf+ga)
    own_overload=any(np.asarray(r['details']['overload_seconds'])[1 if r['swap'] else 0].sum()>0 for r in STATE['ladder'] if r['name']=='candidate')
    gates=dict(defense_improved=ci[0]>0,
        settling_not_worse=c['control']['summary']['settle_rate']>=b['control']['summary']['settle_rate']-.02,
        fast_returns_retained=c['control']['summary']['fast_aimed_rate']>=b['control']['summary']['fast_aimed_rate']-.10,
        immediate_retained=c['immediate']['summary']['block_rate']>=b['immediate']['summary']['block_rate']-.03,
        ladder_retained=ladder_share('candidate')>=ladder_share('baseline')-.05,
        thermal_ok=not own_overload and c['defense']['summary']['overload_seconds']==0 and c['control']['summary']['overload_seconds']==0)
    STATE['gates']=gates;STATE['paired_defense_difference_ci95']=ci
    accepted=all(gates.values())
    STATE['selected']=(winner['checkpoint']+' — qualified simulation candidate; deployment unchanged') if accepted else 'Keep rail30-100-20261001-v1. Experimental candidate did not pass every validation gate.'
    write(OUT/'selection.json',dict(candidate=winner['checkpoint'],accepted=accepted,gates=gates,paired_defense_difference_ci95=ci))
    publish('Complete — '+('candidate improved and passed validation' if accepted else 'review measured improvements and remaining regressions'))


def main(smoke=False):
    publish()
    evaluate('baseline',BASE,smoke=smoke)
    for style in ['deep-bank','forward-bank','deep-direct']:
        r=defense(BASE,count=48 if smoke else 768,seed=20261073,heuristic=style)
        STATE['heuristics'].append(r);write(OUT/(style+'.json'),r);publish()
    for name,changes in (ARMS[:1] if smoke else ARMS):
        train(('smoke-' if smoke else '')+name,changes,131072 if smoke else 20000000,smoke=smoke)
    if smoke:publish('Smoke test complete');return
    best=max(STATE['rows'][1:],key=rank)
    # Isolate network capacity: same parent, settings and transitions, width only.
    for width in (1024,1536):
        train('capacity'+str(width),{**best['changes'],'width':width},20000000,Path(best['checkpoint']))
    candidates=sorted(STATE['rows'][1:],key=rank,reverse=True)
    finalists=[];families=set()
    for r in candidates:
        family=r['name'].split('-agent_step')[0]
        if family in families:continue
        finalists.append(r);families.add(family)
        if len(finalists)==2:break
    for i,r in enumerate(finalists):
        train('long'+str(i),r['changes'],96000000,Path(r['checkpoint']),resume=True)
    winner=max(STATE['rows'],key=rank)
    validate(winner)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a.smoke)
    except BaseException as exc:
        STATE['error']=repr(exc);publish('FAILED: '+str(exc));raise
