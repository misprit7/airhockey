"""Reproducible, simulation-only reward sweep, continuation and held-out ladder.
Run from repo root with PYTHONPATH=ai. No deployment or hardware operations.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'ai'),str(ROOT/'ai/bin')]
from eval_fast_defense import evaluate as defense
from eval_neural_player import load, games, skills
import torch
import numpy as np

BASE=Path('runs/rail30-100-20261001-v1/agent.pt')
OUT=Path('logs/neural-player/defense-science-20261005')
REPORT=Path('ai/airhockey/web/training-report-20261005.html')
REFS=[BASE,Path('runs/possession-20260926-v2/agent.pt'),
 Path('runs/_neural-player-requests-defense30/agent_step_515850240.pt'),
 Path('runs/_neural-player-attack-stage7/agent_step_318898176.pt')]
ARMS=[('control',{}),('depth200',{'defensive-depth-weight':200}),
 ('depth800',{'defensive-depth-weight':800}),
 ('depth200-load16',{'defensive-depth-weight':200,'load-weight':16}),
 ('depth200-lr10',{'defensive-depth-weight':200,'lr':.00001}),
 ('depth200-back20',{'defensive-depth-weight':200,'defensive-depth-target':.20})]
STATE={'status':'Starting','rows':[],'validation':[],'ladder':[],'selected':None}


def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def publish(message=None):
 if message: STATE['status']=message
 STATE['updated']=time.strftime('%Y-%m-%d %H:%M:%S %Z');STATE['pid']=os.getpid()
 write(OUT/'experiment.json',STATE)
 def pct(x):return f'{100*x:.1f}%'
 rows=''
 for r in STATE['rows']:
  d=r['defense'];g=d['groups'];s=d['summary'];links=' '.join(f'<a href="{html.escape(u)}">{i+1}</a>' for i,u in enumerate(d.get('replays',[])))
  rows+=f'<tr><td>{html.escape(r["name"])}</td><td>{pct(s["block_rate"])}</td><td>{pct(g["straight"]["block_rate"])}</td><td>{pct(g["left_bank"]["block_rate"])}</td><td>{pct(g["right_bank"]["block_rate"])}</td><td>{s["mean_release_depth"]:.2f} m</td><td>{s["trials"]}</td><td><a href="{r.get("replay","#")}">match</a> · drills {links}</td></tr>'
 val=''.join(f'<li>{html.escape(r["name"])}: {pct(r["defense"]["summary"]["block_rate"])} blocked ({r["defense"]["summary"]["trials"]} attacks), '+('immediate release' if r['defense']['immediate'] else 'hidden windup')+'</li>' for r in STATE['validation'])
 ladder=''.join(f'<li>{html.escape(r["name"])} vs {html.escape(r["opponent"])}: candidate {r["goals_for"]}, opponent {r["goals_against"]}; {r["seconds"]} s × {r["games"]} games; '+f'<a href="{r["replay"]}">replay</a></li>' for r in STATE['ladder'])
 page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="60"><title>Defense experiments — October 5</title><style>body{background:#111827;color:#e5e7eb;font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}a{color:#7dd3fc}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #374151;text-align:left}.scroll{overflow:auto}p{line-height:1.6}code{color:#a7f3d0}</style></head><body><h1>Defense experiments</h1>'''
 page+=f'<p><b>{html.escape(STATE["status"])}</b><br>Updated {STATE["updated"]}. Refreshes every minute. <a href="/training">Live training/checkpoints</a></p>'
 page+='''<p>Starting from the current production policy. One neural network; reward shaping only. All runs keep the 100 m/s² acceleration cap and 3 cm rail buffer. Nothing is deployed automatically.</p><p><b>Primary metric:</b> percentage of verified on-goal, 10–18 m/s attacks blocked. Straight, left bank and right bank releases are balanced, with hidden timing and target. The policy can prepare for 0.35–1.2 seconds, starting at varied positions and loads. Contact must reverse the goalward flight or stop the puck; goals and unresolved trials fail. Control and counterattacks earn no extra credit. Shots that miss an empty goal are excluded. Separate immediate-release tests distinguish positioning from reaction.</p><p><b>Controlled comparison:</b> identical initialization, seed, practice distribution and training budget for the control, depth200 and depth800 arms. All receive bank preparation drills. The added cost is quadratic beyond 30 cm from the goal, only while the opponent can prepare; it switches off during fast incoming attacks. Other arms separately test lower load shaping, faster learning, and a 20 cm preferred depth. Subsequent continuation selects on development data; the final fresh-seed test is not used to choose a checkpoint.</p><h2>Development results</h2><div class="scroll"><table><tr><th>Checkpoint</th><th>All blocks</th><th>Straight</th><th>Left bank</th><th>Right bank</th><th>Depth at release</th><th>Trials</th><th>Replays</th></tr>'''+rows+'</table></div><p>Drill replays deliberately show a save and a miss when available; they are not an unbiased sample. Match replays use neural policies on both sides.</p><h2>Fresh-seed validation</h2><ul>'+val+'</ul><h2>Past-policy ladder</h2><ul>'+ladder+'</ul>'
 page+=f'<p>Selection: {html.escape(str(STATE["selected"] or "Pending validation; current production model remains selected."))}</p></body></html>'
 tmp=REPORT.with_suffix('.tmp');tmp.write_text(page);tmp.replace(REPORT)


def argv(name,changes,steps,initial=BASE,resume=False,seed=20261050):
 raw=json.loads(Path('ai/recipes/rail30-accel100-anticipation-20261001.json').read_text())['training_argv'][2:]
 def remove(key):
  flag='--'+key
  while flag in raw:
   i=raw.index(flag);n=2 if i+1<len(raw) and not raw[i+1].startswith('--') else 1
   del raw[i:i+n]
 def set_(key,value):
  remove(key)
  if value is not None:raw.extend(['--'+key,str(value)])
 settings={'run-name':'_neural-defense-science-'+name+'-20261005','evaluation-dir':str(OUT/name),
  'steps':steps,'minutes':150,'save-every':4000000,'seed':seed,'lr':.000004,
  'load-weight':32,'readiness-weight':0,'readiness-cost-weight':0,'skill-reference-weight':0,
  'skill-replay-weight':8,'practice-defense-fraction':.65,'defense-windup-fraction':.85,
  'defense-windup-bank-fraction':2/3,'defense-min-speed':10,'defense-max-speed':18,
  'game-fraction':.25,'defensive-depth-weight':0,'defensive-depth-target':.30,
  'defense-clear-reward':300,'skill-concede-weight':1000}
 settings.update(changes)
 remove('resume');remove('initialize-from');settings['resume' if resume else 'initialize-from']=str(initial)
 # Current actor is also a fixed opponent; old workspace policies are transformed by the trainer.
 remove('opponent-checkpoint');settings['opponent-checkpoint']=str(BASE)
 for k,v in settings.items():set_(k,v)
 raw.append('--defense-block-only')
 return [sys.executable,'ai/bin/train_neural_player.py',*raw]


def score_checkpoint(name,path,*,count=768,periodic=False):
 publish('Evaluating '+name)
 dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
 d=defense(path,count=count,seed=20261051,record_prefix=Path('ai/recordings')/('defense-science-'+name))
 write(dest/'fast-defense.json',d)
 net,state=load(path);net.to('cpu');torch.set_num_threads(2)
 replay=Path('ai/recordings')/(path.parent.name+'_'+path.stem+'.json')
 match=games(net,seconds=30 if periodic else 60,n=2 if periodic else 4,seed=20261052,
  checkpoint=path,step=state.get('step',0),record=replay,initial_load=.4,report_sensing=True,continuous_rallies=True)
 write(dest/'selfplay.json',match)
 row=dict(name=name,checkpoint=str(path),defense=d,replay='/?replay='+replay.name,match=match)
 STATE['rows'].append(row)
 # Make checkpoint evaluations discoverable in the existing dashboard.
 metadata=json.loads((path.parent/'run.json').read_text());evaluation=metadata.get('evaluation_dir')
 if evaluation and path.stem.startswith('agent_step_'):
  directory=Path(evaluation);old=[]
  if (directory/'development-selection.json').exists():old=json.loads((directory/'development-selection.json').read_text()).get('rows',[])
  old.append(dict(checkpoint=str(path),sha256=d['sha256'],defense=dict(saved=d['summary']['saved'],trials=d['summary']['trials'],rate=d['summary']['block_rate']),replay=row['replay'],eligible=False))
  write(directory/'development-selection.json',dict(rows=old))
 publish();return row


def train(name,changes,steps,initial=BASE,resume=False,smoke=False):
 command=argv(name,changes,steps,initial,resume)
 run=Path('runs')/command[command.index('--run-name')+1]
 write(OUT/name/'recipe.json',dict(argv=command,initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),simulation_only=True))
 publish('Training '+name)
 with (OUT/name/'train.log').open('w') as log:
  proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='ai',OMP_NUM_THREADS='2'))
  seen=set();last_publish=0
  while proc.poll() is None:
   snapshots=sorted(run.glob('agent_step_*.pt'))
   pending=[p for p in snapshots if p.name not in seen]
   if pending:
    p=pending[-1];seen.update(x.name for x in pending)
    score_checkpoint(name+'-'+p.stem,p,count=48 if smoke else 384,periodic=True)
   if time.time()-last_publish>30:
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    publish(f'Training {name}: {status.get("step",0):,} cumulative steps');last_publish=time.time()
   time.sleep(3)
  if proc.returncode:raise RuntimeError(f'{name} failed; see {OUT/name/"train.log"}')
 return score_checkpoint(name,run/'agent.pt',count=48 if smoke else 768)


def main(smoke=False):
 OUT.mkdir(parents=True,exist_ok=True);publish()
 baseline=score_checkpoint('baseline',BASE,count=48 if smoke else 768)
 first=[]
 for name,changes in (ARMS[:1] if smoke else ARMS):
  first.append((train(('smoke-' if smoke else '')+name,changes,131072 if smoke else 12000000,smoke=smoke),changes))
 if smoke:publish('Smoke test complete');return
 # Use only development block rate to rank defense; thermal violations disqualify.
 ranked=sorted(first,key=lambda x:x[0]['defense']['summary']['block_rate'] if x[0]['defense']['summary']['overload_seconds']==0 else -1,reverse=True)
 finalists=[]
 for row,changes in ranked[:2]:
  finalists.append(train(row['name']+'-long',changes,48000000,Path(row['checkpoint']),resume=True))
 eligible=[r for r in finalists+[x[0] for x in first]+[baseline] if r['defense']['summary']['overload_seconds']==0]
 winner=max(eligible,key=lambda r:r['defense']['summary']['block_rate'])
 # Freeze selection before opening the held-out cohort.
 STATE['selected']='Development winner: '+winner['checkpoint']+'; held-out and ladder validation pending'
 publish('Held-out validation')
 for label,path in [('baseline',BASE),('winner',Path(winner['checkpoint']))]:
  for immediate in [False,True]:
   d=defense(path,count=3072,seed=20261993,immediate=immediate)
   STATE['validation'].append(dict(name=label,defense=d));write(OUT/(label+('-immediate' if immediate else '-heldout')+'.json'),d);publish()
  net,state=load(path);net.to('cpu')
  shots=skills(net,seed=20261994,per_task=256,report_sensing=True,shot_request='random',receiving_speed_range=(3,8),initial_load=.4)
  write(OUT/(label+'-skills.json'),shots)
  for rival in REFS:
   opponent,_=load(rival);opponent.to('cpu')
   for swap in [False,True]:
    replay=Path('ai/recordings')/f'defense-science-{label}_vs_{rival.parent.name}-{int(swap)}.json'
    match=games(net,seconds=120,n=8,seed=20261995,checkpoint=path,opponent_net=opponent,
     opponent_checkpoint=rival,swap_sides=swap,record=replay,initial_load=.4,report_sensing=True,continuous_rallies=True)
    scores=np.array(match['score']);own=1 if swap else 0
    result=dict(name=label,opponent=rival.parent.name,swap=swap,goals_for=int(scores[own].sum()),
     goals_against=int(scores[1-own].sum()),seconds=120,games=8,replay='/?replay='+replay.name,details=match)
    STATE['ladder'].append(result);write(OUT/f'{label}-ladder-{rival.parent.name}-{int(swap)}.json',result);publish()
 winner_path=Path(winner['checkpoint']);STATE['selected']=str(winner_path)+' (experimental; production unchanged)'
 write(winner_path.parent/'review.json',dict(summary='Defense experiment winner on development cohort; see held-out and ladder results in /training/reports/20261005. Not automatically deployed.',report='/training/reports/20261005'))
 publish('Experiment complete — review held-out defense, attack and ladder results before deployment')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
 try:main(a.smoke)
 except BaseException as exc:
  STATE['error']=repr(exc);publish('FAILED: '+str(exc));raise
