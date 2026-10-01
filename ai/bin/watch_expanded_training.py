#!/usr/bin/env python3
"""Evaluate saved simulation checkpoints and publish dashboard replays."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 selection=a.output_dir/'development-selection.json'
 rows=json.loads(selection.read_text())['rows'] if selection.exists() else []
 seen={r['checkpoint'] for r in rows}
 def call(script,ckpt,out,flags):
  if out.exists():return json.loads(out.read_text())
  with out.with_suffix('.log').open('w') as log:
   subprocess.run([sys.executable,str(ROOT/'ai/bin'/script),str(ckpt),'--output',str(out),*flags],
    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'ai'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'))
  return json.loads(out.read_text())
 while True:
  pending=sorted((p for p in a.run.glob('agent_step_*.pt') if str(p) not in seen),key=lambda p:int(p.stem.rsplit('_',1)[1]))
  for ckpt in pending:
   report=a.output_dir/ckpt.stem;report.mkdir(exist_ok=True)
   skill=call('eval_neural_player.py',ckpt,report/'skills.json',['--skills-only','--per-task','128','--request-suite','--receiving-min','3','--receiving-max','8','--seed','20261011'])
   defense=call('eval_neural_preparation.py',ckpt,report/'preparation.json',['--trials','384','--seed','20261012','--lateral-speed','.5','--speed-min','10','--speed-max','16'])
   record=ROOT/'ai/recordings'/f'{a.run.name}_{ckpt.stem}.json'
   match=call('eval_neural_player.py',ckpt,report/'selfplay.json',['--games-only','--games','4','--seconds','90','--record',str(record),'--seed','20261013'])
   fringe=call('eval_neural_fringe.py',ckpt,report/'fringe.json',['--trials','256','--seed','20261015'])
   endurance=call('eval_neural_player.py',ckpt,report/'endurance.json',['--games-only','--games','4','--seconds','600','--initial-load','.8','--thermal-gain','1.3','--seed','20261014'])
   shots={}
   for name,suite in skill['requested_skills'].items():
    shots[name]={kind:{k:suite[kind][k] for k in ('trials','controlled','control_to_requested_fast','first_requested_shots_at_least_6m_s')} for kind in ('stationary','receiving')}
   fast=sum(s[k]['first_requested_shots_at_least_6m_s'] for s in shots.values() for k in ('stationary','receiving'))/768
   saves=defense['saved']/defense['trials'];selfplay=match['selfplay']
   peak=float(np.max(selfplay['peak_load']));overload=float(np.sum(selfplay['overload_seconds']))
   long_peak=float(np.max(endurance['selfplay']['peak_load']))
   long_overload=float(np.sum(endurance['selfplay']['overload_seconds']))
   shot_totals={kind:sum(s[kind]['first_requested_shots_at_least_6m_s'] for s in shots.values()) for kind in ('stationary','receiving')}
   row=dict(checkpoint=str(ckpt),sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),shots=shot_totals,shot_routes=shots,defense=dict(saved=defense['saved'],trials=defense['trials'],rate=saves,forward_at_release=defense['forward_at_release']),
     short_peak=peak,overload_seconds=overload,fringe=fringe['summary'],
     endurance_peak=long_peak,endurance_overload_seconds=long_overload,
     eligible=bool(overload==0 and peak<1 and long_overload==0 and long_peak<1 and fast>.45 and saves>.65),
     development_score=fast+saves,replay='/?replay='+record.name,screen='Development only; independent longer qualification still required.')
   rows.append(row);seen.add(str(ckpt));write(selection,dict(rows=rows))
   print(json.dumps(row),flush=True)
  status=a.run/'status.json'
  if status.exists() and not json.loads(status.read_text()).get('running',True):break
  time.sleep(10)

if __name__=='__main__':main()
