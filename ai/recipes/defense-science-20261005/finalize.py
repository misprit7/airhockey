"""Wait for the experiment, then publish paired uncertainty and regression checks.
No hardware I/O; never promotes a deployment alias.
"""
import html
import json
from pathlib import Path
import time
import numpy as np

OUT=Path('logs/neural-player/defense-science-20261005')
REPORT=Path('ai/airhockey/web/training-report-20261005.html')


def compare(a,b):
    valid=np.array([x['valid'] and y['valid'] for x,y in zip(a['details'],b['details'])])
    delta=np.array([int(y['saved'])-int(x['saved']) for x,y in zip(a['details'],b['details'])])[valid]
    if len(delta)<2:return dict(trials=len(delta),difference=None,ci95=None)
    mean=float(delta.mean());half=float(1.96*delta.std(ddof=1)/np.sqrt(len(delta)))
    return dict(trials=len(delta),difference=mean,ci95=[mean-half,mean+half],
                gained=int((delta>0).sum()),lost=int((delta<0).sum()))


def finalize(state):
    load=lambda name:json.loads((OUT/name).read_text())
    paired={name:compare(load('baseline-'+name+'.json'),load('winner-'+name+'.json'))
            for name in ('heldout','immediate')}
    shots={label:load(label+'-skills.json') for label in ('baseline','winner')}
    attack={label:{task:dict(trials=s[task]['trials'],fast_requested=s[task]['first_requested_shots_at_least_6m_s'],
        rate=s[task]['first_requested_shots_at_least_6m_s']/max(s[task]['trials'],1))
        for task in ('stationary','receiving')} for label,s in shots.items()}
    ladder={}
    for label in ('baseline','winner'):
        rows=[r for r in state['ladder'] if r['name']==label]
        gf=sum(r['goals_for'] for r in rows);ga=sum(r['goals_against'] for r in rows)
        ladder[label]=dict(goals_for=gf,goals_against=ga,goal_share=gf/max(1,gf+ga),
            simulation_minutes=sum(r['seconds']*r['games'] for r in rows)/60,
            overload_seconds=sum(float(np.asarray(r['details']['overload_seconds'])[1 if r['swap'] else 0].sum()) for r in rows))
    attack_ok=all(attack['winner'][task]['rate']>=attack['baseline'][task]['rate']-.10 for task in ('stationary','receiving'))
    bank_positive=paired['heldout']['ci95'][0]>0
    reaction_ok=paired['immediate']['difference']>=-.03
    load_ok=ladder['winner']['overload_seconds']==0
    ladder_ok=ladder['winner']['goal_share']>=ladder['baseline']['goal_share']-.05
    report=dict(paired_defense=paired,attack=attack,ladder=ladder,
        checks=dict(heldout_defense_improves=bank_positive,immediate_reaction_not_materially_worse=reaction_ok,
                    attack_within_10pp=attack_ok,ladder_goal_share_within_5pp=ladder_ok,no_ladder_overloads=load_ok),
        passed=bool(bank_positive and reaction_ok and attack_ok and load_ok and ladder_ok),
        limitation='One training seed per controlled arm; simulator result, not proof of hardware performance. Candidate selected before this fresh-seed cohort; no automatic deployment.')
    (OUT/'qualification-summary.json').write_text(json.dumps(report,indent=2))
    body='<h2>Paired held-out comparison and regression checks</h2><p>Changes below are winner minus baseline on the same eligible attacks. Confidence intervals describe fixture sampling, not variability across training seeds.</p><ul>'
    for name,row in paired.items():
        lo,hi=row['ci95'];body+=f'<li>{name}: {100*row["difference"]:+.1f} percentage points, 95% paired interval [{100*lo:+.1f}, {100*hi:+.1f}]; {row["gained"]} newly blocked, {row["lost"]} newly missed.</li>'
    body+='</ul><p><b>'+('Passed the predeclared regression screen.' if report['passed'] else 'Did not pass every regression check; do not treat this as an overall replacement.')+'</b></p><ul>'
    for label,data in attack.items():
        body+=f'<li>{label} requested fast shots: stationary {data["stationary"]["rate"]:.1%}; receiving {data["receiving"]["rate"]:.1%}.</li>'
    for label,data in ladder.items():
        body+=f'<li>{label} ladder: {data["goals_for"]}–{data["goals_against"]}, goal share {data["goal_share"]:.1%}; simulated overload duration {data["overload_seconds"]:.2f} s.</li>'
    for k,v in report['checks'].items():body+=f'<li>{html.escape(k.replace("_"," "))}: {"pass" if v else "FAIL"}</li>'
    body+='</ul><p>'+html.escape(report['limitation'])+'</p>'
    page=REPORT.read_text().replace('</body>',body+'</body>')
    tmp=REPORT.with_suffix('.tmp');tmp.write_text(page);tmp.replace(REPORT)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    while True:
        try:
            state=json.loads((OUT/'experiment.json').read_text())
            if state['status'].startswith('FAILED'):raise RuntimeError(state['status'])
            if state['status'].startswith('Experiment complete'):break
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(30)
    finalize(state)
