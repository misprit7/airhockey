#!/usr/bin/env python3
"""Render a standalone, read-only training review from a saved result manifest."""
import argparse
import base64
from html import escape
import json
from pathlib import Path


def render(data, load_map=None):
    esc=lambda value: escape(str(value),quote=True)
    rows=lambda values: ''.join('<tr>'+''.join(f'<td>{esc(cell)}</td>' for cell in row)+'</tr>' for row in values)
    bullets=lambda values: '<ul>'+''.join(f'<li>{esc(value)}</li>' for value in values)+'</ul>'
    p=data['policy'];s=data['setup']
    image=''
    if load_map:
        encoded=base64.b64encode(load_map.read_bytes()).decode('ascii')
        image=f'<figure><img alt="Predicted holding load across the expanded workspace, with measured sites marked" src="data:image/png;base64,{encoded}"><figcaption>Predicted steady holding load. Dots are surveyed locations; the outer region uses a conservative prior and still needs physical measurement.</figcaption></figure>'
    setup_rows=[['Acceleration cap',f"{s['acceleration_m_s2']} m/s²"],['Speed ceiling',f"{s['speed_m_s']} m/s"],['Paddle rim clearance','30 mm at the side and back rails'],['Grid-frame center bounds',str(s['workspace_bounds_mm'])+' mm (X min/max, Y min/max)'],['Edge penalty',f"{s['edge_dwell_weight']} reward units/s per rail at the boundary, fading over {100*s['edge_dwell_band_m']:g} cm"],['Pretension assumed by load model','1.5 mm']]
    comparison=rows(data['comparison_rows'])
    endurance=rows(data['endurance_rows'])
    provenance=esc(json.dumps(data['provenance'],indent=2))
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expanded workspace training · {esc(data['date'])}</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#172538;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:36px 24px 70px}}header{{padding:12px 0 24px}}h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;letter-spacing:-.03em;margin:12px 0}}h2{{font-size:22px;margin:0 0 14px}}p{{max-width:86ch}}.eyebrow{{color:#47617a;font-weight:650}}.badge{{display:inline-block;background:#fff2ca;color:#684d0a;border:1px solid #e5c877;border-radius:5px;padding:4px 10px}}.card{{background:white;border:1px solid #d9e2eb;border-radius:10px;padding:24px;margin:18px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.grid .card{{margin:0}}a{{color:#126092}}.actions{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.actions a{{padding:10px 16px;background:#126092;color:white;border-radius:5px;text-decoration:none}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border-bottom:1px solid #e2e8f0;padding:10px 12px;text-align:left;vertical-align:top}}th{{background:#f1f5f9}}.scroll{{overflow-x:auto}}.note,figcaption{{color:#4f6174;font-size:14px}}code,pre{{font-family:ui-monospace,monospace}}code{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}figure{{margin:20px 0 0}}img{{max-width:100%;height:auto}}li{{margin:8px 0}}summary{{cursor:pointer;font-weight:650}}@media(max-width:780px){{.grid{{grid-template-columns:1fr}}main{{padding:20px 14px}}.card{{padding:18px}}}}
</style><main>
<header><div class="eyebrow">TRAINING REVIEW · {esc(data['date'])}</div><h1>30 mm rail clearance.<br>100 m/s² acceleration.</h1><span class="badge">Simulation candidate · physical qualification pending</span><p>{esc(data['summary'])}</p><p><strong>Selected policy:</strong> <code>{esc(p['name'])}</code></p><div class="actions"><a href="{esc(p['replay_url'])}">Watch selected policy</a><a href="/training">Training dashboard</a></div></header>
<section class="card"><h2>What changed</h2>{bullets(data['changes'])}</section>
<div class="grid"><section class="card"><h2>Selected configuration</h2><table>{rows(setup_rows)}</table><p class="note">These are the selected training settings. Existing physical deployment defaults were preserved.</p></section>
<section class="card"><h2>Still one neural policy</h2><p>The actor uses 16 frames of 42 physical features and its own three-way shot request: 675 inputs. Three 1,024-unit hidden layers produce six arrival-action outputs: position, velocity, arrival time, and acceleration effort.</p><p>The opposing player's requested shot is private. A separate value network gets training-only exercise context; that context does not enter the actor. The existing arrival decoder and motion guard enforce the simulated workspace and motion ceilings.</p><p>{esc(data['training_method'])}</p></section></div>
<section class="card"><h2>Compared with the previous best policy</h2><p>{esc(data['comparison_method'])}</p><div class="scroll"><table><thead><tr><th>Test</th><th>Previous policy</th><th>Selected policy</th></tr></thead><tbody>{comparison}</tbody></table></div><p class="note">Requested fast shots mean the first shot was on goal, followed the requested straight/left-bank/right-bank route, and reached at least 6 m/s. Save rates describe the listed simulated fixtures.</p></section>
<section class="card"><h2>Rewards in the selected run</h2><p>The goal remains scoring while defending and managing motor load. Auxiliary rewards guide learning; they do not choose actions at runtime. Event, per-second and potential-difference terms have different units, so their raw weights are not directly comparable.</p><div class="scroll"><table><thead><tr><th>Term</th><th>Weight</th><th>Purpose</th></tr></thead><tbody>{rows(data.get('reward_rows',[]))}</tbody></table></div></section>
<section class="card"><h2>Sustained-load checks</h2><div class="scroll"><table><thead><tr><th>Test conditions</th><th>Previous policy</th><th>Selected policy</th></tr></thead><tbody>{endurance}</tbody></table></div><p>{esc(data['load_conclusion'])}</p><p class="note">A match fails this screen if either simulated player crosses the modeled load threshold. Stress gain 1.3 multiplies estimated current squared, equivalent to about 14% more current. Recorded overload time continues after the first crossing; real drive protection would stop the robot.</p></section>
<section class="card"><h2>Updated motor-load model</h2><p>The model uses post-replacement characterization logs at nine holding locations and 1.5 mm pretension. It combines a spatial holding-current map with direction-dependent acceleration and speed terms, then integrates separate fast and slow load states for each motor.</p><p>All four motors have the same part family. Their logged drive settings still differ: fast RMS limits are approximately 4.1 / 5.8 / 4.1 / 5.8 A; slow limits are 4.0 / 4.1 / 4.0 / 4.1 A. The fit retains those measured settings. Holding behavior outside the surveyed area remains an explicit conservative estimate.</p>{image}</section>
<section class="card"><h2>Limits of this result</h2>{bullets(data['limitations'])}</section>
<section class="card"><h2>Experiment selection</h2><p>{esc(data.get('selection_method','Checkpoints were compared on shooting, defense, edge recovery and sustained motor-load tests.'))}</p>{bullets(data.get('experiment_findings',[]))}</section>
<section class="card"><details><summary>Checkpoint and evidence provenance</summary><p>Checkpoint: <code>{esc(p['source_checkpoint'])}</code></p><p>SHA-256: <code>{esc(p['sha256'])}</code></p><pre>{provenance}</pre></details></section>
</main></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--load-map',type=Path)
    args=parser.parse_args();data=json.loads(args.manifest.read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render(data,args.load_map))
    print(args.output)
