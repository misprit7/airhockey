"""Build a standalone, read-only HTML review from the pinned offline analyses."""
import base64
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('logs/analysis/neural-live-20261002')
load=lambda name: json.loads((out/name).read_text())
r=load('review.json'); m=load('motion-load.json'); s=load('sim-acceleration.json')
with np.load(out/'20261002-114149.npz') as archive:
    a={k:archive[k] for k in ('load_t','rms','puck','cam','ctl')}
active=np.load(out/'active-ticks.npz')
correction=np.load(out/'marker-correction.npz')['poses']
actor=(.05+.95*((active['actions'][:,5]+1)/2)**2)*100
command=active['cmd_accel']/1000
plt.rcParams.update({'figure.facecolor':'#101d30','axes.facecolor':'#101d30',
    'text.color':'#e2eaf4','axes.labelcolor':'#e2eaf4','xtick.color':'#bac9da',
    'ytick.color':'#bac9da','axes.edgecolor':'#68788c','font.size':10})
def png(fig, name):
    fig.tight_layout(); fig.savefig(out/name,dpi=140)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=140);plt.close(fig)
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
fig,ax=plt.subplots(figsize=(10,3.8))
bins=np.arange(0,106,5)
ax.hist(actor,bins=bins,weights=np.full(len(actor),100/len(actor)),alpha=.75,label='Network choice',color='#65b8ff')
ax.hist(command,bins=bins,weights=np.full(len(command),100/len(command)),histtype='step',lw=2,label='Command after guard',color='#ffd166')
ax.set(xlabel='Acceleration cap (m/s²)',ylabel='% of active decisions',title='The actor itself never chose 90–100 m/s² in this session')
ax.legend();hist=png(fig,'acceleration.png')
fig,ax=plt.subplots(figsize=(10,3.8)); q=a['load_t']<850
for k,color in enumerate(['#65b8ff','#ffd166','#fa8d8d','#84e0bb']):
    ax.plot(a['load_t'][q],a['rms'][q,k],label=f'Motor {k}',color=color,lw=1)
ax.axhline(100,color='#fa8d8d',ls='--');ax.set(xlabel='Session time (s)',ylabel='Drive fast RMS (%)',ylim=(0,105),title='Peak fast RMS: 56%, 40%, 68%, 41%; no drive faults')
ax.legend(ncol=4);thermal=png(fig,'thermal.png')

# Continuous line fits use a chronological holdout; do not select physics from goals.
b=r['banks'];train=[v for v in b if v['t']<400];test=[v for v in b if v['t']>=400]
rail=[]
for component,default,sign in [(1,.785,-1),(0,.9,1)]:
    x=np.array([v['incoming'][component]/1000 for v in train]);y=np.array([sign*v['outgoing'][component]/1000 for v in train])
    fitted=float(x@y/(x@x));xt=np.array([v['incoming'][component]/1000 for v in test]);yt=np.array([sign*v['outgoing'][component]/1000 for v in test])
    rail.append(dict(default=default,fitted=fitted,default_rmse=float(np.sqrt(np.mean((yt-default*xt)**2))),fitted_rmse=float(np.sqrt(np.mean((yt-fitted*xt)**2)))))

# Viewer samples require bracketing observations; never hold a missing puck at the goal.
def sample(track,t,columns=(1,2),gap=.04):
    j=int(np.searchsorted(track[:,0],t))
    if j==0 or j==len(track) or track[j,0]-track[j-1,0]>gap:return None
    f=(t-track[j-1,0])/(track[j,0]-track[j-1,0])
    return [round(float(track[j-1,k]*(1-f)+track[j,k]*f),3) for k in columns]
rollouts=load('goal-physics-rollouts.json')
goals=[]
for i,g in enumerate(r['goals']):
    end=float(a['puck'][a['puck'][:,0]<=g['t'],0][-1]);start=g['t']-(1.8 if i==3 else 1.0)
    times=np.arange(start,end,.01); frames=[]
    simulation=next((v for v in rollouts if abs(v['start']-g['crossing_t'])<1e-5),None)
    simtrack=np.array(simulation['frames']) if simulation else np.empty((0,5))
    if len(simtrack):
        for offset in (1,3):
            sx=simtrack[:,offset].copy();sy=simtrack[:,offset+1].copy()
            simtrack[:,offset]=2017.9-sy*1014.6
            simtrack[:,offset+1]=-19+sx*1003.9
    for t in times:
        j=max(0,int(np.searchsorted(active['t_cam'],t,side='right'))-1)
        frames.append(dict(t=round(float(t),4),p=sample(a['puck'],t,gap=.022),
            c=sample(a['cam'],t),k=sample(a['ctl'],t,gap=.04),
            sp=sample(simtrack,t,gap=.025),sr=sample(simtrack,t,columns=(3,4),gap=.025),
            corrected=sample(correction,t,gap=.022),target=[float(active['cmd_x'][j]),float(active['cmd_y'][j])],
            accel=round(float(command[j]),1),actor=round(float(actor[j]),1)))
    goals.append(dict(time=g['t'],frames=frames,depth=round(g['distance_ahead_goal_mm']),
        max_accel=round(g['accel']['max'],1),fallback=i==6))
rows=''.join(f'<tr><td>{g["t"]:.2f} s</td><td>{g["distance_ahead_goal_mm"]/10:.1f} cm{"*" if i==6 else ""}</td><td>{g["accel"]["max"]:.1f}</td><td>{"Slow/earlier contact" if i==3 else "Side-bank approach"}</td></tr>' for i,g in enumerate(r['goals']))
railrows=''.join(f'<tr><td>{name}</td><td>{v["default"]:.3f}</td><td>{v["fitted"]:.3f}</td><td>{v["default_rmse"]:.3f} → {v["fitted_rmse"]:.3f} m/s</td></tr>' for name,v in zip(['Normal restitution','Tangential retention'],rail))
loads=''.join(f'<tr><td>{v["motor"]}</td><td>{v["actual_amps"]["mean"]:.2f} A</td><td>{v["predicted_amps"]["mean"]:.2f} A</td></tr>' for v in m['stationary_load'])
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live play review · October 2</title>
<style>body{margin:0;background:#081321;color:#e2eaf4;font:17px/1.6 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:32px 20px 80px}h1{font-size:clamp(30px,5vw,48px);line-height:1.15}h2{font-size:25px;margin-top:38px}h3{font-size:19px}p{max-width:90ch}a{color:#80c6ff}small,.muted{color:#bac9da}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}.card,section{background:#101d30;border:1px solid #26394f;border-radius:14px;padding:20px;margin:18px 0}.card strong{display:block;font-size:30px;color:#84e0bb}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #304158;padding:10px;text-align:left}img,canvas{max-width:100%;height:auto}button,select,input{font:inherit}button,select{background:#213852;color:white;border:1px solid #6381a3;border-radius:7px;padding:8px}input[type=range]{width:min(480px,100%)}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.tag{color:#ffd166}code{font-size:14px;overflow-wrap:anywhere}li{margin:10px 0}.scroll{overflow:auto}</style>
<main><a href="/training">← Training dashboard</a><p class="tag">PHYSICAL SESSION REVIEW · 2 OCTOBER 2026</p>
<h1>The robot has headroom.<br>The policy rarely asks for it.</h1>
<p>Review of <code>20261002-114149</code> with <code>rail30-100-20261001-v1</code>: 548 seconds of active play, 27,372 decisions, 100 m/s² acceleration cap and 1.5 mm pretension. The idle recording tail is excluded from action statistics.</p>
<div class="cards"><div class="card"><strong>17.0 m/s²</strong>Mean commanded cap<br><small>Median 6.58 · 95th percentile 56.6</small></div><div class="card"><strong>0.205%</strong>Commands at ≥90 m/s²<br><small>All 56 came from the arrival guard</small></div><div class="card"><strong>68.2 cm</strong>Median depth ahead of own goal<br><small>While the opponent prepares slowly</small></div><div class="card"><strong>68%</strong>Highest fast RMS reading<br><small>No observed drive faults or cap violations</small></div></div>
<p><b>Main finding:</b> defensive preparation and effort selection reproduce in simulation. This does not look like a general motor-lag failure or a badly wrong bank reflection model. There is a smaller camera-position bug, corrected below.</p>
<h2>How much acceleration was used?</h2><img src="HIST" alt="Acceleration cap histogram">
<div class="scroll"><table><tr><th>Context</th><th>Physical commands: mean / p95</th><th>Fresh self-play actor: mean / p95</th></tr><tr><td>All active decisions</td><td>17.0 / 56.6 m/s²</td><td>15.8 / 56.5 m/s²</td></tr><tr><td>Incoming puck</td><td>46.2 / 66.3 m/s²</td><td>42.3 / 66.0 m/s²</td></tr><tr><td>Slow opponent preparation</td><td>5.4 / 6.5 m/s²</td><td>5.5 / 7.5 m/s²</td></tr></table></div>
<p>The network's own highest cap in the physical session was <b>86.3 m/s²</b>; it never selected ≥90. Commands reached exactly 100 only after the arrival safety guard raised the cap. Above 75 was just 0.51% of commands; above 50 was 8.8%.</p>
<p>These are <b>requested acceleration limits</b>, not accelerometer measurements. Arrival braking, target distance and jerk limiting can make actual acceleration lower. The fresh simulation comparison used the same weights in eight 60-second self-play games, seed 20261002, initially 40% thermal load. Only one of 48,000 actor decisions exceeded 90 m/s². Mean and percentile comparisons mix different puck trajectories; they establish a behavior pattern, not an identical-input experiment.</p>
<h2>Inspect the likely concessions</h2><section><p>Yellow: measured puck. Blue: recorded paddle centre. Cyan outline: controller position. Green: corrected marker pose, where raw near-contact blobs exist. Purple outlines: simulated puck and paddle after rollout start. Orange cross: commanded target. Robot goal is on the right. Trails show only the past.</p>
<div class="controls"><select id="goal" aria-label="Concession"></select><button id="play">Play</button><label>Speed <select id="speed"><option value="0.25">¼×</option><option value="0.5" selected>½×</option><option value="1">1×</option></select></label></div>
<canvas id="canvas" width="1000" height="520" aria-label="Recorded puck and paddle trajectories"></canvas><div class="controls"><input id="seek" type="range" min="0" step="1" value="0" aria-label="Replay time"><output id="time"></output></div><p id="detail"></p></section>
<div class="scroll"><table><tr><th>Watchdog time</th><th>Paddle ahead of goal at midline</th><th>Peak response cap (m/s²)</th><th>Pattern</th></tr>GOALROWS</table></div>
<p><small>These seven events have near-goal observations consistent with concessions; this is not an official score count. The goal watchdog occurs after the last camera observation. *No midline crossing in the last two seconds for 760.74 s: position is sampled 350 ms before the watchdog instead. Blue is the old biased centroid; green is an offline re-solve of available blobs, not continuous ground truth.</small></p>
<p>Six events show side-bank approaches; the 198.52 s event is a slower possession/contact failure. In the fast examples with a detected midline crossing, only 250–315 ms remained until the goal watchdog, and the paddle started roughly 61–72 cm ahead of its goal. Waiting deeper would provide more time after the bank to cover the goal mouth, at the cost of conceding some early interceptions.</p>
<p>That forward preference also exists in self-play: median preparation depth is <b>68.3 cm</b>. “Opponent preparation” here means puck depth &gt;1 m from the robot's goal and speed &lt;1.5 m/s. Incoming means longitudinal velocity toward the robot faster than 1 m/s, at depth &lt;1.2 m.</p>
<h3>More acceleration helps some trajectories, but not every one</h3>
<p>I replayed the recorded targets open-loop, changing only their cap to 100 m/s². Several missed approaches became possible contacts, but the 518.53 s case became <b>worse</b> (closest centre separation approximately 100 → 162 mm). Faster motion can arrive at the wrong place too soon. These are sensitivity tests, not predicted saves: a changed collision would alter all later observations and actions. The rollouts stop at the last fresh puck sample, avoiding artificial contact from extrapolating a missing puck at the goal.</p>
<h2>Simulation versus physical tracking</h2>
<p><b>All six fast bank cases also concede in the full environment replay.</b> These rollouts initialize once from camera positions and a causal velocity fit, replay the commands, and use measured human motion. Later robot/puck measurements only score the comparison. Mean puck errors are 4–27 mm across cases (largest sample error 62 mm). Camera-initialized paddle mean errors are 9–36 mm, larger than the controller-initialized motion tests below; the 40 ms causal velocity fit and biased initial camera centre are additional sources of replay error. This reproduces the failure pattern, rather than proving exact contact timing.</p>
<ul><li><b>Motion response:</b> 69 independent one-second recorded-command rollouts, initialized from controller state, gave median controller discrepancy 0.57 mm and p95 9.69 mm. Alternating held-out windows gave 0.52 / 4.88 mm. Additional 2–8 ms command delay worsened the fit. Controller agreement alone does not prove physical agreement: camera errors were larger, median 9.5 mm / p95 16.5 mm.</li>
<li><b>Camera centre bug fixed:</b> the runtime tracker averaged an asymmetric three-marker pattern on one height plane. The centre and adjacent arms actually use different planes (65/33 mm). Re-solving 21,629 near-contact frames moved the reported centre by a median 6.8 mm longitudinally. On that selected matched subset, camera/controller median error improved 5.9 → 5.2 mm; p95 13.9 → 10.6 mm. Remaining mismatch needs separate calibration/load checks.</li>
<li><b>Tracking/timing:</b> decisions ran at 49.96 Hz; p99 interval 20.1 ms, isolated maximum 65.1 ms. Puck age exceeded 50 ms on 1.89% of active decisions. All thermal channels were fresh at decisions. Controller pose supplied 96% of active decisions, so the camera correction cannot by itself cure the policy's forward positioning.</li>
<li><b>Replay configuration fixed:</b> new recordings store the actual checkpoint workspace and thermal model. The overlay uses the recorded table configuration and workspace; older neural recordings infer bounds from their checkpoint when available. Previous logs could misleadingly label the load model with the old default.</li></ul>
<h3>Bank physics already fits reasonably well</h3>
<p>302 clean side-rail bounces were fitted away from the robot, requiring continuous samples and &lt;2 mm line-fit residuals. Coefficients fitted on the first 400 seconds were checked on later bounces:</p>
<table><tr><th>Coefficient</th><th>Current simulator</th><th>Fitted</th><th>Held-out velocity RMSE: current → fitted</th></tr>RAILROWS</table>
<p>The changes offer little or no held-out improvement, so I kept the current rail constants. Across 467 near-post-bounce decisions, the actor's lateral velocity input had the wrong sign on nine. That is worth monitoring, but does not explain a persistent forward stance.</p>
<h2>RMS and reward penalties</h2><img src="THERMAL" alt="Drive RMS levels through the played portion of the session">
<p>There was thermal headroom in this session. This does not establish that continuous 100 m/s² play is sustainable. The stationary current model is conservative here, especially on motors 1 and 3:</p>
<table><tr><th>Motor</th><th>Measured mean |current|</th><th>Model mean |current|</th></tr>LOADROWS</table>
<p>These 1,963 stationary samples are concentrated near the home pose. Retraction, calibration and table position affect tension, so fitting the entire workspace to this one-session discrepancy would be premature.</p>
<p>Your penalty hypothesis is plausible, but not proven. The selected recipe uses energy weight 2 and load weight 32. At a nominal central pose, an illustrative 100 ms 100 m/s² burst costs about 1.75 reward units including modeled load/effort, compared with a 600-unit goal. Long-horizon costs and imperfect exploration still matter. Warm-starting from a 60 m/s² policy and later imitation of successful trajectories may also preserve its old effort preference.</p>
<h2>The next training comparison</h2>
<ol><li><b>Teach preparation for banks.</b> The existing recipe's delayed windup drills replaced their launches with direct shots; ordinary defense drills did include banks. I added <code>--defense-windup-bank-fraction</code>, with randomly hidden left/right routes, rail-loss-aware aiming, and re-aiming after lateral drift. Suggested setting: 0.5. This constructs training situations; the single neural actor still chooses every robot action.</li>
<li><b>Make readiness evaluate banks too.</b> The current readiness shaping covers direct threats using the nominal acceleration limit. It can overestimate coverage when the actor later selects roughly half that acceleration. A follow-up should evaluate direct and bank threats over plausible release locations, rewarding coverage instead of hard-coding a parking position.</li>
<li><b>Run an effort ablation.</b> Compare energy weight 2 versus 0.5, retaining overload protection, thermal-state inputs and the small edge penalty. Add coherent high-cap exploration during meaningful shot/block attempts. Check whether faster on-target shots and saves actually improve, instead of maximizing average acceleration.</li>
<li><b>Select on held-out defense and endurance.</b> Keep these real bank trajectories as regression fixtures; add unseen releases from both rails and direct shots. Require improvements in bank save rate, on-target shot speed and goals conceded without degrading possession or hot continuous self-play. Self-play score alone is insufficient.</li></ol>
<section><b>Status:</b> tracking, replay metadata and optional bank-windup training changes are implemented. No new weights have been trained or promoted by this review. The running physical session was not restarted or controlled; the tracking fix loads on the next policy launch. Hardware limits remain 100 m/s² / 12 m/s for this candidate.</section>
<p class="muted">Reproduction scripts: <code>ai/recipes/live-review-20261002</code>. Source checkpoint SHA-256: <code>39aa7a501edc127bc37fd2e500578366c2b0a735a45eb0b628acd07c5bddf629</code>. Report is standalone and has no hardware controls.</p></main>
<script>
const goals=GOALDATA;const select=document.getElementById('goal'),seek=document.getElementById('seek'),play=document.getElementById('play'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');let running=false,elapsed=0,last=0;
goals.forEach((g,i)=>select.add(new Option(`${i+1}: ${g.time.toFixed(2)} s${i===3?' · slow loss':' · bank approach'}`,i)));
const X=x=>12+x/2038*975,Y=y=>12+(y+19)/1004*493;
function circle(p,r,color,fill=true){if(!p)return;ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),r/2038*975,0,2*Math.PI);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;fill?ctx.fill():ctx.stroke()}
function draw(){const g=goals[+select.value],n=+seek.value,f=g.frames[n];ctx.clearRect(0,0,1000,520);ctx.fillStyle='#07101c';ctx.fillRect(0,0,1000,520);ctx.strokeStyle='#73849a';ctx.lineWidth=2;ctx.strokeRect(X(0),Y(-19),X(2017.9)-X(0),Y(984.9)-Y(-19));ctx.setLineDash([7,7]);ctx.beginPath();ctx.moveTo(X(1003.3),Y(-19));ctx.lineTo(X(1003.3),Y(984.9));ctx.stroke();ctx.strokeStyle='#41556d';ctx.strokeRect(X(1200),Y(61.4),X(1937.5)-X(1200),Y(904.5)-Y(61.4));ctx.setLineDash([]);ctx.strokeStyle='#ff9696';ctx.lineWidth=7;ctx.beginPath();ctx.moveTo(X(2017.9),Y(292.95));ctx.lineTo(X(2017.9),Y(672.95));ctx.stroke();
for(const [key,color] of [['p','#ffd166'],['c','#65b8ff'],['sp','#d2a2ff']]){ctx.beginPath();let prev=null;for(let j=0;j<=n;j++){const p=g.frames[j][key];if(p){if(prev)ctx.lineTo(X(p[0]),Y(p[1]));else ctx.moveTo(X(p[0]),Y(p[1]));}prev=p;}ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
circle(f.sp,40.7,'#d2a2ff',false);circle(f.sr,50.4,'#d2a2ff',false);circle(f.p,40.7,'#ffd166');circle(f.c,50.4,'#65b8ff');circle(f.k,50.4,'#7ae7ea',false);circle(f.corrected,50.4,'#84e0bb',false);if(f.target){ctx.strokeStyle='#ffad66';ctx.beginPath();ctx.moveTo(X(f.target[0])-7,Y(f.target[1]));ctx.lineTo(X(f.target[0])+7,Y(f.target[1]));ctx.moveTo(X(f.target[0]),Y(f.target[1])-7);ctx.lineTo(X(f.target[0]),Y(f.target[1])+7);ctx.stroke();}document.getElementById('time').textContent=`${f.t.toFixed(3)} s`;document.getElementById('detail').textContent=`Actor cap ${f.actor} m/s² → command ${f.accel} m/s². ${!f.p?'No fresh puck sample at this time.':''}`;}
function reset(){running=false;play.textContent='Play';seek.max=goals[+select.value].frames.length-1;seek.value=0;elapsed=0;draw()}
select.onchange=reset;seek.oninput=()=>{elapsed=+seek.value*.01;draw()};play.onclick=()=>{if(+seek.value>=+seek.max){seek.value=0;elapsed=0;}running=!running;play.textContent=running?'Pause':'Play';last=performance.now();};function tick(now){if(running){elapsed+=(now-last)/1000*+document.getElementById('speed').value;seek.value=Math.min(+seek.max,Math.floor(elapsed/.01));draw();if(+seek.value>=+seek.max){running=false;play.textContent='Play';}}last=now;requestAnimationFrame(tick)}reset();requestAnimationFrame(tick);
</script></html>'''
for key,value in dict(HIST=hist,THERMAL=thermal,GOALROWS=rows,RAILROWS=railrows,LOADROWS=loads,GOALDATA=json.dumps(goals,separators=(',',':'),allow_nan=False)).items():html=html.replace(key,value)
path=Path('ai/airhockey/web/training-report-20261002.html');path.write_text(html)
print(path, path.stat().st_size)
print('rail holdout:',rail)
