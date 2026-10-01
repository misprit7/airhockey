"use strict";
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const number = value => value == null ? "—" : Intl.NumberFormat(undefined, {maximumFractionDigits: 1, notation: "compact"}).format(value);
const duration = value => value == null ? "—" : value < 60 ? `${Math.round(value)}s` : `${Math.floor(value / 3600)}h ${Math.floor(value % 3600 / 60)}m`;
const time = value => new Date(value * 1000).toLocaleString();
const replayLink = (url, label) => url ? `<a href="${esc(url)}">${esc(label)}</a>` : '<span class="muted">Replay pending</span>';
const expandedRuns = new Set();
let firstPaint = true;
let fetching = false;

function checkpointRow(c) {
    const progress = c.additional_steps == null ? c.name : `+${number(c.additional_steps)} steps`;
    const edge = c.fringe ? `Corner ${c.fringe.corner.recovered}/${c.fringe.corner.trials}; side ${c.fringe.side.recovered}/${c.fringe.side.trials}` : c.edge_trials ? `${c.edges.restored_interior ?? "—"}/${c.edge_trials}` : "—";
    return `<tr><td title="${esc(c.name)}">${esc(progress)}<div class="sub">${esc(c.step.toLocaleString())} ${c.name.startsWith("agent_update_") ? "fitting updates" : "cumulative"}</div></td>
      <td>${esc(c.evaluation)}<div class="sub">${esc(c.screening)}</div></td>
      <td>${esc(edge)}</td><td>${esc(c.edges.controlled_after_restore ?? "—")}</td><td>${esc(c.edges.requested_fast_after_restore ?? "—")}</td>
      <td>${esc(c.shots.stationary ?? "—")} / ${esc(c.shots.receiving ?? "—")}</td>
      <td>${c.recovery ? `${esc(c.recovery.controlled)} / ${esc(c.recovery.controlled_fast)}` : "—"}</td>
      <td>${c.defense ? `${esc(c.defense.saved)}/${esc(c.defense.trials)}<div class="sub">${esc(c.defense.forward_at_release)} at front</div>` : "—"}</td>
      <td>${c.peak_load == null ? "—" : `${(c.peak_load * 100).toFixed(1)}% short`}${c.endurance_peak == null ? "" : `<div class="sub">${(c.endurance_peak * 100).toFixed(1)}% sustained · ${number(c.endurance_overload_seconds)}s overloaded</div>`}</td>
      <td>${replayLink(c.replay, "Watch self-play")}${c.diagnostics.length ? `<details data-section="${esc(c.replay || c.name)}"><summary>${c.diagnostics.length} edge tests</summary><div class="diagnostics">${c.diagnostics.map(d => replayLink(d.url, d.label)).join("")}</div></details>` : ""}</td></tr>`;
}

function runCard(run, index) {
    const open = expandedRuns.has(run.name) || (firstPaint && index === 0);
    if (open) expandedRuns.add(run.name);
    const steps = run.completed_steps == null ? `${number(run.updates)} fitting updates` : `${number(run.completed_steps)} / ${number(run.target_steps)} additional steps`;
    return `<article class="card">
      <div class="run-title"><h2>${esc(run.name)}</h2><span class="badge ${run.training_active ? "active" : ""}">${esc(run.state)}</span></div>
      <p class="outcome">${esc(run.outcome)}</p>
      ${run.progress == null ? "" : `<progress max="1" value="${run.progress}" aria-label="Training progress"></progress>`}
      <div class="facts"><span>${esc(steps)}</span><span>Training time: ${duration(run.elapsed_s)}</span>
      ${run.limits ? `<span>Training limits: ${esc(run.limits.speed_m_s)} m/s · ${esc(run.limits.acceleration_m_s2)} m/s²</span>` : ""}
      ${run.workspace === "rail30" ? `<span>Rail clearance: 30 mm${run.edge_dwell_band ? ` · edge penalty band: ${Math.round(run.edge_dwell_band * 100)} cm` : ""}</span>` : ""}
      ${run.eta_s == null ? "" : `<span>Estimated remaining: ${duration(run.eta_s)}</span>`}
      <span>Evaluations: ${run.evaluated}/${run.checkpoint_count} complete${run.evaluator_active ? " · running" : ""}</span>
      ${run.save_every ? `<span>Checkpoint every ${number(run.save_every)} steps</span>` : ""}</div>
      <p class="sub">Last training update: ${esc(time(run.updated_at))}${run.training_active ? ` · ${duration(run.heartbeat_age_s)} ago` : ""}</p>
      ${run.error ? `<p class="error">${esc(run.error)}</p>` : ""}
      <div class="actions">${replayLink(run.latest_replay, "Watch latest evaluated checkpoint")}</div>
      <details class="checkpoints" data-run="${esc(run.name)}" ${open ? "open" : ""}><summary>All ${run.checkpoint_count} checkpoints & evaluations</summary>
      <p class="sub">Shots: stationary / incoming requested fast-shot counts. Slow drift: controlled / controlled then a requested fast shot. Moving release: saves against a laterally drifting puck before a hidden shot; “at front” means paddle depth above 0.60m. Short load uses 90-second self-play. When shown, sustained load uses 10-minute games starting at 80% modeled load; overload time sums both players across games. These are simulation tests.</p>
      <div class="table-wrap"><table><thead><tr><th>Checkpoint</th><th>Evaluation / screen</th><th>Edges recovered</th><th>Then controlled</th><th>Then fast shot</th><th>Shots</th><th>Slow drift control / shot</th><th>Moving-release saves</th><th>Peak load</th><th>Replays</th></tr></thead>
      <tbody>${run.checkpoints.map(checkpointRow).join("") || '<tr><td colspan="10">Waiting for the first saved checkpoint.</td></tr>'}</tbody></table></div></details>
    </article>`;
}

async function refresh() {
    if (fetching) return;
    fetching = true;
    try {
        const response = await fetch("/api/training", {cache: "no-store"});
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const data = await response.json();
        const expandedSections = new Set([...document.querySelectorAll("details[data-section][open]")].map(el => el.dataset.section));
        document.getElementById("activity").textContent = `${data.active_training ? `${data.active_training} training run(s) active` : "No training currently running"} · ${data.active_evaluations ? `${data.active_evaluations} evaluation worker(s) active` : "No evaluations currently running"}`;
        document.getElementById("refresh").textContent = `Updated ${new Date().toLocaleTimeString()} · refreshes every 5s`;
        const packageContent = p => `<p class="package-name">${esc(p.name)}${p.simulation_candidate ? ' <span class="badge">Simulation candidate</span>' : ""}</p><p class="muted">${esc(p.note)}</p>${p.replay ? `<div class="actions">${replayLink(p.replay, "Watch selected policy")}</div>` : ""}`;
        document.getElementById("packages").innerHTML = data.packages.length ? `<article class="card"><h2>Latest packaged policy</h2>${packageContent(data.packages[0])}</article>${data.packages.length > 1 ? `<details class="card" data-section="packages"><summary>Earlier packaged policies (${data.packages.length - 1})</summary>${data.packages.slice(1).map(packageContent).join("")}</details>` : ""}` : "";
        document.getElementById("runs").innerHTML = data.runs.map(runCard).join("") || "No training runs found.";
        document.querySelectorAll("details.checkpoints").forEach(el => el.addEventListener("toggle", () => {
            if (el.open) expandedRuns.add(el.dataset.run); else expandedRuns.delete(el.dataset.run);
        }));
        document.querySelectorAll("details[data-section]").forEach(el => { el.open = expandedSections.has(el.dataset.section); });
        firstPaint = false;
    } catch (error) {
        document.getElementById("refresh").textContent = `Status unavailable; retrying. ${error.message}`;
        document.getElementById("activity").textContent = "Live status unavailable. Any results below are from the last successful refresh.";
    } finally { fetching = false; }
}
refresh();
setInterval(refresh, 5000);
