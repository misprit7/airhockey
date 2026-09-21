const $ = (id) => document.getElementById(id);
const colors = { real: "#56cdd8", sim: "#ffbb65", human: "#bca4f7" };
let sessions = [],
  data = null,
  segments = [],
  selected = null,
  custom = false,
  time = 0,
  playing = false,
  generation = 0;
let lastFrame = performance.now(),
  pending = null;
const clock = (t) =>
  `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(3).padStart(6, "0")}`;
const label = (t) => clock(t).slice(0, -4);
const mm = (n) => (Number.isFinite(n) ? `${Math.round(n)} mm` : "—");
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
async function api(url, options) {
  const r = await fetch(url, options);
  if (!r.ok) {
    const e = await r.json();
    throw Error(e.detail || r.statusText);
  }
  return r.json();
}
function lookup(a, t) {
  let l = 0,
    h = a.length;
  while (l < h) {
    let m = (l + h) >> 1;
    if (a[m][0] <= t + 1e-8) l = m + 1;
    else h = m;
  }
  return l - 1;
}
function point(track, t, gap = 0.15) {
  if (!track?.length) return null;
  let i = lookup(track, t);
  if (i < 0) return null;
  let a = track[i],
    b = track[i + 1];
  if (Math.abs(t - a[0]) < 1e-6) return a.slice(1);
  if (!b) return t - a[0] <= gap ? a.slice(1) : null;
  if (b[0] - a[0] > gap) return null;
  let u = (t - a[0]) / (b[0] - a[0]);
  return a.slice(1).map((v, j) => v + (b[j + 1] - v) * u);
}
function simPoint(t) {
  if (!selected?.frames.length || t > selected.frames.at(-1)[0] + 1e-6)
    return null;
  return point(selected.frames, t, 0.022);
}
function selectTime(t) {
  if (!data) return;
  time = Math.max(0, Math.min(data.duration, t));
  $("scrub").value = time;
  if (!custom)
    selected = segments[Math.min(segments.length - 1, Math.floor(time / 10))];
  render();
}
function setPlaying(value) {
  playing = value;
  $("play").textContent = value ? "Pause" : "Play";
  lastFrame = performance.now();
}
function renderSessions() {
  const q = $("search").value.toLowerCase();
  $("sessions").replaceChildren();
  for (const s of sessions.filter((s) => s.name.toLowerCase().includes(q))) {
    let b = document.createElement("button");
    b.className = "session" + (data?.name === s.name ? " active" : "");
    let title = document.createElement("strong");
    title.textContent = s.name.replace(/\.(ticks\.csv|replay\.jsonl)$/, "");
    let sub = document.createElement("span");
    sub.textContent = `${s.format} · ${(s.size / 1e6).toFixed(1)} MB`;
    b.append(title, sub);
    b.onclick = () => load(s.name);
    $("sessions").append(b);
  }
  if (!sessions.length)
    $("sessions").textContent = "No hardware recordings yet.";
}
async function refresh() {
  try {
    sessions = await api("/api/replays");
    renderSessions();
    if (!data && sessions.length) await load(sessions[0].name);
  } catch (e) {
    status(e.message, true);
  }
}
async function load(name) {
  const gen = ++generation;
  pending?.abort();
  setPlaying(false);
  status("Loading recording…");
  try {
    const d = await api(`/api/replays/${encodeURIComponent(name)}`);
    if (gen !== generation) return;
    data = d;
    custom = false;
    time = 0;
    selected = null;
    segments = [];
    $("session-title").textContent = name.replace(
      /\.(ticks\.csv|replay\.jsonl)$/,
      "",
    );
    $("session-info").textContent =
      `${label(d.duration)} recorded · ${d.meta.policy} · ${d.tracks.puck.length.toLocaleString()} puck measurements`;
    $("warnings").textContent = d.warnings.join(" ");
    $("warnings").hidden = !d.warnings.length;
    $("scrub").max = d.duration;
    for (const id of ["scrub", "play", "simulate", "grid-reset"])
      $(id).disabled = false;
    renderSessions();
    await buildGrid(gen);
  } catch (e) {
    if (gen === generation && e.name !== "AbortError") status(e.message, true);
  }
}
async function requestSim(starts, signal) {
  return api(`/api/replays/${encodeURIComponent(data.name)}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      starts,
      duration: 10,
      command_offset_ms: Number($("timing").value) || 0,
    }),
  });
}
async function buildGrid(gen) {
  pending?.abort();
  pending = new AbortController();
  const signal = pending.signal;
  segments = [];
  custom = false;
  $("segments").replaceChildren();
  for (let s = 0; s < data.duration; s += 10) {
    let r = { start: s, frames: [], reason: "Calculating…", errors: {} };
    segments.push(r);
    const b = document.createElement("button");
    b.className = "segment";
    b.dataset.start = s;
    const h = document.createElement("strong");
    h.textContent = `${label(s)} – ${label(Math.min(s + 10, data.duration))}`;
    const c = document.createElement("canvas");
    c.width = 260;
    c.height = 360;
    const p = document.createElement("p");
    const reason = document.createElement("p");
    reason.className = "reason";
    b.append(h, c, p, reason);
    b.onclick = () => {
      custom = false;
      selected = segments[Math.floor(s / 10)];
      setPlaying(false);
      selectTime(s);
      $("table").scrollIntoView({ block: "nearest", behavior: "smooth" });
    };
    $("segments").append(b);
  }
  selected = segments[0] || null;
  render();
  updateTiles();
  for (let i = 0; i < segments.length; i += 32) {
    status(
      `Simulating windows ${i + 1}–${Math.min(i + 32, segments.length)} of ${segments.length}…`,
    );
    try {
      const result = await requestSim(
        segments.slice(i, i + 32).map((r) => r.start),
        signal,
      );
      if (gen !== generation) return;
      result.forEach((r, j) => (segments[i + j] = r));
      if (!custom)
        selected =
          segments[Math.min(segments.length - 1, Math.floor(time / 10))];
      updateTiles();
      render();
    } catch (e) {
      if (e.name !== "AbortError") status(e.message, true);
      return;
    }
  }
  status(`${segments.length} independent windows ready`);
}
function updateTiles() {
  document.querySelectorAll(".segment").forEach((tile, i) => {
    const r = segments[i];
    tile.querySelector("p").textContent =
      `Mean puck ${mm(r.errors?.puck?.mean_mm)} · robot ${mm(r.errors?.agent?.mean_mm)}`;
    tile.querySelector(".reason").textContent =
      r.reason === "window complete" ? "" : r.reason;
    drawTable(tile.querySelector("canvas"), r.start, r, true);
  });
}
function drawTable(canvas, t, rollout, thumbnail = false) {
  const ctx = canvas.getContext("2d"),
    w = canvas.width,
    h = canvas.height,
    cfg = data?.table || {
      width: 1,
      height: 2,
      goal_width: 0.38,
      puck_radius: 0.0407,
      paddle_radius: 0.0504,
    };
  ctx.clearRect(0, 0, w, h);
  const pad = 22,
    scale = Math.min((w - 2 * pad) / cfg.width, (h - 2 * pad) / cfg.height),
    ox = (w - cfg.width * scale) / 2,
    oy = (h - cfg.height * scale) / 2;
  const xy = (x, y) => [ox + x * scale, h - oy - y * scale];
  ctx.fillStyle = "#0c151d";
  ctx.fillRect(ox, oy, cfg.width * scale, cfg.height * scale);
  ctx.strokeStyle = "#344a58";
  ctx.lineWidth = 1;
  ctx.strokeRect(ox, oy, cfg.width * scale, cfg.height * scale);
  ctx.strokeStyle = "#1b2d38";
  ctx.beginPath();
  for (let x = 0.1; x < cfg.width; x += 0.1) {
    ctx.moveTo(...xy(x, 0));
    ctx.lineTo(...xy(x, cfg.height));
  }
  for (let y = 0.1; y < cfg.height; y += 0.1) {
    ctx.moveTo(...xy(0, y));
    ctx.lineTo(...xy(cfg.width, y));
  }
  ctx.stroke();
  ctx.strokeStyle = "#914553";
  ctx.beginPath();
  ctx.moveTo(...xy(0, cfg.height / 2));
  ctx.lineTo(...xy(cfg.width, cfg.height / 2));
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(...xy(cfg.width / 2, cfg.height / 2), 0.14 * scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "#7e939f";
  ctx.lineWidth = 4;
  for (const y of [0, cfg.height]) {
    ctx.beginPath();
    ctx.moveTo(...xy((cfg.width - cfg.goal_width) / 2, y));
    ctx.lineTo(...xy((cfg.width + cfg.goal_width) / 2, y));
    ctx.stroke();
  }
  ctx.fillStyle = "#8197a5";
  ctx.font = `${thumbnail ? 9 : 11}px system-ui`;
  ctx.textAlign = "center";
  ctx.fillText("HUMAN", w / 2, oy - 9);
  ctx.fillText("ROBOT", w / 2, h - oy + 15);
  if (!data) return;
  function path(track, color, from, to, indices = [1, 2], dash = false) {
    ctx.strokeStyle = color;
    ctx.globalAlpha = thumbnail ? 0.65 : 0.5;
    ctx.lineWidth = thumbnail ? 1 : 2;
    ctx.setLineDash(dash ? [4, 3] : []);
    ctx.beginPath();
    let last = null;
    for (const f of track) {
      if (f[0] < from || f[0] > to) continue;
      const p = xy(f[indices[0]], f[indices[1]]);
      if (last === null || f[0] - last > 0.15) ctx.moveTo(...p);
      else ctx.lineTo(...p);
      last = f[0];
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }
  const start = thumbnail
      ? rollout.start
      : Math.max(rollout?.start || 0, t - 1),
    end = thumbnail ? Math.min(rollout.start + 10, data.duration) : t;
  if (thumbnail || $("show-trails").checked) {
    if (thumbnail || $("show-real").checked) {
      path(data.tracks.puck, colors.real, start, end);
      path(data.tracks.agent, "#497d85", start, end);
    }
    if (rollout && (thumbnail || $("show-sim").checked)) {
      path(rollout.frames, colors.sim, start, end);
      path(rollout.frames, "#987347", start, end, [3, 4], true);
    }
  }
  function disk(p, r, color, outline = false, robot = false) {
    if (!p) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(outline ? [5, 3] : []);
    ctx.beginPath();
    ctx.arc(...xy(...p), r * scale, 0, Math.PI * 2);
    if (!outline) {
      ctx.globalAlpha = 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (robot) {
      ctx.beginPath();
      const [x, y] = xy(...p);
      ctx.moveTo(x - 4, y);
      ctx.lineTo(x + 4, y);
      ctx.moveTo(x, y - 4);
      ctx.lineTo(x, y + 4);
      ctx.stroke();
    }
  }
  if (!thumbnail) {
    if ($("show-real").checked) {
      disk(point(data.tracks.puck, t, 0.05), cfg.puck_radius, colors.real);
      disk(
        point(data.tracks.agent, t, 0.05),
        cfg.paddle_radius,
        colors.real,
        false,
        true,
      );
    }
    disk(
      point(data.tracks.human, t),
      cfg.paddle_radius,
      colors.human,
      false,
      true,
    );
    if ($("show-sim").checked) {
      const f = simPoint(t);
      if (f) {
        disk(f.slice(0, 2), cfg.puck_radius, colors.sim, true);
        disk(f.slice(2, 4), cfg.paddle_radius, colors.sim, true, true);
      }
    }
  }
}
function render() {
  drawTable($("table"), time, selected);
  $("clock").textContent = clock(time);
  $("scrub").value = time;
  $("rollout-label").textContent = selected
    ? `Simulation starts at ${clock(selected.start)}${custom ? " · selected time" : " · 10 s grid"}`
    : "No simulation selected";
  $("stop-reason").textContent =
    selected?.reason === "window complete"
      ? ""
      : selected?.frames.length
        ? `Ends at ${clock(selected.frames.at(-1)[0])}: ${selected.reason}`
        : selected?.reason || "";
  const s = simPoint(time);
  for (const [kind, offset] of [
    ["puck", 0],
    ["agent", 2],
  ]) {
    const p = data ? point(data.tracks[kind], time, 0.05) : null;
    $(kind + "-error").textContent =
      p && s
        ? mm(Math.hypot(p[0] - s[offset], p[1] - s[offset + 1]) * 1000)
        : "—";
  }
  document
    .querySelectorAll(".segment")
    .forEach((b, i) =>
      b.classList.toggle("active", !custom && segments[i] === selected),
    );
}
$("simulate").onclick = async () => {
  if (!data) return;
  const gen = ++generation;
  pending?.abort();
  pending = new AbortController();
  setPlaying(false);
  status("Simulating from selected state…");
  const start = time;
  try {
    const [r] = await requestSim([start], pending.signal);
    if (gen !== generation) return;
    selected = r;
    custom = true;
    render();
    status(
      r.frames.length ? "Selected rollout ready" : r.reason,
      !r.frames.length,
    );
  } catch (e) {
    if (e.name !== "AbortError") status(e.message, true);
  }
};
$("grid-reset").onclick = () => {
  custom = false;
  selected = segments[Math.min(segments.length - 1, Math.floor(time / 10))];
  render();
  if (segments.some((r) => r.reason === "Calculating…"))
    buildGrid(++generation);
};
$("apply-timing").onclick = () => {
  if (data) buildGrid(++generation);
};
$("scrub").oninput = (e) => {
  setPlaying(false);
  selectTime(Number(e.target.value));
};
$("play").onclick = () => setPlaying(!playing);
$("back").onclick = () => {
  setPlaying(false);
  selectTime(time - 0.02);
};
$("forward").onclick = () => {
  setPlaying(false);
  selectTime(time + 0.02);
};
$("table").addEventListener(
  "wheel",
  (e) => {
    if (!data) return;
    e.preventDefault();
    setPlaying(false);
    selectTime(time + Math.sign(e.deltaY) * (e.shiftKey ? 0.02 : 0.2));
  },
  { passive: false },
);
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "BUTTON"].includes(e.target.tagName) || !data) return;
  if (e.code === "Space") {
    e.preventDefault();
    setPlaying(!playing);
  }
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    setPlaying(false);
    selectTime(
      time + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 1 : 0.02),
    );
  }
});
for (const id of ["show-real", "show-sim", "show-trails"])
  $(id).onchange = render;
$("search").oninput = renderSessions;
$("refresh").onclick = refresh;
function animate(now) {
  if (playing && data) {
    selectTime(time + ((now - lastFrame) / 1000) * Number($("speed").value));
    if (time >= data.duration) setPlaying(false);
  }
  lastFrame = now;
  requestAnimationFrame(animate);
}
render();
refresh();
requestAnimationFrame(animate);
