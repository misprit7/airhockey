const canvas = document.getElementById("table");
const ctx = canvas.getContext("2d");

// State
let config = null;
let frame = null;
let ws = null;
let hardwareRequest = null;
// Whether THIS tab holds the robot for manual control, and callbacks waiting
// for the server's next answer about it (robot.js turns it off before the AI
// starts: the master serves one client).
let hardwareOn = false;
const hardwareModeListeners = [];
// "control" = driving the machine by hand, no world simulated at all;
// "sim" = the full game; "replay" = watching a recording.
const requestedReplay = new URLSearchParams(window.location.search).get("replay");
let mode = requestedReplay ? "replay" : "control";
let replayData = null;
let replayIndex = 0;
let replayPlaying = false;
let replaySpeed = 1;
let replayFps = 60;
let replayPosition = 0;     // fractional frame index while playing
let replayLastTime = null;  // rAF timestamp of the last advance
let scale = 1;
let offsetX = 0;
let offsetY = 0;

// Colors
// Rink livery, matching style.css. The table used to be billiard green with
// a centre line at 8% white — nearly invisible, which is a strange thing to
// do to the one marking the game is organised around. Here it is ice under
// floodlight with the centre line in the table's red, which is also what the
// physical table is painted like.
//
// Keep these in step with the tokens in style.css; they are the same palette
// either side of the canvas boundary.
const COLORS = {
    table: "#16202B",                         // --ice, lifted to read as a surface
    tableBorder: "#3A5064",
    centerLine: "rgba(232, 33, 63, 0.62)",    // --red: the defining marking
    centerCircle: "rgba(232, 33, 63, 0.34)",
    wall: "#46617A",
    goal: "#E8213F",
    goalGlow: "rgba(232, 33, 63, 0.28)",
    puck: "#EAF0F6",
    puckGlow: "rgba(234, 240, 246, 0.38)",
    puckTrail: "rgba(234, 240, 246, 0.10)",
    agent: "#4FA3F7",
    agentGlow: "rgba(79, 163, 247, 0.35)",
    agentRing: "rgba(79, 163, 247, 0.16)",
    opponent: "#E8213F",
    opponentGlow: "rgba(232, 33, 63, 0.35)",
    opponentRing: "rgba(232, 33, 63, 0.16)",
};

// Hardware overlay state
let hwPosition = null; // {x, y} in physics coords, or null if not active
// Reachable box in the SAME coords, straight from the server. The mallet
// stops at this edge, not at the table edge, and without drawing it a
// mallet parked on a software limit is indistinguishable from one against
// the wall.
let hwWorkspace = null;
let replayWorkspace = null;
let showHwOverlay = true;

// Trail buffer
const TRAIL_LENGTH = 12;
let puckTrail = [];

function resize() {
    const container = document.getElementById("canvas-container");
    const cw = container.clientWidth - 48;
    const ch = container.clientHeight - 48;

    if (!config) {
        canvas.width = cw;
        canvas.height = ch;
        return;
    }

    const tableAspect = config.width / config.height;
    const containerAspect = cw / ch;

    let w, h;
    if (containerAspect > tableAspect) {
        h = ch;
        w = h * tableAspect;
    } else {
        w = cw;
        h = w / tableAspect;
    }

    canvas.width = w;
    canvas.height = h;
    scale = w / config.width;
    offsetX = 0;
    offsetY = 0;
}

function tx(x) { return x * scale + offsetX; }
// Flip Y so agent (bottom) is at bottom of screen
function ty(y) { return canvas.height - (y * scale + offsetY); }
function ts(s) { return s * scale; }

function drawTable() {
    if (!config) return;

    // Table surface
    ctx.fillStyle = COLORS.table;
    ctx.beginPath();
    ctx.roundRect(0, 0, canvas.width, canvas.height, ts(0.02));
    ctx.fill();

    // Subtle border
    ctx.strokeStyle = COLORS.tableBorder;
    ctx.lineWidth = ts(0.008);
    ctx.beginPath();
    ctx.roundRect(ts(0.004), ts(0.004), canvas.width - ts(0.008), canvas.height - ts(0.008), ts(0.018));
    ctx.stroke();

    // Center line
    ctx.strokeStyle = COLORS.centerLine;
    ctx.lineWidth = ts(0.003);
    ctx.setLineDash([ts(0.02), ts(0.015)]);
    ctx.beginPath();
    ctx.moveTo(tx(0), ty(config.height / 2));
    ctx.lineTo(tx(config.width), ty(config.height / 2));
    ctx.stroke();
    ctx.setLineDash([]);

    // Center circle
    ctx.strokeStyle = COLORS.centerCircle;
    ctx.lineWidth = ts(0.003);
    ctx.beginPath();
    ctx.arc(tx(config.width / 2), ty(config.height / 2), ts(0.12), 0, Math.PI * 2);
    ctx.stroke();

    // Center dot
    ctx.fillStyle = COLORS.centerCircle;
    ctx.beginPath();
    ctx.arc(tx(config.width / 2), ty(config.height / 2), ts(0.008), 0, Math.PI * 2);
    ctx.fill();

    if (mode === "control") return;   // nothing to score into

    // Goals
    const goalLeft = (config.width - config.goal_width) / 2;
    const goalRight = (config.width + config.goal_width) / 2;
    const goalDepth = ts(0.025);

    // Agent goal (bottom)
    const agentGoalY = ty(0);
    ctx.fillStyle = COLORS.goalGlow;
    ctx.fillRect(tx(goalLeft), agentGoalY - goalDepth / 2, ts(config.goal_width), goalDepth);
    ctx.fillStyle = COLORS.goal;
    ctx.fillRect(tx(goalLeft), agentGoalY - ts(0.005), ts(config.goal_width), ts(0.01));

    // Opponent goal (top)
    const oppGoalY = ty(config.height);
    ctx.fillStyle = COLORS.goalGlow;
    ctx.fillRect(tx(goalLeft), oppGoalY - goalDepth / 2, ts(config.goal_width), goalDepth);
    ctx.fillStyle = COLORS.goal;
    ctx.fillRect(tx(goalLeft), oppGoalY - ts(0.005), ts(config.goal_width), ts(0.01));
}

function drawPuck(x, y) {
    const px = tx(x);
    const py = ty(y);
    const r = ts(config.puck_radius);

    // Update trail
    puckTrail.push({ x: px, y: py });
    if (puckTrail.length > TRAIL_LENGTH) puckTrail.shift();

    // Draw trail
    if (puckTrail.length > 1) {
        for (let i = 1; i < puckTrail.length; i++) {
            const alpha = (i / puckTrail.length) * 0.15;
            const trailR = r * (0.3 + 0.7 * (i / puckTrail.length));
            ctx.fillStyle = `rgba(232, 232, 240, ${alpha})`;
            ctx.beginPath();
            ctx.arc(puckTrail[i].x, puckTrail[i].y, trailR, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // Glow
    const glow = ctx.createRadialGradient(px, py, r * 0.5, px, py, r * 3);
    glow.addColorStop(0, COLORS.puckGlow);
    glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(px, py, r * 3, 0, Math.PI * 2);
    ctx.fill();

    // Puck body
    const bodyGrad = ctx.createRadialGradient(px - r * 0.3, py - r * 0.3, 0, px, py, r);
    bodyGrad.addColorStop(0, "#ffffff");
    bodyGrad.addColorStop(0.7, COLORS.puck);
    bodyGrad.addColorStop(1, "#c0c0c8");
    ctx.fillStyle = bodyGrad;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();

    // Subtle ring
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = ts(0.002);
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.stroke();
}

function drawPaddle(x, y, color, glowColor, ringColor) {
    const px = tx(x);
    const py = ty(y);
    const r = ts(config.paddle_radius);

    // Outer glow
    const glow = ctx.createRadialGradient(px, py, r * 0.5, px, py, r * 3.5);
    glow.addColorStop(0, glowColor);
    glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(px, py, r * 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Ring
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = ts(0.006);
    ctx.beginPath();
    ctx.arc(px, py, r * 1.6, 0, Math.PI * 2);
    ctx.stroke();

    // Paddle body
    const bodyGrad = ctx.createRadialGradient(px - r * 0.25, py - r * 0.25, 0, px, py, r);
    bodyGrad.addColorStop(0, lighten(color, 40));
    bodyGrad.addColorStop(0.8, color);
    bodyGrad.addColorStop(1, darken(color, 30));
    ctx.fillStyle = bodyGrad;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();

    // Inner circle detail
    ctx.strokeStyle = `rgba(255,255,255,0.15)`;
    ctx.lineWidth = ts(0.002);
    ctx.beginPath();
    ctx.arc(px, py, r * 0.55, 0, Math.PI * 2);
    ctx.stroke();

    // Center dot
    ctx.fillStyle = `rgba(255,255,255,0.2)`;
    ctx.beginPath();
    ctx.arc(px, py, r * 0.12, 0, Math.PI * 2);
    ctx.fill();
}

// The robot's reachable box. Dashed because it is a software limit rather
// than a physical edge -- the mallet stopping here means "cannot go further",
// not "hit something".
function drawReachable(ws) {
    ctx.save();
    ctx.strokeStyle = COLORS.hwRing || "rgba(232,33,63,0.45)";
    ctx.lineWidth = Math.max(1, ts(0.003));
    ctx.setLineDash([ts(0.018), ts(0.014)]);
    ctx.strokeRect(tx(ws.min_x), ty(ws.max_y),
                   ts(ws.max_x - ws.min_x), ts(ws.max_y - ws.min_y));
    ctx.restore();
}

function drawHwPaddle(x, y) {
    const px = tx(x);
    const py = ty(y);
    const r = ts(config.paddle_radius);

    ctx.save();
    ctx.globalAlpha = 0.4;

    // Outer glow
    const glow = ctx.createRadialGradient(px, py, r * 0.5, px, py, r * 2.5);
    glow.addColorStop(0, "rgba(50, 220, 120, 0.35)");
    glow.addColorStop(1, "transparent");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(px, py, r * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Ring
    ctx.strokeStyle = "rgba(50, 220, 120, 0.25)";
    ctx.lineWidth = ts(0.006);
    ctx.beginPath();
    ctx.arc(px, py, r * 1.6, 0, Math.PI * 2);
    ctx.stroke();

    // Body
    ctx.fillStyle = "#32dc78";
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();

    // Border
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = ts(0.003);
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
}

function lighten(hex, amt) {
    const num = parseInt(hex.replace("#", ""), 16);
    const r = Math.min(255, (num >> 16) + amt);
    const g = Math.min(255, ((num >> 8) & 0xff) + amt);
    const b = Math.min(255, (num & 0xff) + amt);
    return `rgb(${r},${g},${b})`;
}

function darken(hex, amt) {
    const num = parseInt(hex.replace("#", ""), 16);
    const r = Math.max(0, (num >> 16) - amt);
    const g = Math.max(0, ((num >> 8) & 0xff) - amt);
    const b = Math.max(0, (num & 0xff) - amt);
    return `rgb(${r},${g},${b})`;
}

function render(now) {
    if (mode === "replay") advanceReplay(now ?? performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawTable();

    if (frame) {
        // Control mode simulates no world, so a SIMULATED puck must not be
        // drawn next to a real machine — it invites reading it as real. A
        // CAMERA puck is the opposite case: it is the only thing on the
        // canvas that is measured, so it is drawn whenever the camera is on.
        // The cam_ prefix is what keeps the two apart; they never coexist.
        if (mode !== "control" && frame.puck_x !== undefined) {
            drawPuck(frame.puck_x, frame.puck_y);
            drawPaddle(frame.opponent_x, frame.opponent_y, COLORS.opponent,
                       COLORS.opponentGlow, COLORS.opponentRing);
        } else if (frame.cam_puck_x !== undefined) {
            drawPuck(frame.cam_puck_x, frame.cam_puck_y);
        }
        if (frame.cam_player_x !== undefined) {
            drawPaddle(frame.cam_player_x, frame.cam_player_y,
                       COLORS.opponent, COLORS.opponentGlow,
                       COLORS.opponentRing);
        }
        drawPaddle(frame.agent_x, frame.agent_y, COLORS.agent, COLORS.agentGlow, COLORS.agentRing);
        // The reachable box is a property of the MACHINE, not of whether the
        // drives happen to be connected, so it is drawn whenever the server
        // has told us what it is. Not drawing it made the sim look like it
        // offered the whole half.
        const shownWorkspace = mode === "replay" ? (replayWorkspace || hwWorkspace) : hwWorkspace;
        if (shownWorkspace) drawReachable(shownWorkspace);
        if (hwPosition && showHwOverlay) {
            drawHwPaddle(hwPosition.x, hwPosition.y);
        }
        if (mode === "replay") {
            const requests = ["", "Left bank", "Right bank", "Straight"];
            ctx.save();
            ctx.font = `${Math.max(12, Math.min(18, canvas.width / 25))}px sans-serif`;
            ctx.textAlign = "left";
            if (requests[frame.shot_type_opponent]) {
                ctx.fillStyle = COLORS.opponent;
                ctx.fillText(`Red asked to shoot: ${requests[frame.shot_type_opponent].toLowerCase()}`, 12, 25);
            }
            if (requests[frame.shot_type_agent]) {
                ctx.fillStyle = COLORS.agent;
                ctx.fillText(`Blue asked to shoot: ${requests[frame.shot_type_agent].toLowerCase()}`, 12, canvas.height - 15);
            }
            if (frame.rally_event) {
                ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
                ctx.fillRect(8, 37, canvas.width - 16, 29);
                ctx.fillStyle = "#f2a93b";
                ctx.textAlign = "center";
                ctx.fillText(frame.rally_event, canvas.width / 2, 57, canvas.width - 28);
            }
            ctx.restore();
        }
    }

    requestAnimationFrame(render);
}

// WebSocket
function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/live`);

    ws.onopen = () => {
        document.getElementById("status").textContent = "Connected";
        // Re-assert the mode. The server starts every connection in control,
        // so without this a reconnect while in sim would silently drop the
        // world out from under the UI.
        if (mode !== "replay") {
            ws.send(JSON.stringify({ type: "set_mode", mode: mode }));
            // Same for the side, and for the same reason: the server starts
            // every connection on the robot.
            ws.send(JSON.stringify({ type: "set_players", ...players }));
        }
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);

        if (msg.type === "config") {
            config = msg;
            resize();
        } else if (msg.type === "frame" && mode !== "replay") {
            frame = msg;
            if (msg.hw_ws) hwWorkspace = msg.hw_ws;
            if (msg.hw_x !== undefined && msg.hw_y !== undefined) {
                hwPosition = { x: msg.hw_x, y: msg.hw_y };
            } else {
                hwPosition = null;
            }
            updateLimitState(msg.hw);
            updateFollowTest(msg.follow_test);
            if (mode === "sim") updateScoreboard(msg);
        } else if (msg.type === "game_over") {
            document.getElementById("status").textContent = `Game Over! ${msg.score_agent}-${msg.score_opponent}`;
            loadRecordingsList();
        } else if (msg.type === "saved") {
            document.getElementById("status").textContent = `Saved: ${msg.name}`;
            loadRecordingsList();
        } else if (msg.type === "physics_mode") {
            document.getElementById("btn-physics").textContent =
                msg.instant ? "Physics: Instant" : "Physics: Realistic";
        } else if (msg.type === "limits") {
            showLimitResult(msg);
        } else if (msg.type === "follow_test") {
            showFollowTest(msg);
        } else if (msg.type === "players") {
            if (msg.error) {
                document.getElementById("status").textContent =
                    "player load failed: " + msg.error;
            }
            // The server's word is final (a checkpoint may have failed to
            // load); sync local state and the selects to whatever it acked.
            players = { agent: msg.agent, opponent: msg.opponent };
            controlSide = players.opponent.kind === "human" ? "human" : "robot";
            for (const side of ["agent", "opponent"]) {
                const sel = document.getElementById("sel-side-" + side);
                if (sel.options.length) sel.value = playerValue(players[side]);
            }
        } else if (msg.type === "sim_limits") {
            if (msg.speed) document.getElementById("sim-speed").value = msg.speed;
            if (msg.accel) document.getElementById("sim-accel").value = msg.accel;
        } else if (msg.type === "hardware_mode") {
            const error = document.getElementById("hardware-error");
            // Older running servers acknowledge failure without its reason.
            // Keep that failure visible until the server is restarted too.
            const reason = msg.error || (hardwareRequest === true && !msg.enabled
                ? "The server rejected the enable request. This server version does not send the reason; check logs/ui_server.log. Restart the web server to receive detailed errors here."
                : "");
            error.textContent = reason ? "Hardware enable failed: " + reason : "";
            error.classList.toggle("hidden", !reason);
            hardwareRequest = null;
            hardwareOn = !!msg.enabled;
            hardwareModeListeners.splice(0).forEach((fn) => fn(msg));
            document.getElementById("btn-hardware").disabled = false;
            document.getElementById("btn-hardware").textContent =
                msg.enabled ? "Hardware: ON" : "Hardware: Off";
            // State lives in a class, not an inline colour. Inline styles
            // beat the stylesheet, so the old hardcoded green survived every
            // restyle and sat there looking like the previous design.
            document.getElementById("btn-hardware")
                .classList.toggle("active", !!msg.enabled);
            const toggle = document.getElementById("hw-overlay-toggle");
            if (msg.enabled) {
                toggle.classList.remove("hidden");
            } else {
                toggle.classList.add("hidden");
                hwPosition = null;
            }
        }
    };

    ws.onclose = () => {
        if (hardwareRequest !== null) {
            const error = document.getElementById("hardware-error");
            error.textContent = "Connection lost while changing hardware mode. Hardware state is unknown; check cdpr_master before retrying.";
            error.classList.remove("hidden");
            hardwareRequest = null;
            document.getElementById("btn-hardware").disabled = false;
            document.getElementById("btn-hardware").textContent = "Hardware: Unknown";
        }
        document.getElementById("status").textContent = "Disconnected. Reconnecting...";
        setTimeout(connect, 2000);
    };
}

function updateScoreboard(f) {
    document.getElementById("score-agent").textContent = f.score_agent;
    document.getElementById("score-opponent").textContent = f.score_opponent;
    if (mode === "replay") {
        // The topbar scoreboard is a phone layout; on a desktop this line is
        // the running score and clock of the recording.
        const clock = (t) => `${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`;
        const end = replayData && replayData.length ? replayData[replayData.length - 1].time : undefined;
        document.getElementById("stage-info").textContent =
            `blue ${f.score_agent ?? 0} – ${f.score_opponent ?? 0} red` +
            (f.time !== undefined ? `   ·   ${clock(f.time)}${end !== undefined ? ` / ${clock(end)}` : ""}` : "");
        document.getElementById("stage-info").classList.remove("hidden");
    }
    if (f.time !== undefined) {
        const mins = Math.floor(f.time / 60);
        const secs = Math.floor(f.time % 60).toString().padStart(2, "0");
        document.getElementById("timer").textContent = `${mins}:${secs}`;
    }
    if (f.cumulative_reward !== undefined) {
        const val = f.cumulative_reward;
        const cls = val > 0 ? "positive" : val < 0 ? "negative" : "";
        const text = val.toFixed(1);
        for (const id of ["reward-value", "reward-value-desktop"]) {
            const el = document.getElementById(id);
            if (el) { el.textContent = text; el.className = cls; }
        }
    }
}

// Convert screen coordinates to physics coordinates
function screenToPhysics(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    // Use display size (rect) not internal canvas size for accurate mapping
    const displayScale = rect.width / config.width;
    const mx = (clientX - rect.left) / displayScale;
    const my = config.height - (clientY - rect.top) / displayScale;
    const r = config.paddle_radius;
    const half = config.height / 2;
    // Which half the pointer is allowed into follows which paddle it drives.
    // Control mode has no second paddle, so it is always the robot's.
    const human = controlSide === "human" && mode === "sim";
    const loY = human ? half + r : r;
    const hiY = human ? config.height - r : half - r;
    const clampedX = Math.min(Math.max(mx, r), config.width - r);
    const clampedY = Math.min(Math.max(my, loY), hiY);
    return { x: clampedX, y: clampedY };
}

// Mouse control.
//
// "follow" chases the pointer continuously; "click" only issues a target
// when you commit to one and holds position otherwise. On hardware the
// difference matters: continuous following turns every twitch of the hand
// into a commanded move, which is the wrong thing to be watching while you
// are still checking whether a single commanded move lands where it should.
// Click is the default because it is the safe one: nothing is commanded
// until you commit to a target. Follow turns every twitch of the hand into
// a move, which is wrong while you are still checking whether a single
// commanded move lands where it should.
let controlMode = "click";    // "click" | "follow"

// Who drives each paddle in SIMULATION mode. Each side is the mouse, a
// scripted rule, or a trained checkpoint — any pairing, including two
// checkpoints fighting each other. `controlSide` is derived: it names the
// half the MOUSE currently drives, which is what the pointer clamp needs.
// Control mode ignores all of this: there is no world there, only the
// machine.
let players = {
    agent: { kind: "human" },
    opponent: { kind: "rule", rule: "follow" },
};
let controlSide = "robot";    // derived from `players`, never set directly

function parsePlayerValue(v) {
    if (v === "human") return { kind: "human" };
    if (v.startsWith("rule:")) return { kind: "rule", rule: v.slice(5) };
    return { kind: "agent", run: v.slice(6) };
}
function playerValue(p) {
    if (p.kind === "human") return "human";
    if (p.kind === "rule") return "rule:" + p.rule;
    return "agent:" + p.run;
}

const SIDE_RULES = {
    agent: ["idle", "goalie", "follow"],
    opponent: ["idle", "goalie", "follow", "random"],
};

async function refreshAgentList() {
    let runs = [];
    try {
        runs = await (await fetch("/api/agents")).json();
    } catch (e) { /* server without checkpoints is fine */ }
    for (const side of ["agent", "opponent"]) {
        const sel = document.getElementById("sel-side-" + side);
        const current = playerValue(players[side]);
        sel.innerHTML = "";
        const add = (value, label) => {
            const o = document.createElement("option");
            o.value = value; o.textContent = label;
            sel.appendChild(o);
        };
        add("human", "Mouse");
        for (const r of SIDE_RULES[side]) add("rule:" + r, "Rule: " + r);
        for (const a of runs) add("agent:" + a.run, "Agent: " + a.run);
        // Keep the selection if it still exists, else fall back sanely.
        sel.value = current;
        if (sel.value !== current) sel.value = side === "agent" ? "human" : "rule:follow";
    }
}

function sendPlayers() {
    if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: "set_players", ...players }));
    }
}

function onSideSelect(side) {
    const sel = document.getElementById("sel-side-" + side);
    const cfg = parsePlayerValue(sel.value);
    // One mouse: making this side human bumps the other side off it.
    const other = side === "agent" ? "opponent" : "agent";
    if (cfg.kind === "human" && players[other].kind === "human") {
        players[other] = other === "opponent"
            ? { kind: "rule", rule: "follow" } : { kind: "rule", rule: "goalie" };
        document.getElementById("sel-side-" + other).value = playerValue(players[other]);
    }
    players[side] = cfg;
    sendPlayers();
}

for (const side of ["agent", "opponent"]) {
    document.getElementById("sel-side-" + side)
        .addEventListener("change", () => onSideSelect(side));
}

document.getElementById("btn-sim-limits").addEventListener("click", () => {
    if (!ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({
        type: "sim_limits",
        speed: parseFloat(document.getElementById("sim-speed").value),
        accel: parseFloat(document.getElementById("sim-accel").value),
    }));
});

// `deliberate` distinguishes a click or tap from the pointer merely passing
// over the canvas. Only a deliberate action may pull the UI out of replay —
// otherwise moving the mouse across the field would silently abandon the
// recording you were watching.
function sendTarget(clientX, clientY, deliberate) {
    if (!ws || !config) return;
    if (mode === "replay") {
        if (!deliberate) return;
        setMode("control");
    }
    const pos = screenToPhysics(clientX, clientY);
    lastCommandedSim = pos;
    ws.send(JSON.stringify({ type: "move", ...pos }));
}
let lastCommandedSim = null;

canvas.addEventListener("mousemove", (e) => {
    if (controlMode !== "follow") return;
    sendTarget(e.clientX, e.clientY, false);
});

canvas.addEventListener("click", (e) => {
    // In follow mode the pointer is already there and mousemove has sent it;
    // resending would double up on the wire for no benefit.
    if (controlMode === "follow" && mode !== "replay") return;
    sendTarget(e.clientX, e.clientY, true);
});

// Touch control for mobile. A tap is the touch equivalent of a click, so it
// commits in either mode; dragging only steers in follow mode.
function handleTouch(e) {
    e.preventDefault();
    if (e.type === "touchmove" && controlMode !== "follow") return;
    const touch = e.touches[0];
    if (!touch) return;
    sendTarget(touch.clientX, touch.clientY, e.type === "touchstart");
}
canvas.addEventListener("touchstart", handleTouch, { passive: false });
canvas.addEventListener("touchmove", handleTouch, { passive: false });

function setControlMode(next) {
    controlMode = next;
    const btn = document.getElementById("btn-control");
    btn.textContent = next === "follow" ? "Control: Follow" : "Control: Click";
    btn.classList.toggle("active", next === "click");
    canvas.style.cursor = next === "click" ? "pointer" : "crosshair";
}

document.getElementById("btn-control").addEventListener("click", () => {
    setControlMode(controlMode === "follow" ? "click" : "follow");
});
setControlMode(controlMode);

// Buttons
document.getElementById("btn-reset").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ type: "reset" }));
    puckTrail = [];
});

document.getElementById("btn-save").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ type: "save" }));
});

document.getElementById("btn-physics").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ type: "toggle_physics" }));
});

const startupTension = document.getElementById("startup-tension");
try {
    const saved = localStorage.getItem("airhockey.startupTensionMm");
    if (saved !== null && saved.trim() && Number.isFinite(Number(saved)) && Number(saved) >= 0 && Number(saved) <= 3)
        startupTension.value = saved;
} catch (_) { /* Storage may be unavailable in private browser sessions. */ }
startupTension.addEventListener("change", () => {
    if (startupTension.value.trim() && startupTension.checkValidity()) {
        try { localStorage.setItem("airhockey.startupTensionMm", startupTension.value); } catch (_) {}
    }
});

document.getElementById("btn-hardware").addEventListener("click", () => {
    const button = document.getElementById("btn-hardware");
    const error = document.getElementById("hardware-error");
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        error.textContent = "Hardware request not sent: the web server is disconnected.";
        error.classList.remove("hidden");
        return;
    }
    if (hardwareRequest !== null) return;
    const enabling = !button.classList.contains("active");
    if (enabling && (!startupTension.value.trim() || !startupTension.reportValidity())) return;
    // Driving the machine from replay mode is never what you meant: the
    // field is showing a recording, and every target you set is discarded.
    if (mode === "replay") setMode("control");
    hardwareRequest = enabling;
    button.disabled = true;
    button.textContent = hardwareRequest ? "Hardware: Enabling…" : "Hardware: Disabling…";
    error.textContent = hardwareRequest ? "Starting the motor controller if needed, measuring the mallet and enabling the drives…" : "Disabling hardware…";
    error.classList.remove("hidden");
    ws.send(JSON.stringify({ type: "toggle_hardware", tension_mm: Number(startupTension.value) }));
});

document.getElementById("chk-hw-overlay").addEventListener("change", (e) => {
    showHwOverlay = e.target.checked;
});

// Mode switching
let recordingsRefreshTimer = null;

function setMode(next) {
    mode = next;
    if (next !== "replay") {
        const url = new URL(window.location.href);
        if (url.searchParams.has("replay")) {
            url.searchParams.delete("replay");
            window.history.replaceState(null, "", url);
        }
    }
    document.querySelectorAll(".mode-btn").forEach(
        (b) => b.classList.toggle("active", b.dataset.mode === next));

    // The server needs to know: in control mode it skips the physics
    // entirely rather than simulating a world nobody is looking at. That is
    // not cosmetic — a simulated goal calls env.reset(), which repositions
    // the paddle and would command the hardware to move on its own.
    if (ws && ws.readyState === 1 && next !== "replay") {
        ws.send(JSON.stringify({ type: "set_mode", mode: next }));
    }
    // Scoring and the clock belong to a game: a simulated one, or a recorded
    // one. The robot tab has neither.
    const scored = next === "sim" || next === "replay";
    document.getElementById("scoreboard").style.visibility = scored ? "" : "hidden";
    document.getElementById("topbar-stats").style.visibility = scored ? "" : "hidden";
    if (next === "sim") {
        setScoreLabels("Robot end", "Far end");
        refreshAgentList();
    }
    // Each sidebar section names the tabs it belongs to (data-tabs); a
    // control that means nothing in this tab is not shown in it.
    document.querySelectorAll("#sidebar [data-tabs]").forEach((el) => {
        el.classList.toggle("hidden", !el.dataset.tabs.split(" ").includes(next));
    });
    document.getElementById("mode-hint").textContent = MODE_HINTS[next] || "";

    if (mode === "replay") {
        loadRecordingsList();
        // Scroll the SELECTOR to the top, not the panel: the list then sits
        // right under it and the way back out stays on screen.
        if (isPhone()) document.getElementById("mode-select").scrollIntoView({ block: "start" });
        if (recordingsRefreshTimer) clearInterval(recordingsRefreshTimer);
        recordingsRefreshTimer = setInterval(loadRecordingsList, 15000);
    } else {
        if (recordingsRefreshTimer) clearInterval(recordingsRefreshTimer);
        recordingsRefreshTimer = null;
        stopReplay();
        puckTrail = [];
    }
}

const MODE_HINTS = {
    control: "The real table. Nothing moves until you press a button here.",
    sim: "A simulated table. Blue (bottom) is the robot's end; pick who drives each mallet.",
    replay: "Saved games. Pick one below; it plays on the table at the left.",
};

function setScoreLabels(blue, red) {
    document.getElementById("label-agent").textContent = blue;
    document.getElementById("label-opponent").textContent = red;
}

document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

// Replay
let activeRecordingPath = requestedReplay || null;

// The Recordings tab. Two different things are called recordings and the tab
// keeps them apart: SIMULATED GAMES (ai/recordings, replayed on this page's
// table) and TABLE SESSIONS (the real robot's logs, opened in /sessions/).
// The server says in plain fields who played each simulated game and which
// model version or run it files under (airhockey/recording_index.py); the
// file names themselves are whatever the writing script chose.
let recordings = [];            // /api/recordings, newest first
let tableSessions = [];         // /sessions/api/replays, newest first
let recordingsSignature = "";
let recKind = "sim";            // which list is showing: "sim" | "table"
let recFilter = "";
// Groups remember whether you opened them across refreshes, so the list does
// not fold shut under your pointer.
const openGroups = new Set();
let openGroupsInitialised = false;

const REC_KIND_NOTES = {
    sim: "Games played in the simulator, saved by training and evaluation runs or by you in the Simulation tab. Filed under the model (or training run) that played the blue, robot end. Click one to watch it on the table at the left.",
    table: "The real robot against a person, as the overhead camera recorded it. A session opens in the session viewer, which replays the measured tracks next to a re-simulation of the same moments.",
};
const REC_SECTIONS = [
    ["models", "Released models", "Versions registered in ai/models.json, newest first."],
    ["runs", "Training runs and experiments", "Everything else, by the run that produced it."],
    ["manual", "Played by mouse", "Games saved from the Simulation tab."],
];
// What each kind of game is, for someone who has not read the training notes.
const KIND_NOTES = {
    "self-play": "the model against a copy of itself",
    "cross-play": "two different models against each other",
    "head-to-head": "two different models against each other",
    "human-proxy benchmark": "the model against a scripted stand-in for a person",
    "isolated defense": "one scripted attack on the robot's goal, to test a save",
    "isolated drill": "one scripted situation, not a full game",
    "resting-puck games": "games checked for pucks left lying untouched",
    "resting-puck stall map": "a puck placed at rest, to see whether the robot plays it",
    "training snapshot": "a game saved during training at this step",
};

function fmtStep(step) {
    if (step === null || step === undefined) return "";
    if (step >= 1_000_000) return `${(step / 1_000_000).toFixed(1)}M`;
    if (step >= 1_000) return `${Math.round(step / 1_000)}k`;
    return String(step);
}

function fmtDate(iso) {
    const d = new Date(iso);
    if (isNaN(d)) return "";
    const now = new Date();
    const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return `today ${hm}`;
    const yest = new Date(now); yest.setDate(now.getDate() - 1);
    if (d.toDateString() === yest.toDateString()) return `yesterday ${hm}`;
    return `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${hm}`;
}

function fmtLength(seconds) {
    if (!Number.isFinite(seconds)) return "";
    return seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${Math.round(seconds)} s`;
}

const versionKey = (v) => (v.match(/\d+/g) || []).map(Number);
function compareVersionsDesc(a, b) {
    const x = versionKey(a), y = versionKey(b);
    for (let i = 0; i < 3; i++) if ((x[i] || 0) !== (y[i] || 0)) return (y[i] || 0) - (x[i] || 0);
    return 0;
}

function recordingMatches(rec, words) {
    if (!words.length) return true;
    const hay = `${rec.name} ${rec.group} ${rec.group_alias || ""} ${rec.blue} ${rec.red} ${rec.kind}`.toLowerCase();
    return words.every((w) => hay.includes(w));
}

async function loadRecordingsList() {
    try {
        const [recs, sessions] = await Promise.all([
            fetch("/api/recordings").then((r) => {
                if (!r.ok) throw new Error(`Recording list request failed (${r.status})`);
                return r.json();
            }),
            // The session viewer is a separate small app; without it the
            // simulated games still list.
            fetch("/sessions/api/replays").then((r) => (r.ok ? r.json() : [])).catch(() => []),
        ]);
        const signature = `${recs.length}:${recs[0]?.mtime}:${sessions.length}:${sessions[0]?.name}`;
        if (signature === recordingsSignature) return;     // nothing new: leave the DOM alone
        recordingsSignature = signature;
        recordings = recs;
        tableSessions = sessions;
        document.getElementById("rec-count-sim").textContent = `(${recs.length})`;
        document.getElementById("rec-count-table").textContent = `(${sessions.length})`;
        renderRecordingLists();
        if (activeRecordingPath) describeLoadedRecording();
        // Something to look at straight away: the newest game of a released model.
        if (mode === "replay" && !replayData && !activeRecordingPath && recs.length) {
            const first = recs.find((r) => r.section === "models") || recs[0];
            loadRecording(first.path, null);
        }
    } catch (e) {
        console.error("Failed to load recordings", e);
        const list = document.getElementById("recording-list");
        if (!list.children.length) {
            const message = document.createElement("li");
            message.className = "rec-empty";
            message.textContent = "Could not load the recording list. Retrying…";
            list.appendChild(message);
        }
    }
}

function renderRecordingLists() {
    document.querySelectorAll(".rec-kind-btn").forEach(
        (b) => b.classList.toggle("active", b.dataset.kind === recKind));
    document.getElementById("rec-kind-note").textContent = REC_KIND_NOTES[recKind];
    document.getElementById("recording-list").classList.toggle("hidden", recKind !== "sim");
    document.getElementById("session-list").classList.toggle("hidden", recKind !== "table");
    document.getElementById("rec-filter").placeholder = recKind === "sim"
        ? "Filter by model, run, opponent or kind…" : "Filter by date, model or policy…";
    if (recKind === "sim") renderSimulatedGames(); else renderTableSessions();
}

function renderSimulatedGames() {
    const list = document.getElementById("recording-list");
    const words = recFilter.toLowerCase().split(/\s+/).filter(Boolean);
    const shown = recordings.filter((r) => recordingMatches(r, words));
    if (!openGroupsInitialised && recordings.length) {
        const active = recordings.find((r) => r.path === activeRecordingPath);
        const first = active || recordings.find((r) => r.section === "models") || recordings[0];
        openGroups.add(`${first.section}/${first.group}`);
        openGroupsInitialised = true;
    }
    list.innerHTML = "";
    if (!shown.length) {
        const empty = document.createElement("li");
        empty.className = "rec-empty";
        empty.textContent = recordings.length ? "No recording matches the filter." : "No recordings yet.";
        list.appendChild(empty);
        return;
    }
    for (const [section, title, note] of REC_SECTIONS) {
        const groups = new Map();           // insertion order = newest first
        for (const rec of shown) {
            if (rec.section !== section) continue;
            if (!groups.has(rec.group)) groups.set(rec.group, []);
            groups.get(rec.group).push(rec);
        }
        if (!groups.size) continue;
        const names = [...groups.keys()];
        if (section === "models") names.sort(compareVersionsDesc);
        const head = document.createElement("li");
        head.className = "rec-section";
        head.innerHTML = `<span></span><small></small>`;
        head.firstChild.textContent = title;
        head.lastChild.textContent = note;
        list.appendChild(head);
        for (const name of names) {
            const items = groups.get(name);
            const key = `${section}/${name}`;
            // A narrow filter result is shown unfolded: you searched for it.
            const open = openGroups.has(key) || (words.length > 0 && shown.length <= 40);
            const li = document.createElement("li");
            li.className = "run-group" + (open ? " open" : "");
            const groupHead = document.createElement("div");
            groupHead.className = "run-head";
            const alias = items[0].group_alias;
            groupHead.innerHTML =
                `<span class="run-caret">▸</span><span class="run-name"></span>` +
                `<span class="run-algo"></span><span class="run-meta"></span>`;
            groupHead.children[1].textContent = name;
            groupHead.children[2].textContent = alias ? alias.replace(/^_/, "") : "";
            groupHead.children[2].title = alias ? `run directory runs/${alias}` : "";
            groupHead.children[3].textContent = `${items.length} · ${fmtDate(items[0].date)}`;
            groupHead.addEventListener("click", () => {
                if (openGroups.has(key)) openGroups.delete(key); else openGroups.add(key);
                renderSimulatedGames();
            });
            li.appendChild(groupHead);
            if (open) {
                const ul = document.createElement("ul");
                ul.className = "run-items";
                for (const rec of items) ul.appendChild(recordingItem(rec));
                li.appendChild(ul);
            }
            list.appendChild(li);
        }
    }
}

function recordingItem(rec) {
    const it = document.createElement("li");
    it.className = "rec" + (rec.path === activeRecordingPath ? " active" : "");
    it.dataset.path = rec.path;
    const what = [rec.kind, rec.step !== null && rec.step !== undefined ? `step ${fmtStep(rec.step)}` : "",
                  rec.variant].filter(Boolean).join(" · ");
    it.innerHTML = `<span class="rec-step"></span><span class="rec-score"></span><span class="rec-meta"></span>`;
    // Score from the point of view of the group's model, whichever end it had.
    const mine = rec.score ? rec.score[rec.plays === "red" ? 1 : 0] : null;
    const theirs = rec.score ? rec.score[rec.plays === "red" ? 0 : 1] : null;
    it.children[0].textContent = `vs ${rec.opponent || rec.red}${rec.plays === "red" ? " (playing the red end)" : ""}`;
    it.children[1].textContent = rec.score ? `${mine}–${theirs}` : "";
    it.children[1].title = "final score: this model – its opponent";
    it.children[2].textContent = [what, fmtLength(rec.duration_s), fmtDate(rec.date)].filter(Boolean).join(" · ");
    // The file name is the only thing that tells two games of one pairing apart.
    const file = document.createElement("span");
    file.className = "rec-file";
    file.textContent = rec.name;
    it.appendChild(file);
    it.title = `ai/recordings/${rec.path}`;
    it.addEventListener("click", () => loadRecording(rec.path, it));
    return it;
}

function renderTableSessions() {
    const list = document.getElementById("session-list");
    const words = recFilter.toLowerCase().split(/\s+/).filter(Boolean);
    list.innerHTML = "";
    const describe = (s) => [
        s.model || (s.policy || "unknown policy").replace(/^neural:runs\//, "").replace(/\/agent\.pt$/, ""),
        (s.caps || "").replace(/\^2/g, "\u00B2"), s.live === false ? "dry run" : s.live ? "robot live" : null,
    ].filter(Boolean).join(" · ");
    const shown = tableSessions.filter((s) => {
        const hay = `${s.name} ${s.started || ""} ${describe(s)} ${s.policy || ""}`.toLowerCase();
        return words.every((w) => hay.includes(w));
    });
    if (!shown.length) {
        const empty = document.createElement("li");
        empty.className = "rec-empty";
        empty.textContent = tableSessions.length ? "No session matches the filter."
            : "No table sessions found in logs/run_policy/.";
        list.appendChild(empty);
        return;
    }
    for (const s of shown) {
        const stamp = s.name.replace(/\.(ticks\.csv|replay\.jsonl)$/, "");
        const li = document.createElement("li");
        li.className = "rec session";
        const a = document.createElement("a");
        a.href = `/sessions/?session=${encodeURIComponent(stamp)}`;
        a.title = `logs/run_policy/${s.name} · ${s.format}`;
        a.innerHTML = `<span class="rec-step"></span><span class="rec-score">open ↗</span><span class="rec-meta"></span>`;
        const started = new Date(s.started);
        a.children[0].textContent = [
            isNaN(started) ? stamp : `${started.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })} ` +
                started.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            fmtLength(s.duration_s)].filter(Boolean).join(" · ");
        a.children[2].textContent = describe(s);
        li.appendChild(a);
        list.appendChild(li);
    }
}

document.querySelectorAll(".rec-kind-btn").forEach((b) => b.addEventListener("click", () => {
    recKind = b.dataset.kind;
    renderRecordingLists();
}));
document.getElementById("rec-filter").addEventListener("input", (e) => {
    recFilter = e.target.value;
    renderRecordingLists();
});

// The card above the playback controls: what the loaded recording IS.
let loadedMetadata = null;
function describeLoadedRecording() {
    const box = document.getElementById("rec-now");
    if (!activeRecordingPath || !replayData) { box.classList.add("hidden"); return; }
    const rec = recordings.find((r) => r.path === activeRecordingPath) || {};
    const meta = loadedMetadata || {};
    const blue = rec.blue || meta.run_name || "blue";
    const red = rec.red || meta.opponent || "red";
    const last = replayData[replayData.length - 1] || {};
    const kind = rec.kind || meta.match_type || "";
    document.getElementById("rec-now-title").textContent = `${blue} vs ${red}`;
    const facts = [
        ["What", KIND_NOTES[kind] ? `${kind}: ${KIND_NOTES[kind]}` : kind],
        ["Blue, bottom", `${blue}${rec.group_alias ? ` (run ${rec.group_alias.replace(/^_/, "")})` : ""} — the robot's end`],
        ["Red, top", `${red} — the far end, where a person stands at the real table`],
        ["Sides", meta.policy_sides || ""],
        ["Final score", last.score_agent !== undefined ? `${last.score_agent}–${last.score_opponent} (blue–red)` : ""],
        ["Length", `${fmtLength(last.time ?? replayData.length / replayFps)} of play, ${replayData.length} frames at ${replayFps} per second`],
        ["Robot limits", meta.accel_cap_m_s2 ? `acceleration up to ${meta.accel_cap_m_s2} m/s²` : ""],
        ["Shot requests", meta.shot_request ? `${meta.shot_request} (the route the policy is asked to play each possession; shown on the table)` : ""],
        ["Training step", rec.step !== null && rec.step !== undefined ? fmtStep(rec.step) : (meta.step ? fmtStep(meta.step) : "")],
        ["Saved", rec.date ? new Date(rec.date).toLocaleString() : ""],
        ["File", `ai/recordings/${activeRecordingPath}`],
    ];
    const dl = document.getElementById("rec-now-facts");
    dl.innerHTML = "";
    for (const [name, value] of facts) {
        if (!value) continue;
        const dt = document.createElement("dt"); dt.textContent = name;
        const dd = document.createElement("dd"); dd.textContent = value;
        dl.append(dt, dd);
    }
    // Everything the writer recorded, minus the one field that is a whole
    // motor thermal model.
    const shownMeta = JSON.parse(JSON.stringify(meta));
    if (shownMeta.simulation_environment?.thermal_path) shownMeta.simulation_environment.thermal_path = "(motor thermal model, omitted)";
    document.getElementById("rec-now-meta").textContent = JSON.stringify(shownMeta, null, 1);
    document.getElementById("rec-now-more").classList.toggle("hidden", !Object.keys(meta).length);
    document.getElementById("legend-blue").textContent = `Blue mallet (bottom, the robot's end): ${blue}`;
    document.getElementById("legend-red").textContent = `Red mallet (top, the far end): ${red}`;
    setScoreLabels(blue, red);
    box.classList.remove("hidden");
}

function isPhone() {
    return window.matchMedia("(max-width: 700px)").matches;
}

async function loadRecording(path, li) {
    // On a phone the sidebar covers two thirds of the screen; a tapped
    // recording is there to be watched, so the sidebar gets out of the way.
    if (isPhone()) closeSidebar();
    try {
        const resp = await fetch(`/api/recordings/${encodeURIComponent(path)}`);
        const data = await resp.json();
        if (!resp.ok || data.error || !Array.isArray(data.frames || data)) {
            throw new Error(data.error || "Recording could not be loaded");
        }
        // API returns {frames, metadata} or legacy flat array
        replayData = data.frames || data;
        const metadata = data.metadata || null;
        replayWorkspace = metadata?.workspace_sim || null;
        stopReplay();
        replayFps = metadata?.fps > 0 ? metadata.fps : 60;
        replayIndex = 0;
        puckTrail = [];

        activeRecordingPath = path;
        loadedMetadata = metadata;
        document.querySelectorAll("#recording-list li.rec").forEach(
            (l) => l.classList.toggle("active", l.dataset.path === path));
        describeLoadedRecording();
        const url = new URL(window.location.href);
        url.searchParams.set("replay", path);
        window.history.replaceState(null, "", url);

        const controls = document.getElementById("replay-controls");
        controls.classList.remove("hidden");

        const slider = document.getElementById("replay-slider");
        slider.max = replayData.length - 1;
        slider.value = 0;

        showReplayFrame(0);
    } catch (e) {
        console.error("Failed to load recording", e);
        document.getElementById("status").textContent = `Replay unavailable: ${e.message}`;
    }
}

function showReplayFrame(idx) {
    if (!replayData || idx < 0 || idx >= replayData.length) return;
    replayIndex = idx;
    replayPosition = idx;
    frame = replayData[idx];
    updateScoreboard(frame);
    document.getElementById("replay-slider").value = idx;
}

// Bodies drawn between recorded frames. Recordings are sampled at the
// control rate (50 Hz), so a 10 m/s puck moves 0.2 m per frame; stepping one
// frame per setInterval tick against a 60 Hz display both jumped that far and
// held frames for uneven times. A jump longer than REPLAY_TELEPORT_M in one
// frame is a reset (goal, relaunch) and is shown as a jump.
const REPLAY_BODIES = [["puck_x", "puck_y"], ["agent_x", "agent_y"], ["opponent_x", "opponent_y"]];
const REPLAY_TELEPORT_M = 0.45;

function interpolatedReplayFrame(position) {
    const i = Math.floor(position);
    const a = replayData[i];
    const b = replayData[i + 1];
    const t = position - i;
    if (!b || t <= 0) return a;
    const out = Object.assign({}, a);
    for (const [kx, ky] of REPLAY_BODIES) {
        if (a[kx] === undefined || b[kx] === undefined) continue;
        if (Math.hypot(b[kx] - a[kx], b[ky] - a[ky]) > REPLAY_TELEPORT_M) continue;
        out[kx] = a[kx] + (b[kx] - a[kx]) * t;
        out[ky] = a[ky] + (b[ky] - a[ky]) * t;
    }
    return out;
}

// Advanced from render(), i.e. on the display's own clock.
function advanceReplay(now) {
    if (!replayPlaying || !replayData) return;
    if (replayLastTime === null) replayLastTime = now;
    // A backgrounded tab stops rAF; do not leap ahead when it comes back.
    const dt = Math.min(now - replayLastTime, 100) / 1000;
    replayLastTime = now;
    const last = replayData.length - 1;
    replayPosition = Math.min(last, replayPosition + dt * replayFps * replaySpeed);
    const idx = Math.floor(replayPosition);
    if (idx !== replayIndex) {
        replayIndex = idx;
        updateScoreboard(replayData[idx]);
        document.getElementById("replay-slider").value = idx;
    }
    frame = interpolatedReplayFrame(replayPosition);
    if (replayPosition >= last) stopReplay();
}

function stopReplay() {
    replayPlaying = false;
    replayLastTime = null;
    document.getElementById("btn-play-pause").textContent = "Play";
}

document.getElementById("btn-play-pause").addEventListener("click", () => {
    if (replayPlaying) {
        stopReplay();
    } else if (replayData) {
        if (replayIndex >= replayData.length - 1) showReplayFrame(0);
        replayPlaying = true;
        replayLastTime = null;
        document.getElementById("btn-play-pause").textContent = "Pause";
    }
});

document.getElementById("btn-step-back").addEventListener("click", () => {
    stopReplay();
    showReplayFrame(Math.max(0, replayIndex - 1));
});

document.getElementById("btn-step-fwd").addEventListener("click", () => {
    stopReplay();
    if (replayData) showReplayFrame(Math.min(replayData.length - 1, replayIndex + 1));
});

document.getElementById("replay-slider").addEventListener("input", (e) => {
    stopReplay();
    showReplayFrame(parseInt(e.target.value));
});

document.getElementById("replay-speed").addEventListener("change", (e) => {
    replaySpeed = parseFloat(e.target.value);
});

// Mobile sidebar menu
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("sidebar-overlay");
const menuBtn = document.getElementById("btn-menu");

function openSidebar() {
    sidebar.classList.add("open");
    overlay.classList.add("visible");
}

function closeSidebar() {
    sidebar.classList.remove("open");
    overlay.classList.remove("visible");
}

if (menuBtn) {
    menuBtn.addEventListener("click", () => {
        if (sidebar.classList.contains("open")) {
            closeSidebar();
        } else {
            openSidebar();
        }
    });
}

if (overlay) {
    overlay.addEventListener("click", closeSidebar);
}

// Prevent default touch behaviors on the whole page to avoid scrolling/zooming
document.addEventListener("touchmove", (e) => {
    if (e.target === canvas || e.target === document.body) {
        e.preventDefault();
    }
}, { passive: false });

// Init
window.addEventListener("resize", resize);
resize();
render();
connect();

// The UI opens on the Robot tab: this is a control surface for a machine as
// much as a viewer. A recording is loaded when the Recordings tab is opened
// (the newest game of a released model), or straight away for ?replay=<file>.
setMode(mode);
if (requestedReplay) loadRecording(requestedReplay, null);


// ── motion limits ──────────────────────────────────────────────────
// The profile lives on the Teensy; these push the caps to it and read back
// which one is actually binding. Worth knowing because the two are not
// interchangeable: a move that is accel-limited the whole way never reaches
// its speed cap, so raising the speed changes nothing.
(() => {
    const sp = document.getElementById("lim-speed");
    const ac = document.getElementById("lim-accel");
    const btn = document.getElementById("btn-limits");
    if (!btn) return;

    // The fields are in m/s and m/s²; the firmware speaks mm. Convert at
    // the edge, in both directions, so the number you read is the number
    // the training config uses (12 m/s, 60 m/s²) and not a thousand times it.
    const apply = () => {
        if (!ws || ws.readyState !== 1) return;
        ws.send(JSON.stringify({ type: "set_limits",
                                 speed: parseFloat(sp.value) * 1000,
                                 accel: parseFloat(ac.value) * 1000 }));
    };
    btn.addEventListener("click", apply);

    // Scale-and-apply, so exploring the range is two clicks rather than
    // typing. No ceiling here — ask for whatever, and the firmware's reply
    // says what it accepted. The one clamp lives in one place.
    document.querySelectorAll(".lim-scale").forEach((b) => {
        b.addEventListener("click", () => {
            const el = document.getElementById(b.dataset.for);
            el.value = Math.max(0.01, Math.round(parseFloat(el.value)
                                                 * parseFloat(b.dataset.mul)
                                                 * 100) / 100);
            apply();
        });
    });
})();

const fmtM = (mm) => (mm / 1000).toFixed(2).replace(/\.?0+$/, "");

function showLimitResult(m) {
    const el = document.getElementById("limit-msg");
    if (!el) return;
    if (m.error) {
        el.textContent = m.error;
        el.className = "bad";
        return;
    }
    // Correct a field ONLY where the firmware actually refused the value.
    // Echoing both back meant scaling one of them rewrote the other, so
    // pressing speed x2 silently changed accel to whatever the last status
    // happened to carry \u2014 a value the user never entered, in a field they
    // never touched.
    const clamped = [];
    if (m.clamped_speed) {
        document.getElementById("lim-speed").value = fmtM(m.speed);
        clamped.push(`speed to ${fmtM(m.speed)} m/s`);
    }
    if (m.clamped_accel) {
        document.getElementById("lim-accel").value = fmtM(m.accel);
        clamped.push(`accel to ${fmtM(m.accel)} m/s\u00b2`);
    }
    if (clamped.length) {
        el.textContent = `firmware clamped ${clamped.join(", ")}`;
        el.className = "bad";
    } else {
        el.textContent = `applied ${fmtM(m.speed)} m/s, `
            + `${fmtM(m.accel)} m/s\u00b2`;
        el.className = "";
    }
}

function updateLimitState(hw) {
    const set = (kind, frac, peak, limit, unit) => {
        const fill = document.getElementById(`gauge-${kind}`);
        const mark = document.getElementById(`peak-${kind}`);
        const pct = document.getElementById(`pct-${kind}`);
        if (frac === undefined || frac === null) {
            fill.style.width = "0%";
            mark.style.display = "none";
            pct.textContent = "\u2014";
            return;
        }
        // Clamp the DISPLAY at 100%, not the value: per-axis caps mean a
        // diagonal can exceed 1.0, and a bar that overflows its track just
        // looks broken. The number beside it still tells the truth.
        fill.style.width = `${Math.min(100, frac * 100)}%`;
        pct.textContent = `${Math.round(frac * 100)}%`;
        if (peak === undefined || peak === null || peak <= 0) {
            mark.style.display = "none";
        } else {
            mark.style.display = "block";
            mark.style.left = `calc(${Math.min(100, peak * 100)}% - 1px)`;
            mark.title = `peak ${Math.round(peak * 100)}%`
                + (limit ? ` = ${(peak * limit / 1000).toFixed(2)} ${unit}` : "");
        }
    };
    if (!hw) {
        set("speed", null); set("accel", null);
        return;
    }
    set("speed", hw.speed_frac, hw.speed_peak, hw.speed_limit, "m/s");
    set("accel", hw.accel_frac, hw.accel_peak, hw.accel_limit, "m/s\u00b2");
}

(() => {
    const b = document.getElementById("btn-reset-peaks");
    if (!b) return;
    b.addEventListener("click", () => {
        if (ws && ws.readyState === 1) {
            ws.send(JSON.stringify({ type: "reset_peaks" }));
        }
    });
})();

// \u2500\u2500 tracking test \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
// One button: starts the fixed sequence, or stops it while it runs. The
// verdict and its numbers land in the <pre> below it; the full samples go
// to logs/follow_test/ on the server, and the last line says where.
let followTestRunning = false;

(() => {
    const b = document.getElementById("btn-follow-test");
    if (!b) return;
    b.addEventListener("click", () => {
        if (!ws || ws.readyState !== 1) return;
        if (mode === "replay") setMode("control");
        ws.send(JSON.stringify({ type: "follow_test",
                                 action: followTestRunning ? "stop" : "start" }));
    });
})();

function setFollowTestRunning(on) {
    followTestRunning = on;
    const b = document.getElementById("btn-follow-test");
    if (b) b.textContent = on ? "Stop tracking test" : "Run tracking test";
}

function updateFollowTest(p) {
    // Progress rides on the frame message while the test runs.
    if (!p) return;
    const el = document.getElementById("follow-test-msg");
    if (!el) return;
    const est = p.estimate_s ? ` of ~${Math.round(p.estimate_s)} s` : "";
    const gap = p.gap_mm === null || p.gap_mm === undefined
        ? "" : `   camera gap ${Math.round(p.gap_mm)} mm`;
    el.textContent = `${p.i}/${p.n} ${p.segment}   ${p.elapsed_s.toFixed(1)} s${est}${gap}`;
    el.className = "";
}

function showFollowTest(msg) {
    const el = document.getElementById("follow-test-msg");
    const out = document.getElementById("follow-test-result");
    if (!el || !out) return;
    if (msg.state === "started") {
        setFollowTestRunning(true);
        out.classList.add("hidden");
        el.textContent = msg.camera
            ? `running ${msg.segments} segments`
            : `running ${msg.segments} segments \u2014 camera OFF, drives vs steps only`;
        el.className = msg.camera ? "" : "bad";
    } else if (msg.state === "done") {
        setFollowTestRunning(false);
        const v = msg.summary && msg.summary.verdict;
        el.textContent = `verdict: ${v}`;
        el.className = v === "close" ? "" : "bad";
        out.textContent = msg.text || JSON.stringify(msg.summary, null, 1);
        out.classList.remove("hidden");
    } else {
        setFollowTestRunning(false);
        el.textContent = msg.error || "tracking test failed";
        el.className = "bad";
    }
}

// ── zoom / pan, shared by the tracker view and the state view ──────
//
// Zoom is anchored on the POINTER, not the centre: the thing under the
// cursor stays under the cursor, so you zoom by pointing at what you want
// rather than zooming and then hunting for it. Both views express the
// result the same way — scale about the element centre plus a pan offset —
// which is what CSS transforms do natively and what the canvas can be made
// to do by folding it into its own coordinate mapping.
//
// Deliberately not attached to the field canvas: that one commands the
// machine on click, and a drag that means "pan" in one panel and "move the
// paddle" in another is exactly the ambiguity you don't want near hardware.
const ZOOM_MAX = 12;

function makeZoomable(el, apply) {
    const st = { zoom: 1, panX: 0, panY: 0 };
    let dragging = false, lastX = 0, lastY = 0;

    const reset = () => { st.zoom = 1; st.panX = 0; st.panY = 0; };

    el.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r = el.getBoundingClientRect();
        // Pointer position relative to the element centre, which is what
        // both the CSS transform-origin and the canvas mapping scale about.
        const mx = e.clientX - r.left - r.width / 2;
        const my = e.clientY - r.top - r.height / 2;
        const before = st.zoom;
        // Exponential so each notch is a constant RATIO — linear steps feel
        // glacial when zoomed out and violent when zoomed in.
        st.zoom = Math.min(ZOOM_MAX, Math.max(1,
            st.zoom * Math.exp(-e.deltaY * 0.0015)));
        const k = st.zoom / before;
        st.panX = mx * (1 - k) + st.panX * k;
        st.panY = my * (1 - k) + st.panY * k;
        if (st.zoom === 1) reset();          // snap cleanly back to fitted
        apply(st);
    }, { passive: false });

    // An <img> is natively draggable, so without this the browser starts an
    // HTML5 image drag on the first pointermove and swallows every event
    // after it — the pan moves once and then freezes. Killing dragstart is
    // what actually fixes it; draggable="false" alone is not reliable across
    // browsers.
    el.addEventListener('dragstart', (e) => e.preventDefault());

    el.addEventListener('pointerdown', (e) => {
        if (st.zoom === 1) return;           // nothing to pan while fitted
        e.preventDefault();                  // no text selection, no img drag
        dragging = true;
        lastX = e.clientX; lastY = e.clientY;
        el.setPointerCapture(e.pointerId);
        el.style.cursor = 'grabbing';
    });
    el.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        st.panX += e.clientX - lastX;
        st.panY += e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        apply(st);
    });
    const release = (e) => {
        if (!dragging) return;
        dragging = false;
        try { el.releasePointerCapture(e.pointerId); } catch (_) {}
        el.style.cursor = st.zoom > 1 ? 'grab' : 'default';
    };
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
    el.addEventListener('dblclick', (e) => {
        e.preventDefault();
        reset();
        apply(st);
    });
    return st;
}

const zoomLabel = (st) => (st.zoom > 1.01 ? `   ×${st.zoom.toFixed(1)}` : '');

// Where in an <img> the pointer is, as a fraction of the PICTURE — not of
// the element. object-fit: contain letterboxes the picture inside the box,
// and the zoom transform moves the box, so neither the element rect nor the
// natural size is usable alone. getBoundingClientRect() already reflects
// the transform, and the scale is uniform, so the letterbox maths works
// directly in the transformed rect.
function imageFraction(img, clientX, clientY) {
    const r = img.getBoundingClientRect();
    if (!img.naturalWidth || !r.width) return null;
    const na = img.naturalWidth / img.naturalHeight;
    let cw, ch;
    if (r.width / r.height > na) { ch = r.height; cw = ch * na; }
    else { cw = r.width; ch = cw / na; }
    const u = (clientX - (r.left + (r.width - cw) / 2)) / cw;
    const v = (clientY - (r.top + (r.height - ch) / 2)) / ch;
    return (u < 0 || u > 1 || v < 0 || v > 1) ? null : { u, v };
}

// Bilinear lookup into the server's unprojection grid.
function gridLookup(grid, u, v) {
    if (!grid || !grid.mm) return null;
    const fx = u * (grid.nx - 1), fy = v * (grid.ny - 1);
    const i = Math.min(grid.nx - 2, Math.max(0, Math.floor(fx)));
    const j = Math.min(grid.ny - 2, Math.max(0, Math.floor(fy)));
    const tx = fx - i, ty = fy - j;
    const at = (a, b) => grid.mm[b * grid.nx + a];
    const out = [0, 0];
    for (let k = 0; k < 2; k++) {
        out[k] = at(i, j)[k] * (1 - tx) * (1 - ty)
               + at(i + 1, j)[k] * tx * (1 - ty)
               + at(i, j + 1)[k] * (1 - tx) * ty
               + at(i + 1, j + 1)[k] * tx * ty;
    }
    return out;
}

// Hole coordinates, which are also inches — the grid is a 25.4 mm pitch, so
// dividing by it gives the hole number. Holes are the only ground truth on
// this table, so this is the form you can walk over and check.
//
// x is counted from the FIRST HOLE RIGHT OF THE CENTRE STRIPE, not from the
// far corner. The stripe is painted and visible; the origin hole is 40-odd
// holes away and counting that far by eye is how you end up arguing about a
// 25 mm discrepancy that was an off-by-one. The stripe itself sits BETWEEN
// hole columns 38 and 39, so column 39 is the reference and hole positions
// stay whole numbers.
const X_REF_COL = 40;            // ceil(CENTERLINE_X / 25.4), 80-col grid
const Y_REF_ROW = 0;             // the near-rail hole line

function inches(mm) {
    const hx = mm[0] / 25.4 - X_REF_COL, hy = mm[1] / 25.4 - Y_REF_ROW;
    return `${hx >= 0 ? '+' : ''}${hx.toFixed(2)}, ${hy.toFixed(2)} in`;
}

const cursorLine = (mm, suffix) => (mm === null
    ? 'cursor   —'
    : `cursor   ${mm[0].toFixed(1)}, ${mm[1].toFixed(1)} mm   ${inches(mm)}`
      + (suffix || ''));

// ── live tracker view ──────────────────────────────────────────────
// The camera is NOT started automatically: only one process can hold the
// Spinnaker device, so taking it unasked would break track_mallet and the
// calibration tools.
(() => {
    const btn = document.getElementById('btn-camera');
    const panel = document.getElementById('camera-panel');
    const img = document.getElementById('camera-img');
    const frame = document.getElementById('camera-view');
    const status = document.getElementById('camera-status');
    if (!btn) return;
    let on = false, poll = null, last = null, streamGeneration = null;
    let reconnectAt = 0, polling = false;
    const connectPreview = (s) => {
        if (!on || !s.running || s.stalled || s.error) return;
        if (!img.getAttribute('src') || streamGeneration !== s.generation) {
            img.src = '/camera/stream?t=' + Date.now();
            streamGeneration = s.generation;
        }
    };
    img.addEventListener('error', () => {
        if (on) { img.removeAttribute('src'); reconnectAt = Date.now()+1000; }
    });

    // Scroll to zoom, drag to pan, double-click to fit. The transform goes
    // on the <img>; the wheel listener goes on the clipping frame, which
    // does not move, so the pointer maths stays in one fixed coordinate
    // system however far you have zoomed.
    const zoom = makeZoomable(frame, (st) => {
        img.style.transform =
            `translate(${st.panX}px, ${st.panY}px) scale(${st.zoom})`;
        frame.style.cursor = st.zoom > 1 ? 'grab' : 'default';
        if (last) paint(last);
    });

    const paint = (s) => {
        last = s;
        if (s.error || !s.running || s.stalled) {
            img.removeAttribute('src'); streamGeneration = null;
            status.textContent = s.error ? 'error: '+s.error
                : s.stalled ? 'Camera stalled — no fresh frames ('+Math.round((s.frame_age_ms || 0)/1000)+' s)'
                : 'Camera stopped — preview is not live';
            status.classList.add('bad');
            return;
        }
        if (Date.now() >= reconnectAt) connectPreview(s);
        status.classList.remove('bad');
        const p = s.pose;
        // The puck's corner count is on the line for a reason: it is the one
        // number that says whether a recording is worth making. Two corners
        // resolves to a position by leaning on the previous frame, so its
        // errors correlate instead of averaging out — visible here, invisible
        // in the fitted result.
        const puck = s.puck
            ? `\npuck     ${s.puck.x.toFixed(1)}, ${s.puck.y.toFixed(1)} mm`
              + `   ${inches([s.puck.x, s.puck.y])}`
              + `   ${s.puck.corners}/4 dots   θ ${s.puck.theta_deg.toFixed(1)}°`
            : '';
        const player = s.player
            ? `\nplayer   ${s.player.x.toFixed(1)}, ${s.player.y.toFixed(1)} mm`
              + `   ${inches([s.player.x, s.player.y])}`
            : '';
        status.textContent = (p
            ? `paddle   ${p.x.toFixed(1)}, ${p.y.toFixed(1)} mm   `
              + `${inches([p.x, p.y])}   θ ${p.theta_deg.toFixed(1)}°`
            : (s.note ? s.note : 'searching for the paddle…'))
            + `   ${s.fps} fps` + zoomLabel(zoom)
            + (s.settings ? `\n${s.settings.exposure_us} µs · ${s.settings.gain_db} dB · threshold ${s.settings.threshold}` : '')
            + puck + player
            + '\n' + cursorLine(cursorMm, '  (table surface)');
    };

    // Cursor position in TABLE millimetres, so you can hover an air hole and
    // read what the calibration thinks it is. On the table surface (z = 0),
    // not the mallet plane — the thing you want to check against is the grid.
    let grid = null, cursorMm = null;
    frame.addEventListener('pointermove', (e) => {
        const f = imageFraction(img, e.clientX, e.clientY);
        cursorMm = f && grid ? gridLookup(grid, f.u, f.v) : null;
        if (last) paint(last);
    });
    frame.addEventListener('pointerleave', () => {
        cursorMm = null;
        if (last) paint(last);
    });

    const collapse = document.getElementById('btn-camera-collapse');
    if (collapse) {
        collapse.addEventListener('click', () => {
            const shut = panel.classList.toggle('collapsed');
            collapse.textContent = shut ? '\u25BC' : '\u25B2';
            collapse.setAttribute('aria-expanded', String(!shut));
            collapse.setAttribute('aria-label',
                shut ? 'Expand tracker view' : 'Collapse tracker view');
        });
    }

    btn.addEventListener('click', async () => {
        on = !on;
        btn.textContent = on ? 'Camera: On' : 'Camera: Off';
        panel.classList.toggle('hidden', !on);
        resize();                          // the field canvas re-fits too
        if (on) {
            const s = await (await fetch('/camera/start', {method: 'POST'})).json();
            paint(s);
            connectPreview(s);
            // Fetched per start, not once: re-running the extrinsics changes
            // the mapping, and a stale grid would read plausibly and wrongly.
            try {
                const g = await (await fetch('/camera/unproject')).json();
                grid = g.error ? null : g;
            } catch (_) { grid = null; }
            poll = setInterval(async () => {
                if (polling) return;
                polling = true;
                try {
                    const response = await fetch('/camera/status', {signal: AbortSignal.timeout(2000)});
                    if (!response.ok) throw new Error('Camera status '+response.status);
                    const s = await response.json();
                    if (on) paint(s);
                } catch (e) {
                    if (on) {
                        status.textContent = 'Camera status unavailable — preview may be stale';
                        status.classList.add('bad');
                    }
                } finally { polling = false; }
            }, 500);
        } else {
            clearInterval(poll); poll = null;
            img.removeAttribute('src');
            await fetch('/camera/stop', {method: 'POST'});
        }
    });
})();


// ── live state view ────────────────────────────────────────────────
// A scale drawing of what the software believes, in the table's own frame.
//
// The point is comparison, not decoration. The camera and the step counts
// are two independent answers to "where is the paddle", and the only way to
// tell a kinematics error from a calibration error from a mechanical one is
// to see both answers at once, in millimetres, against the geometry the
// controller is actually using. So both are drawn, and the gap between them
// is stated numerically.
//
// Orientation matches the tracker view beside it: the table's long axis
// (grid x) runs vertically with the robot end at the BOTTOM, and grid y
// increases to the right.
(() => {
    const btn = document.getElementById('btn-state');
    const panel = document.getElementById('state-panel');
    const cv = document.getElementById('state-canvas');
    const readout = document.getElementById('state-readout');
    if (!btn || !cv) return;

    const c = cv.getContext('2d');
    let geom = null, on = false, poll = null, camPose = null, camNote = null;

    const C = {
        felt: '#13251c',
        rail: '#3a7a5a',
        grid: '#5a5a6a',
        stripe: '#c060c0',
        ws: '#6a8a6a',
        motor: '#e6e64a',
        cable: 'rgba(255,176,64,0.55)',
        cableText: 'rgba(255,196,120,0.85)',
        enc: '#ffb040',       // controller / step counts
        cam: '#40d8ff',       // camera
        cmd: '#8a8a9a',
        text: '#c8c8d4',
    };

    // Drawing bounds include the motors, which sit OUTSIDE the rails in y —
    // clipping them to the table would hide the fact that the anchors are
    // wider than the playfield, which is the whole shape of the problem.
    function bounds() {
        const r = geom.rails;
        let x0 = r.min_x, x1 = r.max_x, y0 = r.min_y, y1 = r.max_y;
        for (const m of geom.motors) {
            x0 = Math.min(x0, m.x); x1 = Math.max(x1, m.x);
            y0 = Math.min(y0, m.y); y1 = Math.max(y1, m.y);
        }
        const pad = 45;
        return { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad };
    }

    let T = null;   // grid mm -> css px
    function layout() {
        const box = cv.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        cv.width = Math.max(1, Math.round(box.width * dpr));
        cv.height = Math.max(1, Math.round(box.height * dpr));
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!geom) return;
        const b = bounds();
        const s = Math.min(box.width / (b.y1 - b.y0),
                           box.height / (b.x1 - b.x0));
        const ox = (box.width - (b.y1 - b.y0) * s) / 2;
        const oy = (box.height - (b.x1 - b.x0) * s) / 2;
        T = { s, b, ox, oy, w: box.width, h: box.height };
    }

    // Scroll to zoom, drag to pan, double-click to fit. Folded into the
    // coordinate mapping rather than applied as a canvas transform on
    // purpose: this way GEOMETRY scales but line widths and label text do
    // not, so a zoomed-in view gets more detail instead of fatter strokes
    // and giant text.
    const zoom = makeZoomable(cv, () => {
        cv.style.cursor = zoom.zoom > 1 ? 'grab' : 'default';
    });

    // Cursor in grid millimetres — the exact inverse of P() below, so what
    // it reports is the same coordinate the drawing uses rather than a
    // second, subtly different mapping.
    let cursorMm = null;
    const unproject = (clientX, clientY) => {
        if (!T) return null;
        const r = cv.getBoundingClientRect();
        const cx = T.w / 2, cy = T.h / 2;
        const bx = (clientX - r.left - cx - zoom.panX) / zoom.zoom + cx;
        const by = (clientY - r.top - cy - zoom.panY) / zoom.zoom + cy;
        return [(by - T.oy) / T.s + T.b.x0,      // grid x is VERTICAL
                (bx - T.ox) / T.s + T.b.y0];
    };
    cv.addEventListener('pointermove', (e) => {
        cursorMm = unproject(e.clientX, e.clientY);
    });
    cv.addEventListener('pointerleave', () => { cursorMm = null; });

    // grid (x, y) mm -> canvas px. Grid x is VERTICAL (down), y horizontal.
    const P = (gx, gy) => {
        const bx = T.ox + (gy - T.b.y0) * T.s;
        const by = T.oy + (gx - T.b.x0) * T.s;
        const cx = T.w / 2, cy = T.h / 2;
        return [(bx - cx) * zoom.zoom + cx + zoom.panX,
                (by - cy) * zoom.zoom + cy + zoom.panY];
    };

    function rect(x0, y0, x1, y1, stroke, fill, dash) {
        const [ax, ay] = P(x0, y0), [bx, by] = P(x1, y1);
        c.save();
        c.setLineDash(dash || []);
        if (fill) { c.fillStyle = fill; c.fillRect(ax, ay, bx - ax, by - ay); }
        if (stroke) {
            c.strokeStyle = stroke; c.lineWidth = 1;
            c.strokeRect(ax, ay, bx - ax, by - ay);
        }
        c.restore();
    }

    function line(x0, y0, x1, y1, stroke, width, dash) {
        const [ax, ay] = P(x0, y0), [bx, by] = P(x1, y1);
        c.save();
        c.setLineDash(dash || []);
        c.strokeStyle = stroke; c.lineWidth = width || 1;
        c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
        c.restore();
    }

    function attach(m, gx, gy, thetaRad) {
        const phi = thetaRad + geom.attach_chirality * (Math.PI / 2) * m;
        return [gx + geom.attach_r * Math.cos(phi),
                gy + geom.attach_r * Math.sin(phi)];
    }

    // Free span: the wire actually hanging between the spool and the paddle.
    //
    // Not the same as the length the controller commands. The wire leaves
    // the spool at a TANGENT, so the visible run is sqrt(d^2 - r^2) rather
    // than the centre-to-centre distance; and the controller's number also
    // carries a wrap term for wire wound onto the spool, which is real cable
    // but is not hanging in the air. This is the one you can go and measure
    // with a tape, so it is the one drawn.
    function freeSpan(m, gx, gy, thetaRad) {
        const [ax, ay] = attach(m, gx, gy, thetaRad);
        const dx = ax - geom.motors[m].x, dy = ay - geom.motors[m].y;
        const d = Math.hypot(dx, dy);
        const r = geom.spool_radius;
        return d <= r ? 0 : Math.sqrt(d * d - r * r);
    }

    const spans = (gx, gy, th) =>
        [0, 1, 2, 3].map((m) => freeSpan(m, gx, gy, th));

    // A paddle plus the four cables the model has running to it. Drawing the
    // cables is what makes a wrong pose obvious: they visibly stop pointing
    // at the spools.
    function paddle(gx, gy, thetaRad, colour, label, withCables) {
        if (withCables) {
            c.save();
            c.font = '10px ui-monospace, Menlo, monospace';
            c.fillStyle = C.cableText;
            c.textAlign = 'center';
            for (let m = 0; m < 4; m++) {
                const [ax, ay] = attach(m, gx, gy, thetaRad);
                const mo = geom.motors[m];
                line(ax, ay, mo.x, mo.y, C.cable, 1);
                // Label at 55% toward the spool: far enough from the paddle
                // that the four numbers do not pile up on each other.
                const [lx, ly] = P(ax + (mo.x - ax) * 0.55,
                                   ay + (mo.y - ay) * 0.55);
                c.fillText(freeSpan(m, gx, gy, thetaRad).toFixed(0), lx, ly - 3);
            }
            c.restore();
        }
        const [px, py] = P(gx, gy);
        const r = Math.max(4, geom.attach_r * T.s * zoom.zoom);
        c.save();
        c.strokeStyle = colour; c.lineWidth = 2;
        c.beginPath(); c.arc(px, py, r, 0, Math.PI * 2); c.stroke();
        // Orientation arm — theta is a real degree of freedom here, so it
        // gets drawn rather than assumed away.
        const [tx, ty] = P(gx + geom.attach_r * 1.9 * Math.cos(thetaRad),
                           gy + geom.attach_r * 1.9 * Math.sin(thetaRad));
        c.beginPath(); c.moveTo(px, py); c.lineTo(tx, ty); c.stroke();
        c.fillStyle = colour;
        c.beginPath(); c.arc(px, py, 2.5, 0, Math.PI * 2); c.fill();
        c.font = '10px ui-monospace, Menlo, monospace';
        c.fillText(label, px + r + 4, py - r - 2);
        c.restore();
    }

    function draw() {
        if (!on) return;
        requestAnimationFrame(draw);
        if (!geom || !T) return;
        c.clearRect(0, 0, T.w, T.h);

        const g = geom;
        rect(g.rails.min_x, g.rails.min_y, g.rails.max_x, g.rails.max_y,
             C.rail, C.felt);
        rect(0, 0, g.grid.x, g.grid.y, C.grid);
        line(g.centerline_x, g.rails.min_y, g.centerline_x, g.rails.max_y,
             C.stripe, 1.5);
        const workspace = frame?.hw?.workspace_mm || g.workspace;
        rect(workspace.min_x, workspace.min_y,
             workspace.max_x, workspace.max_y, C.ws, null, [5, 4]);

        c.save();
        c.font = '10px ui-monospace, Menlo, monospace';
        g.motors.forEach((m, i) => {
            const [mx, my] = P(m.x, m.y);
            c.strokeStyle = C.motor; c.lineWidth = 1.5;
            c.beginPath(); c.arc(mx, my, 5, 0, Math.PI * 2); c.stroke();
            c.beginPath();
            c.moveTo(mx - 7, my); c.lineTo(mx + 7, my);
            c.moveTo(mx, my - 7); c.lineTo(mx, my + 7);
            c.stroke();
            // Label toward the table, not outward — the anchors sit at the
            // very edge of the drawing and an outward label gets clipped.
            const inward = m.y > (g.rails.min_y + g.rails.max_y) / 2;
            c.fillStyle = C.motor;
            c.textAlign = inward ? 'right' : 'left';
            c.fillText('M' + i, mx + (inward ? -9 : 9), my + 3);
            c.textAlign = 'left';
        });
        c.restore();

        const hw = frame && frame.hw ? frame.hw : null;
        const nomTh = g.nominal_theta_deg * Math.PI / 180;

        if (hw) {
            const [cx, cy] = P(hw.cmd_x_mm, hw.cmd_y_mm);
            c.save();
            c.strokeStyle = C.cmd; c.lineWidth = 1; c.setLineDash([3, 3]);
            c.beginPath();
            c.moveTo(cx - 6, cy); c.lineTo(cx + 6, cy);
            c.moveTo(cx, cy - 6); c.lineTo(cx, cy + 6);
            c.stroke(); c.restore();
            paddle(hw.x_mm, hw.y_mm, nomTh, C.enc, 'controller', true);
        }
        if (camPose) {
            // With no hardware there is no controller pose to hang the
            // cables off, so draw them from the measured one instead —
            // otherwise the wire lengths are invisible until the drives are
            // up, which is exactly when you want to check them.
            paddle(camPose.x, camPose.y, camPose.theta_deg * Math.PI / 180,
                   C.cam, 'camera', !hw);
        }

        const L = [];
        L.push((camPose
            ? `camera   ${camPose.x.toFixed(1)}, ${camPose.y.toFixed(1)} mm   `
              + `${inches([camPose.x, camPose.y])}   `
              + `θ ${camPose.theta_deg.toFixed(1)}°`
            : `camera   ${camNote || 'not running'}`) + zoomLabel(zoom));
        L.push(cursorLine(cursorMm));
        // "controller", not "encoder": this x/y is the Teensy's integrated
        // trajectory, i.e. where it BELIEVES it stepped the paddle to. The
        // drives' actual encoders are the per-cable row below.
        L.push(hw
            ? `control  x ${hw.x_mm.toFixed(1).padStart(7)}  y ${hw.y_mm.toFixed(1).padStart(6)}   θ ${g.nominal_theta_deg.toFixed(1)}° assumed`
            : 'control  hardware off');
        if (hw && camPose) {
            const dx = camPose.x - hw.x_mm, dy = camPose.y - hw.y_mm;
            L.push(`Δ        x ${dx.toFixed(1).padStart(7)}  y ${dy.toFixed(1).padStart(6)}   |Δ| ${Math.hypot(dx, dy).toFixed(1)} mm`);
        }
        if (hw) {
            L.push(`target   x ${hw.cmd_x_mm.toFixed(1).padStart(7)}  y ${hw.cmd_y_mm.toFixed(1).padStart(6)}   ${hw.speed_mm_s} mm/s`);
        }

        const row = (name, vals, dp) => L.push(
            name.padEnd(9) + vals.map((v) => (v === null || v === undefined
                ? '     --' : v.toFixed(dp).padStart(7))).join(' '));

        if (hw || camPose) {
            L.push('');
            L.push('cable            M0      M1      M2      M3');
            // Wire hanging between each spool and the paddle. Two poses give
            // two answers; if they disagree the cables in the drawing are
            // not the cables on the table.
            if (camPose) {
                row('wire cam', spans(camPose.x, camPose.y,
                                      camPose.theta_deg * Math.PI / 180), 1);
            }
            if (hw) row('wire ctl', spans(hw.x_mm, hw.y_mm, nomTh), 1);
            if (hw && camPose) {
                const a = spans(camPose.x, camPose.y,
                                camPose.theta_deg * Math.PI / 180);
                const b = spans(hw.x_mm, hw.y_mm, nomTh);
                row('wire Δ', a.map((v, i) => v - b[i]), 1);
            }
            // Commanded vs measured, per cable. The Teensy's steps are what
            // it asked for; the drive encoders are what happened. A row that
            // disagrees points at that motor, not at the kinematics.
            if (hw && hw.step_mm) row('stepped', hw.step_mm, 1);
            if (hw && hw.enc_mm) row('measured', hw.enc_mm, 1);
            if (hw && hw.trq_pct) row('torque %', hw.trq_pct, 0);
        }
        readout.textContent = L.join('\n');
    }

    const collapse = document.getElementById('btn-state-collapse');
    if (collapse) {
        collapse.addEventListener('click', () => {
            const shut = panel.classList.toggle('collapsed');
            collapse.textContent = shut ? '▼' : '▲';
            collapse.setAttribute('aria-expanded', String(!shut));
            collapse.setAttribute('aria-label',
                shut ? 'Expand state view' : 'Collapse state view');
        });
    }

    // The canvas resizes whenever ANOTHER panel is toggled, not just on
    // window resize, so watch the element rather than the window.
    if (window.ResizeObserver) {
        new ResizeObserver(() => { if (on && geom) layout(); }).observe(cv);
    } else {
        window.addEventListener('resize', () => { if (on) layout(); });
    }

    btn.addEventListener('click', async () => {
        on = !on;
        btn.textContent = on ? 'State view: On' : 'State view: Off';
        panel.classList.toggle('hidden', !on);
        resize();                          // the field canvas re-fits too
        if (!on) { clearInterval(poll); poll = null; return; }
        if (!geom) {
            try {
                geom = await (await fetch('/api/geometry')).json();
            } catch (e) {
                readout.textContent = 'geometry unavailable: ' + e;
                readout.classList.add('bad');
                return;
            }
        }
        layout();
        // The camera pose comes from the same endpoint the tracker view
        // polls; it is simply null while the camera is off.
        poll = setInterval(async () => {
            try {
                const s = await (await fetch('/camera/status')).json();
                camPose = s.pose || null;
                camNote = s.error || s.note || (s.running ? 'searching…' : 'not running');
            } catch (e) { camPose = null; camNote = 'status unavailable'; }
        }, 150);
        requestAnimationFrame(draw);
    });
})();
