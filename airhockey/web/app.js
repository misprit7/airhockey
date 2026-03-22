const canvas = document.getElementById("table");
const ctx = canvas.getContext("2d");

// State
let config = null;
let frame = null;
let ws = null;
let mode = "live"; // "live" or "replay"
let replayData = null;
let replayIndex = 0;
let replayPlaying = false;
let replaySpeed = 1;
let replayTimer = null;
let scale = 1;
let offsetX = 0;
let offsetY = 0;

// Colors
const COLORS = {
    table: "#1b3a2a",
    tableBorder: "#2a5a3e",
    centerLine: "rgba(255,255,255,0.08)",
    centerCircle: "rgba(255,255,255,0.06)",
    wall: "#3a7a5a",
    goal: "#c0392b",
    goalGlow: "rgba(192, 57, 43, 0.3)",
    puck: "#e8e8f0",
    puckGlow: "rgba(232, 232, 240, 0.4)",
    puckTrail: "rgba(232, 232, 240, 0.08)",
    agent: "#4a9eff",
    agentGlow: "rgba(74, 158, 255, 0.35)",
    agentRing: "rgba(74, 158, 255, 0.15)",
    opponent: "#ff5a6a",
    opponentGlow: "rgba(255, 90, 106, 0.35)",
    opponentRing: "rgba(255, 90, 106, 0.15)",
};

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

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawTable();

    if (frame) {
        drawPuck(frame.puck_x, frame.puck_y);
        drawPaddle(frame.agent_x, frame.agent_y, COLORS.agent, COLORS.agentGlow, COLORS.agentRing);
        drawPaddle(frame.opponent_x, frame.opponent_y, COLORS.opponent, COLORS.opponentGlow, COLORS.opponentRing);
    }

    requestAnimationFrame(render);
}

// WebSocket
function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/live`);

    ws.onopen = () => {
        document.getElementById("status").textContent = "Connected";
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);

        if (msg.type === "config") {
            config = msg;
            resize();
        } else if (msg.type === "frame" && mode === "live") {
            frame = msg;
            updateScoreboard(msg);
        } else if (msg.type === "game_over") {
            document.getElementById("status").textContent = `Game Over! ${msg.score_agent}-${msg.score_opponent}`;
            loadRecordingsList();
        } else if (msg.type === "saved") {
            document.getElementById("status").textContent = `Saved: ${msg.name}`;
            loadRecordingsList();
        }
    };

    ws.onclose = () => {
        document.getElementById("status").textContent = "Disconnected. Reconnecting...";
        setTimeout(connect, 2000);
    };
}

function updateScoreboard(f) {
    document.getElementById("score-agent").textContent = f.score_agent;
    document.getElementById("score-opponent").textContent = f.score_opponent;
    if (f.time !== undefined) {
        const mins = Math.floor(f.time / 60);
        const secs = Math.floor(f.time % 60).toString().padStart(2, "0");
        document.getElementById("timer").textContent = `${mins}:${secs}`;
    }
    if (f.cumulative_reward !== undefined) {
        const el = document.getElementById("reward-value");
        const val = f.cumulative_reward;
        el.textContent = val.toFixed(1);
        el.className = val > 0 ? "positive" : val < 0 ? "negative" : "";
    }
}

// Convert screen coordinates to physics coordinates
function screenToPhysics(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    // Use display size (rect) not internal canvas size for accurate mapping
    const displayScale = rect.width / config.width;
    const mx = (clientX - rect.left) / displayScale;
    const my = config.height - (clientY - rect.top) / displayScale;
    const clampedX = Math.min(Math.max(mx, config.paddle_radius), config.width - config.paddle_radius);
    const clampedY = Math.min(Math.max(my, config.paddle_radius), config.height / 2 - config.paddle_radius);
    return { x: clampedX, y: clampedY };
}

// Mouse control
canvas.addEventListener("mousemove", (e) => {
    if (mode !== "live" || !ws || !config) return;
    const pos = screenToPhysics(e.clientX, e.clientY);
    ws.send(JSON.stringify({ type: "move", ...pos }));
});

// Touch control for mobile
function handleTouch(e) {
    e.preventDefault();
    if (mode !== "live" || !ws || !config) return;
    const touch = e.touches[0];
    const pos = screenToPhysics(touch.clientX, touch.clientY);
    ws.send(JSON.stringify({ type: "move", ...pos }));
}
canvas.addEventListener("touchstart", handleTouch, { passive: false });
canvas.addEventListener("touchmove", handleTouch, { passive: false });

// Buttons
document.getElementById("btn-reset").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ type: "reset" }));
    puckTrail = [];
});

document.getElementById("btn-save").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ type: "save" }));
});

// Mode switching
let recordingsRefreshTimer = null;

document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        mode = btn.dataset.mode;

        const replayPanel = document.getElementById("replay-panel");
        if (mode === "replay") {
            replayPanel.classList.remove("hidden");
            loadRecordingsList();
            // Auto-refresh recording list every 5 seconds
            if (recordingsRefreshTimer) clearInterval(recordingsRefreshTimer);
            recordingsRefreshTimer = setInterval(loadRecordingsList, 5000);
        } else {
            replayPanel.classList.add("hidden");
            if (recordingsRefreshTimer) clearInterval(recordingsRefreshTimer);
            recordingsRefreshTimer = null;
            stopReplay();
            puckTrail = [];
        }
    });
});

// Replay
let activeRecordingPath = null;

async function loadRecordingsList() {
    try {
        const resp = await fetch("/api/recordings");
        const recordings = await resp.json();
        const list = document.getElementById("recording-list");
        list.innerHTML = "";
        recordings.forEach((rec) => {
            const li = document.createElement("li");
            li.textContent = rec.label || rec.name;
            if (rec.path === activeRecordingPath) li.classList.add("active");
            li.addEventListener("click", () => loadRecording(rec.path, li));
            list.appendChild(li);
        });
    } catch (e) {
        console.error("Failed to load recordings", e);
    }
}

async function loadRecording(path, li) {
    try {
        const resp = await fetch(`/api/recordings/${path}`);
        replayData = await resp.json();
        replayIndex = 0;
        puckTrail = [];

        activeRecordingPath = path;
        document.querySelectorAll("#recording-list li").forEach((l) => l.classList.remove("active"));
        li.classList.add("active");

        const controls = document.getElementById("replay-controls");
        controls.classList.remove("hidden");

        const slider = document.getElementById("replay-slider");
        slider.max = replayData.length - 1;
        slider.value = 0;

        showReplayFrame(0);
    } catch (e) {
        console.error("Failed to load recording", e);
    }
}

function showReplayFrame(idx) {
    if (!replayData || idx < 0 || idx >= replayData.length) return;
    replayIndex = idx;
    frame = replayData[idx];
    updateScoreboard(frame);
    document.getElementById("replay-slider").value = idx;
}

function stopReplay() {
    replayPlaying = false;
    if (replayTimer) clearInterval(replayTimer);
    replayTimer = null;
    document.getElementById("btn-play-pause").textContent = "Play";
}

document.getElementById("btn-play-pause").addEventListener("click", () => {
    if (replayPlaying) {
        stopReplay();
    } else {
        replayPlaying = true;
        document.getElementById("btn-play-pause").textContent = "Pause";
        replayTimer = setInterval(() => {
            if (replayIndex < replayData.length - 1) {
                showReplayFrame(replayIndex + 1);
            } else {
                stopReplay();
            }
        }, (1000 / 60) / replaySpeed);
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
    if (replayPlaying) {
        clearInterval(replayTimer);
        replayTimer = setInterval(() => {
            if (replayIndex < replayData.length - 1) {
                showReplayFrame(replayIndex + 1);
            } else {
                stopReplay();
            }
        }, (1000 / 60) / replaySpeed);
    }
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
