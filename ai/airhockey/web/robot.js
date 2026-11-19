// The Robot tab's two actions on the real table: "Start AI" and "Take up
// cable slack". Both are done by the server (airhockey/robot_session.py);
// this file asks and shows what the server reports. No confirmation dialogs
// (removed at the user's request, 2026-11-18): a press is the decision.
//
// Loaded after app.js and shares its globals: ws, mode, hardwareOn,
// hardwareModeListeners, startupTension.
(() => {
    const $ = (id) => document.getElementById(id);
    const aiButton = $("btn-ai");
    const takeupButton = $("btn-takeup");
    const level = $("ai-difficulty");
    let status = null;
    let busy = false;           // a request of ours is in flight
    let levelsKey = "";
    let lastPhase = "idle";

    const LEVEL_NOTES = {
        max: "Full strength.",
        hard: "The full-strength model with its acceleration clipped.",
        medium: "A model trained to play at this acceleration.",
        easy: "A model trained to play at this acceleration.",
        beginner: "A model trained to play at this acceleration.",
    };

    function elapsed(since) {
        const s = Math.max(0, Math.round(Date.now() / 1000 - since));
        return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }

    function fillLevels(modes) {
        const key = JSON.stringify(modes);
        if (key === levelsKey) return;
        levelsKey = key;
        let saved = null;
        try { saved = localStorage.getItem("airhockey.aiLevel"); } catch (_) { /* private mode */ }
        const current = level.value || saved || "max";
        level.innerHTML = "";
        for (const m of modes) {
            const o = document.createElement("option");
            o.value = m.mode;
            o.textContent = `${m.mode} — ${m.model}, up to ${m.accel} m/s²`;
            level.appendChild(o);
        }
        level.value = current;
        if (level.value !== current && modes.length) level.value = modes[0].mode;
        showLevelNote();
    }

    function showLevelNote() {
        const m = (status?.modes || []).find((x) => x.mode === level.value);
        if (!m) { $("ai-mode-note").textContent = ""; return; }
        const clipped = m.accel < m.trained_accel
            ? ` Model ${m.model} trained at ${m.trained_accel} m/s² and is limited to ${m.accel} here.` : "";
        $("ai-mode-note").textContent =
            `${LEVEL_NOTES[m.mode] || ""}${clipped} Levels are defined in ai/models.json.`;
    }
    level.addEventListener("change", () => {
        try { localStorage.setItem("airhockey.aiLevel", level.value); } catch (_) { /* private mode */ }
        showLevelNote();
    });

    function render() {
        if (!status) return;
        fillLevels(status.modes || []);
        const ai = status.ai || { state: "idle" };
        const takeup = status.takeup || { state: "idle" };
        const aiLive = ai.state === "running" || ai.state === "stopping";

        aiButton.textContent = ai.state === "running" ? "Stop AI"
            : ai.state === "stopping" ? "Stopping… (press again to insist)" : "Start AI";
        aiButton.classList.toggle("active", aiLive);
        aiButton.disabled = busy || takeup.state === "running";
        level.disabled = aiLive;

        let text;
        if (ai.state === "running") {
            text = `Running ${ai.difficulty} (${ai.model}, up to ${ai.accel} m/s²) for ${elapsed(ai.started)}. ` +
                "Stop brakes the mallet, returns it to the centre and switches the drives off.";
        } else if (ai.state === "stopping") {
            text = "Stopping: braking, returning to the centre, then switching the drives off…";
        } else if (ai.state === "exited") {
            const how = ai.exit_code === 0 ? "ended normally"
                : ai.exit_code === null || ai.exit_code === undefined ? "ended" : `ended with an error (exit code ${ai.exit_code})`;
            text = `Last session (${ai.difficulty}, ${ai.model}) ${how}. The drives are off.`;
        } else {
            text = "Not running.";
        }
        $("ai-status").textContent = text;
        $("ai-status").classList.toggle("bad", ai.state === "exited" && !!ai.exit_code);

        const log = $("ai-log");
        const lines = (status.log || []).join("\n");
        if (log.textContent !== lines) {
            const pinned = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
            log.textContent = lines;
            if (pinned) log.scrollTop = log.scrollHeight;
        }
        $("ai-log-wrap").classList.toggle("hidden", !lines);
        // Unfold the output when a session starts (the start-up steps are
        // worth watching) and when one fails (it is the only account of why);
        // after that it stays however you left it.
        const phase = ai.state === "exited" && ai.exit_code ? "failed" : aiLive ? "live" : "idle";
        if (phase !== lastPhase && phase !== "idle") $("ai-log-wrap").open = true;
        lastPhase = phase;

        renderMaster(aiLive, takeup.state === "running");
        takeupButton.textContent = takeup.state === "running" ? "Stop taking up slack" : "Take up cable slack";
        takeupButton.classList.toggle("active", takeup.state === "running");
        takeupButton.disabled = busy || aiLive;
        const t = $("takeup-status");
        t.classList.toggle("bad", takeup.state === "error");
        if (takeup.state === "running" && takeup.phase === "starting cdpr_master") {
            t.textContent = "Starting the motor controller…";
        } else if (takeup.state === "running") {
            t.textContent = `Winding in slack for ${elapsed(takeup.started)}… The spools turn slowly; each stops when its cable is taut.`;
        } else if (takeup.state === "done") {
            const mm = takeup.result.wound_mm.map((v) => v.toFixed(1)).join(" / ");
            t.textContent = `Cables taut. Wound in ${mm} mm (motors 0–3). The drives are off again; enable Hardware or start the AI.`;
        } else if (takeup.state === "error" || takeup.state === "stopped") {
            t.textContent = takeup.message;
        } else {
            t.textContent = "";
        }
        applySticky();
    }

    // The background program that talks to the motors. Nobody has to start it:
    // every button that needs it does. This line only says where things stand.
    function renderMaster(aiLive, takeupLive) {
        const line = $("master-status");
        const stop = $("btn-master-stop");
        line.classList.remove("bad");
        line.textContent = !status.master
            ? "Motor controller: off. It starts by itself when you press a button above."
            : aiLive ? "Motor controller: running the AI session."
            : hardwareOn ? "Motor controller: on, drives energized for manual control."
            : "Motor controller: on and idle, drives off.";
        stop.classList.toggle("hidden", !(status.master && status.master_owned));
        stop.disabled = busy || aiLive || takeupLive || hardwareOn;
    }

    // A refusal stays on screen until the next button press; the status poll
    // would otherwise paint over it within a second.
    const sticky = {};
    function showError(where, message) {
        sticky[where] = message;
        applySticky();
    }
    function applySticky() {
        for (const [where, message] of Object.entries(sticky)) {
            const el = $(where);
            el.textContent = message;
            el.classList.add("bad");
        }
    }
    function clearErrors() {
        for (const where of Object.keys(sticky)) delete sticky[where];
    }

    async function refresh() {
        try {
            const r = await fetch("/api/robot");
            if (!r.ok) throw new Error(`status ${r.status}`);
            status = await r.json();
            if (!busy) render();
        } catch (e) {
            if (!busy) $("ai-status").textContent = "Robot status unavailable (is the web server current? restart it).";
        }
    }

    async function post(path, body, where) {
        busy = true;
        aiButton.disabled = takeupButton.disabled = true;
        try {
            const r = await fetch(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body || {}),
            });
            const reply = await r.json();
            if (reply.ai) status = { ...status, ...reply };
            busy = false;
            render();
            if (!reply.ok) showError(where, reply.error || "The server refused the request.");
        } catch (e) {
            busy = false;
            render();
            showError(where, `Request failed: ${e.message}`);
        }
    }

    // The master serves one client. If this tab holds it for manual control,
    // hand it back first and wait for the server to say the drives are off.
    function releaseManualControl() {
        if (!hardwareOn) return Promise.resolve();
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error("Hardware did not turn off within 30 s")), 30000);
            hardwareModeListeners.push((msg) => {
                clearTimeout(timer);
                if (msg.enabled) reject(new Error("Hardware is still on")); else resolve();
            });
            $("btn-hardware").click();
        });
    }

    function tension() {
        const value = Number(startupTension.value);
        return Number.isFinite(value) && value >= 0 && value <= 6 ? value : null;
    }

    aiButton.addEventListener("click", async () => {
        clearErrors();
        const state = status?.ai?.state;
        if (state === "running" || state === "stopping") {
            await post("/api/robot/ai/stop", {}, "ai-status");
            return;
        }
        const m = (status?.modes || []).find((x) => x.mode === level.value);
        const mm = tension();
        if (!m) return showError("ai-status", "No level selected.");
        if (mm === null) return showError("ai-status", "Startup pretension must be between 0 and 6 mm.");
        try {
            await releaseManualControl();
        } catch (e) {
            return showError("ai-status", `Not started: ${e.message}.`);
        }
        await post("/api/robot/ai/start", { confirm: true, difficulty: m.mode, tension_mm: mm }, "ai-status");
    });

    takeupButton.addEventListener("click", async () => {
        clearErrors();
        if (status?.takeup?.state === "running") {
            await post("/api/robot/takeup/stop", {}, "takeup-status");
            return;
        }
        if (hardwareOn) return showError("takeup-status", "Turn Hardware off first: slack is taken up with the drives' own motion, before they are enabled for control.");
        await post("/api/robot/takeup", { confirm: true }, "takeup-status");
    });

    $("btn-master-stop").addEventListener("click", () => {
        clearErrors();
        post("/api/robot/master/stop", {}, "master-status");
    });

    // Once a second while the Robot tab is showing and something is live;
    // slower otherwise. The status call reads files and a process table only.
    async function loop() {
        await refresh();
        const live = status && (status.ai?.state === "running" || status.ai?.state === "stopping"
                                || status.takeup?.state === "running");
        setTimeout(loop, mode === "control" && !document.hidden ? (live ? 1000 : 2000) : 5000);
    }
    loop();
})();
