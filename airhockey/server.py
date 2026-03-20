"""FastAPI WebSocket server for real-time visualization and replay."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from airhockey.dynamics import DelayedDynamics, IdealDynamics
from airhockey.env import AirHockeyEnv
from airhockey.physics import TableConfig
from airhockey.recorder import Recorder

app = FastAPI()

WEB_DIR = Path(__file__).parent / "web"
RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    return HTMLResponse((WEB_DIR / "index.html").read_text())


@app.get("/app.js")
async def app_js():
    return HTMLResponse(
        (WEB_DIR / "app.js").read_text(),
        media_type="application/javascript",
    )


@app.get("/style.css")
async def style_css():
    return HTMLResponse(
        (WEB_DIR / "style.css").read_text(),
        media_type="text/css",
    )


def _recording_label(stem: str) -> str:
    """Turn a filename stem into a readable label.

    e.g. 'ppo_v4_shaped_step_0100000' -> 'ppo_v4_shaped @ step 100k'
         'game_1773964952' -> 'game_1773964952'
    """
    if "_step_" in stem:
        parts = stem.rsplit("_step_", 1)
        run_name = parts[0]
        step_num = int(parts[1])
        if step_num >= 1_000_000:
            step_label = f"{step_num / 1_000_000:.1f}M"
        elif step_num >= 1_000:
            step_label = f"{step_num // 1_000}k"
        else:
            step_label = str(step_num)
        return f"{run_name} @ {step_label}"
    return stem


@app.get("/api/recordings")
async def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.json"), reverse=True)
    return [{"name": f.stem, "path": f.name, "label": _recording_label(f.stem)} for f in files]


@app.get("/api/recordings/{filename}")
async def get_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if not path.exists():
        return {"error": "not found"}
    return json.loads(path.read_text())


@app.websocket("/ws/live")
async def live_game(ws: WebSocket):
    """Run a live game and stream frames to the client."""
    await ws.accept()

    env = AirHockeyEnv(
        agent_dynamics=DelayedDynamics(max_speed=3.0, max_accel=30.0),
        opponent_policy="follow",
        record=True,
        action_dt=1 / 60,
    )
    obs, info = env.reset()

    cfg = env.table_config
    await ws.send_json({
        "type": "config",
        "width": cfg.width,
        "height": cfg.height,
        "puck_radius": cfg.puck_radius,
        "paddle_radius": cfg.paddle_radius,
        "goal_width": cfg.goal_width,
    })

    target_x = cfg.width / 2
    target_y = cfg.height * 0.15

    try:
        while True:
            # Check for client messages (paddle target from mouse)
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                if msg.get("type") == "move":
                    target_x = msg["x"]
                    target_y = msg["y"]
                elif msg.get("type") == "save":
                    recording = env.get_recording()
                    if recording:
                        rec = Recorder()
                        rec._current = recording
                        ts = int(time.time())
                        rec.save(RECORDINGS_DIR / f"game_{ts}.json")
                        await ws.send_json({"type": "saved", "name": f"game_{ts}"})
                elif msg.get("type") == "reset":
                    obs, info = env.reset()
                    target_x = cfg.width / 2
                    target_y = cfg.height * 0.15
            except (TimeoutError, asyncio.TimeoutError):
                pass

            action = np.array([target_x, target_y], dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)

            state = env.engine.state
            await ws.send_json({
                "type": "frame",
                "puck_x": state.puck.x,
                "puck_y": state.puck.y,
                "agent_x": state.paddle_agent.x,
                "agent_y": state.paddle_agent.y,
                "opponent_x": state.paddle_opponent.x,
                "opponent_y": state.paddle_opponent.y,
                "score_agent": state.score_agent,
                "score_opponent": state.score_opponent,
                "time": round(state.time, 2),
            })

            if terminated or truncated:
                await ws.send_json({"type": "game_over", **info})
                # Auto-save
                recording = env.get_recording()
                if recording:
                    rec = Recorder()
                    rec._current = recording
                    ts = int(time.time())
                    rec.save(RECORDINGS_DIR / f"game_{ts}.json")
                obs, info = env.reset()
                target_x = cfg.width / 2
                target_y = cfg.height * 0.15

            await asyncio.sleep(1 / 60)

    except WebSocketDisconnect:
        pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8420)


if __name__ == "__main__":
    main()
