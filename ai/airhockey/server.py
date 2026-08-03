"""FastAPI WebSocket server for real-time visualization and replay."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from airhockey.dynamics import DelayedDynamics, HardwareDynamics, IdealDynamics
from airhockey.env import AirHockeyEnv
from airhockey.recorder import Recorder
from airhockey.vision_service import SERVICE as VISION

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
    files = sorted(RECORDINGS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for f in files:
        entry = {"name": f.stem, "path": f.name, "label": _recording_label(f.stem)}
        # Extract metadata (stage info) if present
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and "metadata" in data:
                entry["metadata"] = data["metadata"]
        except Exception:
            pass
        result.append(entry)
    return result


@app.get("/api/recordings/{filename}")
async def get_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if not path.exists():
        return {"error": "not found"}
    # Replace literal NaN/Infinity/-Infinity tokens with null during parse so
    # the response is JSON-spec compliant (Starlette uses allow_nan=False).
    data = json.loads(path.read_text(), parse_constant=lambda _c: None)
    # Convert columnar format to row format for frontend
    if isinstance(data, dict) and "columns" in data:
        fields = data["fields"]
        columns = data["columns"]
        n = len(columns[fields[0]])
        frames = [{f: columns[f][i] for f in fields} for i in range(n)]
        metadata = data.get("metadata")
        return {"frames": frames, "metadata": metadata}
    # Old row-based format (no metadata)
    return {"frames": data, "metadata": None}


@app.get("/api/geometry")
async def geometry():
    """The canonical geometry, for the browser to draw the state view with.

    Served rather than duplicated in JS on purpose: shared/cdpr_geometry.py
    is the one Python copy of the header, and a fourth transcription in the
    front end is exactly the drift the shared header exists to prevent.
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "shared"))
    import cdpr_geometry as g

    return {
        "grid": {"x": g.GRID_X_MM, "y": g.GRID_Y_MM},
        "rails": {"min_x": g.RAIL_MIN_X, "max_x": g.RAIL_MAX_X,
                  "min_y": g.RAIL_MIN_Y, "max_y": g.RAIL_MAX_Y},
        "centerline_x": g.CENTERLINE_X,
        "workspace": {"min_x": g.WS_MIN_X, "max_x": g.WS_MAX_X,
                      "min_y": g.WS_MIN_Y, "max_y": g.WS_MAX_Y},
        "motors": [{"x": g.MOTOR_X[m], "y": g.MOTOR_Y[m]} for m in range(4)],
        "home": {"x": g.HOME_X, "y": g.HOME_Y},
        "attach_r": g.ATTACH_R_MM,
        "attach_chirality": g.ATTACH_CHIRALITY,
        "nominal_theta_deg": math.degrees(g.MALLET_THETA_RAD),
        "spool_radius": g.SPOOL_RADIUS_MM,
    }


@app.get("/camera/status")
async def camera_status():
    return VISION.status()


@app.post("/camera/{action}")
async def camera_control(action: str):
    """start/stop the camera. It is not started automatically: only one
    process can hold the Spinnaker device, so grabbing it unasked would
    break every other vision tool."""
    if action == "start":
        VISION.start()
        for _ in range(40):                      # let it produce a frame
            if VISION.frame_jpeg() or VISION.error:
                break
            await asyncio.sleep(0.05)
    elif action == "stop":
        VISION.stop()
    else:
        return {"ok": False, "error": f"unknown action {action}"}
    return VISION.status()


@app.get("/camera/stream")
async def camera_stream():
    """MJPEG. An <img> consumes this natively — no polling in the UI."""
    async def frames():
        blank = 0
        while VISION.running and blank < 100:
            jpg = VISION.frame_jpeg()
            if jpg is None:
                blank += 1
                await asyncio.sleep(0.05)
                continue
            blank = 0
            yield (b"--f\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode()
                   + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(1.0 / 15)
    return StreamingResponse(frames(),
                             media_type="multipart/x-mixed-replace; boundary=f")


@app.websocket("/ws/live")
async def live_game(ws: WebSocket):
    """Run a live game and stream frames to the client."""
    await ws.accept()

    use_instant = False
    use_hardware = False
    hardware_dynamics = None
    agent_dynamics = DelayedDynamics(max_speed=5.0, max_accel=60.0, time_constant=0.01)
    env = AirHockeyEnv(
        agent_dynamics=agent_dynamics,
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
            # Drain all pending messages, keeping only the latest mouse position.
            try:
                while True:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                    msg_type = msg.get("type")
                    if msg_type == "move":
                        target_x = msg["x"]
                        target_y = msg["y"]
                    elif msg_type == "save":
                        recording = env.get_recording()
                        if recording:
                            rec = Recorder()
                            rec._current = recording
                            ts = int(time.time())
                            rec.save(RECORDINGS_DIR / f"game_{ts}.json")
                            await ws.send_json({"type": "saved", "name": f"game_{ts}"})
                    elif msg_type == "toggle_physics":
                        use_instant = not use_instant
                        if use_instant:
                            env.agent_dynamics = IdealDynamics()
                        else:
                            env.agent_dynamics = DelayedDynamics(max_speed=5.0, max_accel=60.0, time_constant=0.01)
                        env.agent_dynamics.reset(
                            env.engine.state.paddle_agent.x,
                            env.engine.state.paddle_agent.y,
                        )
                        await ws.send_json({"type": "physics_mode", "instant": use_instant})
                    elif msg_type == "toggle_hardware":
                        use_hardware = not use_hardware
                        if use_hardware:
                            try:
                                # Measure where the mallet ACTUALLY is before
                                # energizing. Assuming it sits at the centre
                                # of the robot half offsets every subsequent
                                # command by however wrong that guess was, so
                                # this fails closed rather than guessing.
                                import sys as _sys
                                from pathlib import Path as _P
                                _sys.path.insert(0, str(
                                    _P(__file__).resolve().parents[2]
                                    / "vision" / "bin"))
                                import track_mallet as _tm

                                import math as _math
                                live = VISION.latest_pose()
                                if live is not None:
                                    mx, my, mth = live
                                else:
                                    mx, my, mth = _tm.measure()
                                cal_pose = (mx, my, _math.degrees(mth))
                                print(f"  HW: paddle measured at "
                                      f"({mx:.1f}, {my:.1f}) mm, "
                                      f"{cal_pose[2]:.2f} deg")
                                hardware_dynamics = HardwareDynamics(
                                    sim_width=cfg.width,
                                    sim_height=cfg.height,
                                    cal_pose_mm=cal_pose,
                                )
                                # Target the paddle where it ALREADY IS, not
                                # the middle of the workspace. Enabling used
                                # to command a move to HOME the instant the
                                # drives came up — so the first thing the rig
                                # ever did was move, before anyone had asked
                                # it to and before there was any chance to
                                # see whether it was going to behave. Now
                                # nothing is commanded until you click, which
                                # also makes "does it misbehave while merely
                                # energized?" a question you can answer.
                                sx, sy = hardware_dynamics._mm_to_sim(mx, my)
                                env.agent_dynamics = hardware_dynamics
                                env.agent_dynamics.reset(sx, sy)
                                target_x = sx
                                target_y = sy
                                print(f"  HW: holding at the measured pose, "
                                      f"sim ({sx:.3f}, {sy:.3f}) — nothing "
                                      f"commanded until you click")
                            except Exception as e:
                                print(f"Hardware connect failed: {e}")
                                use_hardware = False
                                hardware_dynamics = None
                        else:
                            if hardware_dynamics:
                                try:
                                    hardware_dynamics.client.disable()
                                    hardware_dynamics.client.close()
                                except Exception:
                                    pass
                                hardware_dynamics = None
                            env.agent_dynamics = DelayedDynamics(max_speed=5.0, max_accel=60.0, time_constant=0.01)
                            env.agent_dynamics.reset(
                                env.engine.state.paddle_agent.x,
                                env.engine.state.paddle_agent.y,
                            )
                        await ws.send_json({"type": "hardware_mode", "enabled": use_hardware})
                    elif msg_type == "reset":
                        obs, info = env.reset()
                        target_x = cfg.width / 2
                        target_y = cfg.height * 0.15
            except (TimeoutError, asyncio.TimeoutError):
                pass

            # Convert physics coords to normalized [-1, 1] action space
            norm_x = (target_x - env._action_low[0]) / (env._action_high[0] - env._action_low[0]) * 2 - 1
            norm_y = (target_y - env._action_low[1]) / (env._action_high[1] - env._action_low[1]) * 2 - 1
            action = np.array([norm_x, norm_y], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)

            state = env.engine.state
            frame_msg = {
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
            }
            if use_hardware and hardware_dynamics:
                frame_msg["hw_x"] = hardware_dynamics.x
                frame_msg["hw_y"] = hardware_dynamics.y
                hw_x_mm, hw_y_mm = hardware_dynamics.get_hw_position_mm()
                frame_msg["hw_x_mm"] = round(hw_x_mm, 1)
                frame_msg["hw_y_mm"] = round(hw_y_mm, 1)
                frame_msg["hw"] = hardware_dynamics.hw_state()
            await ws.send_json(frame_msg)

            if terminated or truncated:
                await ws.send_json({"type": "game_over", **info})
                obs, info = env.reset()
                target_x = cfg.width / 2
                target_y = cfg.height * 0.15

            await asyncio.sleep(1 / 60)

    except WebSocketDisconnect:
        pass
    finally:
        if hardware_dynamics:
            try:
                hardware_dynamics.client.disable()
                hardware_dynamics.client.close()
            except Exception:
                pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8420)


if __name__ == "__main__":
    main()
