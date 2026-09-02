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

from airhockey.dynamics import (DelayedDynamics, HardwareDynamics,  # noqa: F401
                                ProfileDynamics,
                                IdealDynamics, table_mm_to_sim)
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


# One parse per file per mtime: the list refreshes every 5 s in the UI and a
# recording is ~700 KB of JSON, so re-reading every file on every poll would
# turn a directory of a few hundred games into a seconds-long request.
_REC_CACHE: dict[str, tuple[float, dict]] = {}


def _recording_entry(f: Path) -> dict:
    st = f.stat()
    cached = _REC_CACHE.get(f.name)
    if cached and cached[0] == st.st_mtime:
        return cached[1]
    stem = f.stem
    run, step, opponent = stem, None, None
    if "_step_" in stem:
        run, tail = stem.rsplit("_step_", 1)
        try:
            step = int(tail)
        except ValueError:
            pass
    elif "_vs_" in stem:
        # Benchmark games ("<run>_vs_<opponent>") file under their run
        # rather than each becoming a one-item group of its own.
        run, opponent = stem.rsplit("_vs_", 1)
    entry = {
        "name": stem, "path": f.name, "label": _recording_label(stem),
        "run": run, "step": step,
        "mtime": st.st_mtime,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
        "size": st.st_size,
    }
    try:
        data = json.loads(f.read_text(), parse_constant=lambda _c: None)
        if isinstance(data, dict) and "columns" in data:
            cols = data["columns"]
            n = len(cols[data["fields"][0]])
            entry["frames"] = n
            if n and "time" in cols:
                entry["duration_s"] = round(float(cols["time"][-1]), 1)
            if n and "score_agent" in cols and "score_opponent" in cols:
                entry["score"] = [int(cols["score_agent"][-1]),
                                  int(cols["score_opponent"][-1])]
            if "metadata" in data:
                entry["metadata"] = data["metadata"]
        elif isinstance(data, list):
            entry["frames"] = len(data)
    except Exception:            # noqa: BLE001 — a corrupt file still lists
        pass
    if opponent and "opponent" not in entry.get("metadata", {}):
        entry.setdefault("metadata", {})["opponent"] = opponent
    _REC_CACHE[f.name] = (st.st_mtime, entry)
    return entry


@app.get("/api/recordings")
async def list_recordings():
    """Every recording with the metadata the replay menu groups and sorts
    by: run name, training step, wall-clock date, final score, duration."""
    files = sorted(RECORDINGS_DIR.glob("*.json"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return [_recording_entry(f) for f in files]


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


@app.get("/camera/unproject")
async def camera_unproject(z: float = 0.0):
    """Grid mapping the tracker view to table mm, for the cursor readout."""
    from airhockey.vision_service import unproject_grid
    try:
        return unproject_grid(z=z)
    except Exception as e:                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/api/agents")
async def list_agents():
    """Trained checkpoints the sim tab can load as a player."""
    from airhockey.policy_loader import list_checkpoints
    return list_checkpoints()


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


def _camera_objects():
    """The real puck and the player's mallet, in SIM coordinates.

    Under `cam_` names so they can never be mistaken for the simulated puck.
    Only sent in CONTROL mode, where no world is being simulated and there is
    therefore nothing for them to contradict — drawing a camera puck next to a
    physics puck in sim mode would be two pucks and no way to tell which one
    the game believes in.

    Empty when the camera is not running. The camera is never started
    automatically: only one process can hold the Spinnaker device, so grabbing
    it unasked would break record_puck.py and every other vision tool.
    """
    if not VISION.running:
        return {}
    out = {}
    p = VISION.latest_puck()
    if p:
        sx, sy = table_mm_to_sim(p["x"], p["y"])
        out["cam_puck_x"] = round(sx, 4)
        out["cam_puck_y"] = round(sy, 4)
        out["cam_puck_n"] = p["n"]
    q = VISION.latest_player()
    if q:
        sx, sy = table_mm_to_sim(q["x"], q["y"])
        out["cam_player_x"] = round(sx, 4)
        out["cam_player_y"] = round(sy, 4)
    return out


@app.websocket("/ws/live")
async def live_game(ws: WebSocket):
    """Run a live game and stream frames to the client."""
    await ws.accept()

    use_instant = False
    use_hardware = False
    ui_mode = "control"     # "control" | "sim"; replay never reaches here

    # Who drives each paddle in SIM mode. Each side is one of:
    #   {"kind": "human"}                       the mouse
    #   {"kind": "rule",  "rule": "idle"|...}   a scripted policy
    #   {"kind": "agent", "run": "<run name>"}  a trained checkpoint
    # Any pairing goes: play the robot against a checkpoint, watch two
    # checkpoints fight, put a rule on either side. Control mode ignores all
    # of this -- there is no world there, only the machine.
    players = {
        "agent": {"kind": "human"},
        "opponent": {"kind": "rule", "rule": "follow"},
    }
    # Loaded checkpoints, cached per run name: loading costs seconds (torch,
    # CUDA, weights), switching back and forth should not.
    policy_cache: dict[str, object] = {}
    agent_t0 = True     # first act() after a load/reset warm-starts nothing
    opp_t0 = True

    def _policy(run_name):
        return policy_cache[run_name]
    hardware_dynamics = None
    # The UI drives the SAME control law the Teensy runs, so what you see
    # dragging the mouse is what the machine will do -- jerk-limited, with the
    # firmware's parking rule. A first-order lag looked close and was not.
    agent_dynamics = ProfileDynamics()
    env = AirHockeyEnv(
        agent_dynamics=agent_dynamics,
        opponent_policy="follow",
        record=True,
        # 100 Hz: the measured 7.7 ms loop latency is shorter than a 60 Hz
        # step, so at 60 the delay cannot be represented at all.
        action_dt=1 / 100,
        # A moving start, as in training: the robot's box does not include
        # the table centre, so a puck parked dead centre is unreachable and
        # the game never begins. Also re-enables the stuck-puck relaunch.
        still_puck=False,
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
                            env.agent_dynamics = ProfileDynamics()
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
                            env.agent_dynamics = ProfileDynamics()
                            env.agent_dynamics.reset(
                                env.engine.state.paddle_agent.x,
                                env.engine.state.paddle_agent.y,
                            )
                        await ws.send_json({"type": "hardware_mode", "enabled": use_hardware})
                    elif msg_type == "set_limits":
                        hd = hardware_dynamics
                        if not hd:
                            await ws.send_json({"type": "limits",
                                                "error": "hardware is off"})
                        else:
                            try:
                                r = hd.set_limits(
                                    float(msg.get("speed", hd.speed)),
                                    float(msg.get("accel", 400.0)))
                                await ws.send_json({"type": "limits", **r})
                            except Exception as e:      # noqa: BLE001
                                await ws.send_json({"type": "limits",
                                                    "error": str(e)})
                    elif msg_type == "reset_peaks":
                        if hardware_dynamics:
                            try:
                                hardware_dynamics.reset_peaks()
                            except Exception as e:      # noqa: BLE001
                                print(f"reset_peaks failed: {e}")
                    elif msg_type == "set_mode":
                        # "control" = human driving the machine, no world.
                        # "sim" = the full game. Replay never reaches here.
                        ui_mode = ("control" if msg.get("mode") == "control"
                                   else "sim")
                    elif msg_type == "set_players":
                        new_players = {
                            "agent": msg.get("agent") or players["agent"],
                            "opponent": (msg.get("opponent")
                                         or players["opponent"]),
                        }
                        # Load any checkpoints first, off the event loop --
                        # torch + CUDA + weights cost seconds, and a stalled
                        # loop here would freeze the canvas mid-switch.
                        err = None
                        for side_cfg in new_players.values():
                            run = side_cfg.get("run")
                            if (side_cfg.get("kind") == "agent"
                                    and run not in policy_cache):
                                try:
                                    from airhockey.policy_loader import load_agent
                                    policy_cache[run] = await asyncio.to_thread(
                                        load_agent, run)
                                except Exception as e:  # noqa: BLE001
                                    err = f"{run}: {type(e).__name__}: {e}"
                        if err:
                            await ws.send_json({"type": "players",
                                                "error": err, **players})
                        else:
                            players = new_players
                            # The opponent slot of the ENV: rules run inside
                            # _opponent_action; mouse and checkpoints both
                            # come in as external targets.
                            opp = players["opponent"]
                            env.opponent_policy = (opp["rule"]
                                                   if opp["kind"] == "rule"
                                                   else "external")
                            agent_t0 = opp_t0 = True
                            # Park the mouse cursor in whichever half it now
                            # drives, so the switch does not command a dash
                            # from wherever the last click landed.
                            if players["opponent"]["kind"] == "human":
                                target_x, target_y = cfg.width / 2, cfg.height * 0.85
                            else:
                                target_x, target_y = cfg.width / 2, cfg.height * 0.15
                            await ws.send_json({"type": "players", **players})
                    elif msg_type == "sim_limits":
                        # The SIM robot's caps, not the hardware's ("limits"
                        # is the message that reaches the Teensy). Applied to
                        # the live dynamics object; the obs cap features read
                        # the same attributes, so a loaded policy is TOLD the
                        # machine it is now driving.
                        dyn = env.agent_dynamics
                        if msg.get("speed") is not None:
                            dyn.max_speed = max(0.1, float(msg["speed"]))
                        if msg.get("accel") is not None:
                            dyn.max_accel = max(0.5, float(msg["accel"]))
                        await ws.send_json({
                            "type": "sim_limits",
                            "speed": getattr(dyn, "max_speed", None),
                            "accel": getattr(dyn, "max_accel", None),
                        })
                    elif msg_type == "reset":
                        obs, info = env.reset()
                        target_x = cfg.width / 2
                        # Back to the half the mouse drives.
                        target_y = (cfg.height * 0.85
                                    if players["opponent"]["kind"] == "human"
                                    else cfg.height * 0.15)
                        agent_t0 = opp_t0 = True
            except (TimeoutError, asyncio.TimeoutError):
                pass

            terminated = truncated = False
            if ui_mode == "control":
                # No physics at all. A human driving the machine does not
                # want a simulated puck in the way, and it is not merely
                # cosmetic: a goal calls env.reset(), which repositions the
                # agent paddle and would command the hardware to move on its
                # own. Drive the dynamics directly and skip the world.
                # Clamp to the machine's reachable box BEFORE the dynamics.
                # This path bypasses the env's action space entirely, so it
                # was the one place the mouse could still drag the paddle
                # somewhere the robot cannot go -- and with hardware enabled
                # the display then disagreed with the machine, which clamps
                # in _sim_to_mm regardless.
                tx = min(max(target_x, env._ws["min_x"]), env._ws["max_x"])
                ty = min(max(target_y, env._ws["min_y"]), env._ws["max_y"])
                ax, ay = env.agent_dynamics.update(tx, ty, env.action_dt)
                frame_msg = {
                    "type": "frame",
                    "control": True,
                    "agent_x": ax,
                    "agent_y": ay,
                }
                if use_hardware and hardware_dynamics:
                    frame_msg["hw_x"] = hardware_dynamics.x
                    frame_msg["hw_y"] = hardware_dynamics.y
                    hx, hy = hardware_dynamics.get_hw_position_mm()
                    frame_msg["hw_x_mm"] = round(hx, 1)
                    frame_msg["hw_y_mm"] = round(hy, 1)
                    frame_msg["hw"] = hardware_dynamics.hw_state()
                    frame_msg["hw_ws"] = hardware_dynamics.workspace_in_sim()
                frame_msg["hw_ws"] = env._ws     # draw the limit always, not
                frame_msg.update(_camera_objects())   # only in hardware mode
                await ws.send_json(frame_msg)
                await asyncio.sleep(1 / 60)
                continue

            # ── ROBOT side: mouse, rule, or checkpoint ───────────────────
            akind = players["agent"]["kind"]
            if akind == "agent":
                import torch
                with torch.no_grad():
                    a = _policy(players["agent"]["run"]).act(
                        torch.from_numpy(obs).float(),
                        t0=agent_t0, eval_mode=True)
                agent_t0 = False
                action = a.numpy().astype(np.float32)
            else:
                if akind == "human":
                    agent_target = (target_x, target_y)
                else:
                    rule = players["agent"].get("rule", "idle")
                    pa = env.engine.state.paddle_agent
                    if rule == "goalie":
                        # Guard the goal mouth from the closest reachable
                        # line -- the box floor, since the machine cannot
                        # stand on its own goal line.
                        agent_target = (cfg.width / 2, env._ws["min_y"])
                    elif rule == "follow":
                        agent_target = (env.engine.state.puck.x,
                                        env._ws["min_y"] + 0.08)
                    else:               # idle: hold station
                        agent_target = (pa.x, pa.y)
                # Convert physics coords to normalized [-1, 1] action space
                norm_x = (agent_target[0] - env._action_low[0]) / (env._action_high[0] - env._action_low[0]) * 2 - 1
                norm_y = (agent_target[1] - env._action_low[1]) / (env._action_high[1] - env._action_low[1]) * 2 - 1
                action = np.clip(np.array([norm_x, norm_y], dtype=np.float32),
                                 -1.0, 1.0)

            # ── HUMAN side: mouse or checkpoint (rules run inside the env) ─
            okind = players["opponent"]["kind"]
            if okind == "human":
                # NOT bound by the robot's workspace -- a hand can reach its
                # own goal line and the machine cannot -- so clamped only by
                # the table and the halfway line.
                r = cfg.paddle_radius
                env._external_opponent_target = (
                    min(max(target_x, r), cfg.width - r),
                    min(max(target_y, cfg.height / 2 + r), cfg.height - r),
                )
            elif okind == "agent":
                import torch
                with torch.no_grad():
                    oa = _policy(players["opponent"]["run"]).act(
                        torch.from_numpy(env.mirror_obs(obs)).float(),
                        t0=opp_t0, eval_mode=True)
                opp_t0 = False
                tx, ty = env.mirror_action_to_opponent(oa.numpy())
                env.set_opponent_action(tx, ty)

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
                # Which half the MOUSE drives, for the client's clamp.
                "side": ("human" if players["opponent"]["kind"] == "human"
                         else "robot"),
            }
            if use_hardware and hardware_dynamics:
                frame_msg["hw_x"] = hardware_dynamics.x
                frame_msg["hw_y"] = hardware_dynamics.y
                hw_x_mm, hw_y_mm = hardware_dynamics.get_hw_position_mm()
                frame_msg["hw_x_mm"] = round(hw_x_mm, 1)
                frame_msg["hw_y_mm"] = round(hw_y_mm, 1)
                frame_msg["hw"] = hardware_dynamics.hw_state()
                frame_msg["hw_ws"] = hardware_dynamics.workspace_in_sim()
            await ws.send_json(frame_msg)

            if terminated or truncated:
                await ws.send_json({"type": "game_over", **info})
                obs, info = env.reset()
                agent_t0 = opp_t0 = True
                target_x = cfg.width / 2
                # Back to the half the mouse drives.
                target_y = (cfg.height * 0.85
                            if players["opponent"]["kind"] == "human"
                            else cfg.height * 0.15)

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
