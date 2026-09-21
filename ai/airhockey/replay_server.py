"""Offline comparison UI. No camera, controller socket, or policy is opened."""

from pathlib import Path
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from airhockey.hardware_replay import list_sessions, load_session, simulate

app = FastAPI()
WEB = Path(__file__).parent / "web" / "compare"
SIM_LOCK = asyncio.Lock()


class SimulationRequest(BaseModel):
    starts: list[float] = Field(min_length=1, max_length=32)
    duration: float = Field(default=10, gt=0, le=30)
    command_offset_ms: float = Field(default=0, ge=-100, le=100, allow_inf_nan=False)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/compare.js")
def script():
    return FileResponse(WEB / "compare.js", media_type="application/javascript")


@app.get("/compare.css")
def style():
    return FileResponse(WEB / "compare.css", media_type="text/css")


@app.get("/api/replays")
def recordings():
    return list_sessions()


@app.get("/api/replays/{name}")
def recording(name: str):
    try:
        return load_session(name)
    except FileNotFoundError:
        raise HTTPException(404, "Recording not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/replays/{name}/simulate")
async def rollout(name: str, request: SimulationRequest):
    try:
        session = await asyncio.to_thread(load_session, name)
        async with SIM_LOCK:
            return await asyncio.to_thread(
                simulate,
                session,
                request.starts,
                request.duration,
                request.command_offset_ms,
            )
    except FileNotFoundError:
        raise HTTPException(404, "Recording not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8422)
