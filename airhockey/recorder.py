"""Game recording and replay."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FrameData:
    time: float
    puck_x: float
    puck_y: float
    puck_vx: float
    puck_vy: float
    agent_x: float
    agent_y: float
    opponent_x: float
    opponent_y: float
    score_agent: int
    score_opponent: int


class Recorder:
    """Records game frames for later replay."""

    def __init__(self) -> None:
        self._episodes: list[list[FrameData]] = []
        self._current: list[FrameData] = []

    def start_episode(self) -> None:
        if self._current:
            self._episodes.append(self._current)
        self._current = []

    def record(self, frame: FrameData) -> None:
        self._current.append(frame)

    def get_episode(self, index: int = -1) -> list[FrameData]:
        if index == -1 and self._current:
            return self._current
        return self._episodes[index]

    def save(self, path: str | Path, episode_index: int = -1) -> None:
        episode = self.get_episode(episode_index)
        data = [asdict(f) for f in episode]
        Path(path).write_text(json.dumps(data))

    @staticmethod
    def load(path: str | Path) -> list[FrameData]:
        data = json.loads(Path(path).read_text())
        return [FrameData(**f) for f in data]
