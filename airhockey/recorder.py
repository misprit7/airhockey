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
    reward: float = 0.0
    cumulative_reward: float = 0.0


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
        if not episode:
            return
        # Columnar format: keys once, then arrays of values (much smaller)
        fields = list(asdict(episode[0]).keys())
        columns = {f: [] for f in fields}
        for frame in episode:
            d = asdict(frame)
            for f in fields:
                v = d[f]
                columns[f].append(round(float(v), 4) if isinstance(v, float) else v)
        data = {"fields": fields, "columns": columns}
        Path(path).write_text(json.dumps(data, separators=(",", ":")))

    @staticmethod
    def load(path: str | Path) -> list[FrameData]:
        data = json.loads(Path(path).read_text())
        # Support both row-based (old) and columnar (new) formats
        if isinstance(data, list):
            return [FrameData(**f) for f in data]
        fields = data["fields"]
        columns = data["columns"]
        n = len(columns[fields[0]])
        return [FrameData(**{f: columns[f][i] for f in fields}) for i in range(n)]
