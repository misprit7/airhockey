"""Episode-safe replay with O(batch) sampling, independent of buffer capacity.

Episodes are appended as array blocks, not per-step TensorDict objects.
CPU generation/episode IDs validate sequence boundaries before GPU gathers.
No sequence can cross a terminal reset or the ring's overwrite frontier.
"""

import numpy as np
import torch


class SequenceReplay:
    def __init__(
        self,
        capacity,
        obs_dim,
        action_dim,
        horizon,
        batch_size,
        device="cuda",
        seed=0,
        terminal_fraction=0.0,
    ):
        if capacity <= horizon + 1:
            raise ValueError("capacity must exceed sequence length")
        self.capacity = capacity
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.data = {
            k: torch.empty((capacity, *shape), dtype=torch.float32, device=self.device)
            for k, shape in dict(
                obs=(obs_dim,), action=(action_dim,), reward=(), terminated=(), demo=()
            ).items()
        }
        self.episode = np.full(capacity, -1, np.int64)
        self.generation = np.full(capacity, -capacity, np.int64)
        self.written = 0
        self.num_eps = 0
        self.size = 0
        if not 0 <= terminal_fraction <= 1:
            raise ValueError("terminal fraction must be in [0,1]")
        self.terminal_fraction = terminal_fraction
        self.terminal_starts = np.empty(capacity // (horizon + 1) + 2, np.int64)
        self.terminal_head = self.terminal_tail = 0

    def add(self, episode):
        n = len(episode["obs"])
        if n <= self.horizon:
            return
        if n > self.capacity:
            raise ValueError("episode exceeds replay capacity")
        idx = (np.arange(n) + self.written) % self.capacity
        ti = torch.as_tensor(idx, device=self.device)
        for key, dest in self.data.items():
            v = np.asarray(episode[key], dtype=np.float32)
            dest[ti] = torch.as_tensor(v, device=self.device)
        self.episode[idx] = self.num_eps
        self.generation[idx] = np.arange(self.written, self.written + n)
        self.written += n
        self.size = min(self.capacity, self.written)
        self.num_eps += 1
        # Ring of terminal-window starts in global generation coordinates.
        # Prune overwritten anchors incrementally; sampling stays O(batch).
        frontier = self.written - self.capacity
        ring = self.terminal_starts
        while (
            self.terminal_head < self.terminal_tail
            and ring[self.terminal_head % len(ring)] < frontier
        ):
            self.terminal_head += 1
        if episode["terminated"][-1]:
            ring[self.terminal_tail % len(ring)] = self.written - 1 - self.horizon
            self.terminal_tail += 1
            if self.terminal_tail - self.terminal_head > len(ring):
                raise RuntimeError("terminal anchor ring overflow")

    def sample_with_demo(self):
        if not self.num_eps:
            raise RuntimeError("empty replay")
        chosen = []
        terminal_n = (
            round(self.batch_size * self.terminal_fraction)
            if self.terminal_tail > self.terminal_head
            else 0
        )
        needed = self.batch_size - terminal_n
        if terminal_n:
            positions = self.rng.integers(
                self.terminal_head, self.terminal_tail, terminal_n
            )
            chosen.append(
                self.terminal_starts[positions % len(self.terminal_starts)]
                % self.capacity
            )
        for _ in range(100):
            if not needed:
                break
            start = self.rng.integers(0, self.size, max(needed * 2, 32))
            end = (start + self.horizon) % self.capacity
            valid = (self.episode[start] == self.episode[end]) & (
                self.generation[end] - self.generation[start] == self.horizon
            )
            take = start[valid][:needed]
            chosen.append(take)
            needed -= len(take)
            if not needed:
                break
        if needed:
            raise RuntimeError("not enough intact replay sequences")
        start = np.concatenate(chosen)
        ix = (start[None, :] + np.arange(self.horizon + 1)[:, None]) % self.capacity
        ix = torch.as_tensor(ix, device=self.device)
        d = {k: v[ix] for k, v in self.data.items()}
        return (
            d["obs"],
            d["action"][1:],
            d["reward"][1:, :, None],
            d["terminated"][1:, :, None],
            None,
            d["demo"][1:, :, None],
        )


class EpisodeBatch:
    def __init__(self, n, obs_dim, action_dim, max_steps=1500):
        self.n = n
        self.max_steps = max_steps
        self.data = {
            k: np.empty((max_steps + 1, n, *shape), np.float32)
            for k, shape in dict(
                obs=(obs_dim,), action=(action_dim,), reward=(), terminated=(), demo=()
            ).items()
        }
        self.length = np.zeros(n, dtype=int)
        self.idx = np.arange(n)

    def reset(self, obs, mask=None):
        mask = np.ones(self.n, bool) if mask is None else mask
        self.length[mask] = 0
        self.data["obs"][0, mask] = obs[mask]
        for k in ("action", "reward", "terminated", "demo"):
            self.data[k][0, mask] = 0

    def append(self, obs, action, reward, terminated, demo=0.0):
        self.length += 1
        if np.any(self.length > self.max_steps):
            raise RuntimeError("episode staging overflow")
        for k, v in dict(
            obs=obs, action=action, reward=reward, terminated=terminated, demo=demo
        ).items():
            self.data[k][self.length, self.idx] = v

    def episode_at(self, i):
        return {k: v[: self.length[i] + 1, i].copy() for k, v in self.data.items()}
