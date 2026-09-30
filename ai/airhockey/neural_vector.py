"""Persistent simulation workers; no CUDA or hardware inside worker functions."""

import multiprocessing as mp
import traceback

import numpy as np


def _worker(connection, count, options):
    from airhockey.neural_training import NeuralTrainingEnv

    try:
        env = NeuralTrainingEnv(count, **options)
        last_obs = last_opp = None
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                mask = payload
                if mask is None:
                    mask = np.ones(count, bool)
                if mask.any():
                    obs = env.reset(
                        seed=options.get("seed", 0) if last_obs is None else None,
                        mask=mask,
                    )
                    opp = env.opponent_obs()
                    if last_opp is not None:
                        opp[~mask] = last_opp[~mask]
                    last_obs, last_opp = obs, opp
                connection.send((last_obs, last_opp, env.critic_context()))
            elif command == "step":
                action, opponent = payload
                if opponent is not None:
                    env.set_opponent_action(opponent)
                result = env.step(action)
                last_obs = result[0]
                last_opp = env.opponent_obs()
                connection.send((*result, last_opp))
            else:
                raise ValueError(command)
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class ParallelNeuralEnv:
    def __init__(self, n_envs, workers, **options):
        if n_envs % workers:
            raise ValueError("environment count must divide evenly across workers")
        self.n_envs, self.workers = n_envs, workers
        context = mp.get_context("spawn")
        self.connections, self.processes = [], []
        for i in range(workers):
            parent, child = context.Pipe()
            config = dict(options, seed=options.get("seed", 0) + i * 100003)
            process = context.Process(
                target=_worker, args=(child, n_envs // workers, config), daemon=True
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        self._opponent_action = None
        self._opponent_obs = None
        self._critic_context = None

    def receive(self):
        result = [c.recv() for c in self.connections]
        for r in result:
            if isinstance(r, dict) and "error" in r:
                raise RuntimeError(r["error"])
        return result

    def reset(self, *, seed=None, mask=None):
        # Seeds are independent and fixed at construction; no global reseeding.
        parts = [None] * self.workers if mask is None else np.split(mask, self.workers)
        for c, part in zip(self.connections, parts):
            c.send(("reset", part))
        result = self.receive()
        self._opponent_obs = np.concatenate([r[1] for r in result])
        self._critic_context = np.concatenate([r[2] for r in result])
        return np.concatenate([r[0] for r in result])

    def opponent_obs(self):
        return self._opponent_obs

    def critic_context(self):
        return self._critic_context

    def set_opponent_action(self, action):
        self._opponent_action = action

    def step(self, action):
        opponent = (
            [None] * self.workers
            if self._opponent_action is None
            else np.split(self._opponent_action, self.workers)
        )
        for c, part, opp in zip(
            self.connections, np.split(action, self.workers), opponent
        ):
            c.send(("step", (part, opp)))
        result = self.receive()
        info = {k: np.concatenate([r[4][k] for r in result]) for k in result[0][4]}
        self._opponent_obs = np.concatenate([r[5] for r in result])
        self._critic_context = info["critic_context"]
        return *(np.concatenate([r[j] for r in result]) for j in range(4)), info

    def close(self):
        for c in self.connections:
            try:
                c.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for c in self.connections:
            c.close()
