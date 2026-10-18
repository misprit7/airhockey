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
            elif command in ("step", "step_reset"):
                action, opponent = payload
                if opponent is not None:
                    env.set_opponent_action(opponent)
                result = env.step(action)
                last_obs = result[0]
                last_opp = env.opponent_obs()
                reply = (*result, last_opp)
                if command == "step_reset":
                    # Exactly the calls a following ("reset", term|trunc)
                    # command makes, in the same order, without the round trip.
                    done = result[2] | result[3]
                    restarted = None
                    if done.any():
                        obs = env.reset(mask=done)
                        opp = env.opponent_obs()
                        opp[~done] = last_opp[~done]
                        last_obs, last_opp = obs, opp
                        restarted = (obs, opp, env.critic_context())
                    reply = (*reply, restarted)
                connection.send(reply)
            else:
                raise ValueError(command)
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class ParallelNeuralEnv:
    """NeuralTrainingEnv split across spawn workers.

    ``auto_reset=True`` makes each worker reset its finished environments
    right after the step that ended them. The caller still calls
    ``reset(mask=terminated | truncated)`` after every step with a finished
    environment, as with ``auto_reset=False``; that call then returns the
    worker's result without another round trip. Observations, rewards and
    random streams are identical either way.
    ``step_async``/``step_wait`` let the caller work while the workers step.
    """

    # Process entry point. Subclasses with a different message protocol
    # supply their own and override reset/step (airhockey.distillation).
    worker = staticmethod(_worker)

    def __init__(self, n_envs, workers, auto_reset=False, **options):
        if n_envs % workers:
            raise ValueError("environment count must divide evenly across workers")
        self.n_envs, self.workers = n_envs, workers
        self.auto_reset = bool(auto_reset)
        context = mp.get_context("spawn")
        self.connections, self.processes = [], []
        for i in range(workers):
            parent, child = context.Pipe()
            config = dict(options, seed=options.get("seed", 0) + i * 100003)
            process = context.Process(
                target=self.worker, args=(child, n_envs // workers, config), daemon=True
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        self._opponent_action = None
        self._opponent_obs = None
        self._critic_context = None
        self._stepping = False
        self._restarted = None

    def receive(self):
        result = [c.recv() for c in self.connections]
        for r in result:
            if isinstance(r, dict) and "error" in r:
                raise RuntimeError(r["error"])
        return result

    def reset(self, *, seed=None, mask=None):
        if self._stepping:
            raise RuntimeError("reset while a step is in flight")
        if self._restarted is not None:
            done, obs, opp, context = self._restarted
            if mask is None or not np.array_equal(np.asarray(mask, bool), done):
                raise RuntimeError("auto-reset environments must be reset with the step's done mask")
            self._restarted = None
            self._opponent_obs, self._critic_context = opp, context
            return obs
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

    def step_async(self, action):
        if self._stepping:
            raise RuntimeError("a step is already in flight")
        if self._restarted is not None:
            raise RuntimeError("finished environments were not reset after the previous step")
        opponent = (
            [None] * self.workers
            if self._opponent_action is None
            else np.split(self._opponent_action, self.workers)
        )
        command = "step_reset" if self.auto_reset else "step"
        for c, part, opp in zip(
            self.connections, np.split(action, self.workers), opponent
        ):
            c.send((command, (part, opp)))
        self._stepping = True

    def step_wait(self):
        if not self._stepping:
            raise RuntimeError("no step in flight")
        self._stepping = False
        result = self.receive()
        info = {k: np.concatenate([r[4][k] for r in result]) for k in result[0][4]}
        self._opponent_obs = np.concatenate([r[5] for r in result])
        self._critic_context = info["critic_context"]
        out = tuple(np.concatenate([r[j] for r in result]) for j in range(4))
        if self.auto_reset:
            done = out[2] | out[3]
            if done.any():
                # Workers without a finished environment return their
                # post-step arrays, exactly as a "reset" command would.
                count = self.n_envs // self.workers
                parts = [(r[6] if r[6] is not None else
                          (r[0], r[5], info["critic_context"][i * count:(i + 1) * count]))
                         for i, r in enumerate(result)]
                self._restarted = (done, *(np.concatenate([p[j] for p in parts]) for j in range(3)))
        return *out, info

    def step(self, action):
        self.step_async(action)
        return self.step_wait()

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
