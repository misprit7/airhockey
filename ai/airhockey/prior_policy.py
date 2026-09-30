"""Optional simulation inference acceleration for frozen deterministic priors."""

import torch


class FrozenPrior:
    """Evaluate only the actor mean; retain the original policy for MPC/sampling.

    Freezing is a caller contract: compiled graphs must not be used to update
    the underlying actor while they execute. No policy weights are changed.
    """

    def __init__(self, policy, *, compile=True):
        self.policy, self.cfg = policy, policy.cfg
        self.device = policy.device
        self.training_algorithm = getattr(policy, "training_algorithm", "Frozen prior")
        self.inference_mode = "prior"

        def mean(observation):
            return policy.model._pi(policy.model.encode(observation, None))[
                ..., : self.cfg.action_dim
            ].tanh()

        self.mean = torch.compile(mean, mode="reduce-overhead") if compile else mean

    @property
    def _prev_mean_batch(self):
        return self.policy._prev_mean_batch

    @_prev_mean_batch.setter
    def _prev_mean_batch(self, value):
        self.policy._prev_mean_batch = value

    @torch.no_grad()
    def act(self, observation, t0=False, eval_mode=True):
        if self.cfg.mpc or not eval_mode:
            return self.policy.act(observation, t0=t0, eval_mode=eval_mode)
        single = observation.ndim == 1
        obs = observation[None] if single else observation
        torch.compiler.cudagraph_mark_step_begin()
        action = self.mean(obs.to(self.device)).cpu()
        return action[0] if single else action
