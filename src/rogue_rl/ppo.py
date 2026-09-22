"""PPO with frozen-prior KL and time-limit-correct generalized advantages."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .policy import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.05
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("epochs and minibatch_size must be positive")
        for name in ("learning_rate", "clip_coef", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("value_coef", "entropy_coef", "kl_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class Rollout:
    """Time-major tensors, optionally with a parallel-environment dimension.

    next_values[t] MUST value the actual successor/final observation before an
    automatic reset. A time limit bootstraps from that value; a terminal battle
    does not. Both boundaries stop GAE from leaking into the next episode.
    """

    features: Tensor
    priors: Tensor
    masks: Tensor
    actions: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    next_values: Tensor


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    if rewards.ndim < 1 or rewards.shape[0] == 0:
        raise ValueError("A rollout must contain at least one time step")
    if any(t.shape != rewards.shape for t in (values, next_values, terminated, truncated)):
        raise ValueError("GAE tensors must have matching time-major shapes")
    terminated, truncated = terminated.bool(), truncated.bool()
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for t in reversed(range(rewards.shape[0])):
        bootstrap = (~terminated[t]).to(values.dtype)
        continuation = (~(terminated[t] | truncated[t])).to(values.dtype)
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, policy: ActorCritic, config: PPOConfig | None = None):
        self.policy = policy
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.generator = torch.Generator().manual_seed(self.config.seed)

    def update(self, rollout: Rollout) -> dict[str, float]:
        cfg = self.config
        shape = rollout.rewards.shape
        if any(
            getattr(rollout, name).shape != shape
            for name in ("actions", "old_log_probs", "values", "terminated", "truncated", "next_values")
        ):
            raise ValueError("Rollout scalar tensors must have matching shapes")
        if rollout.features.shape != (*shape, self.policy.feature_dim):
            raise ValueError("Rollout feature shape is inconsistent")
        if any(t.shape != (*shape, self.policy.action_dim) for t in (rollout.priors, rollout.masks)):
            raise ValueError("Rollout prior/mask shape is inconsistent")
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.next_values,
            rollout.terminated,
            rollout.truncated,
            cfg.gamma,
            cfg.gae_lambda,
        )
        advantages, returns = advantages.flatten().detach(), returns.flatten().detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        features = rollout.features.detach().reshape(-1, self.policy.feature_dim)
        priors = rollout.priors.detach().reshape(-1, self.policy.action_dim)
        masks = rollout.masks.detach().reshape(-1, self.policy.action_dim)
        actions = rollout.actions.detach().flatten().long()
        old_log_probs = rollout.old_log_probs.detach().flatten()
        count = actions.numel()
        if (actions < 0).any() or (actions >= self.policy.action_dim).any():
            raise ValueError("Rollout action is outside the 10-slot action space")
        if not masks.gather(1, actions[:, None]).bool().all():
            raise ValueError("Rollout contains an illegal action")
        totals: dict[str, float] = {}
        weight = 0
        self.policy.train()
        for _ in range(cfg.epochs):
            order = torch.randperm(count, generator=self.generator).to(features.device)
            for start in range(0, count, cfg.minibatch_size):
                indices = order[start : start + cfg.minibatch_size]
                distribution, values, extras = self.policy(features[indices], priors[indices], masks[indices])
                log_ratio = distribution.log_prob(actions[indices]) - old_log_probs[indices]
                ratio = log_ratio.exp()
                surrogate = ratio * advantages[indices]
                clipped = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages[indices]
                policy_loss = -torch.minimum(surrogate, clipped).mean()
                value_loss = (values - returns[indices]).square().mean()
                entropy = distribution.entropy().mean()
                prior_kl = extras["kl"].mean()
                # Scratch is the unregularized matched-capacity PPO baseline.
                prior_coef = 0.0 if self.policy.mode == "scratch" else cfg.kl_coef
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                    + prior_coef * prior_kl
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "prior_kl": prior_kl,
                    "grad_norm": grad_norm,
                    "approx_kl": ((ratio - 1) - log_ratio).mean(),
                    "clip_fraction": ((ratio - 1).abs() > cfg.clip_coef).float().mean(),
                }
                size = indices.numel()
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
                weight += size
        return {name: value / weight for name, value in totals.items()}
