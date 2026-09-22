"""Masked categorical actor/critic; slots 0:4 moves and 4:10 party switches."""

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Small MLP correction to a frozen, externally supplied action prior.

    Priors are probabilities, never logits. The residual actor starts exactly at
    the normalized legal prior (zero legal probabilities receive a 1e-8 floor).
    Scratch has identical capacity but replaces every prior input with uniform
    legal probabilities and uses correction scale 1. No Laya information reaches
    its actor or critic. Optional ``gated`` is a residual-logit gate ablation,
    not a mixture of independently normalized policies.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int = 10,
        hidden_dim: int = 128,
        mode: str = "residual",
        alpha: float = 1.0,
    ):
        super().__init__()
        if action_dim != 10:
            raise ValueError("Emerald Rogue uses exactly 10 action slots")
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if mode not in {"scratch", "residual", "gated"}:
            raise ValueError(f"Unknown policy mode: {mode}")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.mode = mode
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + 2 * action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 1) if mode == "gated" else None
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self, features: Tensor, prior: Tensor, mask: Tensor
    ) -> tuple[Categorical, Tensor, dict[str, Tensor]]:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("Wrong feature dimension")
        if prior.shape != mask.shape or prior.shape[-1] != self.action_dim:
            raise ValueError("Prior and mask must have matching 10-slot shapes")
        if features.shape[:-1] != prior.shape[:-1]:
            raise ValueError("Features and priors must have matching batch dimensions")
        mask = mask.bool()
        if not mask.any(dim=-1).all():
            raise ValueError("Every decision must have at least one legal action")
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")
        if not torch.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("Priors must be finite nonnegative probabilities")
        features = features.to(dtype=self.actor.weight.dtype)
        uniform = mask.to(features.dtype) / mask.sum(dim=-1, keepdim=True)
        if self.mode == "scratch":
            prior_probs = uniform
        else:
            prior_probs = prior.detach().to(features.dtype).clamp_min(1e-8) * mask
            prior_probs = prior_probs / prior_probs.sum(dim=-1, keepdim=True)
        hidden = self.encoder(torch.cat((features, prior_probs, mask.to(features.dtype)), dim=-1))
        delta = self.actor(hidden)
        gate = (
            self.gate(hidden).sigmoid().squeeze(-1)
            if self.gate is not None
            else torch.ones_like(delta[..., 0])
        )
        log_prior = prior_probs.clamp_min(1e-8).log()
        scale = 1.0 if self.mode == "scratch" else self.alpha
        logits = log_prior + scale * gate.unsqueeze(-1) * delta
        distribution = Categorical(logits=logits.masked_fill(~mask, -torch.inf))
        log_ratio = distribution.logits.masked_fill(~mask, 0) - log_prior.masked_fill(~mask, 0)
        kl = (distribution.probs * log_ratio).sum(dim=-1)
        return (
            distribution,
            self.critic(hidden).squeeze(-1),
            {
                "prior_probs": prior_probs,
                "kl": kl,
                "gate": gate,
            },
        )
