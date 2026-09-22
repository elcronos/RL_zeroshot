"""Numerical policy fixtures only; these are not simulated Pokémon battles."""

import pytest
import torch

from rogue_rl.policy import ActorCritic


def inputs():
    features = torch.tensor([[0.2, 0.3, -0.1], [1.0, 0.0, 0.5]])
    prior = torch.tensor([[0.2, 0.3, 0.5, 0, 0, 0, 0, 0, 0, 0], [0.8, 0, 0, 0, 0, 0, 0, 0, 0.2, 0]])
    return features, prior, prior > 0


@pytest.mark.parametrize("mode", ["residual", "gated"])
def test_initial_actor_matches_prior_and_preserves_mask(mode):
    features, prior, mask = inputs()
    policy = ActorCritic(3, mode=mode)
    distribution, values, extras = policy(features, prior, mask)
    torch.testing.assert_close(distribution.probs, prior)
    torch.testing.assert_close(extras["kl"], torch.zeros(2), atol=1e-6, rtol=0)
    assert values.shape == (2,)
    assert (distribution.probs[~mask] == 0).all()
    samples = distribution.sample((100,))
    assert mask.gather(1, samples.T).all()
    loss = distribution.entropy().sum() + extras["kl"].sum() + values.sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_scratch_actor_and_critic_cannot_use_prior():
    features, prior, mask = inputs()
    policy = ActorCritic(3, mode="scratch")
    with torch.no_grad():
        policy.actor.weight.normal_()
    first, first_value, _ = policy(features, prior, mask)
    second, second_value, _ = policy(features, torch.ones_like(prior), mask)
    torch.testing.assert_close(first.probs, second.probs)
    torch.testing.assert_close(first_value, second_value)
    residual = ActorCritic(3, mode="residual")
    assert sum(p.numel() for p in residual.parameters()) == sum(p.numel() for p in policy.parameters())


def test_prior_is_frozen_and_input_is_not_mutated():
    features, prior, mask = inputs()
    original = prior.clone()
    prior.requires_grad_()
    policy = ActorCritic(3)
    dist, value, extras = policy(features, prior, mask)
    (dist.log_prob(torch.tensor([0, 0])).sum() + value.sum() + extras["kl"].sum()).backward()
    assert prior.grad is None
    torch.testing.assert_close(prior, original)


def test_single_legal_action_with_zero_prior_is_finite():
    policy = ActorCritic(3)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[7] = True
    dist, value, extras = policy(torch.zeros(3), torch.zeros(10), mask)
    assert dist.sample().item() == 7
    assert dist.entropy().item() == 0
    assert torch.isfinite(value)
    assert extras["kl"].item() == 0


def test_no_legal_actions_rejected():
    policy = ActorCritic(3)
    with pytest.raises(ValueError, match="legal action"):
        policy(torch.zeros(3), torch.ones(10), torch.zeros(10, dtype=torch.bool))


def test_zero_legal_prior_probability_is_smoothed():
    policy = ActorCritic(3)
    mask = torch.tensor([True, True] + [False] * 8)
    prior = torch.tensor([1.0] + [0.0] * 9)
    dist, _, _ = policy(torch.zeros(3), prior, mask)
    assert dist.probs[1] > 0
    assert (dist.probs[2:] == 0).all()


def test_regularizer_is_forward_kl_to_frozen_prior():
    features, prior, mask = inputs()
    policy = ActorCritic(3)
    with torch.no_grad():
        policy.actor.bias[0] = 1.5
    dist, _, extras = policy(features, prior, mask)
    reference = torch.distributions.Categorical(probs=prior)
    expected = torch.distributions.kl_divergence(dist, reference)
    torch.testing.assert_close(extras["kl"], expected)
    assert (extras["kl"] > 0).all()


@pytest.mark.parametrize("action_dim", [4, 9, 11])
def test_action_contract(action_dim):
    with pytest.raises(ValueError, match="exactly 10"):
        ActorCritic(3, action_dim=action_dim)
