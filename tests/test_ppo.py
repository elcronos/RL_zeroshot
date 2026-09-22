"""PPO mathematical tests and an explicitly synthetic categorical fixture."""

from copy import deepcopy

import pytest
import torch

from rogue_rl.policy import ActorCritic
from rogue_rl.ppo import PPOConfig, PPOTrainer, Rollout, compute_gae


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_gae_bootstraps_truncation_but_stops_across_reset():
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 2.0, 10.0]),
        values=torch.tensor([2.0, 3.0, 5.0]),
        next_values=torch.tensor([3.0, 4.0, 999.0]),
        terminated=torch.tensor([False, False, True]),
        truncated=torch.tensor([False, True, False]),
        gamma=0.9,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([4.04, 2.6, 5.0]))
    torch.testing.assert_close(returns, torch.tensor([6.04, 5.6, 10.0]))


def test_gae_rollout_end_bootstraps_and_supports_parallel_envs():
    advantages, _ = compute_gae(
        rewards=torch.tensor([[1.0, 2.0]]),
        values=torch.tensor([[0.5, 1.0]]),
        next_values=torch.tensor([[3.0, 100.0]]),
        terminated=torch.tensor([[False, True]]),
        truncated=torch.tensor([[False, False]]),
        gamma=0.9,
    )
    torch.testing.assert_close(advantages, torch.tensor([[3.2, 1.0]]))


def categorical_fixture(policy, count=128):
    """One-step numerical bandit: action 0 pays +1; action 1 pays -1."""
    features = torch.ones(count, policy.feature_dim)
    priors = torch.zeros(count, 10)
    priors[:, :2] = 0.5
    masks = priors > 0
    with torch.no_grad():
        dist, values, _ = policy(features, priors, masks)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)
    return Rollout(
        features,
        priors,
        masks,
        actions,
        old_log_probs,
        values,
        torch.where(actions == 0, 1.0, -1.0),
        torch.ones(count, dtype=torch.bool),
        torch.zeros(count, dtype=torch.bool),
        torch.zeros(count),
    )


def test_ppo_update_is_finite_and_preserves_rollout_priors():
    torch.manual_seed(15)
    policy = ActorCritic(3)
    trainer = PPOTrainer(policy, PPOConfig(epochs=2, minibatch_size=13))
    rollout = categorical_fixture(policy, 31)
    saved_prior = rollout.priors.clone()
    before = deepcopy(policy.state_dict())
    metrics = trainer.update(rollout)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    torch.testing.assert_close(rollout.priors, saved_prior)
    assert any(not torch.equal(before[name], value) for name, value in policy.state_dict().items())


def test_seeded_minibatches_reproduce_update():
    torch.manual_seed(42)
    first = ActorCritic(3, hidden_dim=16)
    second = deepcopy(first)
    rollout = categorical_fixture(first, 32)
    config = PPOConfig(epochs=2, minibatch_size=8, seed=77)
    PPOTrainer(first, config).update(rollout)
    PPOTrainer(second, config).update(rollout)
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["scratch", "residual"])
def test_ppo_learns_synthetic_categorical_fixture(mode):
    """Optimizer smoke only; this provides no evidence of Pokémon performance."""
    torch.manual_seed(123)
    policy = ActorCritic(3, hidden_dim=16, mode=mode)
    trainer = PPOTrainer(policy, PPOConfig(learning_rate=0.003, epochs=4, minibatch_size=64))
    for _ in range(8):
        trainer.update(categorical_fixture(policy))
    with torch.no_grad():
        example = categorical_fixture(policy, 1)
        dist, _, _ = policy(example.features, example.priors, example.masks)
    assert dist.probs[0, 0] > 0.85
    assert (dist.probs[0, 2:] == 0).all()


def test_ppo_rejects_illegal_rollout_actions():
    policy = ActorCritic(3)
    rollout = categorical_fixture(policy, 8)
    rollout.actions[0] = 7
    with pytest.raises(ValueError, match="illegal action"):
        PPOTrainer(policy).update(rollout)


def test_kl_penalty_pulls_actor_toward_prior_without_reward_signal():
    torch.manual_seed(50)
    policy = ActorCritic(3, hidden_dim=16)
    with torch.no_grad():
        policy.actor.bias[0] = 1.0
        policy.critic.weight.zero_()
        policy.critic.bias.zero_()
    rollout = categorical_fixture(policy, 32)
    rollout.rewards.zero_()
    before = policy(rollout.features, rollout.priors, rollout.masks)[2]["kl"].mean().item()
    trainer = PPOTrainer(
        policy,
        PPOConfig(
            learning_rate=0.01,
            epochs=1,
            minibatch_size=32,
            value_coef=0,
            entropy_coef=0,
            kl_coef=1,
        ),
    )
    trainer.update(rollout)
    after = policy(rollout.features, rollout.priors, rollout.masks)[2]["kl"].mean().item()
    assert after < before
