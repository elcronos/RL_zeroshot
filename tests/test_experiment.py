"""Orchestration tests replay protocol fixtures, not Pokémon simulations.

The transcript ends on schedule independently of chosen actions. These tests
verify bookkeeping, boundaries and frozen evaluation, never game performance.
"""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from rogue_rl import experiment
from rogue_rl.experiment import (
    ExperimentConfig,
    ShuffledPrior,
    UniformPrior,
    checked_prior,
    evaluate,
    load_checkpoint,
    make_policy,
    select_prior,
    train,
)
from rogue_rl.manifest import Corpus, sha256_file
from rogue_rl.observations import FEATURE_DIM
from rogue_rl.policy import ActorCritic
from rogue_rl.ppo import PPOConfig


def observation(battle_id, index, *, terminal=False):
    return {
        "battle_id": battle_id,
        "decision_id": index,
        "turn": index,
        "phase": "terminal" if terminal else "action",
        "outcome": "win" if terminal else None,
        "legal_actions": [False] * 10 if terminal else [True, True] + [False] * 8,
        "player": {"hp_fraction": 0.75},
        "opponent": {"hp_fraction": 0.5},
        "party": [],
        "moves": [{"name": "fixture A"}, {"name": "fixture B"}],
    }


class TranscriptFixture:
    """Finite protocol transcript whose output does not model any game rule."""

    def __init__(self, length=2):
        self.length = length
        self.resets, self.actions = [], []

    def reset(self, battle):
        self.battle, self.index = battle, 0
        self.resets.append(battle["id"])
        return observation(battle["id"], 0)

    def step(self, action):
        assert action in {0, 1}
        self.actions.append((self.battle["id"], action))
        self.index += 1
        return observation(self.battle["id"], self.index, terminal=self.index >= self.length)


class FixedPrior:
    def __init__(self):
        self.calls = 0

    @property
    def provenance(self):
        return {"calls_observed": self.calls}

    def probabilities(self, obs):
        self.calls += 1
        return np.asarray([0.8, 0.2] + [0.0] * 8, dtype=np.float32)


def corpus_fixture(tmp_path):
    rows = []
    for i, split in enumerate(("train", "validation", "test")):
        state = tmp_path / f"placeholder-{split}.bin"
        state.write_bytes(f"non-ROM test fixture {split}".encode())
        rows.append(
            {
                "id": split,
                "split": split,
                "scenario_group": split,
                "seed": i,
                "state_path": state.name,
                "state_sha256": sha256_file(state),
            }
        )
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "source_revision": "fixture", "rom_sha256": "0" * 64, "battles": rows}
        )
    )
    return Corpus.load(path)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_training_checkpoint_evaluation_lifecycle(tmp_path, monkeypatch):
    corpus = corpus_fixture(tmp_path)
    config = ExperimentConfig(
        mode="residual",
        total_steps=7,
        rollout_steps=2,
        eval_every=3,
        max_decisions=10,
        hidden_dim=16,
        ppo=PPOConfig(epochs=1, minibatch_size=2),
    )
    batches = []
    original_update = experiment.PPOTrainer.update

    def capture_update(self, rollout):
        batches.append(deepcopy(rollout))
        return original_update(self, rollout)

    monkeypatch.setattr(experiment.PPOTrainer, "update", capture_update)
    env = TranscriptFixture(length=2)
    output = tmp_path / "run"
    final = train(env, corpus, config, output, FixedPrior(), provenance={"fixture": True})
    policy, loaded_config, saved = load_checkpoint(final)
    assert loaded_config == config
    assert saved["steps"] == 7
    assert saved["provenance"]["fixture"] is True
    assert saved["provenance"]["prior"]["calls_observed"] > 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["provenance"]["prior"]["calls_observed"] > 0
    assert metadata["dataset"] == {
        "training_battles": 1,
        "validation_battles": 1,
        "test_battles": 1,
        "training_battles_used": 1,
    }
    assert policy is not None and not policy.training
    assert sum(batch.actions.numel() for batch in batches) == 7
    assert [row["train_steps"] for row in read_rows(output / "progress.jsonl")] == [0, 3, 6, 7]
    assert {row["split"] for row in read_rows(output / "evaluation.jsonl")} == {"validation"}
    assert "test" not in env.resets
    episodes = read_rows(output / "episodes.jsonl")
    assert any(row["truncation_reason"] == "evaluation_boundary" for row in episodes)
    assert sum(int(batch.truncated.sum()) for batch in batches) == 3
    # Initial zero residual is genuinely different from trained weights.
    initial = load_checkpoint(output / "checkpoint-000000000.pt")[0]
    assert any(not torch.equal(a, b) for a, b in zip(initial.parameters(), policy.parameters()))
    before = deepcopy(policy.state_dict())
    rows = evaluate(TranscriptFixture(), corpus.split("test"), FixedPrior(), policy, seed=0, repeats=2)
    assert len(rows) == 2 and {row["split"] for row in rows} == {"test"}
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_evaluation_pairs_frozen_and_initial_residual_and_restores_mode(tmp_path):
    corpus = corpus_fixture(tmp_path)
    policy = ActorCritic(FEATURE_DIM, hidden_dim=16)
    first_env, second_env = TranscriptFixture(4), TranscriptFixture(4)
    evaluate(first_env, corpus.split("test"), FixedPrior(), None, seed=42, repeats=5)
    evaluate(second_env, corpus.split("test"), FixedPrior(), policy, seed=42, repeats=5)
    assert first_env.actions == second_env.actions
    assert policy.training
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_evaluation_timeout_is_recorded_as_truncation(tmp_path):
    corpus = corpus_fixture(tmp_path)
    row = evaluate(
        TranscriptFixture(10), corpus.split("test"), UniformPrior(), None, seed=0, max_decisions=3
    )[0]
    assert row["outcome"] == "truncated" and row["return"] == 0 and row["decisions"] == 3


def test_frozen_training_has_no_optimizer_or_trainable_parameters(tmp_path):
    corpus = corpus_fixture(tmp_path)
    config = ExperimentConfig(mode="frozen", total_steps=2, rollout_steps=2, eval_every=2)
    output = tmp_path / "frozen"
    checkpoint = train(TranscriptFixture(), corpus, config, output, FixedPrior())
    assert load_checkpoint(checkpoint)[0] is None
    assert not (output / "updates.jsonl").exists()
    assert json.loads((output / "metadata.json").read_text())["trainable_parameters"] == 0


def test_prior_selection_never_silently_falls_back():
    for mode in ("frozen", "residual", "gated", "shuffled_residual"):
        with pytest.raises(ValueError, match="real frozen decision prior"):
            select_prior(ExperimentConfig(mode=mode), None)
    for mode in ("scratch", "uniform_residual"):
        assert isinstance(select_prior(ExperimentConfig(mode=mode), None), UniformPrior)
    assert make_policy(ExperimentConfig(mode="scratch")).mode == "scratch"


def test_shuffled_prior_is_stable_without_battle_identifiers():
    prior = ShuffledPrior(FixedPrior(), 17)
    first = observation("one", 1)
    second = deepcopy(first)
    second.update(battle_id="other", decision_id=100)
    probs = prior.probabilities(first)
    np.testing.assert_array_equal(probs, prior.probabilities(second))
    np.testing.assert_allclose(sorted(probs[:2]), [0.2, 0.8])
    assert (probs[2:] == 0).all()


def test_prior_illegal_mass_is_rejected():
    class IllegalPrior:
        def probabilities(self, obs):
            return np.ones(10)

    with pytest.raises(ValueError, match="illegal actions"):
        checked_prior(IllegalPrior(), observation("fixture", 0))
