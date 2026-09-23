"""Training/evaluation orchestration; evaluation never mutates learner weights."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .manifest import Corpus, sha256_file
from .observations import FEATURE_DIM, FEATURE_VERSION, FeatureEncoder, public_observation, terminal_reward
from .policy import ActorCritic
from .ppo import PPOConfig, PPOTrainer, Rollout
from .scoring import battle_quality


class Environment(Protocol):
    def reset(self, battle: dict) -> dict: ...
    def step(self, action: int) -> dict: ...


class Prior(Protocol):
    def probabilities(self, observation: dict) -> np.ndarray: ...


class UniformPrior:
    """Explicit no-model baseline. Never a fallback for a failing Laya model."""

    def probabilities(self, observation: dict) -> np.ndarray:
        mask = np.asarray(observation["legal_actions"], dtype=np.float32)
        return mask / mask.sum()


class ShuffledPrior:
    """Seeded semantic-destruction ablation, stable per observable state."""

    def __init__(self, base: Prior, seed: int):
        self.base, self.seed = base, seed

    def probabilities(self, observation: dict) -> np.ndarray:
        probs = checked_prior(self.base, observation).copy()
        visible = {k: v for k, v in observation.items() if k not in {"battle_id", "decision_id"}}
        digest = hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little") ^ self.seed)
        legal = np.flatnonzero(observation["legal_actions"])
        probs[legal] = probs[rng.permutation(legal)]
        return probs


def checked_prior(prior: Prior, obs: dict) -> np.ndarray:
    probs = np.asarray(prior.probabilities(obs), dtype=np.float32)
    mask = np.asarray(obs["legal_actions"], dtype=bool)
    if probs.shape != (10,) or not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Prior must return ten finite nonnegative probabilities")
    if probs[~mask].sum() > 1e-6 or probs[mask].sum() <= 0:
        raise ValueError("Prior assigns mass to illegal actions or no mass to legal actions")
    probs = np.where(mask, np.maximum(probs, 1e-8), 0)
    return probs / probs.sum()


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "residual"
    seed: int = 0
    total_steps: int = 100_000
    rollout_steps: int = 256
    eval_every: int = 5_000
    eval_repeats: int = 1
    max_decisions: int = 500
    hidden_dim: int = 128
    alpha: float = 1.0
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        if self.mode not in {
            "frozen",
            "scratch",
            "residual",
            "gated",
            "uniform_residual",
            "shuffled_residual",
        }:
            raise ValueError("Unsupported experiment mode")
        for name in (
            "total_steps",
            "rollout_steps",
            "eval_every",
            "eval_repeats",
            "max_decisions",
            "hidden_dim",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _tensor(array: Any, *, boolean: bool = False) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.bool if boolean else torch.float32)


def action_distribution(
    policy: ActorCritic | None, features: np.ndarray, prior: np.ndarray, mask: list[bool]
) -> tuple[np.ndarray, float, float]:
    if policy is None:
        return prior, 0.0, 0.0
    with torch.no_grad():
        dist, value, extras = policy(_tensor(features), _tensor(prior), _tensor(mask, boolean=True))
    return dist.probs.numpy(), float(value), float(extras["kl"])


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    # Shared uniform draws provide a reproducible paired evaluation convention.
    legal = np.flatnonzero(probs > 0)
    index = np.searchsorted(np.cumsum(probs[legal], dtype=np.float64), rng.random(), side="right")
    return int(legal[min(int(index), len(legal) - 1)])


def _append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def evaluate(
    env: Environment,
    battles: list[dict],
    prior: Prior,
    policy: ActorCritic | None,
    *,
    seed: int,
    repeats: int = 1,
    max_decisions: int = 500,
    log_path: Path | None = None,
    train_steps: int = 0,
    trace_path: Path | None = None,
) -> list[dict]:
    if not battles or repeats < 1 or max_decisions < 1:
        raise ValueError("Evaluation requires battles and positive limits")
    was_training = policy.training if policy is not None else False
    if policy is not None:
        policy.eval()
    rows = []
    try:
        for battle in battles:
            for repeat in range(repeats):
                key = hashlib.sha256(f"{seed}:{battle['id']}:{repeat}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
                obs = public_observation(env.reset(battle))
                encoder = FeatureEncoder()
                start = time.perf_counter()
                kls, latencies, changes = [], [], []
                decisions = 0
                actions: list[int] = []
                while obs["phase"] != "terminal" and decisions < max_decisions:
                    decision_start = time.perf_counter()
                    features = encoder.encode(obs)
                    base = checked_prior(prior, obs)
                    probs, _, kl = action_distribution(policy, features, base, obs["legal_actions"])
                    latencies.append(1000 * (time.perf_counter() - decision_start))
                    kls.append(kl)
                    changes.append(int(probs.argmax() != base.argmax()))
                    action = sample_action(probs, rng)
                    actions.append(action)
                    if trace_path is not None and battle is battles[0] and repeat == 0:
                        _append(
                            trace_path,
                            {
                                "battle_id": battle["id"],
                                "agent_seed": seed,
                                "train_steps": train_steps,
                                "decision_id": obs["decision_id"],
                                "observation": obs,
                                "prior": base.tolist(),
                                "final": probs.tolist(),
                                "action": action,
                                "prior_kl": kl,
                            },
                        )
                    encoder.record_action(obs, action)
                    obs = public_observation(env.step(action))
                    decisions += 1
                party = obs.get("party", [])
                fractions = [
                    float(mon.get("hp", 0)) / float(mon["max_hp"])
                    for mon in party
                    if mon.get("species") and float(mon.get("max_hp", 0)) > 0
                ]
                final_party_hp_fraction = float(np.mean(fractions)) if fractions else None
                quality = battle_quality(obs["outcome"] if obs["phase"] == "terminal" else "truncated", final_party_hp_fraction, obs["turn"])
                row = {
                    "battle_id": battle["id"],
                    "scenario_group": battle["scenario_group"],
                    "split": battle["split"],
                    "battle_seed": battle["seed"],
                    "agent_seed": seed,
                    "repeat": repeat,
                    "train_steps": train_steps,
                    "outcome": obs["outcome"] if obs["phase"] == "terminal" else "truncated",
                    "return": terminal_reward(obs),
                    "decisions": decisions,
                    "turns": obs["turn"],
                    "distinct_actions": len(set(actions)),
                    "switch_action_rate": float(np.mean([action >= 4 for action in actions])) if actions else 0.0,
                    "final_party_hp_fraction": final_party_hp_fraction,
                    **quality,
                    "wall_seconds": time.perf_counter() - start,
                    "mean_prior_kl": float(np.mean(kls)) if kls else 0,
                    "argmax_override_rate": float(np.mean(changes)) if changes else 0,
                    "decision_ms_p50": float(np.median(latencies)) if latencies else 0,
                    "decision_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0,
                }
                rows.append(row)
                if log_path is not None:
                    _append(log_path, row)
    finally:
        if policy is not None:
            policy.train(was_training)
    return rows


def make_policy(config: ExperimentConfig) -> ActorCritic | None:
    if config.mode == "frozen":
        return None
    mode = config.mode if config.mode in {"scratch", "gated"} else "residual"
    return ActorCritic(FEATURE_DIM, hidden_dim=config.hidden_dim, mode=mode, alpha=config.alpha)


def select_prior(config: ExperimentConfig, prior_source: Prior | None) -> Prior:
    if config.mode in {"scratch", "uniform_residual"}:
        return UniformPrior()
    if prior_source is None:
        raise ValueError("This arm requires a real frozen decision prior")
    return ShuffledPrior(prior_source, config.seed) if config.mode == "shuffled_residual" else prior_source


def train(
    env: Environment,
    corpus: Corpus,
    config: ExperimentConfig,
    output: Path,
    prior_source: Prior | None,
    *,
    provenance: dict | None = None,
) -> Path:
    train_battles, validation = corpus.split("train"), corpus.split("validation")
    if not train_battles or not validation:
        raise ValueError("Training requires separate train and validation battle sets")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)
    policy = make_policy(config)
    trainer = PPOTrainer(policy, config.ppo) if policy is not None else None
    prior = select_prior(config, prior_source)
    metadata = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "config": asdict(config),
        "corpus_sha256": sha256_file(corpus.path),
        "rom_sha256": corpus.rom_sha256,
        "source_revision": corpus.source_revision,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "provenance": provenance or {},
        "trainable_parameters": 0 if policy is None else sum(p.numel() for p in policy.parameters()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rng = np.random.default_rng(config.seed)
    # Same shuffled scenario schedule across arms, independent of action RNG.
    schedule_rng = np.random.default_rng(config.seed)
    schedule: list[dict] = []
    episode = 0
    completed = 0
    steps = 0
    started = time.perf_counter()

    def checkpoint() -> None:
        path = output / f"checkpoint-{steps:09d}.pt"
        torch.save(
            {
                "policy": None if policy is None else policy.state_dict(),
                "config": asdict(config),
                "steps": steps,
                "feature_version": FEATURE_VERSION,
                "corpus_sha256": metadata["corpus_sha256"],
                "rom_sha256": corpus.rom_sha256,
                "provenance": provenance or {},
            },
            path,
        )
        rows = evaluate(
            env,
            validation,
            prior,
            policy,
            seed=config.seed,
            repeats=config.eval_repeats,
            max_decisions=config.max_decisions,
            log_path=output / "evaluation.jsonl",
            train_steps=steps,
            trace_path=output / "decision-traces.jsonl",
        )
        _append(
            output / "progress.jsonl",
            {
                "train_steps": steps,
                "episodes": completed,
                "validation_win_rate": sum(row["outcome"] == "win" for row in rows) / len(rows),
                "truncation_rate": sum(row["outcome"] == "truncated" for row in rows) / len(rows),
                "wall_seconds": time.perf_counter() - started,
            },
        )

    def reset_episode() -> tuple[dict, FeatureEncoder]:
        nonlocal schedule, episode
        if not schedule:
            schedule = [train_battles[int(i)] for i in schedule_rng.permutation(len(train_battles))]
        battle = schedule.pop()
        episode += 1
        observation = public_observation(env.reset(battle))
        if observation["phase"] == "terminal":
            raise ValueError("Training save state must begin at a battle decision, not terminal")
        return observation, FeatureEncoder()

    checkpoint()
    obs, encoder = reset_episode()
    episode_steps = 0
    next_eval = min(config.eval_every, config.total_steps)
    # Evaluation uses the same emulator; run it only at forced rollout boundaries
    # and explicitly end the current episode with a bootstrap before reset.
    while steps < config.total_steps:
        size = min(config.rollout_steps, config.total_steps - steps, next_eval - steps)
        batch: dict[str, list] = {name: [] for name in Rollout.__dataclass_fields__}
        for index in range(size):
            features = encoder.encode(obs)
            base = checked_prior(prior, obs)
            probs, value, _ = action_distribution(policy, features, base, obs["legal_actions"])
            action = sample_action(probs, rng)
            encoder.record_action(obs, action)
            successor = public_observation(env.step(action))
            episode_steps += 1
            steps += 1
            terminated = successor["phase"] == "terminal"
            evaluation_boundary = steps == next_eval
            truncated = not terminated and (episode_steps >= config.max_decisions or evaluation_boundary)
            next_value = 0.0
            if not terminated and policy is not None:
                successor_prior = checked_prior(prior, successor)
                _, next_value, _ = action_distribution(
                    policy, encoder.encode(successor), successor_prior, successor["legal_actions"]
                )
            values = {
                "features": features,
                "priors": base,
                "masks": obs["legal_actions"],
                "actions": action,
                "old_log_probs": float(np.log(probs[action])),
                "values": value,
                "rewards": terminal_reward(successor),
                "terminated": terminated,
                "truncated": truncated,
                "next_values": next_value,
            }
            for name, val in values.items():
                batch[name].append(val)
            if terminated or truncated:
                completed += 1
                _append(
                    output / "episodes.jsonl",
                    {
                        "episode": episode,
                        "train_steps": steps,
                        "battle_id": obs["battle_id"],
                        "decisions": episode_steps,
                        "outcome": successor["outcome"] if terminated else "truncated",
                        "truncation_reason": None
                        if terminated
                        else ("evaluation_boundary" if evaluation_boundary else "decision_limit"),
                    },
                )
                if not evaluation_boundary:
                    obs, encoder = reset_episode()
                    episode_steps = 0
            else:
                obs = successor
        if trainer is not None:
            rollout = Rollout(
                **{
                    name: torch.as_tensor(
                        np.asarray(vals),
                        dtype=(
                            torch.bool
                            if name in {"masks", "terminated", "truncated"}
                            else torch.long
                            if name == "actions"
                            else torch.float32
                        ),
                    )
                    for name, vals in batch.items()
                }
            )
            metrics = trainer.update(rollout)
            _append(output / "updates.jsonl", {"train_steps": steps, **metrics})
        if steps == next_eval:
            checkpoint()
            next_eval = min(next_eval + config.eval_every, config.total_steps)
            if steps < config.total_steps:
                obs, encoder = reset_episode()
                episode_steps = 0
    final = output / f"checkpoint-{steps:09d}.pt"
    (output / "completed.json").write_text(json.dumps({"final_checkpoint": final.name, "train_steps": steps}))
    return final


def load_checkpoint(path: Path) -> tuple[ActorCritic | None, ExperimentConfig, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["feature_version"] != FEATURE_VERSION:
        raise ValueError("Checkpoint uses an incompatible feature version")
    raw = dict(saved["config"])
    raw["ppo"] = PPOConfig(**raw["ppo"])
    config = ExperimentConfig(**raw)
    policy = make_policy(config)
    if policy is not None:
        policy.load_state_dict(saved["policy"])
        policy.eval()
    return policy, config, saved
