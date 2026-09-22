"""Frozen local CoreML priors; no generated text, training, or implicit fallback."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np

UPSTREAM_SHA = "12b7501583c7f03a6b2e49ebe118a2c6302505b9"
PRESETS = {
    "ane96": (
        "aac6fef/laya-multilingual-coreml-ane",
        "39d6a9b3d0f67f06da74fbade6121ea134cbdb21",
        "cpu_ne",
        96,
    ),
    "general1024": (
        "aac6fef/laya-multilingual-coreml",
        "8139e9089273319512c730218903784074133187",
        "cpu_gpu",
        1024,
    ),
}
TYPE_NAMES = (
    "Normal",
    "Fighting",
    "Flying",
    "Poison",
    "Ground",
    "Rock",
    "Bug",
    "Ghost",
    "Steel",
    "Mystery",
    "Fire",
    "Water",
    "Grass",
    "Electric",
    "Psychic",
    "Ice",
    "Dragon",
    "Dark",
    "Fairy",
)
PROMPT_VERSION = "rogue-public-action-v1"


class LayaCapacityError(ValueError):
    """The complete semantic input cannot fit the selected artifact."""


@dataclass(frozen=True)
class LayaConfig:
    preset: str = "ane96"
    strategy: str = "action_scores"
    model_dir: str | None = None
    cache_size: int = 4096
    probability_floor: float = 1e-6

    def __post_init__(self):
        if self.preset not in PRESETS or self.strategy not in {"action_scores", "joint"}:
            raise ValueError("Unknown Laya preset or strategy")
        if self.cache_size < 0 or not 0 < self.probability_floor < 1:
            raise ValueError("Invalid cache size or probability floor")


class Predictor(Protocol):
    def predict(self, state: str, questions: dict) -> dict: ...
    def token_count(self, state: str, questions: dict) -> int: ...


class CoreMLPredictor:
    """Adapter to pinned laya-coreml 0.1.0. Downloads are setup-only."""

    def __init__(self, config: LayaConfig):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Laya CoreML requires Apple Silicon and macOS 15+")
        try:
            import laya_coreml
        except ImportError as exc:
            raise RuntimeError("Install the Laya extra and run scripts/setup_laya.py first") from exc
        installed = importlib.metadata.version("laya-coreml")
        if installed != "0.1.0":
            raise RuntimeError(f"Expected laya-coreml 0.1.0, got {installed}")
        repo, revision, compute, capacity = PRESETS[config.preset]
        source = config.model_dir or repo
        if config.model_dir:
            receipt = Path(source) / "rogue_laya_provenance.json"
            if not receipt.is_file():
                raise ValueError("Local model lacks setup_laya.py provenance receipt")
            pinned = json.loads(receipt.read_text())
            if (pinned.get("repo_id"), pinned.get("revision")) != (repo, revision):
                raise ValueError("Local model receipt does not match requested pinned preset")
        self.agent = laya_coreml.load(source, revision=revision, local_files_only=True, compute_units=compute)
        if self.agent.shape["max_length"] != capacity:
            raise ValueError("Model capacity does not match pinned preset")
        manifest = Path(self.agent.model_dir) / "coreml_config.json"
        self.provenance = {
            "implementation": "laya-coreml",
            "package_version": installed,
            "source_commit": UPSTREAM_SHA,
            "model_repo": repo,
            "model_revision": revision,
            "compute_units": compute,
            "capacity": capacity,
            "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        }

    def token_count(self, state: str, questions: dict) -> int:
        """Count before upstream's prefix/state truncation can occur."""
        if len(questions) != 1:
            raise ValueError("One question per inference is required")
        from laya_coreml.common import render_options

        q = self.agent._to_internal(next(iter(questions.values())))
        tok = self.agent.tok

        def count(value):
            return len(tok(value.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"])

        head = count(f"{q['t']} question: {q['ins']}")
        options = [1 + count(" " + text) for text in render_options(q)]
        budget = self.agent.cfg.get("head_max_len", 192) - sum(options)
        if any(length > 49 for length in options) or budget < 16 or head > max(8, budget):
            raise LayaCapacityError(
                "Laya would truncate the question/options; simplify semantic names explicitly"
            )
        total = head + sum(options) + count(state) + 4
        capacity = min(self.agent.shape["max_length"], self.agent.cfg.get("max_len", 512))
        if total > capacity:
            raise LayaCapacityError(
                f"Laya input needs {total} tokens; selected artifact supports {capacity}. Use explicit general1024 preset or revise the prompt; no truncation was applied."
            )
        return total

    def predict(self, state: str, questions: dict) -> dict:
        return self.agent.predict(state, questions)


def _types(mon: dict) -> str:
    values = mon.get("types", mon.get("type_names", mon.get("type_ids", [])))
    if isinstance(values, (str, int)):
        values = [values]
    names = []
    for value in values:
        name = TYPE_NAMES[value] if isinstance(value, int) and 0 <= value < len(TYPE_NAMES) else str(value)
        if name not in names:
            names.append(name)
    return "/".join(names) or "unknown"


def _hp(mon: dict) -> str:
    if "hp_fraction" in mon:
        fraction = float(mon["hp_fraction"])
    elif float(mon.get("max_hp", 0)) > 0:
        fraction = float(mon.get("hp", 0)) / float(mon["max_hp"])
    else:
        return "?"
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("HP fraction must be finite and in [0,1]")
    return str(round(100 * fraction)) + "%"


def _mon(mon: dict) -> str:
    # Explicit public allowlist. Hidden enemy moves/ability/items/stats never enter text.
    name = mon.get("species_name", mon.get("name", ""))
    status = mon.get("status", 0)
    return f"{name} {_types(mon)} HP{_hp(mon)}".strip() + (f" status{status}" if status else "")


def semantic_request(observation: dict, action: int | None = None) -> tuple[str, dict]:
    """Versioned compact projection, deliberately omitting IDs, hidden state and history."""
    legal = _legal(observation)
    state = f"Pokemon. You {_mon(observation['player'])}; foe {_mon(observation['opponent'])}."
    descriptions = {}
    for slot in np.flatnonzero(legal):
        slot = int(slot)
        if slot < 4:
            move = observation["moves"][slot]
            kind = move.get("type_name", move.get("type", "unknown"))
            if isinstance(kind, int) and 0 <= kind < len(TYPE_NAMES):
                kind = TYPE_NAMES[kind]
            descriptions[str(slot)] = f"use {move.get('name', 'move')} {kind} power{move.get('power', 0)}"
        else:
            descriptions[str(slot)] = "switch " + _mon(observation["party"][slot - 4])
    if action is None:
        return state, {
            "action": {"type": "choice", "instructions": "Best action to win?", "criteria": descriptions}
        }
    if str(action) not in descriptions:
        raise ValueError("Cannot score an illegal action")
    state += " " + descriptions[str(action)] + "."
    return state, {
        "action": {
            "type": "noul",
            "instructions": "Good action to win?",
            "criteria": {"false": "no", "true": "yes"},
        }
    }


def _legal(observation: dict) -> np.ndarray:
    legal = np.asarray(observation["legal_actions"])
    if legal.shape != (10,) or not np.isin(legal, [False, True]).all():
        raise ValueError("legal_actions must contain exactly ten booleans")
    legal = legal.astype(bool)
    if not legal.any():
        raise ValueError("A prior requires at least one legal action; terminal observations have none")
    return legal


class LayaPrior:
    def __init__(self, config: LayaConfig | None = None, *, predictor: Predictor | None = None):
        self.config = config or LayaConfig()
        self.predictor = predictor if predictor is not None else CoreMLPredictor(self.config)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.calls = self.cache_hits = 0
        self.inference_seconds = 0.0
        self.last_metrics: dict[str, Any] = {}

    @property
    def provenance(self) -> dict:
        return {
            "config": asdict(self.config),
            "prompt_version": PROMPT_VERSION,
            "backend": getattr(self.predictor, "provenance", {"implementation": "injected_test_predictor"}),
        }

    def _predict(self, state, questions):
        key = json.dumps([PROMPT_VERSION, state, questions], sort_keys=True, separators=(",", ":"))
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.predictor.token_count(state, questions)  # capacity errors propagate, never a fake fallback
        start = time.perf_counter()
        result = self.predictor.predict(state, questions)
        self.inference_seconds += time.perf_counter() - start
        self.calls += 1
        if result.get("usage", {}).get("output_tokens", 0) != 0:
            raise ValueError("Frozen typed backend must not generate output tokens")
        if self.config.cache_size:
            self._cache[key] = result
            while len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)
        return result

    def probabilities(self, observation: dict) -> np.ndarray:
        legal = _legal(observation)
        before = self.calls, self.cache_hits, self.inference_seconds
        start = time.perf_counter()
        values = np.zeros(10, dtype=np.float64)
        if legal.sum() == 1:
            values[legal] = 1.0
        elif self.config.strategy == "action_scores":
            for slot in np.flatnonzero(legal):
                state, questions = semantic_request(observation, int(slot))
                values[slot] = float(self._predict(state, questions)["answers"]["action"]["noul"])
        else:
            state, questions = semantic_request(observation)
            answer = self._predict(state, questions)["answers"]["action"]["probabilities"]
            if set(answer) != {str(i) for i in np.flatnonzero(legal)}:
                raise ValueError("Laya probability labels do not match legal action slots")
            for slot in np.flatnonzero(legal):
                values[slot] = float(answer[str(slot)])
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Laya returned invalid probabilities")
        values[legal] = np.maximum(values[legal], self.config.probability_floor)
        values /= values.sum()
        self.last_metrics = {
            "model_calls": self.calls - before[0],
            "cache_hits": self.cache_hits - before[1],
            "inference_seconds": self.inference_seconds - before[2],
            "total_seconds": time.perf_counter() - start,
            "strategy": self.config.strategy,
        }
        return values
