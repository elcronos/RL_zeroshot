"""TypeSafe Jev action chooser through OpenRouter's System One endpoint."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .laya import semantic_request


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    timeout_seconds: float = 45.0
    endpoint: str = "https://openrouter.ai/api/alpha/decisions"

    def __post_init__(self) -> None:
        if not self.model or self.timeout_seconds <= 0:
            raise ValueError("OpenRouter model and timeout must be positive")


class OpenRouterPrior:
    """Typed Jev Choice distribution; errors never fall back to another policy."""

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY in the environment before selecting --prior jev")
        self.calls = self.cache_hits = 0
        self.inference_seconds = 0.0
        self.served_models: set[str] = set()

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "openrouter_chat_completion",
            "config": asdict(self.config),
            "served_models": sorted(self.served_models),
        }

    def probabilities(self, observation: dict) -> np.ndarray:
        legal = np.asarray(observation["legal_actions"], dtype=bool)
        state, questions = semantic_request(observation)
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(
                {
                    "model": self.config.model,
                    "state": state,
                    "questions": questions,
                }
            ).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"OpenRouter request failed: {type(exc).__name__}") from exc
        self.inference_seconds += time.perf_counter() - started
        self.calls += 1
        if isinstance(payload.get("model"), str):
            self.served_models.add(payload["model"])
        try:
            answer = payload["answers"]["action"]
            choice = int(answer["choice"])
            raw_probabilities = answer["probabilities"]
            values = np.asarray(
                [float(raw_probabilities.get(str(index), 0.0)) for index in range(len(legal))],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Jev did not return the required typed Choice answer") from exc
        if not 0 <= choice < len(legal) or not legal[choice] or values.shape != legal.shape:
            raise RuntimeError("Jev selected an illegal action")
        if not np.isfinite(values).all() or (values < 0).any() or values[~legal].sum() > 1e-6:
            raise RuntimeError("Jev returned invalid action probabilities")
        if values[legal].sum() <= 0:
            raise RuntimeError("Jev returned no probability for any legal action")
        return values / values.sum()
