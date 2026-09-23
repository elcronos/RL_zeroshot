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
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    endpoint: str = "https://openrouter.ai/api/alpha/decisions"

    def __post_init__(self) -> None:
        if (
            not self.model
            or self.timeout_seconds <= 0
            or type(self.max_retries) is not int
            or self.max_retries < 0
            or self.retry_backoff_seconds < 0
        ):
            raise ValueError("OpenRouter model, timeout, retry count, and backoff must be valid")


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
        payload = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw_payload = response.read()
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.config.max_retries:
                    self.inference_seconds += time.perf_counter() - started
                    raise RuntimeError(f"OpenRouter request failed with HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                retryable = isinstance(exc.reason, TimeoutError)
                if not retryable or attempt == self.config.max_retries:
                    self.inference_seconds += time.perf_counter() - started
                    raise RuntimeError(f"OpenRouter request failed: {type(exc).__name__}") from exc
            except TimeoutError as exc:
                if attempt == self.config.max_retries:
                    self.inference_seconds += time.perf_counter() - started
                    raise RuntimeError("OpenRouter request failed: TimeoutError") from exc
            time.sleep(self.config.retry_backoff_seconds * 2**attempt)
        self.inference_seconds += time.perf_counter() - started
        try:
            payload = json.loads(raw_payload.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OpenRouter returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("OpenRouter returned a non-object response")  # noqa: TRY004
        self.calls += 1
        served_model = payload.get("model")
        if not isinstance(served_model, str) or not served_model:
            raise RuntimeError("OpenRouter response omitted its served model identifier")
        self.served_models.add(served_model)
        try:
            answer = payload["answers"]["action"]
            choice = int(answer["choice"])
            raw_probabilities = answer["probabilities"]
            if not isinstance(raw_probabilities, dict):
                raise TypeError("probabilities must be an object")
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
