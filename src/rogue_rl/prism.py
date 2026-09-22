"""Frozen PrismNLI action prior built from public battle observations."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .laya import _legal, semantic_request

PRISM_MODEL = "Jaehun/PrismNLI-0.4B"
PROMPT_VERSION = "rogue-prism-nli-v1"


@dataclass(frozen=True)
class PrismConfig:
    model_id: str = PRISM_MODEL
    device: str | None = None
    probability_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not self.model_id or not 0 < self.probability_floor < 1:
            raise ValueError("Prism model id and probability floor must be valid")


class PrismPrior:
    """Score each legal action with PrismNLI's frozen entailment logit.

    The premise is exactly the public semantic projection used by Laya/Jev;
    only the typed-decision backend differs.  It is never fine-tuned here.
    """

    def __init__(self, config: PrismConfig | None = None, *, classifier: Any | None = None):
        self.config = config or PrismConfig()
        self.calls = self.cache_hits = 0
        self.inference_seconds = 0.0
        self.last_metrics: dict[str, Any] = {}
        self._classifier = classifier if classifier is not None else self._load_classifier()

    def _load_classifier(self):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install the Prism extra: uv sync --extra prism") from exc
        device = self.config.device
        if device is None:
            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        return pipeline("text-classification", model=self.config.model_id, device=device)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "transformers_text_classification",
            "model_id": self.config.model_id,
            "prompt_version": PROMPT_VERSION,
            "config": asdict(self.config),
            "frozen": True,
        }

    def probabilities(self, observation: dict) -> np.ndarray:
        legal = _legal(observation)
        before = self.calls, self.inference_seconds
        started = time.perf_counter()
        values = np.zeros(10, dtype=np.float64)
        for slot in np.flatnonzero(legal):
            state, question = semantic_request(observation, int(slot))
            hypothesis = question["action"]["instructions"] + ": true."
            result = self._classifier({"text": state, "text_pair": hypothesis}, top_k=None)
            # The model card declares label 0 to be entailment.  Avoid relying
            # on provider-specific pipeline ordering or label text formatting.
            entailment = [
                float(row["score"])
                for row in result
                if str(row["label"]).lower() in {"label_0", "entailment"}
            ]
            if len(entailment) != 1:
                raise RuntimeError("PrismNLI did not return exactly one entailment label")
            values[int(slot)] = entailment[0]
            self.calls += 1
        values[legal] = np.maximum(values[legal], self.config.probability_floor)
        values /= values.sum()
        elapsed = time.perf_counter() - started
        self.inference_seconds += elapsed
        self.last_metrics = {
            "model_calls": self.calls - before[0],
            "cache_hits": 0,
            "inference_seconds": self.inference_seconds - before[1],
            "total_seconds": elapsed,
            "strategy": "per-action-entailment",
        }
        return values
