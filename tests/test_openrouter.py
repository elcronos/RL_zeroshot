import json

import numpy as np
import pytest

from rogue_rl.openrouter import OpenRouterConfig, OpenRouterPrior


class Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def observation():
    return {
        "legal_actions": [True, True] + [False] * 8,
        "player": {"species_name": "Bulbasaur", "type_ids": [11], "hp": 20, "max_hp": 30},
        "opponent": {"species_name": "Pansage", "type_ids": [12], "hp_fraction": 1.0},
        "moves": [
            {"name": "Tackle", "type": 0, "power": 40, "pp": 35, "max_pp": 35},
            {"name": "Vine Whip", "type": 11, "power": 45, "pp": 25, "max_pp": 25},
        ],
        "party": [],
    }


def test_openrouter_prior_uses_environment_secret_and_returns_one_legal_choice(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    captured = {}

    def urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data)
        assert timeout == 45
        return Response(
            {
                "model": "provider/jev-snapshot",
                "answers": {
                    "action": {
                        "choice": "1",
                        "probabilities": {"0": 0.25, "1": 0.75},
                    }
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    prior = OpenRouterPrior(OpenRouterConfig(model="provider/jev-current"))
    values = prior.probabilities(observation())
    assert np.array_equal(values, np.asarray([0.25, 0.75] + [0.0] * 8))
    assert captured["authorization"] == "Bearer test-secret"
    assert captured["payload"]["model"] == "provider/jev-current"
    assert captured["payload"]["questions"]["action"]["type"] == "choice"
    assert prior.provenance["served_models"] == ["provider/jev-snapshot"]


def test_openrouter_prior_rejects_missing_secret_and_illegal_action(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterPrior(OpenRouterConfig(model="provider/jev-current"))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(
            {"answers": {"action": {"choice": "9", "probabilities": {"9": 1.0}}}}
        ),
    )
    with pytest.raises(RuntimeError, match="illegal"):
        OpenRouterPrior(OpenRouterConfig(model="provider/jev-current")).probabilities(observation())
