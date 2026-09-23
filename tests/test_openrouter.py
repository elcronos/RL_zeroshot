import json
import urllib.error

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
            {
                "model": "provider/jev-snapshot",
                "answers": {"action": {"choice": "9", "probabilities": {"9": 1.0}}},
            }
        ),
    )
    with pytest.raises(RuntimeError, match="illegal"):
        OpenRouterPrior(OpenRouterConfig(model="provider/jev-current")).probabilities(observation())


def test_openrouter_prior_retries_transient_timeout(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    attempts = 0

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError
        return Response(
            {
                "model": "provider/jev-snapshot",
                "answers": {
                    "action": {
                        "choice": "0",
                        "probabilities": {"0": 0.75, "1": 0.25},
                    }
                }
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    prior = OpenRouterPrior(
        OpenRouterConfig(model="provider/jev-current", max_retries=1, retry_backoff_seconds=0)
    )

    assert prior.probabilities(observation())[0] == pytest.approx(0.75)
    assert attempts == 2 and prior.calls == 1


def test_openrouter_prior_does_not_retry_authentication_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    attempts = 0

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError("url", 401, "unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: pytest.fail("authentication error retried"))

    with pytest.raises(RuntimeError, match="HTTP 401"):
        OpenRouterPrior(OpenRouterConfig(model="provider/jev-current")).probabilities(observation())
    assert attempts == 1


def test_openrouter_prior_does_not_retry_malformed_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    attempts = 0

    class MalformedResponse(Response):
        def read(self):
            return b"{not-json"

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return MalformedResponse(None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: pytest.fail("malformed JSON retried"))

    with pytest.raises(RuntimeError, match="malformed JSON"):
        OpenRouterPrior(OpenRouterConfig(model="provider/jev-current")).probabilities(observation())
    assert attempts == 1


def test_openrouter_prior_rejects_non_object_and_missing_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response([]))
    prior = OpenRouterPrior(OpenRouterConfig(model="provider/jev-current"))
    with pytest.raises(RuntimeError, match="non-object"):
        prior.probabilities(observation())

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(
            {"answers": {"action": {"choice": "0", "probabilities": {"0": 1.0}}}}
        ),
    )
    with pytest.raises(RuntimeError, match="served model identifier"):
        prior.probabilities(observation())


def test_openrouter_prior_rejects_non_object_probabilities(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(
            {
                "model": "provider/jev-snapshot",
                "answers": {"action": {"choice": "0", "probabilities": []}},
            }
        ),
    )
    prior = OpenRouterPrior(OpenRouterConfig(model="provider/jev-current"))
    with pytest.raises(RuntimeError, match="required typed Choice"):
        prior.probabilities(observation())


def test_openrouter_prior_does_not_retry_non_timeout_url_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    attempts = 0

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("certificate failure")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: pytest.fail("non-timeout error retried"))
    with pytest.raises(RuntimeError, match="URLError"):
        OpenRouterPrior(OpenRouterConfig(model="provider/jev-current")).probabilities(observation())
    assert attempts == 1
