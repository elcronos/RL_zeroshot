import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rogue_rl.laya import CoreMLPredictor, LayaCapacityError, LayaConfig, LayaPrior, semantic_request


def observation():
    return {
        "battle_id": "a",
        "decision_id": 1,
        "phase": "action",
        "turn": 1,
        "player": {"species_name": "Pikachu", "type_ids": [13, 13], "hp": 50, "max_hp": 100},
        "opponent": {"species_name": "Squirtle", "type_ids": [11, 11], "hp_fraction": 0.75},
        "moves": [
            {"name": "Thunderbolt", "type": 13, "power": 90},
            {"name": "Tackle", "type": 0, "power": 40},
        ],
        "party": [
            {"type_ids": [13, 13], "hp": 50, "max_hp": 100},
            {"type_ids": [12, 3], "hp": 90, "max_hp": 100},
        ],
        "legal_actions": [True, True, False, False, False, True, False, False, False, False],
    }


class FakePredictor:
    def __init__(self, scores=(0.8, 0.2, 0.6)):
        self.scores = iter(scores)
        self.requests = []

    def token_count(self, state, questions):
        return 40

    def predict(self, state, questions):
        self.requests.append((state, questions))
        definition = questions["action"]
        answer = (
            {"noul": next(self.scores)}
            if definition["type"] == "noul"
            else {"probabilities": {"0": 0.5, "1": 0.2, "5": 0.3}}
        )
        return {"answers": {"action": answer}, "usage": {"output_tokens": 0}}


def test_scores_normalized_fixed_slots_and_cache():
    predictor = FakePredictor()
    prior = LayaPrior(predictor=predictor)
    probabilities = prior.probabilities(observation())
    np.testing.assert_allclose(probabilities[[0, 1, 5]], [0.5, 0.125, 0.375])
    assert probabilities[~np.array(observation()["legal_actions"])].sum() == 0
    changed_ids = observation() | {"decision_id": 100, "battle_id": "new"}
    np.testing.assert_array_equal(prior.probabilities(changed_ids), probabilities)
    assert prior.last_metrics["model_calls"] == 0
    assert prior.last_metrics["cache_hits"] == 3
    assert prior.calls == 3
    probabilities[:] = 0  # caller mutation cannot poison cache
    assert prior.probabilities(observation()).sum() == pytest.approx(1)


def test_joint_choice_uses_exact_slot_labels():
    predictor = FakePredictor()
    prior = LayaPrior(LayaConfig(strategy="joint"), predictor=predictor)
    np.testing.assert_allclose(prior.probabilities(observation())[[0, 1, 5]], [0.5, 0.2, 0.3])
    assert len(predictor.requests) == 1


def test_only_allowed_public_enemy_fields_enter_prompt():
    original = observation()
    private = copy.deepcopy(original)
    private["opponent"].update(
        moves=["SECRET"], ability="SECRET", held_item="SECRET", attack=999, trainer_party="SECRET"
    )
    private.update(rng_seed="SECRET", battle_id="SECRET", hidden_state="SECRET")
    assert semantic_request(original, 0) == semantic_request(private, 0)
    text, _ = semantic_request(original, 0)
    assert "Electric" in text and "Water" in text and "Thunderbolt" in text
    assert "SECRET" not in text


def test_forced_single_action_skips_model():
    prior = LayaPrior(predictor=FakePredictor(scores=()))
    obs = observation()
    obs["legal_actions"] = [False] * 10
    obs["legal_actions"][5] = True
    obs["phase"] = "forced_switch"
    assert prior.probabilities(obs)[5] == 1
    assert prior.calls == 0


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1, float("inf")])
def test_invalid_probabilities_fail_closed(score):
    prior = LayaPrior(predictor=FakePredictor([score] * 3))
    with pytest.raises(ValueError, match="invalid probabilities"):
        prior.probabilities(observation())


def test_zero_rounding_scores_receive_documented_floor():
    prior = LayaPrior(predictor=FakePredictor([0, 0, 0]))
    np.testing.assert_allclose(prior.probabilities(observation())[[0, 1, 5]], [1 / 3] * 3)


def test_capacity_error_never_triggers_inference_or_fallback():
    predictor = FakePredictor()

    def reject(*args):
        raise LayaCapacityError("too long")

    predictor.token_count = reject
    with pytest.raises(LayaCapacityError):
        LayaPrior(predictor=predictor).probabilities(observation())
    assert predictor.requests == []


def test_terminal_and_wrong_mask_rejected():
    prior = LayaPrior(predictor=FakePredictor())
    for mask in [[False] * 10, [True] * 9, [2] * 10]:
        with pytest.raises(ValueError):
            prior.probabilities(observation() | {"legal_actions": mask})


def test_lru_eviction():
    prior = LayaPrior(LayaConfig(cache_size=1), predictor=FakePredictor([0.5] * 6))
    prior.probabilities(observation())
    prior.probabilities(observation())
    assert prior.calls == 6
    assert len(prior._cache) == 1


class WordTokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


def test_real_adapter_counts_before_upstream_truncation():
    # No CoreML import/model load: exercise the actual adapter's preflight contract.
    import sys
    from unittest.mock import patch

    fake_common = SimpleNamespace(render_options=lambda question: ["no", "yes"])
    adapter = CoreMLPredictor.__new__(CoreMLPredictor)
    adapter.agent = SimpleNamespace(
        tok=WordTokenizer(),
        cfg={"max_len": 1024},
        shape={"max_length": 96},
        _to_internal=lambda question: {"t": "noul", "ins": "Good action?"},
    )
    with patch.dict(sys.modules, {"laya_coreml.common": fake_common}):
        assert adapter.token_count("short state", {"action": {}}) == 14
        with pytest.raises(LayaCapacityError, match="supports 96"):
            adapter.token_count("word " * 100, {"action": {}})
