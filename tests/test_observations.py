"""Public-information and numeric encoding contract tests."""

from copy import deepcopy

import numpy as np
import pytest

from rogue_rl.observations import FEATURE_DIM, FeatureEncoder, public_observation, terminal_reward


def raw_observation():
    return {
        "battle_id": "schema-fixture",
        "decision_id": 0,
        "phase": "action",
        "turn": 1,
        "legal_actions": [True, True] + [False] * 8,
        "player": {"hp": 10, "max_hp": 20, "type_ids": [10], "species": 1, "name": "own"},
        "opponent": {
            "hp_fraction": 0.75,
            "hp": 300,
            "max_hp": 400,
            "species": 2,
            "type_ids": [12],
            "moves": ["hidden"],
            "ability": "hidden",
            "rng": 12345,
            "speed": 999,
            "held_item": "hidden",
        },
        "party": [{"hp_fraction": 0.5}],
        "moves": [
            {
                "id": 1,
                "name": "first",
                "pp": 5,
                "max_pp": 10,
                "power": 50,
                "accuracy": 100,
                "category": 0,
                "type": 10,
            },
            {"id": 2, "name": "second", "pp": 8, "max_pp": 10, "power": 70},
        ],
        "opponent_party": ["hidden"],
        "rng_state": "hidden",
        "future_ai_action": 0,
    }


def test_allowlist_removes_unobserved_opponent_information():
    raw = raw_observation()
    public = public_observation(raw)
    assert not {"opponent_party", "rng_state", "future_ai_action"} & public.keys()
    assert not {"hp", "max_hp", "moves", "ability", "rng", "speed", "held_item"} & public["opponent"].keys()
    assert public["opponent"]["hp_fraction"] == 0.75
    assert "hp" in raw["opponent"]


def test_exact_opponent_hp_cannot_substitute_for_visible_fraction():
    raw = raw_observation()
    del raw["opponent"]["hp_fraction"]
    with pytest.raises(ValueError, match="publicly observed"):
        public_observation(raw)


def test_numeric_features_are_name_free_and_fixed_size():
    first = public_observation(raw_observation())
    second = deepcopy(first)
    second["player"].update(species=999, species_name="other", name="other")
    second["opponent"].update(species=999, species_name="other", name="other")
    for move in second["moves"]:
        move.update(id=999, name="unseen name")
    encoder = FeatureEncoder()
    features = encoder.encode(first)
    assert features.shape == (FEATURE_DIM,)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, encoder.encode(second))


def test_history_is_episode_local_and_encoding_is_idempotent():
    first = public_observation(raw_observation())
    encoder = FeatureEncoder()
    initial = encoder.encode(first)
    encoder.record_action(first, 1)
    second = deepcopy(first)
    second["opponent"]["hp_fraction"] = 0.25
    features = encoder.encode(second)
    np.testing.assert_allclose(features[-2:], [0, -0.5])
    assert features[-11] == 1  # Previous-action one-hot index 1.
    np.testing.assert_array_equal(features, encoder.encode(second))
    encoder.reset()
    np.testing.assert_array_equal(initial, encoder.encode(first))


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda raw: raw.update(legal_actions=[True] * 9), "ten booleans"),
        (lambda raw: raw.update(legal_actions=[1] + [False] * 9), "ten booleans"),
        (lambda raw: raw.update(phase="forced_switch"), "forced switch"),
        (lambda raw: raw.update(legal_actions=[False] * 9 + [True]), "missing move or party"),
        (lambda raw: raw.update(outcome="win"), "Nonterminal"),
        (lambda raw: raw["opponent"].update(hp_fraction=float("nan")), "HP fraction"),
    ],
)
def test_invalid_observation_contract_rejected(change, match):
    raw = raw_observation()
    change(raw)
    with pytest.raises(ValueError, match=match):
        public_observation(raw)


@pytest.mark.parametrize("outcome,reward", [("win", 1), ("loss", -1)])
def test_terminal_reward_and_empty_terminal_mask(outcome, reward):
    raw = raw_observation()
    raw.update(phase="terminal", outcome=outcome, legal_actions=[False] * 10)
    assert terminal_reward(public_observation(raw)) == reward


def test_engine_draw_must_be_mapped_to_single_player_loss_by_bridge():
    raw = raw_observation()
    raw.update(phase="terminal", outcome="draw", legal_actions=[False] * 10)
    with pytest.raises(ValueError, match="win/loss"):
        public_observation(raw)
