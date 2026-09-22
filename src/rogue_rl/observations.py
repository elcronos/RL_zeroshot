"""Public-information boundary and fixed, name-free learner features.

Schema 1 uses stable move slots 0..3 and absolute party slots 4..9.
Species/move identities are intentionally excluded from numerical features.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

ACTION_DIM = 10
FEATURE_VERSION = 1
TYPE_COUNT = 19
PUBLIC_MON_FIELDS = {
    "species",
    "species_name",
    "name",
    "types",
    "type_ids",
    "hp_fraction",
    "status",
    "stat_stages",
}
OWN_MON_FIELDS = PUBLIC_MON_FIELDS | {"hp", "max_hp", "level", "speed", "active"}
MOVE_FIELDS = {"id", "name", "type", "type_name", "power", "category", "pp", "max_pp", "accuracy"}


def hp_fraction(mon: Mapping[str, Any]) -> float:
    value = mon.get("hp_fraction")
    if value is None:
        value = float(mon.get("hp", 0)) / max(1, float(mon.get("max_hp", 1)))
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("HP fraction must be finite and within [0,1]")
    return value


def public_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build an allowlisted copy, never forwarding raw emulator memory to Laya."""
    phase = raw.get("phase")
    if phase not in {"action", "forced_switch", "terminal"}:
        raise ValueError(f"Unsupported battle phase: {phase!r}")
    legal = raw.get("legal_actions", [])
    if len(legal) != ACTION_DIM or any(type(x) is not bool for x in legal):
        raise ValueError("legal_actions must be exactly ten booleans")
    if phase != "terminal" and not any(legal):
        raise ValueError("Nonterminal battle has no legal actions")
    if phase == "forced_switch" and any(legal[:4]):
        raise ValueError("Moves cannot be legal during a forced switch")
    outcome = raw.get("outcome")
    if phase == "terminal" and outcome not in {"win", "loss", "draw"}:
        raise ValueError("Terminal battle requires win/loss/draw outcome")
    if phase != "terminal" and outcome is not None:
        raise ValueError("Nonterminal battle cannot have an outcome")
    party, moves = raw.get("party", []), raw.get("moves", [])
    if len(party) > 6 or len(moves) > 4:
        raise ValueError("Only singles with six party members and four move slots are supported")
    for index, allowed in enumerate(legal):
        if allowed and (index < 4 and index >= len(moves) or index >= 4 and index - 4 >= len(party)):
            raise ValueError("Legal action refers to a missing move or party slot")
    opponent = {k: v for k, v in raw.get("opponent", {}).items() if k in PUBLIC_MON_FIELDS}
    # Never derive opponent fraction from hidden exact HP supplied by an untrusted reader.
    if "hp_fraction" not in opponent and phase != "terminal":
        raise ValueError("Opponent requires publicly observed hp_fraction")
    result = {
        "battle_id": str(raw["battle_id"]),
        "decision_id": int(raw["decision_id"]),
        "phase": phase,
        "turn": int(raw.get("turn", 0)),
        "outcome": outcome,
        "player": {k: v for k, v in raw.get("player", {}).items() if k in OWN_MON_FIELDS},
        "opponent": opponent,
        "party": [{k: v for k, v in mon.items() if k in OWN_MON_FIELDS} for mon in party],
        "moves": [{k: v for k, v in move.items() if k in MOVE_FIELDS} for move in moves],
        "legal_actions": list(legal),
    }
    for mon in [result["player"], opponent, *result["party"]]:
        hp_fraction(mon)
    return result


def _mon_features(mon: Mapping[str, Any]) -> list[float]:
    types = [0.0] * TYPE_COUNT
    for type_id in mon.get("type_ids", []):
        if isinstance(type_id, int) and 0 <= type_id < TYPE_COUNT:
            types[type_id] = 1.0
    # Gen III visible major-status bits; multiple sleep-counter values map to one flag.
    status = int(mon.get("status", 0))
    status_flags = [float(bool(status & bits)) for bits in (7, 8, 16, 32, 64, 128)]
    stages = list(mon.get("stat_stages", [6] * 7))[:7]
    stages += [6] * (7 - len(stages))
    return [
        hp_fraction(mon),
        float(bool(mon)),
        *types,
        *status_flags,
        *[(float(stage) - 6) / 6 for stage in stages],
    ]


MON_DIM = 2 + TYPE_COUNT + 6 + 7
MOVE_DIM = 5 + TYPE_COUNT
FEATURE_DIM = MON_DIM * 8 + MOVE_DIM * 4 + 3 + ACTION_DIM + 2


class FeatureEncoder:
    """Episode-local observed history; reset on every battle and evaluation episode."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous_hp: tuple[float, float] | None = None
        self.previous_action: int | None = None

    def encode(self, obs: Mapping[str, Any]) -> np.ndarray:
        values = _mon_features(obs["player"]) + _mon_features(obs["opponent"])
        party = obs["party"]
        for slot in range(6):
            values += _mon_features(party[slot] if slot < len(party) else {})
        moves = obs["moves"]
        for slot in range(4):
            move = moves[slot] if slot < len(moves) else {}
            types = [0.0] * TYPE_COUNT
            type_id = move.get("type")
            if isinstance(type_id, int) and 0 <= type_id < TYPE_COUNT:
                types[type_id] = 1.0
            values += [
                float(bool(move)),
                float(move.get("power", 0)) / 250,
                float(move.get("pp", 0)) / max(1, float(move.get("max_pp", 1))),
                float(move.get("category", 0)) / 2,
                float(move.get("accuracy", 0)) / 100,
                *types,
            ]
        values += [float(obs["phase"] == "forced_switch"), min(int(obs["turn"]), 500) / 500, 1.0]
        values += [float(self.previous_action == index) for index in range(ACTION_DIM)]
        hp = (hp_fraction(obs["player"]), hp_fraction(obs["opponent"]))
        values += [0.0, 0.0] if self.previous_hp is None else [hp[i] - self.previous_hp[i] for i in (0, 1)]
        features = np.asarray(values, dtype=np.float32)
        if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
            raise ValueError("Invalid numeric observation features")
        return features

    def record_action(self, obs: Mapping[str, Any], action: int) -> None:
        self.previous_hp = (hp_fraction(obs["player"]), hp_fraction(obs["opponent"]))
        self.previous_action = action


def terminal_reward(obs: Mapping[str, Any]) -> float:
    """Sparse win/loss reward avoids shaping-induced strategic confounds."""
    return {"win": 1.0, "loss": -1.0, "draw": 0.0, None: 0.0}[obs["outcome"]]
