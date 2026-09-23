"""Transparent post-battle quality metrics; these do not shape PPO reward."""

from __future__ import annotations

import math


def battle_quality(outcome: str, party_hp_fraction: float | None, turns: int, *, target_turns: int = 6) -> dict:
    """Return lexicographic outcome plus a granular survival/speed diagnostic.

    A win always scores above every loss or truncation.  Within wins,
    retained party HP carries 65% weight and speed carries 35%, decaying over a
    six-turn reference battle.  This is an evaluation diagnostic only: PPO
    still trains on the preregistered sparse +1/-1 game reward, with zero on
    an incomplete truncation.
    """
    if outcome not in {"win", "loss", "truncated"}:
        raise ValueError("Unknown terminal outcome")
    if type(turns) is not int or turns < 0 or type(target_turns) is not int or target_turns < 1:
        raise ValueError("Turns and target_turns must be nonnegative/positive integers")
    hp = 0.0 if party_hp_fraction is None else float(party_hp_fraction)
    if not 0 <= hp <= 1:
        raise ValueError("Party HP fraction must be in [0, 1]")
    speed = math.exp(-turns / target_turns)
    win_quality = 100 * (0.65 * hp + 0.35 * speed)
    if outcome == "win":
        score = 50 + 0.5 * win_quality
    elif outcome == "loss":
        score = 20 * hp
    else:
        score = 0.0
    return {
        "battle_score": score,
        "win_quality": win_quality if outcome == "win" else None,
        "survival_component": 100 * hp,
        "speed_component": 100 * speed,
    }
