import pytest

from rogue_rl.scoring import battle_quality


def test_fast_full_hp_win_scores_above_slow_damaged_win():
    fast = battle_quality("win", 1.0, 2)
    slow = battle_quality("win", 0.5, 10)
    assert fast["battle_score"] > slow["battle_score"]
    assert fast["win_quality"] > slow["win_quality"]


def test_outcome_order_is_preserved():
    assert battle_quality("win", 0.0, 100)["battle_score"] > battle_quality("loss", 1.0, 0)["battle_score"]
    assert battle_quality("loss", 1.0, 0)["battle_score"] > battle_quality("truncated", 1.0, 0)["battle_score"]


@pytest.mark.parametrize(
    "outcome,hp,turns",
    [("bad", 0.5, 1), ("draw", 0.5, 1), ("win", 1.1, 1), ("win", 0.5, -1)],
)
def test_quality_rejects_invalid_inputs(outcome, hp, turns):
    with pytest.raises(ValueError):
        battle_quality(outcome, hp, turns)
