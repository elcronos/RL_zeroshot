import pytest

from rogue_rl.report import paired_comparison


def rows(win):
    return [
        {
            "agent_seed": seed,
            "battle_id": f"battle-{group}",
            "scenario_group": str(group),
            "repeat": 0,
            "outcome": "win" if win else "loss",
        }
        for seed in (0, 1)
        for group in (0, 1, 2)
    ]


def test_pairing_and_seed_scenario_interval():
    result = paired_comparison(rows(True), rows(False), samples=100)
    assert result["mean_win_rate_difference"] == 1
    assert result["ci95"] == [1, 1]
    assert result["agent_seeds"] == 2 and result["scenario_groups"] == 3


def test_pairing_rejects_different_episodes():
    with pytest.raises(ValueError, match="exactly the same"):
        paired_comparison(rows(True), rows(False)[:-1])


def test_pairing_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        paired_comparison(rows(True) * 2, rows(False) * 2)


def test_no_seed_replication_means_no_inferential_interval():
    result = paired_comparison(rows(True)[:3], rows(False)[:3])
    assert result["ci95"] is None and "warning" in result
