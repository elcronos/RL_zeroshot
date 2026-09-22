"""Assign a captured Rogue corpus to disjoint, team/trainer-grouped splits.

The unit of assignment is a complete player-team/opponent-trainer matchup, so
RNG variants of a matchup can never appear in more than one split.  Assignment
is deterministic from the group name and is balanced within each player team.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rogue_rl.manifest import Corpus


def split_groups(groups: list[str]) -> dict[str, str]:
    """Give every team at least one held-out group when possible."""
    ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    if len(ordered) < 3:
        raise ValueError("Each player team needs at least three trainer matchups")
    # With Rogue's common four-trainer pool this is 2/1/1: fifty percent of
    # samples train, but every team is represented in both held-out cohorts.
    validation_count = 1
    test_count = 1
    train_count = len(ordered) - validation_count - test_count
    return {
        **{group: "train" for group in ordered[:train_count]},
        **{group: "validation" for group in ordered[train_count : train_count + validation_count]},
        **{group: "test" for group in ordered[train_count + validation_count :]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/battles.json"))
    parser.add_argument("--min-train", type=int, default=50)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    by_team: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for battle in payload["battles"]:
        team = tuple(battle.get("capture", {}).get("player_team", []))
        if len(team) != 3:
            raise ValueError(f"{battle['id']} has no recorded three-Pokémon team")
        by_team[team].add(battle["scenario_group"])
    assignments: dict[str, str] = {}
    for groups in by_team.values():
        assignments.update(split_groups(list(groups)))
    for battle in payload["battles"]:
        battle["split"] = assignments[battle["scenario_group"]]
    counts = Counter(battle["split"] for battle in payload["battles"])
    if counts["train"] < args.min_train or not counts["validation"] or not counts["test"]:
        raise ValueError(f"Insufficient split sizes: {dict(counts)}")
    payload["cohort"] = (
        "six declared three-Pokémon player teams against ordinary Rogue route trainers; "
        "team/trainer matchups are disjoint across train, validation, and test"
    )
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    Corpus.load(args.manifest)
    print(json.dumps({"splits": dict(counts), "teams": len(by_team), "groups": len(assignments)}, indent=2))


if __name__ == "__main__":
    main()
