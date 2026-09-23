"""Learning curves and paired seed/scenario bootstrap; no turn-level pseudoreplication."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def baseline_summary(
    rows: list[dict], *, planned_episodes: int, status: str, model_metrics: dict | None = None
) -> dict:
    """Describe actual frozen-policy episodes, clustering uncertainty by scenario.

    Repeats of one save state do not create independent battle scenarios. When
    fewer than two scenario groups finish, the interval is deliberately absent.
    """
    if planned_episodes < 1 or status not in {"complete", "failed"}:
        raise ValueError("Invalid baseline plan or status")
    if len(rows) > planned_episodes or (status == "complete" and len(rows) != planned_episodes):
        raise ValueError("Completed baseline rows do not match the planned run")
    outcomes = {name: 0 for name in ("win", "loss", "truncated")}
    groups: dict[str, list[float]] = defaultdict(list)
    keys = set()
    for row in rows:
        outcome = row["outcome"]
        if outcome not in outcomes:
            raise ValueError(f"Unknown battle outcome: {outcome}")
        key = (row["battle_id"], row["repeat"])
        if key in keys:
            raise ValueError("Duplicate baseline episode")
        keys.add(key)
        outcomes[outcome] += 1
        groups[row["scenario_group"]].append(float(outcome == "win"))
    rates = np.asarray([np.mean(scores) for scores in groups.values()], dtype=float)
    ci = None
    if len(rates) >= 2:
        rng = np.random.default_rng(2026)
        means = rates[rng.integers(len(rates), size=(2000, len(rates)))].mean(axis=1)
        ci = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "status": status,
        "planned_episodes": planned_episodes,
        "completed_episodes": len(rows),
        "scenario_groups_completed": len(groups),
        "outcome_counts": outcomes,
        "episode_win_rate": outcomes["win"] / len(rows) if rows else None,
        "scenario_weighted_win_rate": float(rates.mean()) if len(rates) else None,
        "scenario_bootstrap_ci95": ci,
        "ci_note": (
            "Two or more completed scenario groups are required for a cluster interval."
            if ci is None
            else "Percentile bootstrap over scenario groups (2000 draws); exploratory with few groups."
        ),
        "mean_decisions": float(np.mean([row["decisions"] for row in rows])) if rows else None,
        "mean_turns": float(np.mean([row["turns"] for row in rows])) if rows else None,
        "mean_final_party_hp_fraction": (
            float(np.mean([row["final_party_hp_fraction"] for row in rows if row["final_party_hp_fraction"] is not None]))
            if any(row["final_party_hp_fraction"] is not None for row in rows)
            else None
        ),
        "mean_win_quality": (
            float(np.mean([row["win_quality"] for row in rows if row["win_quality"] is not None]))
            if any(row["win_quality"] is not None for row in rows)
            else None
        ),
        "mean_battle_score": float(np.mean([row["battle_score"] for row in rows])) if rows else None,
        "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in rows])) if rows else None,
        "median_battle_decision_ms_p50": (
            float(np.median([row["decision_ms_p50"] for row in rows])) if rows else None
        ),
        "model_metrics": model_metrics or {},
        "note": "Truncations are non-wins, reported separately. Failed runs summarize completed episodes only.",
    }


def compare_baselines(run_a: Path, run_b: Path, *, samples: int = 2000, seed: int = 2026) -> dict:
    """Compare two frozen-policy baselines on exactly matched captured states."""
    meta_a = json.loads((run_a / "metadata.json").read_text())
    meta_b = json.loads((run_b / "metadata.json").read_text())
    for field in ("rom_sha256", "source_revision", "corpus_sha256", "battle_ids", "repeats", "max_decisions"):
        if meta_a.get(field) != meta_b.get(field):
            raise ValueError(f"Baselines disagree on {field}; comparison requires matched conditions")
    rows_a = read_jsonl(run_a / "episodes.jsonl")
    rows_b = read_jsonl(run_b / "episodes.jsonl")
    key = lambda row: (row["battle_id"], row["repeat"])
    if {key(row) for row in rows_a} != {key(row) for row in rows_b}:
        raise ValueError("Baselines must complete exactly the same battle/repeat rows")
    groups_a: dict[str, list[float]] = defaultdict(list)
    groups_b: dict[str, list[float]] = defaultdict(list)
    for row in rows_a:
        groups_a[row["scenario_group"]].append(float(row["outcome"] == "win"))
    for row in rows_b:
        groups_b[row["scenario_group"]].append(float(row["outcome"] == "win"))
    if groups_a.keys() != groups_b.keys():
        raise ValueError("Baselines disagree on scenario groups")
    groups = sorted(groups_a)
    delta = np.asarray([np.mean(groups_a[group]) - np.mean(groups_b[group]) for group in groups])
    ci = None
    if len(delta) >= 2:
        rng = np.random.default_rng(seed)
        means = delta[rng.integers(len(delta), size=(samples, len(delta)))].mean(axis=1)
        ci = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "baseline_a": {"directory": str(run_a), "mode": meta_a.get("mode")},
        "baseline_b": {"directory": str(run_b), "mode": meta_b.get("mode")},
        "completed_episodes": len(rows_a),
        "scenario_groups": len(groups),
        "scenario_weighted_win_rate_difference_a_minus_b": float(delta.mean()),
        "scenario_bootstrap_ci95": ci,
        "per_scenario_win_rate_difference_a_minus_b": dict(zip(groups, delta.tolist(), strict=True)),
        "note": "Repeats are averaged within a scenario group; this pilot interval is exploratory.",
    }


def _group_scores(rows: list[dict]) -> dict[tuple[int, str], float]:
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        # Truncations count as non-wins and are also separately reported.
        groups[(row["agent_seed"], row["scenario_group"])].append(float(row["outcome"] == "win"))
    return {key: float(np.mean(values)) for key, values in groups.items()}


def paired_comparison(a: list[dict], b: list[dict], *, samples: int = 2000, seed: int = 2026) -> dict:
    key = lambda row: (row["agent_seed"], row["battle_id"], row["repeat"])
    if len({key(row) for row in a}) != len(a) or len({key(row) for row in b}) != len(b):
        raise ValueError("Duplicate evaluation rows; supply one checkpoint per arm and seed")
    if {key(row) for row in a} != {key(row) for row in b}:
        raise ValueError("Paired arms must evaluate exactly the same seeds, battles, and repeats")
    ag, bg = _group_scores(a), _group_scores(b)
    if ag.keys() != bg.keys():
        raise ValueError("Paired arms have inconsistent scenario groups")
    seeds = sorted({x[0] for x in ag})
    groups = sorted({x[1] for x in ag})
    if any((s, g) not in ag for s in seeds for g in groups):
        raise ValueError("Every agent seed must evaluate the same scenario groups")
    delta = np.asarray([[ag[s, g] - bg[s, g] for g in groups] for s in seeds])
    result = {
        "mean_win_rate_difference": float(delta.mean()),
        "agent_seeds": len(seeds),
        "scenario_groups": len(groups),
        "ci95": None,
        "weighting": "equal agent seed, equal scenario group; repeats averaged within group",
    }
    if len(seeds) < 2 or len(groups) < 2:
        result["warning"] = (
            "At least two agent seeds and two scenario groups are required for an interval; "
            "the talk-sized single-seed result is descriptive only."
        )
        return result
    rng = np.random.default_rng(seed)
    bootstrap = [
        delta[
            np.ix_(rng.integers(len(seeds), size=len(seeds)), rng.integers(len(groups), size=len(groups)))
        ].mean()
        for _ in range(samples)
    ]
    result["ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return result


def summarize(run_dirs: list[Path], *, split: str = "validation") -> dict:
    curves: dict[tuple[str, int], list[dict]] = defaultdict(list)
    final: dict[str, list[dict]] = defaultdict(list)
    metadata = []
    seen = set()
    for directory in run_dirs:
        meta = json.loads((directory / "metadata.json").read_text())
        metadata.append(meta)
        mode, seed = meta["config"]["mode"], meta["config"]["seed"]
        prior_name = meta.get("provenance", {}).get("prior_name", "uniform")
        arm = f"{prior_name}_{mode}"
        if (arm, seed) in seen:
            raise ValueError("Duplicate arm/seed runs; choose one preregistered run per arm/seed")
        seen.add((arm, seed))
        rows = read_jsonl(directory / ("test-evaluation.jsonl" if split == "test" else "evaluation.jsonl"))
        rows = [row for row in rows if row["split"] == split]
        if not rows:
            raise ValueError(f"No {split} evaluation in {directory}")
        final_step = max(row["train_steps"] for row in rows)
        if final_step != meta["config"]["total_steps"]:
            raise ValueError("Final result must use the complete, fixed-budget checkpoint")
        for row in rows:
            curves[arm, row["train_steps"]].append(row)
            if row["train_steps"] == final_step:
                final[arm].append(row)
    for field in ("corpus_sha256", "rom_sha256", "source_revision", "feature_version"):
        if len({str(meta[field]) for meta in metadata}) > 1:
            raise ValueError(f"Runs disagree on {field}")
    for field in ("total_steps", "max_decisions", "eval_repeats"):
        if len({meta["config"][field] for meta in metadata}) > 1:
            raise ValueError(f"Runs have unequal {field}")
    points = []
    for (mode, steps), rows in sorted(curves.items()):
        scores = _group_scores(rows)
        party_hp = [
            row["final_party_hp_fraction"]
            for row in rows
            if row["final_party_hp_fraction"] is not None
        ]
        points.append(
            {
                "mode": mode,
                "train_steps": steps,
                "win_rate": float(np.mean(list(scores.values()))),
                "agent_seeds": len({key[0] for key in scores}),
                "episodes": len(rows),
                "truncation_rate": float(np.mean([row["outcome"] == "truncated" for row in rows])),
                "mean_decisions": float(np.mean([row["decisions"] for row in rows])),
                "mean_turns": float(np.mean([row["turns"] for row in rows])),
                "mean_final_party_hp_fraction": float(np.mean(party_hp)) if party_hp else None,
                "mean_battle_score": float(np.mean([row["battle_score"] for row in rows])),
                "mean_prior_kl": float(np.mean([row["mean_prior_kl"] for row in rows])),
            }
        )
    comparisons = {}
    for prior_name in ("laya", "prism", "jev"):
        residual = f"{prior_name}_residual"
        frozen = f"{prior_name}_frozen"
        if residual in final and frozen in final:
            comparisons[f"{residual}_minus_{frozen}"] = paired_comparison(
                final[residual], final[frozen]
            )
    auc = {}
    for mode in final:
        series = sorted((point for point in points if point["mode"] == mode), key=lambda x: x["train_steps"])
        if len(series) > 1 and series[0]["train_steps"] == 0:
            xs = np.array([point["train_steps"] for point in series])
            ys = np.array([point["win_rate"] for point in series])
            auc[mode] = float(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2) / xs[-1])
    return {
        "split": split,
        "curves": points,
        "normalized_learning_auc": auc,
        "paired_final": comparisons,
        "note": "Validation curves guide development. Only untouched fixed-budget test evaluation supports final claims.",
    }
