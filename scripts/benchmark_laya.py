#!/usr/bin/env python3
"""Benchmark real frozen CoreML on semantic fixtures, never simulated game results."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from rogue_rl.laya import LayaConfig, LayaPrior, semantic_request


def fixture():
    return {
        "player": {"species_name": "Gyarados", "type_ids": [11, 2], "hp_fraction": 0.61},
        "opponent": {"species_name": "Jolteon", "type_ids": [13], "hp_fraction": 0.82},
        "moves": [
            {"name": "Waterfall", "type": 11, "power": 80},
            {"name": "Ice Fang", "type": 15, "power": 65},
            {"name": "Dragon Dance", "type": 16, "power": 0},
            {"name": "Earthquake", "type": 4, "power": 100},
        ],
        "party": [
            {"species_name": name, "type_ids": types, "hp_fraction": hp}
            for name, types, hp in [
                ("Gyarados", [11, 2], 0.61),
                ("Swampert", [11, 4], 0.92),
                ("Breloom", [12, 1], 0.47),
                ("Skeledirge", [10, 7], 0.81),
                ("Corviknight", [8, 2], 0.74),
                ("Alolan Ninetales", [15, 18], 0.62),
            ]
        ],
        "legal_actions": [True] * 4 + [False] + [True] * 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/laya-ane96")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("runs/laya-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    start = time.perf_counter()
    prior = LayaPrior(LayaConfig(model_dir=args.model, cache_size=0))
    load_seconds = time.perf_counter() - start
    obs = fixture()
    lengths = [
        prior.predictor.token_count(*semantic_request(obs, int(action)))
        for action in np.flatnonzero(obs["legal_actions"])
    ]
    prior.probabilities(obs)  # warmup; cache disabled in measurement
    elapsed = []
    for _ in range(args.repeats):
        probs = prior.probabilities(obs)
        elapsed.append(1000 * prior.last_metrics["total_seconds"])
    result = {
        "fixture_only": True,
        "game_emulation": False,
        "platform": platform.platform(),
        "load_seconds": load_seconds,
        "legal_actions": sum(obs["legal_actions"]),
        "tokens_per_action": lengths,
        "repeats": args.repeats,
        "cache_enabled": False,
        "full_decision_ms_p50": float(np.median(elapsed)),
        "full_decision_ms_p95": float(np.percentile(elapsed, 95)),
        "probabilities": probs.tolist(),
        "provenance": prior.provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
