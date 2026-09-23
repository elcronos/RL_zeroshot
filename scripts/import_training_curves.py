"""Import or verify PPO update logs in the checked-in result registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_METRICS = {
    "train_steps",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "prior_kl",
    "approx_kl",
    "grad_norm",
    "clip_fraction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_updates(path: Path) -> list[dict]:
    points = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not points:
        raise ValueError(f"no PPO updates in {path}")
    for index, point in enumerate(points, start=1):
        if set(point) != REQUIRED_METRICS:
            raise ValueError(f"unexpected metrics in {path} line {index}")
    return points


def curve_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("curve must use MODEL=PATH")
        model, raw_path = value.split("=", 1)
        if not model or not raw_path or model in result:
            raise ValueError("curve models and paths must be non-empty and unique")
        result[model] = Path(raw_path)
    return result


def update_registry(registry: dict, sources: dict[str, Path], check: bool) -> None:
    curves = {curve["model"]: curve for curve in registry.get("training_curves", [])}
    if set(curves) != set(sources):
        raise ValueError("curve arguments must cover every registry training model")

    for model, source in sources.items():
        points = load_updates(source)
        digest = sha256(source)
        curve = curves[model]
        if check:
            if curve.get("points") != points or curve.get("updates_sha256") != digest:
                raise RuntimeError(f"{model} registry curve does not match {source}")
        else:
            curve["updates_artifact"] = source.name
            curve["updates_sha256"] = digest
            curve["points"] = points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/results.json"))
    parser.add_argument("--curve", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--check", action="store_true", help="verify without changing the registry")
    args = parser.parse_args()
    registry = json.loads(args.results.read_text())
    update_registry(registry, curve_paths(args.curve), args.check)
    if not args.check:
        args.results.write_text(json.dumps(registry, indent=2) + "\n")


if __name__ == "__main__":
    main()
