import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "import_training_curves", Path(__file__).parents[1] / "scripts/import_training_curves.py"
)
assert spec is not None and spec.loader is not None
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)


def write_updates(path: Path) -> list[dict]:
    points = [
        {
            "train_steps": 256,
            "loss": 0.1,
            "policy_loss": -0.01,
            "value_loss": 0.2,
            "entropy": 1.5,
            "prior_kl": 0.001,
            "grad_norm": 0.3,
            "approx_kl": 0.001,
            "clip_fraction": 0.0,
        }
    ]
    path.write_text("".join(json.dumps(point) + "\n" for point in points))
    return points


def test_update_registry_imports_and_then_verifies_exact_log(tmp_path):
    source = tmp_path / "updates.jsonl"
    points = write_updates(source)
    registry = {"training_curves": [{"model": "Laya", "learner_seed": 0, "points": []}]}

    curves.update_registry(registry, {"Laya": source}, check=False)

    assert registry["training_curves"][0]["points"] == points
    assert registry["training_curves"][0]["updates_artifact"] == "updates.jsonl"
    assert registry["training_curves"][0]["updates_sha256"] == curves.sha256(source)
    curves.update_registry(registry, {"Laya": source}, check=True)


def test_update_registry_check_detects_changed_log(tmp_path):
    source = tmp_path / "updates.jsonl"
    write_updates(source)
    registry = {
        "training_curves": [
            {"model": "Laya", "points": [], "updates_sha256": "wrong"},
        ]
    }

    with pytest.raises(RuntimeError, match="does not match"):
        curves.update_registry(registry, {"Laya": source}, check=True)
