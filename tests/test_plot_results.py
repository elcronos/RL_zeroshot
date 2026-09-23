import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "plot_results", Path(__file__).parents[1] / "scripts/plot_results.py"
)
assert spec is not None and spec.loader is not None
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def curve(model: str, steps: tuple[int, ...] = (256, 512)) -> dict:
    return {
        "model": model,
        "learner_seed": 0,
        "points": [
            {
                "train_steps": step,
                "loss": 0.1 / index,
                "policy_loss": -0.01 * index,
                "value_loss": 0.2 / index,
                "entropy": 1.5,
                "prior_kl": 0.001,
                "approx_kl": 0.001,
                "grad_norm": 0.3,
                "clip_fraction": 0.0,
            }
            for index, step in enumerate(steps, start=1)
        ],
    }


def test_training_loss_plot_renders_registry_curves(tmp_path):
    output = tmp_path / "training-loss.png"
    data = {
        "policy_protocol": {
            "train_steps": 512,
            "rollout_steps": 256,
            "local_models": ["Laya", "PrismNLI-0.4B"],
            "learner_seeds": [0],
        },
        "training_curves": [curve("Laya"), curve("PrismNLI-0.4B")],
    }

    plots.training_loss(data, output)

    with Image.open(output) as image:
        assert image.size == (1040, 825)
        assert image.mode == "RGB"


def test_training_loss_plot_rejects_misaligned_update_steps(tmp_path):
    data = {
        "policy_protocol": {
            "train_steps": 512,
            "rollout_steps": 256,
            "local_models": ["Laya", "PrismNLI-0.4B"],
            "learner_seeds": [0],
        },
        "training_curves": [
            curve("Laya"),
            curve("PrismNLI-0.4B", steps=(256, 768)),
        ]
    }

    with pytest.raises(ValueError, match="every protocol update step"):
        plots.training_loss(data, tmp_path / "training-loss.png")


def test_training_loss_plot_rejects_invalid_auxiliary_diagnostic(tmp_path):
    data = {
        "policy_protocol": {
            "train_steps": 512,
            "rollout_steps": 256,
            "local_models": ["Laya", "PrismNLI-0.4B"],
            "learner_seeds": [0],
        },
        "training_curves": [curve("Laya"), curve("PrismNLI-0.4B")],
    }
    data["training_curves"][0]["points"][0]["clip_fraction"] = 1.1

    with pytest.raises(ValueError, match="invalid clip_fraction"):
        plots.training_loss(data, tmp_path / "training-loss.png")
