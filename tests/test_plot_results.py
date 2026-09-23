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


def evaluation_data() -> dict:
    zero = [
        {
            "model": model,
            "status": "not_run" if model == "Jev" else "complete",
            "battle_score": None if model == "Jev" else score,
            "wins": 18,
            "truncations": 0,
            "test_battles": 18,
        }
        for model, score in (("PrismNLI-0.4B", 92.1), ("Laya", 85.3), ("Jev", None))
    ]
    zero.append({"model": "Uniform (random)", "status": "complete", "battle_score": 87.4})
    trained = [
        {
            "base_model": model,
            "status": "not_run" if model == "Jev" else "complete",
            "battle_score": None if model == "Jev" else score,
            "wins": 18,
            "truncations": 0,
            "test_battles": 18,
            "training_battles": training,
        }
        for model, score, training in (
            ("PrismNLI-0.4B", 92.3, 144),
            ("Laya", 88.1, 138),
            ("Jev", None, None),
        )
    ]
    return {"dataset": {"test_battles": 18}, "zero_shot": zero, "trained_policy": trained}


def test_model_comparison_renders_three_models_and_pending_result(tmp_path):
    output = tmp_path / "model-comparison.png"

    plots.model_comparison(evaluation_data(), output)

    with Image.open(output) as image:
        assert image.size == (1280, 820)
        assert image.mode == "RGB"


def test_model_comparison_rejects_nonfinite_completed_score(tmp_path):
    data = evaluation_data()
    data["zero_shot"][0]["battle_score"] = float("nan")

    with pytest.raises(ValueError, match="finite battle score"):
        plots.model_comparison(data, tmp_path / "model-comparison.png")


def test_model_comparison_renders_low_score_on_full_scale(tmp_path):
    data = evaluation_data()
    data["zero_shot"][0]["battle_score"] = 10.0

    plots.model_comparison(data, tmp_path / "model-comparison.png")


def test_model_comparison_keeps_completed_zero_shot_when_policy_is_pending(tmp_path):
    data = evaluation_data()
    jev = next(row for row in data["zero_shot"] if row["model"] == "Jev")
    jev.update(status="complete", battle_score=80.0, wins=15)
    output = tmp_path / "model-comparison.png"

    plots.model_comparison(data, output)

    with Image.open(output) as image:
        assert image.getpixel((950, 600)) == (56, 189, 248)


def test_model_comparison_rejects_unknown_status(tmp_path):
    data = evaluation_data()
    data["trained_policy"][0]["status"] = "typo"

    with pytest.raises(ValueError, match="status"):
        plots.model_comparison(data, tmp_path / "model-comparison.png")


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
