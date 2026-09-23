"""Render evaluation and PPO training plots from the checked-in result registry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#0e1726"
TEXT = "#f8fafc"
MUTED = "#b9c6d8"
GRID = "#334155"
COLORS = {
    "PrismNLI-0.4B": "#93c5fd",
    "Laya": "#6ee7b7",
    "Jev": "#fda4af",
    "Uniform (random)": "#fbbf24",
}


def _canvas(title: str, subtitle: str, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont]:
    image = Image.new("RGB", (1040, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((32, 26), title, fill=TEXT, font=font)
    draw.text((32, 48), subtitle, fill=MUTED, font=font)
    return image, draw, font


def _bar(draw: ImageDraw.ImageDraw, y: int, score: float, color: str) -> None:
    left, right = 230, 830
    draw.rectangle((left, y, right, y + 22), fill="#1f2937")
    draw.rectangle((left, y, left + int((right - left) * score / 100), y + 22), fill=color)


def zero_shot(data: dict, output: Path) -> None:
    rows = data["zero_shot"]
    image, draw, font = _canvas(
        "Zero-shot held-out battle quality",
        f"0 training battles | {data['dataset']['test_battles']} test battles in "
        f"{data['dataset']['test_scenario_groups']} held-out scenario groups",
        140 + 78 * len(rows),
    )
    for index, row in enumerate(rows):
        y = 96 + 78 * index
        draw.text((32, y + 3), row["model"], fill=TEXT, font=font)
        if row["status"] != "complete":
            draw.rectangle((230, y, 830, y + 22), outline=GRID, width=2)
            draw.text((850, y + 3), "NOT RUN", fill=MUTED, font=font)
            continue
        _bar(draw, y, row["battle_score"], COLORS[row["model"]])
        draw.text((850, y + 3), f"{row['battle_score']:.1f}/100", fill=TEXT, font=font)
        details = (
            f"wins {row['wins']}/{row['test_battles']}  |  turns {row['mean_turns']:.2f}  |  "
            f"party HP {row['final_party_hp_percent']:.1f}%"
        )
        draw.text((230, y + 34), details, fill=MUTED, font=font)
    image.save(output)


def trained(data: dict, output: Path) -> None:
    zero = {row["model"]: row for row in data["zero_shot"]}
    rows = data["trained_policy"]
    protocol = data.get("policy_protocol", {})
    wall_minutes = protocol.get("end_to_end_wall_seconds", 0) / 60
    image, draw, font = _canvas(
        "Frozen prior vs learned residual policy",
        f"{protocol.get('train_steps', '?')} decisions | seed 0 | "
        f"{data['dataset']['validation_battles']} validation + "
        f"{data['dataset']['test_battles']} held-out test battles | local study {wall_minutes:.1f} min",
        160 + 112 * len(rows),
    )
    for index, row in enumerate(rows):
        y = 94 + 112 * index
        base = zero[row["base_model"]]
        draw.text((32, y + 3), row["base_model"], fill=TEXT, font=font)
        if base["status"] == "complete":
            _bar(draw, y, base["battle_score"], COLORS[row["base_model"]])
            draw.text((850, y + 3), f"frozen {base['battle_score']:.1f}", fill=TEXT, font=font)
        else:
            draw.rectangle((230, y, 830, y + 22), outline=GRID, width=2)
            draw.text((850, y + 3), "frozen NOT RUN", fill=MUTED, font=font)
        policy_y = y + 40
        if row["status"] == "complete":
            _bar(draw, policy_y, row["battle_score"], "#c4b5fd")
            draw.text((850, policy_y + 3), f"trained {row['battle_score']:.1f}", fill=TEXT, font=font)
            details = (
                f"wins {row['wins']}/{row['test_battles']}  |  turns {row['mean_turns']:.2f}  |  "
                f"party HP {row['final_party_hp_percent']:.1f}%  |  "
                f"distinct train battles {row['training_battles']}"
            )
            draw.text((230, policy_y + 30), details, fill=MUTED, font=font)
        else:
            draw.rectangle((230, policy_y, 830, policy_y + 22), outline=GRID, width=2)
            draw.text((850, policy_y + 3), "trained NOT RUN", fill=MUTED, font=font)
    image.save(output)


def _validated_training_curves(data: dict) -> list[dict]:
    curves = data.get("training_curves", [])
    if not curves:
        raise ValueError("results registry has no training_curves")

    protocol = data.get("policy_protocol", {})
    total_steps = protocol.get("train_steps")
    rollout_steps = protocol.get("rollout_steps")
    if not isinstance(total_steps, int) or not isinstance(rollout_steps, int):
        raise TypeError("policy protocol needs integer train_steps and rollout_steps")
    if total_steps < 1 or rollout_steps < 1 or total_steps % rollout_steps:
        raise ValueError("training steps must be a positive multiple of rollout steps")
    expected_steps = list(range(rollout_steps, total_steps + 1, rollout_steps))
    expected_models = set(protocol.get("local_models", []))
    expected_seeds = set(protocol.get("learner_seeds", []))
    observed_pairs: set[tuple[str, int]] = set()

    for curve in curves:
        model = curve.get("model")
        seed = curve.get("learner_seed")
        if model not in expected_models or seed not in expected_seeds:
            raise ValueError("training curve model and seed must match the policy protocol")
        pair = (model, seed)
        if pair in observed_pairs:
            raise ValueError("training curve model and seed pairs must be unique")
        observed_pairs.add(pair)
        points = curve.get("points", [])
        if len(points) < 2:
            raise ValueError(f"{model} needs at least two loss points")
        steps = [point["train_steps"] for point in points]
        if steps != expected_steps:
            raise ValueError("training curves must contain every protocol update step")
        for point in points:
            for metric in (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "prior_kl",
                "approx_kl",
                "grad_norm",
                "clip_fraction",
            ):
                if not math.isfinite(point[metric]):
                    raise ValueError(f"non-finite {metric} at step {point['train_steps']}")
            for metric in ("value_loss", "entropy", "prior_kl", "approx_kl", "grad_norm"):
                if point[metric] < 0:
                    raise ValueError(f"negative {metric} at step {point['train_steps']}")
            if not 0 <= point["clip_fraction"] <= 1:
                raise ValueError(f"invalid clip_fraction at step {point['train_steps']}")

    expected_pairs = {(model, seed) for model in expected_models for seed in expected_seeds}
    if observed_pairs != expected_pairs:
        raise ValueError("training curves must cover every local model and learner seed")
    return curves


def _line_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    curves: list[dict],
    metric: str,
    title: str,
    top: int,
) -> None:
    left, right, bottom = 122, 1000, top + 168
    values = [point[metric] for curve in curves for point in curve["points"]]
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    padding = max((high - low) * 0.12, 0.001)
    low, high = low - padding, high + padding
    steps = [point["train_steps"] for point in curves[0]["points"]]

    def x_position(step: int) -> float:
        return left + (right - left) * (step - steps[0]) / (steps[-1] - steps[0])

    def y_position(value: float) -> float:
        return bottom - (bottom - top) * (value - low) / (high - low)

    draw.text((32, top - 20), title, fill=TEXT, font=font)
    for index in range(5):
        value = low + (high - low) * index / 4
        y = y_position(value)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((32, y - 5), f"{value:.3f}", fill=MUTED, font=font)
    zero_y = y_position(0.0)
    draw.line((left, zero_y, right, zero_y), fill="#64748b", width=2)

    for step in steps:
        x = x_position(step)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = str(step)
        label_width = draw.textlength(label, font=font)
        draw.text((x - label_width / 2, bottom + 8), label, fill=MUTED, font=font)

    for curve in curves:
        color = COLORS[curve["model"]]
        plotted = [
            (x_position(point["train_steps"]), y_position(point[metric]))
            for point in curve["points"]
        ]
        draw.line(plotted, fill=color, width=4)
        for x, y in plotted:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=BACKGROUND, width=2)


def training_loss(data: dict, output: Path) -> None:
    curves = _validated_training_curves(data)
    protocol = data.get("policy_protocol", {})
    seeds = ", ".join(str(seed) for seed in protocol["learner_seeds"])
    seed_label = "seed" if len(protocol["learner_seeds"]) == 1 else "seeds"
    image, draw, font = _canvas(
        "Residual PPO training diagnostics",
        f"{protocol.get('train_steps', '?')} decisions | {protocol.get('rollout_steps', '?')}-step updates | "
        f"{seed_label} {seeds} | fresh on-policy batch at each point",
        825,
    )
    legend_x = 690
    for index, curve in enumerate(curves):
        y = 74 + index * 18
        color = COLORS[curve["model"]]
        draw.line((legend_x, y + 5, legend_x + 24, y + 5), fill=color, width=4)
        draw.text((legend_x + 34, y), curve["model"], fill=TEXT, font=font)

    _line_panel(draw, font, curves, "loss", "Combined PPO objective", 130)
    _line_panel(draw, font, curves, "policy_loss", "Policy surrogate loss", 372)
    _line_panel(draw, font, curves, "value_loss", "Value loss (weighted x0.5 in objective)", 614)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zero_shot(data, args.output_dir / "zero-shot-performance.png")
    trained(data, args.output_dir / "trained-policy-performance.png")
    training_loss(data, args.output_dir / "training-loss.png")


if __name__ == "__main__":
    main()
