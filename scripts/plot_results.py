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
SERIES_COLORS = {"Zero shot": "#38bdf8", "Residual PPO": "#a78bfa"}
MODEL_ORDER = ("PrismNLI-0.4B", "Laya", "Jev")


def _canvas(title: str, subtitle: str, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont]:
    image = Image.new("RGB", (1040, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((32, 26), title, fill=TEXT, font=font)
    draw.text((32, 48), subtitle, fill=MUTED, font=font)
    return image, draw, font


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    value: str,
    font: ImageFont.ImageFont,
    color: str,
) -> None:
    width = draw.textlength(value, font=font)
    draw.text((center_x - width / 2, y), value, font=font, fill=color)


def _validated_evaluation_rows(data: dict) -> tuple[dict[str, dict], dict[str, dict], dict]:
    zero = {row["model"]: row for row in data.get("zero_shot", [])}
    trained = {row["base_model"]: row for row in data.get("trained_policy", [])}
    if not set(MODEL_ORDER) <= zero.keys() or not set(MODEL_ORDER) <= trained.keys():
        raise ValueError("comparison requires zero-shot and policy rows for PrismNLI, Laya, and Jev")
    uniform = zero.get("Uniform (random)")
    if not uniform or uniform.get("status") != "complete":
        raise ValueError("comparison requires the completed uniform control")
    for row in [uniform, *(zero[model] for model in MODEL_ORDER), *(trained[model] for model in MODEL_ORDER)]:
        if row.get("status") not in {"complete", "not_run"}:
            raise ValueError("comparison status must be complete or not_run")
        if row.get("status") == "complete":
            score = row.get("battle_score")
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError("completed comparison rows need a finite battle score in [0, 100]")
    return zero, trained, uniform


def model_comparison(data: dict, output: Path) -> None:
    zero, trained, uniform = _validated_evaluation_rows(data)
    image = Image.new("RGB", (1280, 820), "#0b1220")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = _font(40, bold=True), _font(19)
    label_font, small_font = _font(18, bold=True), _font(15)
    score_font, model_font = _font(24, bold=True), _font(25, bold=True)

    draw.text((64, 34), "Can a learned policy improve battle quality?", fill=TEXT, font=title_font)
    subtitle = (
        f"Frozen zero shot vs residual PPO  •  {data['dataset']['test_battles']} held-out battles  •  "
        "higher is better"
    )
    draw.text((66, 88), subtitle, fill=MUTED, font=subtitle_font)

    legend_y = 132
    legend_x = 66
    for name in ("Zero shot", "Residual PPO"):
        color = SERIES_COLORS[name]
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 28, legend_y + 18), radius=5, fill=color)
        draw.text((legend_x + 40, legend_y - 2), name, fill=TEXT, font=small_font)
        legend_x += 176
    draw.line((410, legend_y + 9, 446, legend_y + 9), fill=COLORS["Uniform (random)"], width=3)
    draw.text((458, legend_y - 2), "Uniform random reference", fill=TEXT, font=small_font)

    left, right, top, bottom = 104, 1216, 196, 650
    completed_scores = [float(uniform["battle_score"])]
    for model in MODEL_ORDER:
        completed_scores.extend(
            float(row["battle_score"])
            for row in (zero[model], trained[model])
            if row["status"] == "complete"
        )
    y_min, y_max = (50.0 if min(completed_scores) >= 50 else 0.0), 100.0

    def y_position(score: float) -> float:
        return bottom - (score - y_min) * (bottom - top) / (y_max - y_min)

    tick_step = 10 if y_min == 50 else 20
    for value in range(int(y_min), 101, tick_step):
        y = y_position(value)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = str(value)
        draw.text((left - 18 - draw.textlength(label, font=small_font), y - 9), label, fill=MUTED, font=small_font)
    draw.text((34, 176), "BATTLE SCORE", fill=MUTED, font=_font(13, bold=True))

    reference_y = y_position(float(uniform["battle_score"]))
    for x in range(left, right, 18):
        draw.line((x, reference_y, min(x + 10, right), reference_y), fill=COLORS["Uniform (random)"], width=3)
    reference = f"Uniform random  {uniform['battle_score']:.1f}"
    ref_width = draw.textlength(reference, font=small_font)
    draw.rounded_rectangle(
        (right - ref_width - 22, reference_y - 29, right, reference_y - 5),
        radius=7,
        fill="#3b3218",
    )
    draw.text((right - ref_width - 11, reference_y - 27), reference, fill="#fde68a", font=small_font)

    centers = (290, 650, 1010)
    bar_width, gap = 92, 20
    for center, model in zip(centers, MODEL_ORDER, strict=True):
        base, policy = zero[model], trained[model]
        draw.rounded_rectangle(
            (center - 160, top - 10, center + 160, bottom + 118),
            radius=18,
            fill="#101b2d",
            outline="#1e293b",
            width=2,
        )
        rows = ((base, "Zero shot", center - bar_width - gap / 2), (policy, "Residual PPO", center + gap / 2))
        for row, series, x in rows:
            if row["status"] == "complete":
                score = float(row["battle_score"])
                y = y_position(score)
                draw.rounded_rectangle(
                    (x, y, x + bar_width, bottom),
                    radius=11,
                    fill=SERIES_COLORS[series],
                )
                _centered_text(draw, x + bar_width / 2, y - 34, f"{score:.1f}", score_font, TEXT)
            else:
                pending_top = bottom - 132
                draw.rounded_rectangle(
                    (x, pending_top, x + bar_width, bottom),
                    radius=11,
                    fill="#131c2b",
                    outline=SERIES_COLORS[series],
                    width=3,
                )
                _centered_text(draw, x + bar_width / 2, pending_top + 44, "PENDING", _font(13, bold=True), MUTED)

        if base["status"] == "complete" and policy["status"] == "complete":
            delta = float(policy["battle_score"]) - float(base["battle_score"])
            delta_y = min(y_position(float(base["battle_score"])), y_position(float(policy["battle_score"]))) - 71
            delta_text = f"{delta:+.1f}"
            delta_width = draw.textlength(delta_text, font=label_font)
            draw.rounded_rectangle(
                (center - delta_width / 2 - 13, delta_y, center + delta_width / 2 + 13, delta_y + 31),
                radius=15,
                fill="#21304a",
                outline="#64748b",
            )
            _centered_text(draw, center, delta_y + 4, delta_text, label_font, TEXT)
            detail = f"{policy['wins']}/{policy['test_battles']} wins  •  {policy['truncations']} truncated"
            training = f"0 → {policy['training_battles']} training states"
        elif base["status"] == "complete":
            detail = f"Zero shot: {base['wins']}/{base['test_battles']} wins"
            training = "Residual policy pending"
        elif policy["status"] == "complete":
            detail = f"Residual PPO: {policy['wins']}/{policy['test_battles']} wins"
            training = f"Zero shot pending  •  {policy['training_battles']} training states"
        else:
            detail = "Both results pending"
            training = "No score inferred"

        _centered_text(draw, center, bottom + 24, model, model_font, COLORS[model])
        _centered_text(draw, center, bottom + 61, detail, small_font, MUTED)
        _centered_text(draw, center, bottom + 86, training, small_font, "#94a3b8")

    scale_note = "Win range shown (50–100)" if y_min == 50 else "Full score range shown (0–100)"
    footer = f"{scale_note}  •  one learner seed  •  underlying model weights remain frozen"
    _centered_text(draw, 640, 786, footer, small_font, MUTED)
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
    expected_models = set(protocol.get("trained_models", []))
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
        raise ValueError("training curves must cover every trained model and learner seed")
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
    model_comparison(data, args.output_dir / "model-comparison.png")
    training_loss(data, args.output_dir / "training-loss.png")


if __name__ == "__main__":
    main()
