"""Render zero-shot and frozen-vs-trained plots from the checked-in result registry."""

from __future__ import annotations

import argparse
import json
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
    image, draw, font = _canvas(
        "Frozen prior vs learned residual policy",
        f"Policy: {data['dataset']['train_battles']} train + {data['dataset']['validation_battles']} validation; "
        f"final comparison: {data['dataset']['test_battles']} held-out test battles",
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
        else:
            draw.rectangle((230, policy_y, 830, policy_y + 22), outline=GRID, width=2)
            draw.text((850, policy_y + 3), "trained NOT RUN", fill=MUTED, font=font)
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


if __name__ == "__main__":
    main()
