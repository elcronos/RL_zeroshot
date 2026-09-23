"""Render a compact, checked-in zero-shot baseline figure from run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for directory in args.runs:
        summary = json.loads((directory / "summary.json").read_text())
        rows.append((summary["prior"], summary))
    width, height = 940, 270 + 70 * len(rows)
    image = Image.new("RGB", (width, height), "#0e1726")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 26), "Held-out zero-shot Rogue battles", fill="white", font=font)
    draw.text((30, 46), "Win rate ceiling is visible; battle score summarizes retained HP and game turns.", fill="#b9c6d8", font=font)
    colors = {"laya": "#6ee7b7", "prism": "#93c5fd", "uniform": "#fbbf24", "jev": "#fda4af"}
    for index, (name, summary) in enumerate(rows):
        y = 100 + 70 * index
        win = float(summary["episode_win_rate"] or 0)
        draw.text((30, y), name.upper(), fill="white", font=font)
        draw.rectangle((180, y, 680, y + 20), fill="#1f2937")
        draw.rectangle((180, y, 180 + int(500 * win), y + 20), fill=colors.get(name, "#d1d5db"))
        draw.text((700, y), f"{win:.0%} wins", fill="white", font=font)
        draw.text((180, y + 30), f"battle score {summary['mean_battle_score']:.1f}/100", fill="#b9c6d8", font=font)
        draw.text((400, y + 30), f"turns {summary['mean_turns']:.2f}", fill="#b9c6d8", font=font)
        draw.text((570, y + 30), f"party HP {100 * summary['mean_final_party_hp_fraction']:.1f}%", fill="#b9c6d8", font=font)
    panel_y = 120 + 70 * len(rows)
    draw.rectangle((30, panel_y, 910, panel_y + 84), outline="#475569", width=2)
    draw.text((50, panel_y + 16), "POLICY LEARNING / RESIDUAL PPO", fill="white", font=font)
    draw.text(
        (50, panel_y + 38),
        "Not run: uniform, Laya, and PrismNLI each won every held-out fixture battle.",
        fill="#fbbf24",
        font=font,
    )
    draw.text(
        (50, panel_y + 58),
        "Train residual arms only after a harder corpus permits a paired held-out win-rate comparison.",
        fill="#b9c6d8",
        font=font,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
