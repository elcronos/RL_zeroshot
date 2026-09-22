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
    width, height = 940, 170 + 70 * len(rows)
    image = Image.new("RGB", (width, height), "#0e1726")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 26), "Held-out zero-shot Rogue battles", fill="white", font=font)
    draw.text((30, 46), "Win rate ceiling is visible; lower decisions are descriptive, not a quality ranking.", fill="#b9c6d8", font=font)
    colors = {"laya": "#6ee7b7", "prism": "#93c5fd", "uniform": "#fbbf24", "jev": "#fda4af"}
    for index, (name, summary) in enumerate(rows):
        y = 100 + 70 * index
        win = float(summary["episode_win_rate"] or 0)
        draw.text((30, y), name.upper(), fill="white", font=font)
        draw.rectangle((180, y, 680, y + 20), fill="#1f2937")
        draw.rectangle((180, y, 180 + int(500 * win), y + 20), fill=colors.get(name, "#d1d5db"))
        draw.text((700, y), f"{win:.0%} wins", fill="white", font=font)
        draw.text((180, y + 30), f"mean decisions {summary['mean_decisions']:.2f}", fill="#b9c6d8", font=font)
        draw.text((430, y + 30), f"p50 decision {summary['median_battle_decision_ms_p50']:.2f} ms", fill="#b9c6d8", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
