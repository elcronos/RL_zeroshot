"""Merge independently captured corpus manifests without changing battle records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogue_rl.manifest import Corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output in args.manifests:
        parser.error("Output must differ from every input manifest")

    payloads = [json.loads(path.read_text()) for path in args.manifests]
    reference = payloads[0]
    for payload in payloads[1:]:
        for key in ("schema_version", "rom_sha256", "source_revision"):
            if payload[key] != reference[key]:
                raise ValueError(f"Cannot merge manifests with different {key}")
    battles = [battle for payload in payloads for battle in payload["battles"]]
    ids = [battle["id"] for battle in battles]
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot merge duplicate battle IDs")
    reference["battles"] = battles
    reference["cohort"] = "merged capture lanes; split assignment pending"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2) + "\n")
    Corpus.load(args.output)
    print(json.dumps({"manifest": str(args.output), "battles": len(battles)}, indent=2))


if __name__ == "__main__":
    main()
