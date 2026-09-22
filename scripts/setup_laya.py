#!/usr/bin/env python3
"""Download an exact upstream release; never download weights during experiments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rogue_rl.laya import PRESETS, UPSTREAM_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="ane96")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    repo, revision, _, _ = PRESETS[args.preset]
    destination = args.output or Path("models") / ("laya-" + args.preset)
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "rogue_laya_provenance.json"
    expected = {"repo_id": repo, "revision": revision, "source_commit": UPSTREAM_SHA}
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != expected:
        raise SystemExit("Destination belongs to a different model revision; choose another directory")
    snapshot_download(repo_id=repo, revision=revision, local_dir=destination)
    # Upstream ANE manifests include checksum inventories for all inference artifacts.
    from laya_coreml.artifacts import verify_files

    manifest = json.loads((destination / "coreml_config.json").read_text())
    if manifest.get("files"):
        verify_files(destination, manifest["files"])
    receipt_path.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"Downloaded verified pinned bundle to {destination.resolve()}")


if __name__ == "__main__":
    main()
