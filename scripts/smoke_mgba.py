#!/usr/bin/env python3
"""Boot the real ROM, verify the actual Lua HELLO, and capture a boot frame.

This validates transport and ROM execution, not battle mechanics or learning.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from rogue_rl.manifest import sha256_file
from rogue_rl.mgba import MGBAEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=Path(".cache/mgba-runner"))
    parser.add_argument("--rom", type=Path, default=Path("data/rogue-research.gba"))
    parser.add_argument("--profile", type=Path, default=Path("data/rogue-profile.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/mgba-smoke"))
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if sha256_file(args.rom) != profile["rom_sha256"]:
        raise ValueError("ROM SHA256 differs from profile")
    args.output.mkdir(parents=True, exist_ok=True)
    screenshot = (args.output / "boot.png").resolve()
    screenshot.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "smoke.lua"
        script.write_text(
            f"dofile({json.dumps(str(args.profile.with_suffix('.lua').resolve()))})\n"
            "local frames=0\ncallbacks:add('frame',function()\n"
            " frames=frames+1\n"
            f" if frames==1200 then emu:screenshot({json.dumps(str(screenshot))}) end\n"
            "end)\n"
        )
        with (args.output / "emulator.log").open("w") as log:
            process = subprocess.Popen(
                [str(args.host.resolve()), str(args.rom.resolve()), str(script)],
                stdout=log,
                stderr=log,
            )
            try:
                deadline = time.monotonic() + 20
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("mGBA exited; inspect emulator.log")
                    try:
                        with MGBAEnv(profile_path=args.profile, port=profile["port"], timeout=5) as env:
                            hello = env.describe()
                        break
                    except ConnectionRefusedError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
                while not screenshot.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Boot capture did not complete; inspect emulator.log")
                    time.sleep(0.05)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"mGBA exited with {process.returncode}; inspect emulator.log")
    if not screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("mGBA did not produce a PNG screenshot")
    result = {
        "scope": "real ROM boot and Lua transport only; no battle validation",
        "hello": hello,
        "host_sha256": sha256_file(args.host),
        "profile_sha256": sha256_file(args.profile),
        "screenshot": str(screenshot),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
