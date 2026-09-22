#!/usr/bin/env python3
"""Capture reproducible, first-action Rogue trainer battles with mGBA.

Each state starts from a fresh game.  The instrumented ROM selects one of six
declared three-Pokémon player teams and Rogue selects an ordinary route trainer.
The manifest records both choices so train/validation/test grouping can prevent
the same team/trainer matchup from crossing a split.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rogue_rl.manifest import Corpus, sha256_file

MAGIC = 0x524C5247
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def symbol_address(symbols: Path, name: str) -> int:
    matches = re.findall(rf"^([0-9a-fA-F]+)\s+\w\s+{re.escape(name)}$", symbols.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} in {symbols}")
    return int(matches[0], 16)


def capture_script(
    mailbox: int, rng_address: int, trainer_address: int, state: Path, image: Path, delay: int
) -> str:
    # Button timing is fixed and declared. Only the initial wait varies, which
    # samples the game's frame-cycled RNG before a real Rogue trainer is chosen.
    return f"""
local base={mailbox}
local rng={rng_address}
local trainer={trainer_address}
local state={json.dumps(str(state))}
local image={json.dumps(str(image))}
local shift={delay}
local saved=false
local captured=false
local frame=0
callbacks:add('frame',function()
  frame=frame+1
  local phase=emu:read32(base+12)
  if not saved and emu:read32(base)=={MAGIC} and
     emu:read32(base+8)==0 and phase==1 then
    -- Save before enabling the mailbox: this is the reproducible state used
    -- by every policy.  The following disposable frame only reads metadata.
    emu:setKeys(0)
    assert(emu:saveStateFile(state,31), 'saveStateFile failed')
    emu:write32(base+8,1)
    emu:write32(base+12,0) -- make the hook publish the first policy boundary
    saved=true
    return
  end
  if saved and not captured and emu:read32(base+8)==1 and phase==1 then
    local seed=emu:read32(rng)
    local trainer_id=emu:read16(trainer)
    local team={{}}
    for i=0,2 do table.insert(team, emu:read32(base+4*(44+i*16))) end
    emu:setKeys(0)
    emu:screenshot(image)
    console:log('ROGUE_RL_CAPTURE seed='..seed..' trainer='..trainer_id..' frame='..frame..' team='..table.concat(team,','))
    captured=true
  end
  if captured then emu:setKeys(0); return end
  local key=0
  if frame>=1120+shift and frame<1131+shift then key=8 end -- title Start
  if frame>=1500+shift and frame<2100+shift and frame%40<8 then key=1 end
  for _,f in ipairs({{2200,2240,2280,2320}}) do
    if frame>=f+shift and frame<f+shift+7 then key=128 end
  end -- difficulty Save & Exit
  if frame>=2400+shift and frame<6200+shift and frame%40<8 then key=1 end
  for _,f in ipairs({{6300,6340,6380,6420,6460}}) do
    if frame>=f+shift and frame<f+shift+7 then key=128 end
  end -- outfit Save & Exit
  if frame>=6520+shift and frame<6530+shift then key=1 end
  if frame>=6600+shift and frame%40<8 then key=1 end
  emu:setKeys(key)
end)
"""


def capture_one(
    host: Path,
    rom: Path,
    profile: dict,
    symbols: Path,
    output: Path,
    index: int,
    prefix: str,
    frame_limit: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    state = (output / f"{prefix}-{index:03d}.ss0").resolve()
    image = (output / f"{prefix}-{index:03d}.png").resolve()
    log = (output / f"{prefix}-{index:03d}.log").resolve()
    if any(path.exists() for path in (state, image, log)):
        raise FileExistsError(f"Capture {index} already exists; never overwrite a saved battle")
    delay = index * 37
    script_text = capture_script(
        profile["mailbox_address"],
        symbol_address(symbols, "gRngValue"),
        symbol_address(symbols, "gTrainerBattleOpponent_A"),
        state,
        image,
        delay,
    )
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        clean_rom = directory / "rogue-research.gba"
        shutil.copyfile(rom, clean_rom)
        script = directory / "capture.lua"
        script.write_text(script_text)
        with log.open("w") as handle:
            result = subprocess.run(
                [str(host), str(clean_rom), str(script), str(frame_limit)],
                stdout=handle,
                stderr=handle,
                timeout=60,
                check=False,
            )
    if result.returncode or not state.is_file() or not image.is_file():
        raise RuntimeError(f"Battle capture {index} failed; inspect {log}")
    transcript = log.read_text()
    match = re.search(r"ROGUE_RL_CAPTURE seed=(\d+) trainer=(\d+) frame=(\d+) team=([\d,]+)", transcript)
    if not match:
        raise RuntimeError(f"Battle capture {index} has no mailbox receipt; inspect {log}")
    seed, trainer, frame = map(int, match.groups()[:3])
    team_ids = [int(value) for value in match.group(4).split(",")]
    species = profile["species"]
    team = [species.get(str(value), f"species-{value}") for value in team_ids if value]
    if len(team) != 3:
        raise RuntimeError(f"Battle capture {index} did not expose a three-Pokémon team")
    if not image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Battle capture {index} has no valid screenshot")
    return {
        "id": f"{prefix}-trainer{trainer}-rng{seed}-{index:03d}",
        "split": "train",
        "scenario_group": f"fixture-{'-'.join(team).lower().replace(' ', '-')}-route-trainer-{trainer}",
        "seed": seed,
        "state_path": str(state),
        "state_sha256": sha256_file(state),
        "capture": {
            "input_delay_frames": delay,
            "capture_frame": frame,
            "trainer_id": trainer,
            "screenshot": str(image),
            "player_team": team,
            "player_team_species": team_ids,
            "research_fixture": "one of six level-10 three-Pokémon teams; first Rogue route trainer after BeginRogueRun",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--frame-limit", type=int, default=30000)
    parser.add_argument("--host", type=Path, default=Path(".cache/mgba-runner"))
    parser.add_argument("--rom", type=Path, default=Path("data/rogue-research.gba"))
    parser.add_argument("--profile", type=Path, default=Path("data/rogue-profile.json"))
    parser.add_argument("--symbols", type=Path, default=Path("data/rogue-symbols.txt"))
    parser.add_argument("--states", type=Path, default=Path("data/states"))
    parser.add_argument("--manifest", type=Path, default=Path("data/battles.json"))
    parser.add_argument("--append", action="store_true", help="Add new states to an existing pilot corpus")
    parser.add_argument("--replace", action="store_true", help="Replace an existing generated manifest")
    parser.add_argument("--prefix", default="benchmark", help="Filename and id prefix for this capture batch")
    parser.add_argument("--index-offset", type=int, default=0, help="Disjoint deterministic RNG-delay range")
    args = parser.parse_args()
    if args.append and args.replace:
        parser.error("--append and --replace cannot be used together")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.prefix):
        parser.error("--prefix must contain lowercase letters, digits, and hyphens")
    if not 1 <= args.count <= 100 or not 1000 <= args.frame_limit <= 1_000_000:
        parser.error("count must be 1–100 and frame limit 1,000–1,000,000")
    profile = json.loads(args.profile.read_text())
    if sha256_file(args.rom) != profile["rom_sha256"]:
        raise ValueError("Research ROM differs from its generated profile")
    previous = []
    if args.manifest.exists():
        if args.replace:
            previous = []
        elif not args.append:
            raise FileExistsError(f"Corpus already exists: {args.manifest}; use --append or --replace explicitly")
        else:
            existing = Corpus.load(args.manifest)
            existing.verify_rom(args.rom)
            if existing.source_revision != profile["source_revision"]:
                raise ValueError("Existing corpus source revision differs from this ROM")
            previous = json.loads(args.manifest.read_text())["battles"]
    elif args.append:
        raise FileNotFoundError(f"Cannot append to missing corpus: {args.manifest}")
    if args.index_offset < 0:
        parser.error("--index-offset must be nonnegative")
    start_index = args.index_offset + len(previous)
    states = [
        capture_one(
            args.host.resolve(),
            args.rom.resolve(),
            profile,
            args.symbols.resolve(),
            args.states,
            start_index + index,
            args.prefix,
            args.frame_limit,
        )
        for index in range(args.count)
    ]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    for item in states:
        item["state_path"] = os.path.relpath(item["state_path"], args.manifest.parent.resolve())
        item["capture"]["screenshot"] = os.path.relpath(
            item["capture"]["screenshot"], args.manifest.parent.resolve()
        )
    payload = {
        "schema_version": 1,
        "rom_sha256": profile["rom_sha256"],
        "source_revision": profile["source_revision"],
        "cohort": "six-team Rogue route-trainer benchmark; split assignment pending",
        "battles": previous + states,
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    corpus = Corpus.load(args.manifest)
    corpus.verify_rom(args.rom)
    print(json.dumps({"manifest": str(args.manifest), "battles": len(corpus.battles)}, indent=2))


if __name__ == "__main__":
    main()
