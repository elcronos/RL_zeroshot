#!/usr/bin/env python3
"""Install checked research hooks, then generate a profile from the actual build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zlib
from pathlib import Path

REVISION = "a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014"
HERE = Path(__file__).resolve().parent


def install(source: Path) -> None:
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(
            f"Expected Rogue expansion {REVISION}, got {revision}; port and review hooks before using another revision"
        )
    changes = {
        "src/battle_controller_player.c": [
            (
                "static void PlayerBufferExecCompleted(u32 battler);",
                'static void PlayerBufferExecCompleted(u32 battler);\n#include "rogue_rl.inc.c"',
            ),
            (
                "static void PlayerHandleChooseAction(u32 battler)\n{",
                "static void PlayerHandleChooseAction(u32 battler)\n{\n    if (RogueRLChooseAction(battler)) return;",
            ),
            (
                "static void HandleInputChooseAction(u32 battler)\n{",
                "static void HandleInputChooseAction(u32 battler)\n{\n    if (RogueRLChooseAction(battler)) return;",
            ),
            (
                "static void PlayerHandleChooseMove(u32 battler)\n{",
                "static void PlayerHandleChooseMove(u32 battler)\n{\n    if (RogueRLChooseMove(battler)) return;",
            ),
            (
                "static void PlayerHandleChoosePokemon(u32 battler)\n{",
                "static void PlayerHandleChoosePokemon(u32 battler)\n{\n    if (RogueRLChoosePokemon(battler)) return;",
            ),
            (
                "void PlayerHandleExpUpdate(u32 battler)\n{",
                "void PlayerHandleExpUpdate(u32 battler)\n{\n    if (gRogueRL.enabled) { PlayerBufferExecCompleted(battler); return; }",
            ),
        ],
        "src/battle_main.c": [
            (
                "void BattleMainCB2(void)\n{",
                "extern bool32 RogueRLTick(void);\nvoid BattleMainCB2(void)\n{\n    if (RogueRLTick()) return;",
            ),
        ],
        "src/rogue_controller.c": [
            (
                "void Rogue_OnWarpIntoMap(void)\n{",
                "/* Research ROM only: bootstrap a fresh lab party and a Rogue battle. */\n#define ROGUE_RL_AUTOSTART_BATTLE 1\nextern void RogueRLPrepareBattle(void);\nextern void RogueRLPrepareFixture(void);\nvoid Rogue_OnWarpIntoMap(void)\n{",
            ),
            (
                "        if(!Rogue_IsRunActive())\n        {\n            BeginRogueRun();\n        }\n    }\n    else if(gMapHeader.mapLayoutId == LAYOUT_ROGUE_ADVENTURE_PATHS)",
                "        if(!Rogue_IsRunActive())\n        {\n            BeginRogueRun();\n#if ROGUE_RL_AUTOSTART_BATTLE\n            RogueRLPrepareBattle();\n#endif\n        }\n    }\n#if ROGUE_RL_AUTOSTART_BATTLE\n    else if(gMapHeader.mapLayoutId == LAYOUT_ROGUE_AREA_LABS\n        && VarGet(VAR_ROGUE_INTRO_STATE) <= ROGUE_INTRO_STATE_EXPLORE)\n    {\n        RogueRLPrepareFixture();\n    }\n#endif\n    else if(gMapHeader.mapLayoutId == LAYOUT_ROGUE_ADVENTURE_PATHS)",
            ),
        ],
        "data/maps/Rogue_Area_Labs/scripts.pory": [
            (
                "script Rogue_Area_Labs_WarpState0\n{\n    if(var(VAR_ROGUE_INTRO_STATE) == ROGUE_INTRO_STATE_SPAWN)",
                "script Rogue_Area_Labs_WarpState0\n{\n    if(var(VAR_UNUSED_0x40D2) == 1)\n    {\n        setvar(VAR_UNUSED_0x40D2, 0)\n        setvar(VAR_RETURN_STATE, 1)\n        warpsilent(MAP_ROGUE_HUB_TRANSITION, 0)\n        waitstate\n        return\n    }\n    if(var(VAR_ROGUE_INTRO_STATE) == ROGUE_INTRO_STATE_SPAWN)",
            ),
        ],
        "data/maps/Rogue_HubTransition/scripts.pory": [
            (
                "script Rogue_HubTransition_OnEnter\n{\n    setvar(VAR_TEMP_1, 1)\n    end\n}",
                "script Rogue_HubTransition_OnEnter\n{\n    setvar(VAR_TEMP_1, 1)\n    if(var(VAR_UNUSED_0x40D1) == 1)\n    {\n        setvar(VAR_UNUSED_0x40D1, 0)\n        trainerbattle(TRAINER_BATTLE_SINGLE_NO_INTRO_TEXT, VAR_ROGUE_SPECIAL_ENCOUNTER_DATA, 0, gPlaceholder_Trainer_PostBattleTaunt)\n    }\n    end\n}",
            ),
        ],
        "include/random.h": [
            ("#define GUARD_RANDOM_H", "#define GUARD_RANDOM_H\nextern bool32 RogueRLIsEnabled(void);"),
            (
                "#define CycleRandom()       Random32()",
                "#define CycleRandom()       (RogueRLIsEnabled() ? 0 : Random32())",
            ),
            (
                "#define CycleRandom()       Random()",
                "#define CycleRandom()       (RogueRLIsEnabled() ? 0 : Random())",
            ),
        ],
    }
    prepared = {}
    for relative, replacements in changes.items():
        path = source / relative
        content = path.read_text()
        for old, new in replacements:
            if new in content:
                continue
            if content.count(old) != 1:
                raise ValueError(f"Unique source anchor missing: {relative}: {old}")
            content = content.replace(old, new)
        prepared[path] = content
    for path, content in prepared.items():
        path.write_text(content)
    shutil.copyfile(HERE / "rogue_rl.inc.c", source / "src/rogue_rl.inc.c")


def constant_names(path: Path, prefix: str) -> dict[str, str]:
    # Direct numeric constants only; unsupported expressions are not guessed.
    pattern = re.compile(r"^#define\s+" + re.escape(prefix) + r"([A-Z0-9_]+)\s+(\d+)\s*(?://.*)?$")
    result = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            result.setdefault(match[2], match[1].replace("_", " ").title())
    return result


def make_profile(source: Path, rom: Path, symbols: Path, output: Path, port: int) -> dict:
    """symbols is output of arm-none-eabi-nm -n <built ELF>."""
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Profile source revision differs from the supported Rogue revision")
    if not 1 <= port <= 65535:
        raise ValueError("Bridge port must be within [1,65535]")
    hook = source / "src/rogue_rl.inc.c"
    if not hook.is_file() or hook.read_bytes() != (HERE / "rogue_rl.inc.c").read_bytes():
        raise ValueError("Install current research hooks and rebuild before generating the profile")
    matches = re.findall(r"^([0-9a-fA-F]+)\s+\w\s+gRogueRL$", symbols.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Expected exactly one gRogueRL symbol in nm output")
    address = int(matches[0], 16)
    if not 0x02000000 <= address <= 0x02040000 - 688:
        raise ValueError("gRogueRL must fit in GBA EWRAM")
    data = rom.read_bytes()
    if len(data) < 192:
        raise ValueError("ROM is too small")
    profile = {
        "schema_version": 1,
        "abi": 1,
        "source_revision": REVISION,
        "rom_sha256": hashlib.sha256(data).hexdigest(),
        "rom_crc32": f"{zlib.crc32(data):08x}",
        "rom_size": len(data),
        "mailbox_address": address,
        "port": port,
        "hook_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
        "species": constant_names(source / "include/constants/species.h", "SPECIES_"),
        "moves": constant_names(source / "include/constants/moves.h", "MOVE_"),
        "types": constant_names(source / "include/constants/pokemon.h", "TYPE_"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n")
    lua = (
        "ROGUE_RL_PROFILE = {\n"
        + "\n".join(
            f"  {key} = {json.dumps(profile[key])},"
            for key in (
                "abi",
                "source_revision",
                "rom_sha256",
                "rom_crc32",
                "rom_size",
                "mailbox_address",
                "port",
            )
        )
        + "\n}\n"
    )
    # A single absolute-path launcher avoids relying on mGBA's current directory.
    lua += "dofile(" + json.dumps(str(HERE / "bridge.lua"), ensure_ascii=False) + ")\n"
    output.with_suffix(".lua").write_text(lua)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    patch = sub.add_parser("install")
    patch.add_argument("source", type=Path)
    profile = sub.add_parser("profile")
    profile.add_argument("source", type=Path)
    profile.add_argument("rom", type=Path)
    profile.add_argument("symbols", type=Path)
    profile.add_argument("output", type=Path)
    profile.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "install":
        install(args.source)
    else:
        make_profile(args.source, args.rom, args.symbols, args.output, args.port)


if __name__ == "__main__":
    main()
