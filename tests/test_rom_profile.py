import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_rom", Path(__file__).resolve().parents[1] / "mgba/prepare_rom.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_profile_uses_real_symbol_and_checksums(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: prepare.REVISION)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/rogue_rl.inc.c").write_bytes((prepare.HERE / "rogue_rl.inc.c").read_bytes())
    constants = tmp_path / "include/constants"
    constants.mkdir(parents=True)
    for name, text in {
        "species": "#define SPECIES_PIKACHU 25",
        "moves": "#define MOVE_THUNDERBOLT 85",
        "pokemon": "#define TYPE_ELECTRIC 13",
    }.items():
        (constants / f"{name}.h").write_text(text)
    rom = tmp_path / "fixture.bin"
    rom.write_bytes(bytes(512))
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("02001200 B gRogueRL\n")
    out = tmp_path / "profile.json"
    # Profile generation requires an actual revision; test supplies its checked identity.
    result = prepare.make_profile(tmp_path, rom, symbols, out, 8765)
    assert result["mailbox_address"] == 0x02001200
    assert result["rom_size"] == 512 and len(result["rom_sha256"]) == 64
    assert result["species"]["25"] == "Pikachu"
    assert json.loads(out.read_text()) == result
    assert "dofile(" in out.with_suffix(".lua").read_text()
    symbols.write_text("08001200 B gRogueRL\n")
    with pytest.raises(ValueError, match="EWRAM"):
        prepare.make_profile(tmp_path, rom, symbols, out, 8765)


def test_install_rejects_different_source_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "wrong-commit\n")
    with pytest.raises(ValueError, match="Expected Rogue"):
        prepare.install(tmp_path)
    assert not list(tmp_path.iterdir())


def _make_source_fixture(tmp_path):
    files = {
        "src/battle_controller_player.c": (
            "static void PlayerBufferExecCompleted(u32 battler);\n"
            "static void PlayerHandleChooseAction(u32 battler)\n{\n"
            "static void HandleInputChooseAction(u32 battler)\n{\n"
            "static void PlayerHandleChooseMove(u32 battler)\n{\n"
            "static void PlayerHandleChoosePokemon(u32 battler)\n{\n"
            "void PlayerHandleExpUpdate(u32 battler)\n{"
        ),
        "src/battle_main.c": "void BattleMainCB2(void)\n{",
        "src/rogue_controller.c": (
            "void Rogue_OnWarpIntoMap(void)\n{\n"
            "        if(!Rogue_IsRunActive())\n"
            "        {\n"
            "            BeginRogueRun();\n"
            "        }\n"
            "    }\n"
            "    else if(gMapHeader.mapLayoutId == LAYOUT_ROGUE_ADVENTURE_PATHS)"
        ),
        "include/random.h": (
            "#define GUARD_RANDOM_H\n"
            "#define CycleRandom()       Random32()\n"
            "#define CycleRandom()       Random()\n"
        ),
        "data/maps/Rogue_HubTransition/scripts.pory": (
            "script Rogue_HubTransition_OnEnter\n{\n    setvar(VAR_TEMP_1, 1)\n    end\n}"
        ),
        "data/maps/Rogue_Area_Labs/scripts.pory": (
            "script Rogue_Area_Labs_WarpState0\n"
            "{\n"
            "    if(var(VAR_ROGUE_INTRO_STATE) == ROGUE_INTRO_STATE_SPAWN)"
        ),
    }
    for relative, content in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return list(files)


def test_install_autostart_hooks_are_checked_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: prepare.REVISION)
    paths = _make_source_fixture(tmp_path)
    prepare.install(tmp_path)
    script = (tmp_path / "data/maps/Rogue_HubTransition/scripts.pory").read_text()
    lab_script = (tmp_path / "data/maps/Rogue_Area_Labs/scripts.pory").read_text()
    controller = (tmp_path / "src/rogue_controller.c").read_text()
    assert "trainerbattle(TRAINER_BATTLE_SINGLE_NO_INTRO_TEXT" in script
    assert "VAR_UNUSED_0x40D1" in script
    assert "BeginRogueRun();\n#if ROGUE_RL_AUTOSTART_BATTLE\n" in controller
    assert "RogueRLPrepareBattle();" in controller
    assert "RogueRLPrepareFixture();" in controller
    assert "warpsilent(MAP_ROGUE_HUB_TRANSITION, 0)" in lab_script
    assert "VAR_UNUSED_0x40D2" in lab_script
    assert "Rogue_ChooseRouteTrainers(&trainer, 1);" in (tmp_path / "src/rogue_rl.inc.c").read_text()
    before = {relative: (tmp_path / relative).read_bytes() for relative in paths}
    before["src/rogue_rl.inc.c"] = (tmp_path / "src/rogue_rl.inc.c").read_bytes()
    prepare.install(tmp_path)
    assert before == {relative: (tmp_path / relative).read_bytes() for relative in before}


def test_install_missing_autostart_anchor_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: prepare.REVISION)
    paths = _make_source_fixture(tmp_path)
    script = tmp_path / "data/maps/Rogue_HubTransition/scripts.pory"
    script.write_text(
        script.read_text().replace("script Rogue_HubTransition_OnEnter", "script RenamedOnEnter")
    )
    before = {relative: (tmp_path / relative).read_bytes() for relative in paths}
    with pytest.raises(ValueError, match="Unique source anchor missing"):
        prepare.install(tmp_path)
    assert before == {relative: (tmp_path / relative).read_bytes() for relative in paths}
    assert not (tmp_path / "src/rogue_rl.inc.c").exists()
