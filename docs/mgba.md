# mGBA / Emerald Rogue integration

## Status and supported build

The bridge targets the expansion source at [Pokabbie/pokeemerald-rogue commit a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014](https://github.com/Pokabbie/pokeemerald-rogue/tree/a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014). mGBA 0.10.5 is installed locally; the bridge uses the documented [Lua API](https://mgba.io/docs/scripting.html).

**The research ROM is built locally and boots in mGBA.** `data/rogue-research.gba`, `data/rogue-profile.json`, and its Lua launcher are ready in this workspace; they are ignored by Git. The actual Python/Lua HELLO succeeds. A battle corpus is still missing, and battle reset/action/terminal behavior remains to be validated. Ordinary released ROMs and states from other builds will not work with this profile.

## Prepare the research build

Follow the pinned source's [build instructions](https://github.com/Pokabbie/pokeemerald-rogue/blob/a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014/INSTALL.md) for an ARM GCC/newlib toolchain, the C/C++ asset tools and Rogue's Poryscript dependency. Use an isolated checkout; the installer modifies three upstream files. You need the game assets required by that source tree. This repository does not fetch a commercial ROM.

From this experiment repository:

```sh
git clone https://github.com/Pokabbie/pokeemerald-rogue.git vendor/rogue
git -C vendor/rogue checkout a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014
uv run python mgba/prepare_rom.py install vendor/rogue
```

The default source Makefile uses modern ARM GCC and expansion mode. It assumes a Linux Poryscript executable on non-Windows systems; on macOS build/download Poryscript for macOS and override `PORYSCRIPT` with its actual absolute path. Do not attempt to execute the Linux binary on macOS. Build only after the upstream tools and generated headers are available:

```sh
# Example after completing upstream toolchain setup:
make -C vendor/rogue RELEASE=1 PORYSCRIPT=/absolute/path/to/poryscript -j4
arm-none-eabi-nm -n vendor/rogue/pokeemerald.elf > data/rogue-symbols.txt
cp vendor/rogue/pokeemerald.gba data/rogue-research.gba
uv run python mgba/prepare_rom.py profile vendor/rogue data/rogue-research.gba data/rogue-symbols.txt data/rogue-profile.json
```

This full build was executed successfully with Arm GNU 15.3.rel1 and Poryscript 3.0.2. Regenerate all states/profile after any hook, source, build-flag, or ROM change. The profile contains the actual `gRogueRL` EWRAM address from the built ELF, ROM SHA256/CRC32/size, source and hook identities, and move/species name catalogs. No vanilla Emerald addresses are assumed.

`install` checks the exact upstream commit and unique insertion anchors before writing. Repeating it on the same instrumented checkout is safe. Do not apply it to a different source revision without porting and reviewing the hooks. `profile` checks the source revision/current hook copy; the caller must ensure the ELF, ROM, and symbol dump came from the same completed build.

### Build details for this Apple Silicon workspace

The extracted ARM toolchain is `.cache/arm-toolchain/Payload`; its binaries are in `bin`. Poryscript was built with Go from commit `1ff8b70e50dbcd0dd5527581b3f9892dda2277b5` into `vendor/poryscript/poryscript`. Native asset tools were built using `make -C vendor/rogue -f make_tools.mk -j4`. The source dependency scanner expects newlib headers at `vendor/rogue/tools/agbcc/include`; here that is a symlink to the toolchain's `arm-none-eabi/include`. Do not replace an existing agbcc checkout's headers blindly.

The successful build command, from the experiment root, was:

```sh
make -C vendor/rogue RELEASE=1 TOOLCHAIN="$PWD/.cache/arm-toolchain/Payload" PORYSCRIPT="$PWD/vendor/poryscript/poryscript" -j6
.cache/arm-toolchain/Payload/bin/arm-none-eabi-nm -n vendor/rogue/pokeemerald.elf > data/rogue-symbols.txt
cp vendor/rogue/pokeemerald.gba data/rogue-research.gba
uv run python mgba/prepare_rom.py profile vendor/rogue data/rogue-research.gba data/rogue-symbols.txt data/rogue-profile.json
```

Build log: `.cache/rogue-build.log`. The final optional ROM-size display emits macOS `stat`/missing `numfmt` warnings; compilation, linking, ROM conversion and profile generation succeeded. The built ROM SHA256 is recorded in `docs/verification.md`.

### Optional headless mGBA host

The host uses the official mGBA core and the same Lua bridge as the GUI, runs uncapped, and responds to Ctrl-C/SIGTERM. It does not load your desktop mGBA settings. On macOS, rebuild it with CMake, Clang, Homebrew and Lua 5.4 available:

```sh
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install lua@5.4
bash scripts/build_mgba_host.sh
uv run python scripts/smoke_mgba.py
.cache/mgba-runner data/rogue-research.gba data/rogue-profile.lua
```

`build_mgba_host.sh` pins mGBA 0.10.5, builds into `.cache`, and does not replace the desktop application. It uses Lua 5.4 explicitly; the machine's default Lua 5.5 is not used for this host. The optional third runner argument limits frames for diagnostics (`0` or omitted runs until stopped). Use only one emulator per profile port. The smoke check launches and stops its own instance; close a running bridge first. Its boot PNG and JSON receipt are written to `runs/mgba-smoke/`.

## Capture battle states

### Research fixture collector

The instrumented research ROM includes a deliberately narrow, reproducible
fresh-game collector: it creates a level-10 Bulbasaur, enters Rogue's normal
run transition, lets Rogue select a valid route trainer, and invokes the
ordinary trainer-battle script in singles/SET mode. The player party is a
fixture; the Rogue trainer selection, party generation and battle engine are
the game's own. It is appropriate for checking the bridge and comparing action
priors, and is not a representative general-play corpus.

Build the headless host, then capture distinct route-trainer states:

```sh
bash scripts/build_mgba_host.sh
uv run python scripts/collect_battles.py --count 10
uv run rogue-rl validate-corpus
uv run rogue-rl verify-game --steps 100
```

The collector saves states, screenshots and capture receipts under
`data/states/`, records their SHA256 hashes in `data/battles.json`, and refuses
to overwrite a corpus. Use `--append` only to add additional captures from the
same ROM. Its varied title-screen wait samples the game RNG; capture metadata
records the resulting RNG seed and selected trainer. The first policy action is
still captured with research mode disabled.

To run a direct prior baseline on that corpus:

```sh
uv run rogue-rl baseline --prior laya --split train --output runs/laya-pilot
uv run rogue-rl baseline --prior uniform --split train --output runs/uniform-pilot
uv run rogue-rl compare-baselines runs/laya-pilot runs/uniform-pilot --output runs/comparison.json
uv run rogue-rl visual --prior laya --split train --battle-id pilot-trainer191-rng2303321540-000 --output runs/visual-laya
```

The current results are recorded in `docs/pilot-results.md`. Do not promote
this fixture corpus to validation/test evidence. Capture normal runs with a
range of parties, route tiers and battle mechanics before the residual-RL
experiment.

### Manual collection for the full experiment

1. Open the **research ROM** in mGBA and enter a Rogue run normally, before loading the Lua bridge.
2. Enable **set battle style** in Rogue settings. Use ordinary trainer singles, without special battle formats or the auto-battler campaign/auto-move curse.
3. At the **first Fight/Bag/Pokémon/Run action menu**, save an mGBA state to a separate file. Capture before research mode is enabled; the bridge deliberately rejects states captured during a research episode. The mailbox is initialized when the game's battle loop starts.
4. Repeat across varied procedural runs/teams/opponents and RNG realizations. Include representative switching, fainting, trapping, and Struggle fixtures for integration checks. Exclude Mega/Z/Dynamax-dependent setups and mechanics requiring unsupported controller menus, such as Revival Blessing, from this v1 corpus.
5. Register each file in `data/battles.json`; `state_path` is relative to that manifest. Record exact SHA256 with `shasum -a 256 path/to/file`. Put every RNG variant of the same team/opponent scenario into one `scenario_group` and one split. The `seed` is generation provenance, not an arbitrary label that changes the saved RNG. Use `null` and record why in `capture` when a user-supplied state has no recoverable generation seed.

Obtaining diverse states is an explicit data-collection step. The implementation does not navigate the overworld, construct teams, or automatically generate a thousand battles from one state. A research ROM may still require normal progression or source-supported debug tools to reach useful battles.

## Connect and validate

Load `data/rogue-profile.lua` in mGBA's **Tools → Scripting** window. It loads `mgba/bridge.lua` by absolute path and listens only on `127.0.0.1:8765`. Keep emulation running; pausing the GUI prevents frame/socket callbacks from responding. Use one runner per emulator/port. If reloading a script causes an address-in-use error, close its prior scripting session or restart that emulator instance first.

```sh
uv run rogue-rl doctor
uv run rogue-rl validate-corpus
uv run rogue-rl verify-game --steps 500
```

`verify-game` replays the first training state twice with the same deterministic legal actions, once immediately and once with a 50 ms delay before each action. The observations must match exactly. It reports whether a terminal was reached. Also verify each supported mechanical case visually before collecting evidence:

| Fixture | Required behavior |
|---|---|
| Normal move | Correct slot/target submitted, exactly one successor decision |
| Voluntary switch | Correct absolute party member selected |
| Fainted active | Only living bench slots legal; next action is a forced switch |
| Disable/Encore/Choice/Taunt/PP | Mask matches the game's restrictions |
| Trapped active | No voluntary switches; forced replacement still legal |
| No usable moves | Slot 0 executes Struggle |
| Terminal win/loss/draw | One terminal result, no progression into overworld |
| Artificial inference delay | Replay observations/outcome unchanged |

A failed bridge request aborts the run. Frame timeouts are **integration errors**, not losses or truncated training rewards. A policy's decision-count cap is separately recorded as an RL truncation.

## Protocol and instrumentation

Python sends newline-delimited tab-separated requests; arbitrary strings are UTF-8 hex, so file names/battle IDs cannot break framing. Lua replies with one JSON object per line: `{seq,result,error}`. Only one request is active. Both request sequence and decision ID are checked. Python closes on timeout/malformed/stale responses; reset is required after reconnection.

```
HELLO  request_sequence
RESET  request_sequence  hex_battle_id  hex_absolute_state_path  max_frames
STEP   request_sequence  decision_id  action_slot  max_frames
```

HELLO checks the loaded ROM CRC32/size against the profile and echoes its identity. Python checks corpus ROM SHA256 and state SHA256 before loading. Commands do not expose a network service beyond loopback. Lua binds one client at a time, uses nonblocking socket callbacks, caps input size, and advances only animation/text input between policy decisions. It clears controller keys at boundaries and disconnects.

The ABI is 172 aligned 32-bit words (688 bytes): header, active player/opponent, six party records, four move records. It is generated by `rogue_rl.inc.c`, not extracted from guessed struct layouts. The player controller uses the existing engine's move-limit and switching checks and emits its normal controller return values. Only supported trainer singles inside a Rogue run are accepted.

Research mode disables EXP updates, blocks the battle loop while waiting for a decision, suppresses passive `CycleRandom()` calls, and halts before battle resources are freed on terminal outcome. Actual battle `Random()` calls remain active. This equalizes inference-speed effects but is a documented gameplay instrumentation change. Save-state RNG is restored without rewriting it. A legal mask can reveal engine constraints; it is supplied equally to every experimental arm.

Hook error codes: `1` unsupported mode/setup; `2` stale action decision; `3` illegal action; `4` unexpected move controller; `5` unsupported party-controller operation; `6` unsupported terminal outcome. Fix the fixture or extend and retest the integration; never convert these to fabricated battle outcomes.
