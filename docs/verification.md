# Verification evidence

Validated on Apple Silicon, macOS 15.6.1, Python 3.12.0. The research ROM is built locally and a narrow, hash-locked trainer-battle pilot has been collected.

- Python unit/integration suite: **83 passed**. Coverage includes policy initialization/masking, PPO and KL updates, GAE terminal/truncation handling, immutable evaluation, checkpoint round trips, frozen and uniform baselines, corpus hashes/splits, public-information filtering, paired reporting, CLI error handling, real loopback TCP with fixture replies, and the real Lua bridge executed against a mock mGBA API.
- The synthetic learning test is a categorical optimization fixture. Protocol transcripts have no Pokémon battle dynamics. Neither supplies experimental performance evidence.
- mGBA application version checked: **0.10.5** (`26b7884bc25a5933960f3cdcd98bac1ae14d42e2`). Lua syntax passes `luac -p mgba/bridge.lua`.
- Hook installer applied to actual pinned Rogue expansion source. Its generated quest/decoration/custom-mon headers were produced by the upstream CustomJson tool. Modified `battle_controller_player.c` and `battle_main.c` pass ARM-targeted Clang syntax checks with upstream C headers. This caught and fixed an incorrect battle-style field reference. The complete ROM subsequently compiled and linked with Arm GNU 15.3.rel1 and Poryscript 3.0.2. The mailbox occupies 688 bytes at `0x0201c080`.
- Source review found per-frame RNG cycling in both main/battle VBlank callbacks. The installer now guards `CycleRandom()` while research mode is enabled. Live replay verification is still required; the hook alone is not proof of complete determinism.
- Pinned Laya ANE FP16 model downloaded and checksum-verified (approximately 649 MiB). Actual local CoreML inference succeeds. The nine-action semantic fixture fits in **50–56 tokens per query** against the 96-token limit.

Real inference benchmark, 20 repetitions after warmup, cache disabled, nine legal actions: **74.80 ms median / 75.62 ms p95 per full decision**. Initial model loading took **30.20 seconds**. This is a semantic fixture, not a game benchmark; performance depends on hardware, prompts, and load. Raw locally generated evidence is `runs/laya-benchmark.json`, and `scripts/benchmark_laya.py` reproduces the measurement. A separate three-action first-call fixture took 37.57 ms after load.

CoreML emitted a warning that installed Torch 2.14 is newer than its conversion-tested Torch 2.7. The tested path loads an already-converted CoreML artifact and does not convert Torch models; inference succeeded. Runtime/dependency versions are recorded in the lockfile and run metadata.

## Actual ROM and mGBA smoke check

The locally built ROM is `data/rogue-research.gba` (32 MiB), SHA256 `42dc79af70487a152098b9a1886dca86e599e8617d0122a393685a79132cda42`. Its matching profile and ELF symbol dump are in `data/`; all are ignored by Git.

The official mGBA 0.10.5 core was built with Lua 5.4 and hosted by `mgba/runner.c`. The real game reached the Emerald Rogue title screen. The actual Lua socket bridge answered Python's HELLO with the expected ROM CRC32, SHA256 identity, ABI and source revision. Reproduce with `uv run python scripts/smoke_mgba.py`; output is `runs/mgba-smoke/result.json`, `boot.png`, and `emulator.log`. The subsequent route-trainer pilot verifies reset, legal move stepping, terminal outcome collection and delay-invariant replay for its captured states. It does not cover switching, trapping, fainting, Struggle or other unsupported menus. The desktop file dialog timed out, so GUI operation was not verified.

An independent integration review ran 10 targeted tests and checked the runner/bridge against mGBA source. It identified excessive emulator logging; the runner now retains warnings/errors and scripting information. The broader independent learning review could not finish because the agent hit its usage limit; no review approval is claimed.

## Measured pilot

The research-only fresh-game entry creates a level-10 Bulbasaur, begins a normal Rogue run, selects a Rogue route trainer and launches the game's normal trainer battle. Ten states across five trainer groups were captured at the first action menu. Frozen Laya finished 7 wins and 3 losses, while a matched uniform-legal-action control finished 9 wins and 1 loss. Scenario-weighted rates were 0.667 and 0.800 respectively; the matched difference was −0.133 with scenario-bootstrap interval [−0.400, 0.000]. Complete receipts, traces and constraints are in `docs/pilot-results.md`.

## Outstanding real-game checks

Capture battle states; verify normal moves, switches, fainting, trapping, move restrictions, Struggle and terminal handling; then run delay-invariant replay. None of these are replaced by the protocol mocks or syntax check. Follow `docs/mgba.md`.

`rogue-rl doctor` reports the ROM, profile and Laya model present. The current corpus is pilot-only and has no validation/test partition. No trained adapter, held-out win rate, sample-efficiency gain, or general experimental conclusion is claimed.
