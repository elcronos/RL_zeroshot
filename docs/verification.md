# Verification evidence and current limits

The current implementation was verified on Apple Silicon with mGBA 0.10.5 and
Python 3.12. The reproducible checks currently pass:

- **101 Python tests** covering policy masking, PPO/GAE, frozen evaluation,
  battle scoring, model adapters, corpus isolation, bridge transport, and run
  metadata.
- `ruff check src scripts tests` passes.
- The 172-state corpus matches its research-ROM hash and contains 144 train,
  10 validation, and 18 held-out test battles.
- Whole player-team/opponent-trainer scenario groups are isolated between
  splits.
- A real mGBA replay produced the same transcript with and without artificial
  inference delay.
- Laya and PrismNLI inference run locally; the Jev adapter fails explicitly
  without a valid hosted credential.

PrismNLI, Laya, and uniform each won 18/18 zero-shot test battles. The 1,024-step
Laya and PrismNLI policies also completed all 18 battles. Their turns, final
party HP, battle scores, and 13-minute-17-second end-to-end runtime are recorded
in [results.md](results.md). Jev remains `not_run` without a valid credential.

The research ROM uses a pinned Emerald Rogue expansion revision and mGBA bridge.
It controls real game moves, switches, damage, trainer parties, and terminal
outcomes, but it is an instrumented research variant: EXP updates are suppressed
and passive per-frame RNG cycling is paused at policy boundaries. This is not a
pixels-only or byte-identical retail-game benchmark.

The current level-10 route-trainer fixture is easy enough for uniform random to
reach the win-rate ceiling. Battle score provides within-win resolution, but a
stronger claim about general play or policy improvement needs harder held-out
states, a completed Jev baseline, and multiple preregistered learner seeds at a
larger policy budget.
