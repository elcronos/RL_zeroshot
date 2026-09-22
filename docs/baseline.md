# Frozen Laya battle baseline

The `baseline` command measures whether the pinned fast Laya CoreML model can play **real Emerald Rogue trainer singles** through the instrumented mGBA bridge. It requires actual pre-battle mGBA save states listed in the hash-locked corpus. It does not train a policy or load a checkpoint. It samples from Laya's legal-action distribution with a recorded seed, using the same action rule as later evaluation.

Start the research ROM and generated Lua bridge as described in [mGBA setup](mgba.md). The optional headless mGBA host can run uncapped; a desktop mGBA session also works. Then run:

```sh
uv run rogue-rl validate-corpus
uv run rogue-rl verify-game --steps 500
uv run rogue-rl baseline --prior laya --split train --max-battles 5 --repeats 1 --max-decisions 500 --seed 0 --output runs/laya-pilot
uv run rogue-rl baseline --prior uniform --split train --max-battles 5 --repeats 1 --max-decisions 500 --seed 0 --output runs/uniform-pilot
uv run rogue-rl compare-baselines runs/laya-pilot runs/uniform-pilot --output runs/comparison.json
```

Omit `--max-battles` to play the full selected split. `--split` accepts `train`, `validation`, or `test` and defaults to `train`; `train` is a sensible exploratory pilot. If you inspect or iterate against test results, those battles are no longer an untouched final test set. The output directory must be new so previous evidence is never silently replaced.

Each completed battle writes an `episodes.jsonl` row with battle ID, scenario group, saved battle seed, model action seed, repeat, win/loss/draw/truncated outcome, sparse return, decision count, wall time, and per-battle decision latency percentiles. `first-battle-trace.jsonl` records public observations, action probabilities and selected actions for the first battle's first repeat. `metadata.json` records ROM, corpus, source, mGBA profile, model/prompt identities, selected states, limits, and bridge handshake. `summary.json` counts outcomes and reports episode and equally scenario-weighted win rates, model call/cache/compute totals, average decision count, and average wall time. The terminal game outcome comes from the Rogue bridge; hitting the decision cap is reported as `truncated` and counts as a non-win.

The 95% interval resamples **scenario groups**, averaging repeats inside each group. With fewer than two completed groups it is null. Even with two or three groups the interval is exploratory; diverse independent battles are needed to estimate general play strength. Repeating the same save state with another action seed estimates action variability, not scenario generalization. A pilot can establish that the control loop runs and yield concrete outcomes, but it cannot alone establish the full residual-RL hypothesis.

An emulator, state, or model failure leaves completed rows on disk and writes `summary.json` with `status: failed`, the completed/planned count, and the error. It exits nonzero. A failed or partial run should not be combined with complete runs as though it had played every planned battle. Missing battle states produce an explicit error; no wins or losses are fabricated.

`--prior uniform` is a matched no-model control. `compare-baselines` requires the two completed baseline directories to have identical ROM, corpus, selected states, repeat count and decision cap; it averages repeats inside a scenario group before reporting a paired scenario-bootstrap interval.

Future model adapters such as PrismNLI-0.4 or Jev can implement the same `probabilities(public_observation) -> 10 legal-action probabilities` interface while retaining the ROM corpus, mGBA bridge, and episode report. Preserve each model's prompt, artifact hash, hardware and latency in provenance so comparisons remain meaningful.

## Visual one-battle replay

The `visual` command uses the same real save-state reset and legal-action controller. At normal move decisions it temporarily opens mGBA's own Fight menu, saves the original game screen with its move names and PP/type panel under `raw-*.png`, restores the exact decision, and only then lets the policy act. Enlarged frames under `frames/` retain the policy's selected action and probability panel; `decisions.jsonl` records the public observation and `battle.gif` is the replay. It is useful for inspecting one Laya or uniform trajectory before interpreting aggregate results.

```sh
uv run rogue-rl visual --prior laya --split train --battle-id pilot-trainer191-rng2303321540-000 --seed 0 --output runs/visual-laya
uv run rogue-rl visual --prior uniform --split train --battle-id pilot-trainer191-rng2303321540-000 --seed 0 --output runs/visual-uniform
```

Run either command while the GUI bridge or the headless mGBA host is running. `result.json` records the terminal outcome, decision count, frame paths, and model timing. A decision cap is marked `truncated`; it is never presented as a win.

For a live run, start the visible mGBA app instead of the headless host:

```sh
scripts/start_live_mgba.sh
```

In the app, load `data/rogue-profile.lua` through **Tools → Scripting** once. Keep the mGBA window visible and unpaused, then run the same `visual` command in a terminal. The bridge submits decisions while the game animation remains visible in the mGBA window.

## Jev/OpenRouter adapter

`--prior jev` is a TypeSafe typed-choice baseline. It sends only the public observation and legal action labels, receives Jev's full legal-action probability distribution, and fails the run when the provider is unavailable or returns an invalid action; it never silently substitutes Laya or uniform. The default is OpenRouter's current `~typesafe/jev-latest` alias; override it to pin a dated model version:

```sh
export OPENROUTER_API_KEY='replacement-key-from-your-secret-manager'
uv run rogue-rl visual --prior jev --openrouter-model '~typesafe/jev-latest' \
  --split train --battle-id pilot-trainer191-rng2303321540-000 --output runs/visual-jev
```

The API key is read from the environment and is never written into the repository or experiment metadata. TypeSafe documents Jev's typed Choice request/response, including full probabilities, in its [quickstart](https://docs.typesafe.ai/introduction/quickstart) and [Choice reference](https://docs.typesafe.ai/primitives/choice); OpenRouter documents its Typesafe model aliases on [the Typesafe model page](https://openrouter.ai/typesafe).
