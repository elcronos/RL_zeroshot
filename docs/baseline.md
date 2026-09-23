# Zero-shot battle baselines

The `baseline` command evaluates a frozen model or uniform random legal actions
without training. It uses hash-locked mGBA save states and the same stochastic
action-sampling rule used for trained-policy evaluation.

Start the research ROM and Lua bridge, validate the corpus, then run all four
models on the same test split:

```sh
uv run rogue-rl validate-corpus
uv run rogue-rl verify-game --steps 500
uv run rogue-rl baseline --prior prism --split test --output runs/prism-zero-shot-test
uv run rogue-rl baseline --prior laya --split test --output runs/laya-zero-shot-test
uv run rogue-rl baseline --prior uniform --split test --output runs/uniform-zero-shot-test
export OPENROUTER_API_KEY="your-key"
uv run rogue-rl baseline --prior jev --split test --output runs/jev-zero-shot-test
```

Every zero-shot summary records `training_battles_used: 0`, the number of test
battles, outcomes, game turns, final party HP, battle score, latency, and model
call statistics. The output directory also contains raw episode rows and the
first battle's decision trace. A provider or emulator failure leaves a failed
partial run and never falls back to another prior.

The 95% win-rate interval resamples scenario groups, averaging RNG variants
inside each group. Battle score explains tied outcomes but does not replace the
raw win count. See [scoring.md](scoring.md).

## Visual replay

The `visual` command resets the same real state, temporarily opens mGBA's own
Fight menu, captures the real move names and PP/type panel, restores the exact
decision boundary, and then lets the policy act.

```sh
uv run rogue-rl visual --prior prism --split test --output runs/visual-prism
uv run rogue-rl visual --prior laya --split test --output runs/visual-laya
uv run rogue-rl visual --prior uniform --split test --output runs/visual-uniform
uv run rogue-rl visual --prior jev --split test --output runs/visual-jev
```

For a visible live run, start `scripts/start_live_mgba.sh`, load
`data/rogue-profile.lua` through **Tools → Scripting**, and issue the visual
command from a terminal.
