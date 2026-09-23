# Experimental protocol

## Question and scope

How well do frozen PrismNLI-0.4B, Laya, Jev, and uniform random legal actions play held-out **Pokémon Emerald Rogue trainer singles battles**? After that zero-shot comparison, does a small residual policy improve each real model without fine-tuning it?

This follows the [shared proposal](https://chatgpt.com/share/6ab143f4-8ef8-83ec-a2f7-e6632c449cdc), with two explicit adaptations: only Rogue battles are included; compact per-action scores make the ANE L96 model usable. Battle Factory and all other games are excluded.

## Policies

Let `m` be the engine's legal-action mask, `x` the public numerical features, and `p0` the frozen prior. Invalid actions have exactly zero probability. On legal actions:

```
delta, V = MLP(x, p0, m)
pi = softmax(log(max(p0, epsilon)) + alpha * delta)
```

The MLP has two 128-unit tanh layers with actor and critic heads. The residual actor head starts at zero, so the initial corrected distribution equals the frozen prior. The foundation model is outside PyTorch's optimization graph. The baseline scratch actor/critic has the same parameter count; it receives a uniform legal prior and uses alpha=1 and no prior KL. It never loads Laya.

| Arm | Prior | Learning |
|---|---|---|
| Zero-shot PrismNLI | Frozen entailment action scores | None; 0 training battles |
| Zero-shot Laya | Frozen Noul action scores | None; 0 training battles |
| Zero-shot Jev | Frozen typed Choice probabilities | None; 0 training battles |
| Uniform random | Uniform over legal actions | None; zero-shot control only |
| Trained PrismNLI policy | Frozen PrismNLI prior | Residual PPO + KL(pi || p0) |
| Trained Laya policy | Frozen Laya prior | Residual PPO + KL(pi || p0) |
| Trained Jev policy | Frozen Jev prior | Residual PPO + KL(pi || p0) |

The gated ablation scales `delta`; it is **not** the probability-mixture gate discussed as another option in the conversation. It has extra gate parameters and is not a primary capacity-matched comparison. Supervised adapters are outside this implementation because no demonstration corpus is available.

The ANE prior uses `p0(a) = q(good | compact state, action a) / sum_a q(good | state, action a)`, with a probability floor of 1e-6. This is not a calibrated probability of winning, nor an exact joint choice distribution. Each action sees the active matchup and its own description; opportunity cost and the other alternatives are omitted. Keep this projection fixed across frozen and residual arms. An explicit 1024-token joint-choice ablation can test the representation tradeoff. Never attribute gains solely to model pretraining without the uniform/shuffled controls.

## Observation and control boundary

The task is partially observable. Laya receives compact active-species/type/HP/status and candidate move or switch descriptions. The MLP receives HP fractions, one-hot type IDs, visible status/stages, move power/PP/category/accuracy, the legal mask, previous action, and the last observed active HP changes. Numerical features exclude species IDs, species names, move IDs, and move names. HP changes after switches are observations, not claimed damage estimates. Weather, hazards, full revealed-move history, and long-term memory are not encoded in v1; this limits tactical strength.

Opponent HP is quantized to 48 health-bar units. Opponent private stats, held items, abilities, moves, unshown team, and RNG are excluded by an allowlist. Species typing uses the displayed species, including Illusion. Current type changes not inferable from species are not represented. The legal mask is engine-derived and can reveal trapping constraints; all arms receive the same mask. This is an assisted structured-state benchmark, not a pixels-only human-information benchmark.

Actions 0–3 select move slots. Actions 4–9 select absolute party slots 0–5. The active/fainted/empty slots are masked. At forced replacement only switches are legal. If all moves are unusable, move slot 0 represents Struggle. The protocol fails on unsupported battle modes or controller states. Mega/Z/Dynamax activation is outside the ten-action interface; build the corpus without teams/opponents relying on those formats.

Research instrumentation suppresses EXP updates, pauses battle logic at policy boundaries, and disables passive per-frame RNG cycling while preserving actual battle RNG calls. Apply it identically to every arm. This is a controlled research variant of Rogue, not byte-identical retail gameplay. Engine rules, moves, damage, opponents, and battle outcomes still run inside the GBA game in mGBA.

## Corpus and splits

Before model development, build a corpus of diverse independent trainer battles from procedural Rogue runs, including ordinary trainers and supported boss singles, a range of team compositions/types/levels, and forced-switch situations. Capture pre-research first-action-menu save states. Keep team composition/lead fixed within an episode. Use the same saved game RNG for paired evaluations; the manifest `seed` records provenance and does not secretly reseed RAM.

Suggested full study: 1,000 training scenario groups, 200 validation groups, 200 test groups, with several predeclared RNG realizations per scenario if available. These counts are a **data-collection plan**, not supplied assets. For initial engineering, a handful of representative fixtures is sufficient. A single battle or merely replaying identical teams with new RNG does not establish procedural generalization.

Partition by `scenario_group` (team/opponent encounter identity) so all RNG variants of a scenario remain together. Also separate originating Rogue runs/trainers where possible. The manifest enforces declared group separation and duplicate-byte hashes; it cannot detect falsely labelled or strategically equivalent scenarios. Audit corpus provenance manually before the study. Freeze manifest bytes and hashes before training. The talk-sized protocol uses learner seed 0, distinct from battle-generation seeds.

## Optimization and evaluation

Train each arm for exactly 1,024 policy decisions; checkpoint and validate at steps 0 and 1,024. This four-rollout budget is the repository's talk-sized protocol and targets less than 30 minutes for both local arms and their test evaluations on a 10-core M1 Max with 64 GB RAM. PPO uses 256-step rollouts, gamma=.99, lambda=.95, clip=.2, Adam 3e-4, four epochs, minibatches of 64, value coefficient .5, entropy .01, KL .05, and gradient norm .5. Use terminal win +1, loss -1, draw 0, and no dense shaping. Discounting favors earlier outcomes and is held fixed across arms. Report the fixed-budget result rather than selecting whichever checkpoint wins on test. Hosted Jev latency is recorded separately and is not controlled by local hardware.

After every 256-decision update, `updates.jsonl` records the combined PPO objective, policy loss, value loss, entropy, KL to the frozen prior, approximate update KL, gradient norm, and clip fraction. Import these points and their source SHA-256 with `scripts/import_training_curves.py`, preserve them in the checked-in result registry, and plot the combined objective plus its policy and value terms. Because each update uses a fresh on-policy rollout, the total loss is a stability diagnostic and need not decrease monotonically.

A 500-decision cap is a truncation, not a game loss or draw. GAE bootstraps from the last real observation on truncation; it never bootstraps true terminals or carries advantages across episodes. Validation shares the emulator, so an active training battle at a validation boundary is explicitly truncated and bootstrapped before evaluation; the next training battle starts from a fresh save state. Rollout boundaries without evaluation do not reset the battle. Report truncation rates, including the separate evaluation-boundary reason.

Evaluation uses no gradients or optimizer updates. It samples the policy with reproducible action RNG paired by learner seed, battle ID, and repeat. It does not switch to greedy selection for only one arm. Same seed/checkpoint repeats make frozen Laya curves flat under deterministic game replay. Training scenarios use a seed-controlled shuffled schedule independent of action randomness; their order is shared, but different battle lengths mean fixed-step budgets may complete different numbers of episodes.

Primary metrics are scenario-weighted wins and the transparent battle-quality table: training battles, validation battles, test battles, wins, game turns, final party HP, and battle score. Learning curves use validation only. The final test compares every trained residual policy with its own frozen zero-shot prior. Also report wall-clock cost, model initialization, cache behavior, trainable parameter count, truncation rate, KL from prior, action diversity, and argmax override rate.

Final comparisons require exactly paired seed/battle/repeat rows. The default has one learner seed, so the implementation suppresses confidence intervals and the result is descriptive. A publication study should preregister multiple learner seeds and a larger decision budget, then bootstrap learner seeds and scenario groups as crossed clusters while preserving pairing across arms. That longer protocol is intentionally separate from the reproducible 30-minute demo.

The talk-sized run uses the checked-in hyperparameters without a search. Validation is a diagnostic, then untouched test runs once at the fixed budget. A null or negative result is valid. No synthetic fixture scores can be used as evidence for this hypothesis.

## Provenance

Record model repo/revision, prompt version and scoring strategy, source revision, hook and ROM hashes, generated profile, corpus/state hashes, configuration, learner seeds, library versions, and hardware. `metadata.json` plus checkpoints record the runtime identities; preserve the local `uv.lock` and corpus alongside results. Use `verify-game` to compare deterministic action replay with and without host inference delay before collecting results.

Primary integration sources: [Laya-CoreML](https://github.com/mizorewww/laya-coreml), [mGBA scripting API](https://mgba.io/docs/scripting.html), and [Emerald Rogue source](https://github.com/Pokabbie/pokeemerald-rogue/tree/a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014).
