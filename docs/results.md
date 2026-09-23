# Held-out zero-shot and 1,024-step policy results

This report was generated from `data/battles.json` with 18 test save states in
six disjoint player-team/opponent-trainer scenario groups. The corpus contains
172 states total: 144 train, 10 validation, and 18 test. The test group has all
six declared player teams.

![Zero-shot versus residual PPO battle-quality comparison](assets/model-comparison.png)

![PPO training loss over policy decisions](assets/training-loss.png)

The training chart reports the four 256-decision PPO updates for learner seed
0. The combined objective adds policy loss, `0.5 × value_loss`,
`-0.01 × entropy`, and `0.05 × KL(policy || frozen_prior)`. Every point uses a
new on-policy batch, so it is not expected to decrease monotonically. Negative
total loss values are valid; the component curves and the exact diagnostics in
`results.json` make the behavior auditable. Each curve also records its raw
`updates.jsonl` SHA-256; `scripts/import_training_curves.py --check` verifies
the checked-in numbers against local run artifacts.

| Model / policy | Training battles | Validation battles | Test battles | Wins | Truncations | Mean turns | Final party HP | Battle score | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PrismNLI-0.4B zero shot | 0 | 0 | 18 | 18/18 | 0/18 | 2.61 | 92.4% | 92.1 | Complete |
| Laya zero shot | 0 | 0 | 18 | 18/18 | 0/18 | 6.17 | 83.5% | 85.3 | Complete |
| Uniform random zero shot | 0 | 0 | 18 | 18/18 | 0/18 | 4.17 | 84.5% | 87.4 | Complete |
| Jev zero shot | 0 | 0 | 18 | 18/18 | 0/18 | 2.83 | 90.2% | 92.9 | Complete |
| PrismNLI + residual PPO | 144 | 10 | 18 | 18/18 | 0/18 | 2.72 | 91.6% | 92.3 | Complete |
| Laya + residual PPO | 138 | 10 | 18 | 18/18 | 0/18 | 5.39 | 86.3% | 88.1 | Complete |
| Jev + residual PPO | 144 | 10 | 18 | 18/18 | 0/18 | 1.56 | 91.6% | 94.1 | Complete |

Training-battle counts are distinct save states actually sampled within the
fixed decision budget, not the 144-state pool size.

The primary score is equal-scenario-group held-out win rate; see
[the scoring rule](scoring.md). Since every completed arm has the same 100%
win rate, the battle scores are descriptive within-win diagnostics only. They
show that this particular held-out fixture was completed fastest by the Jev
residual policy, but do not establish broader Pokémon ability.

The frozen model weights were never fine-tuned. A 69,643-parameter residual
PPO policy was trained for 1,024 decisions with learner seed 0. Laya improved
from 85.3 to 88.1 battle score, with fewer turns and more party HP. PrismNLI
was effectively flat at 92.1 versus 92.3, with slightly more turns and less HP.
That apparent score increase comes from averaging the nonlinear speed term per
battle. Zero shot contributed 30.02 HP points and 12.10 speed points; the
residual contributed 29.76 HP points and 12.55 speed points. The residual had
more zero- and one-turn wins but also several long outliers, so its mean turns
worsened even though its mean `exp(-turns / 6)` improved. The resulting 92.309
versus 92.118 difference is mathematically consistent but too small and too
dependent on this scoring shape to call an improvement. Jev improved from
92.9 to 94.1, cutting mean turns from 2.83 to 1.56 while increasing retained
party HP from 90.2% to 91.6%.

The complete two-model local run took 796.56 seconds (13 minutes 17 seconds) on
a 10-core M1 Max with 64 GB RAM. The hosted Jev train, validation, and test run
took 1,704.82 seconds (28 minutes 25 seconds); provider latency can vary.

This is deliberately descriptive: one learner seed cannot support a confidence
interval, and all policies reached the win-rate ceiling, including uniform.
The fixture needs harder held-out states
(for example, stronger opposing trainers, unfavorable type matchups, and
resource-constrained party HP) before residual-training test results can
meaningfully test the steering hypothesis.

The exact raw episode records are created under ignored `runs/` directories by
the corresponding baseline, train, and evaluate commands. The checked-in GIFs
are replays from the same held-out protocol and preserve the real in-game move
menu.
