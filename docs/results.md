# Held-out zero-shot and 1,024-step policy results

This report was generated from `data/battles.json` with 18 test save states in
six disjoint player-team/opponent-trainer scenario groups. The corpus contains
172 states total: 144 train, 10 validation, and 18 test. The test group has all
six declared player teams.

![Held-out zero-shot comparison](assets/zero-shot-performance.png)

![Frozen models versus trained policies](assets/trained-policy-performance.png)

| Model / policy | Training battles | Validation battles | Test battles | Wins | Mean turns | Final party HP | Battle score | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PrismNLI-0.4B zero shot | 0 | 0 | 18 | 18/18 | 2.61 | 92.4% | 92.1 | Complete |
| Laya zero shot | 0 | 0 | 18 | 18/18 | 6.17 | 83.5% | 85.3 | Complete |
| Uniform random zero shot | 0 | 0 | 18 | 18/18 | 4.17 | 84.5% | 87.4 | Complete |
| Jev zero shot | 0 | 0 | 18 | — | — | — | — | Credential required |
| PrismNLI + residual PPO | 144 | 10 | 18 | 18/18 | 2.72 | 91.6% | 92.3 | Complete |
| Laya + residual PPO | 138 | 10 | 18 | 18/18 | 5.39 | 86.3% | 88.1 | Complete |
| Jev + residual PPO | — | — | — | — | — | — | — | Not run |

Training-battle counts are distinct save states actually sampled within the
fixed decision budget, not the 144-state pool size.

The primary score is equal-scenario-group held-out win rate; see
[the scoring rule](scoring.md). Since every completed arm has the same 100%
win rate, the battle scores are descriptive within-win diagnostics only. They
show that this particular held-out fixture was completed faster and with more
HP by PrismNLI, but do not establish broader Pokémon ability.

The frozen model weights were never fine-tuned. A 69,643-parameter residual
PPO policy was trained for 1,024 decisions with learner seed 0. Laya improved
from 85.3 to 88.1 battle score, with fewer turns and more party HP. PrismNLI
was effectively flat at 92.1 versus 92.3, with slightly more turns and less HP.
The complete two-model local run took 796.56 seconds (13 minutes 17 seconds) on
a 10-core M1 Max with 64 GB RAM.

This is deliberately descriptive: one learner seed cannot support a confidence
interval, and all completed policies reached the win-rate ceiling, including
uniform. The fixture needs harder held-out states
(for example, stronger opposing trainers, unfavorable type matchups, and
resource-constrained party HP) before residual-training test results can
meaningfully test the steering hypothesis.

The exact raw episode records are created under ignored `runs/` directories by
the corresponding baseline, train, and evaluate commands. The checked-in GIFs
are replays from the same held-out protocol and preserve the real in-game move
menu.
