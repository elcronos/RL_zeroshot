# First held-out baseline

This report was generated from `data/battles.json` with 18 test save states in
six disjoint player-team/opponent-trainer scenario groups. The corpus contains
172 states total: 144 train, 10 validation, and 18 test. The test group has all
six declared player teams.

![Held-out baseline comparison](assets/zero-shot-baselines.png)

| Frozen policy | Test wins | Mean turns | Final party HP | Mean battle score | Decision p50 | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Laya CoreML ANE | 18 / 18 | 6.17 | 83.5% | 85.3 / 100 | 51.08 ms | Complete |
| PrismNLI-0.4B | 18 / 18 | 2.61 | 92.4% | 92.1 / 100 | 171.87 ms | Complete |
| Uniform legal action | 18 / 18 | 4.17 | 84.5% | 87.4 / 100 | 0.10 ms | Complete |
| Jev (OpenRouter) | — | — | — | — | — | Not run: valid API credential required |

The primary score is equal-scenario-group held-out win rate; see
[the scoring rule](scoring.md). Since every completed arm has the same 100%
win rate, the battle scores are descriptive within-win diagnostics only. They
show that this particular held-out fixture was completed faster and with more
HP by PrismNLI, but do not establish broader Pokémon ability.

No policy was trained for this report, so there is no learned-policy panel to
compare yet. Laya and PrismNLI are frozen zero-shot baselines; uniform is a
no-model control. This is deliberately not presented as evidence that one
model plays better: all completed policies reached the win-rate ceiling,
including uniform. The fixture needs harder held-out states
(for example, stronger opposing trainers, unfavorable type matchups, and
resource-constrained party HP) before residual-training test results can
meaningfully test the steering hypothesis.

The exact raw episode records are created under ignored `runs/` directories by
the baseline command. The checked-in GIFs are replays from the same held-out
protocol, and preserve the real in-game move menu.
