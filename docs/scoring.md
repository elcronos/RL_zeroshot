# How a policy scores better

There is no hidden reward score in this benchmark. A model is judged by a
predeclared sequence of held-out measures, including the explicit diagnostic
battle score defined below.

## 1. Primary: wins on held-out scenario groups

For each saved battle, the benchmark records one terminal outcome: win or
loss. Emerald's internal simultaneous-exhaustion outcome is treated as a
single-player defeat by the game itself, so the bridge records it as a loss.
A decision-limit truncation is reported separately and counts as a non-win for
the primary rate. For a scenario group `g`, first average repeated RNG
variants; then give every scenario group equal weight:

```
test win rate = mean_g(mean_repeat(terminal outcome is win))
```

This prevents a team/trainer matchup with many saved RNG variants from
dominating the result. It is the only primary ranking metric.

For policy learning, compare a residual policy with **its own frozen prior**
on exactly the same test battle IDs, learner seeds, and repeats:

```
improvement = test_win_rate(residual) - test_win_rate(frozen prior)
```

With at least two learner seeds, the comparison is paired and bootstrapped over
learner seeds and scenario groups. A policy is called statistically better only
when this difference is positive and its 95% paired interval excludes zero.
The talk-sized protocol has one seed, suppresses the interval, and reports only
the observed descriptive change.

## 2. Guardrails

A run cannot be called better if it has more than a predeclared practical
truncation increase (default: 2 percentage points) or uses a different ROM,
profile, corpus hash, action cap, decision sampling rule, or fixed training
budget. Failed provider calls do not receive replacement actions.

If uniform legal action sampling reaches the win-rate ceiling, win rate cannot
rank the models. That is the current pilot outcome: Laya, PrismNLI, and uniform
all won 18/18 held-out episodes. It means the fixture is too easy, not that
the models are equally capable. Collect harder states before interpreting a
residual-training comparison.

## Granular score within a battle outcome

Every evaluated episode also receives a **battle score** from 0 to 100. It is a
readable diagnostic, not PPO's reward and not a replacement for the primary
win-rate comparison.

For a win, calculate the final fraction of HP across the player party (`H`) and
the game-reported battle turns (`T`). Six turns is the declared reference
length. The within-win quality is:

```
win_quality = 100 * (0.65 * H + 0.35 * exp(-T / 6))
battle_score = 50 + 0.5 * win_quality
```

The 65/35 split gives survival more importance than speed. A two-turn win with
the whole party healthy receives about 90/100 quality and 95/100 battle score;
a ten-turn win with half the party HP receives about 39/100 quality and 70/100
battle score. Any win still scores above a loss or truncation:

| Outcome | Battle score |
| --- | --- |
| Win | `50 + 0.5 * win_quality` (50–100) |
| Loss | `20 * H` (0–20) |
| Incomplete (truncated) | 0 |

“Truncated” means the battle did not reach the game's win/loss state before
the evaluator's fixed 500-decision safety cap. During training only, an active
battle can also be truncated at a validation boundary so validation starts
from clean save states. It is an experiment-control event, not a Pokémon
battle outcome. The reason is stored with every truncated training episode.

This creates the granular explanation you asked for: a policy can win equally
often but conserve more HP or finish in fewer turns. Report `mean_win_quality`
only over wins and `mean_battle_score` over all episodes, alongside the raw
outcome counts.

## 3. Secondary diagnostics

These metrics explain a tied or improved win rate; they never override the
primary result.

| Metric | Better direction | Meaning |
| --- | --- | --- |
| Decisions to terminal | Lower, conditional on the same outcome | Battle efficiency; do not compare a fast loss with a win. |
| Final party HP fraction | Higher, conditional on a win | Remaining team health for the next battle. |
| Truncation rate | Lower | Whether the fixed decision cap is adequate. |
| Switch-action rate and distinct actions | Descriptive | Checks for degenerate always-attack or always-switch behaviour. |
| Decision latency / wall-clock | Lower | Deployment cost, reported separately from game quality. |
| Residual argmax override rate / KL | Descriptive | How much the learned policy changes its frozen prior. |

## Required plot

Every final report has two held-out panels with identical y-axis and test rows:

1. **Zero shot:** Laya, PrismNLI, Jev, and uniform.
2. **Policy learning:** each frozen prior beside its residual PPO policy.

The talk-sized plot reports battle score beside the auditable wins, turns, and
party HP values and does not draw a confidence interval for one seed. A larger
multi-seed report should add scenario-weighted win-rate intervals. Missing arms
are shown as `not run`; they are never plotted as zero or inferred from another
model.
