# How a policy scores better

There is no hidden composite reward score in this benchmark. A model is judged
by a predeclared sequence of held-out measures, in this order.

## 1. Primary: wins on held-out scenario groups

For each saved battle, the game supplies one terminal outcome: win, loss, or
draw. A decision-limit truncation is reported separately and counts as a
non-win for the primary rate. For a scenario group `g`, first average repeated
RNG variants; then give every scenario group equal weight:

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

The comparison is paired and bootstrapped over learner seeds and scenario
groups. A policy is called better only when this difference is positive and
its 95% paired interval excludes zero. Otherwise the result is reported as
inconclusive, even if the point estimate is positive.

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

Each bar reports scenario-weighted win rate and its paired interval. A table
below it reports decisions, party HP, truncations, and latency. Missing arms are
shown as `not run`; they are never plotted as zero or inferred from another
model.
