# Frozen Laya pilot: route-trainer fixture

Date: 2026-09-22. This is an engineering and measurement pilot, not a held-out
evaluation of general Emerald Rogue play.

The research ROM starts a fresh game with a documented level-10 Bulbasaur and
then enters a normal Rogue run. Rogue selects and generates the first route
trainer; the battle itself is started by the game's standard trainer-battle
script in singles/SET mode. Ten pre-policy mGBA save states were captured from
that fixture. They contain five route-trainer scenario groups; repeated states
within a group are RNG variants, not independent scenarios.

The capture screenshot below is one actual state. It shows level-10 Bulbasaur
against Pansage, before the policy takes its first action.

![Captured battle state](/Users/camilopestana/Documents/ChatGPT/RL_zeroshot/data/states/pilot-000.png)

| Policy | Episodes | Scenario groups | Wins–losses | Scenario-weighted win rate | Median decision latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen Laya ANE-96 | 10 | 5 | 7–3 | 0.667 | 35.9 ms |
| Uniform legal action | 10 | 5 | 9–1 | 0.800 | 0.11 ms |

The paired scenario-weighted difference, Laya minus uniform, was **−0.133**
with a scenario-bootstrap 95% interval of **[−0.400, 0.000]**. Only the
Pansage route-trainer group differed materially: Laya won one of three saved
RNG variants while uniform won all three. All other group means matched.

Laya’s pilot involved 352 action-score calls, 32 cache hits and 3.21 seconds
of model compute. Its ten battles averaged 9.6 decisions and 1.69 seconds of
wall time. Uniform averaged 5.5 decisions and 0.71 seconds. Both arms used the
same ROM, hashed saved states, bridge, action cap of 100, and action seed 0.
No episode truncated or reached a protocol error.

The pilot establishes that the actual mGBA reset, action stepping, terminal
outcome collection, screenshot capture, and result recording work together.
It cannot distinguish game competence from luck on this small, highly specific
starter/trainer fixture. A proper comparison needs fresh player parties,
different route tiers and bosses, several independent run seeds, and a
validation/test split frozen before model selection.

Evidence is retained in:

- [corpus manifest](/Users/camilopestana/Documents/ChatGPT/RL_zeroshot/data/battles.json)
- [Laya episode results](/Users/camilopestana/Documents/ChatGPT/RL_zeroshot/runs/laya-pilot-ten/episodes.jsonl)
- [Laya summary](/Users/camilopestana/Documents/ChatGPT/RL_zeroshot/runs/laya-pilot-ten/summary.json)
- [uniform summary](/Users/camilopestana/Documents/ChatGPT/RL_zeroshot/runs/uniform-pilot-ten/summary.json)
- [paired comparison](/Users/camilopestana/Documents/ChatGPT/RL_zeroshot/runs/pilot-laya-vs-uniform.json)

To compare a future adapter such as PrismNLI-0.4 or Jev, give it the same
`probabilities(public_observation) -> 10 legal-action probabilities` interface
and run it on a newly held-out corpus. Record the model artifact, prompts,
hardware, cache policy and wall-clock behavior in its `metadata.json`.
