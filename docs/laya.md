# Frozen fast Laya prior

The default uses the actual **Laya multilingual CoreML ANE FP16 L96** bundle,
not a generative language model or an approximation implemented in this project.
Its parameters remain frozen. The upstream runtime returns typed probabilities
and zero generated tokens. No fallback model, synthetic probabilities, or remote
API is used if artifacts are missing or inference fails.

## Reproducible installation

Requirements: Apple Silicon, macOS 15+, Python 3.11–3.13. Install this project's
Laya extra, then download the separately hosted weights:

```sh
python -m pip install -e '.[laya]'
python scripts/setup_laya.py --preset ane96
```

The download is approximately 700 MB (consult the upstream artifact inventory for
exact size). It happens only in this explicit setup command. The experiment loads
with `local_files_only=True`; no network is used during inference. Configure
`model_dir="models/laya-ane96"` when using the setup directory. Alternatively leave
`model_dir=None` to use an already populated Hugging Face cache for the pinned revision.
A local model directory must include the setup script's provenance receipt.

Pinned runtime: `laya-coreml==0.1.0`. Source inspected at
[`12b7501583c7f03a6b2e49ebe118a2c6302505b9`](https://github.com/mizorewww/laya-coreml/tree/12b7501583c7f03a6b2e49ebe118a2c6302505b9).
Pinned ANE model revision:
`aac6fef/laya-multilingual-coreml-ane@39d6a9b3d0f67f06da74fbade6121ea134cbdb21`.
Pinned optional general model revision:
`aac6fef/laya-multilingual-coreml@8139e9089273319512c730218903784074133187`.

Official sources: [API and capacity](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/docs/USAGE.md),
[pinned releases](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/docs/RELEASE.md),
[benchmark conditions](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/docs/ANE_BENCHMARKS.md).

## Decision mapping and capacity adaptation

The policy has ten stable slots: moves 0–3 and switches to party slots 0–5 in
positions 4–9. The observation supplies legality. Illegal entries always receive
exactly zero probability; a sole legal action needs no model call.

The fast artifact fits **96 tokens total**, including instructions, option
markers, and state. A complete battle encoded as one ten-option question often
cannot fit. The default `strategy="action_scores"` asks a compact independent
`noul` (boolean) question for each legal action: “Good action to win?” It then
normalizes those affirmative probabilities over legal actions. This is a
**constructed action prior**, not Laya's original joint categorical distribution.
Its interpretation and calibration differ from a single choice head and this
adaptation must be stated in reports. The raw rounded scores are floored at
`1e-6` before normalization for finite log priors.

Each request includes only active friendly/visible enemy names, types, HP
percentage, status, and the candidate move's name/type/power or candidate switch's
name/types/HP/status. It deliberately omits detailed stat stages, speed, move
accuracy, PP amounts (legality handles exhausted moves), full history, and opposing
unrevealed moves/items/ability/stats/bench. This is a fixed versioned semantic
projection (`rogue-public-action-v1`), not runtime text truncation. The numeric
policy can have a richer observation. Fields named `name`/`species_name` must be
public catalog values rather than hidden enemy metadata. Type IDs follow Rogue's
Gen 3/expansion convention (Fairy = 18).

The adapter counts tokens **before** upstream's prefix/state truncation and fails
with a diagnostic if any question, option, or complete state would be truncated.
Long species or move names can still overflow; no automatic switch of model or
strategy is allowed. The explicit `preset="general1024", strategy="joint"`
configuration offers a joint legal-choice ablation at a larger capacity. Download
it separately with `python scripts/setup_laya.py --preset general1024`.

## Timing and reproducibility

Upstream's roughly 5 ms number measures one short question on its M3 Max, excluding
load/warmup. Here a decision may need several sequential questions, plus emulator,
feature, and policy work. It is **not** a promised 5 ms battle decision. Report
measured warmup separately and collect `LayaPrior.last_metrics`: model calls, cache
hits, inference seconds, and total prior time. Identical semantic requests reuse
an in-process bounded LRU cache; battle/decision IDs are intentionally not cache
inputs. Report cache setting and hit rate with speed measurements.

`LayaPrior.provenance` records configuration, prompt version, artifact revision,
runtime version, source commit, compute units, and manifest SHA256. Dependency-
injected predictors are explicitly labeled as test predictors. Model conversion
fidelity is not evidence of Pokémon competence; actual held-out battle results
are required before claiming any benefit.
