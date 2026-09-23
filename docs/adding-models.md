# Adding a model to the benchmark

This guide adds a **frozen decision prior**. It does not fine-tune the model
and it does not change the PPO learner. The benchmark then evaluates the new
prior zero-shot and, optionally, trains a separate residual PPO policy above
its fixed action distribution.

## The contract

Implement one object with this method:

```python
def probabilities(self, observation: dict) -> np.ndarray:
    """Return one non-negative probability for each of ten action slots."""
```

The returned array must have shape `(10,)`. It must place zero mass on illegal
slots and positive total mass over the legal slots. Slots `0..3` are the four
move positions; slots `4..9` are absolute party positions used for switches.
Always use `checked_prior()` at the evaluation boundary; it rejects malformed
model output instead of silently substituting uniform action selection.

Your model receives only the `public_observation()` projection. Do not inspect
emulator memory directly, use state IDs as features, or add private opponent
information. The shared compact language projection is available through:

```python
from rogue_rl.laya import semantic_request
state, questions = semantic_request(observation)
```

Using this projection gives text-based models the same visible battle facts and
legal action descriptions as Laya and Jev. A numerical adapter may use the
normalized `FeatureEncoder` representation instead, but must document that
choice and never add fields unavailable to other models.

## Minimal adapter

Create `src/rogue_rl/my_model.py`:

```python
import numpy as np

from .laya import _legal, semantic_request


class MyModelPrior:
    def __init__(self, model):
        self.model = model
        self.calls = self.cache_hits = 0
        self.inference_seconds = 0.0

    @property
    def provenance(self):
        return {
            "backend": "my-runtime",
            "model_id": "publisher/model@immutable-revision",
            "prompt_version": "rogue-public-action-v1",
            "frozen": True,
        }

    def probabilities(self, observation):
        legal = _legal(observation)
        state, question = semantic_request(observation)
        # Convert the model's output to scores for exactly the legal slots.
        scores = self.model.score(state, question["action"]["criteria"])
        values = np.zeros(10, dtype=np.float64)
        for slot in np.flatnonzero(legal):
            values[slot] = max(0.0, float(scores[str(slot)]))
        if values[legal].sum() == 0:
            raise RuntimeError("MyModel returned no score for a legal action")
        self.calls += 1
        return values / values.sum()
```

Pin a model revision or artifact checksum. Store its identifier, runtime
version, prompt version, device, and inference configuration in `provenance`.
This metadata is written beside each baseline and PPO checkpoint and is checked
again before held-out test evaluation.

## Wire it into the CLI

1. Add its optional package dependencies to `pyproject.toml`, for example a
   `my-model` extra. Do not put credentials in the repository or config file.
2. Add `my-model` to the `--prior` choices in `src/rogue_rl/cli.py`.
3. Construct `MyModelPrior` in `_prior()`. Raise a clear error when a required
   local artifact, package, or environment credential is missing.
4. Add `tests/test_my_model.py` using an injected fake backend. Test legal-mask
   handling, normalization, an all-zero/error response, and provenance.
5. Run `uv run pytest -q` and `uv run ruff check src scripts tests`.

Do not add a fallback to Laya, PrismNLI, uniform, or a heuristic. A failed
provider must produce a failed/incomplete run, because a fallback invalidates a
comparison.

## Run the comparison

First validate the exact corpus and replay control:

```sh
uv run rogue-rl validate-corpus
uv run rogue-rl verify-game --steps 500
```

Use a matched, untouched test baseline only once the protocol is fixed:

```sh
uv run rogue-rl baseline --prior my-model --split test \
  --output runs/my-model-zero-shot-test
```

For policy steering, the prior remains frozen and only PPO is optimized from
the training split:

```sh
uv run rogue-rl train --mode residual --prior my-model --seed 0 \
  --output runs/my-model-residual-0
uv run rogue-rl evaluate --checkpoint runs/my-model-residual-0/checkpoint-000001024.pt
```

Run the same learner seeds, step budget, action cap, ROM/profile, and corpus as
the comparison arms. Tune only on validation. Evaluate the final checkpoint on
test once, then record win rate, outcome counts, decisions, final party HP,
switch/action diversity, latency, and residual override rate. If the uniform
control reaches the win-rate ceiling, report that the fixture cannot rank model
quality and collect harder held-out battle states before drawing conclusions.
