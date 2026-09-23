import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from rogue_rl.cli import _write_test_evaluation
from rogue_rl.prism import PrismConfig

spec = importlib.util.spec_from_file_location(
    "run_policy_study", Path(__file__).parents[1] / "scripts/run_policy_study.py"
)
assert spec is not None and spec.loader is not None
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
expected_config = study.expected_config
file_sha256 = study.file_sha256
bind_prior_provenance = study.bind_prior_provenance
raise_if_interrupted = study.raise_if_interrupted
validate_completed_run = study.validate_completed_run
validate_test_evaluation = study.validate_test_evaluation


def settings_fixture(tmp_path):
    corpus = tmp_path / "battles.json"
    rom = tmp_path / "game.gba"
    profile = tmp_path / "profile.json"
    corpus.write_text('{"battles": [{"split": "test"}, {"split": "test"}]}')
    rom.write_bytes(b"rom")
    profile.write_text("{}")
    return {
        "rom_path": str(rom),
        "corpus_path": str(corpus),
        "mgba": {"profile_path": str(profile)},
        "experiment": {
            "total_steps": 7,
            "rollout_steps": 2,
            "eval_every": 3,
            "eval_repeats": 1,
            "max_decisions": 10,
            "hidden_dim": 16,
            "alpha": 1.0,
            "ppo": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_coef": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "kl_coef": 0.05,
                "max_grad_norm": 0.5,
                "epochs": 1,
                "minibatch_size": 2,
            },
        },
    }


def completed_run(tmp_path, settings, seed=3):
    output = tmp_path / "prism-residual-3"
    output.mkdir()
    checkpoint = output / "checkpoint-000000007.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "config": expected_config(settings, seed),
        "corpus_sha256": file_sha256(tmp_path / "battles.json"),
        "rom_sha256": file_sha256(tmp_path / "game.gba"),
        "provenance": {
            "prior_name": "prism",
            "prior": {"config": asdict(PrismConfig())},
            "mgba_profile_sha256": file_sha256(tmp_path / "profile.json"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata))
    (output / "completed.json").write_text(
        json.dumps({"train_steps": 7, "final_checkpoint": checkpoint.name})
    )
    return output, checkpoint


def test_completed_run_is_bound_to_config_seed_and_artifacts(tmp_path):
    settings = settings_fixture(tmp_path)
    output, checkpoint = completed_run(tmp_path, settings)
    assert validate_completed_run(output, "prism", 3, settings, "unused") == checkpoint

    changed = deepcopy(settings)
    changed["experiment"]["max_decisions"] = 11
    with pytest.raises(RuntimeError, match="config and seed"):
        validate_completed_run(output, "prism", 3, changed, "unused")


def test_expected_config_materializes_dataclass_defaults():
    config = expected_config({"experiment": {"total_steps": 7}}, seed=2)
    assert config["total_steps"] == 7
    assert config["rollout_steps"] == 256
    assert config["ppo"]["gamma"] == 0.99
    assert config["ppo"]["seed"] == config["seed"] == 2


def test_study_manifest_binds_full_prior_provenance(tmp_path):
    path = tmp_path / "study.json"
    study_inputs = {"config_sha256": "one"}
    provenance = {"prompt_version": "v1", "backend": {"artifact": "abc"}}
    bind_prior_provenance(path, study_inputs, "laya", provenance)
    bind_prior_provenance(path, study_inputs, "laya", provenance)
    with pytest.raises(RuntimeError, match="provenance differs"):
        bind_prior_provenance(
            path,
            study_inputs,
            "laya",
            {"prompt_version": "v2", "backend": {"artifact": "abc"}},
        )


def test_interrupted_state_prevents_next_launch():
    with pytest.raises(InterruptedError, match="signal 15"):
        raise_if_interrupted({"signal": 15})


def test_partial_or_checkpoint_mismatched_test_evaluation_is_rejected(tmp_path):
    settings = settings_fixture(tmp_path)
    output, checkpoint = completed_run(tmp_path, settings)
    log = output / "test-evaluation.jsonl"
    rows = [
        {"battle_id": "one", "repeat": 0, "split": "test"},
        {"battle_id": "two", "repeat": 0, "split": "test"},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "episodes": 2,
        "result_sha256": file_sha256(log),
    }
    (output / "test-evaluation-metadata.json").write_text(json.dumps(manifest))
    assert validate_test_evaluation(output, checkpoint, 2)

    log.write_text(json.dumps(rows[0]) + "\n")
    assert not validate_test_evaluation(output, checkpoint, 2)
    (output / "test-evaluation-metadata.json").write_text("[]")
    assert not validate_test_evaluation(output, checkpoint, 2)
    (output / "test-evaluation-metadata.json").write_text(json.dumps(manifest))
    log.write_text("1\n")
    assert not validate_test_evaluation(output, checkpoint, 2)


def test_test_evaluation_writer_publishes_log_and_binding_atomically(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    log = tmp_path / "test-evaluation.jsonl"
    rows = [{"battle_id": "held-out", "repeat": 0, "split": "test"}]
    manifest_path = _write_test_evaluation(log, rows, checkpoint)
    manifest = json.loads(manifest_path.read_text())
    assert json.loads(log.read_text()) == rows[0]
    assert manifest["checkpoint_sha256"] == file_sha256(checkpoint)
    assert manifest["result_sha256"] == file_sha256(log)
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="already exist"):
        _write_test_evaluation(log, rows, checkpoint)
