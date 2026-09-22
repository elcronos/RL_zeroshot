"""Baseline bookkeeping uses replay fixtures, never synthetic game claims."""

import json
from pathlib import Path

import numpy as np
import pytest

from rogue_rl import cli
from rogue_rl.manifest import sha256_file
from rogue_rl.report import baseline_summary, compare_baselines


class FixturePrior:
    def __init__(self):
        self.provenance = {"test_fixture": True}
        self.calls = 0
        self.cache_hits = 0
        self.inference_seconds = 0.0

    def probabilities(self, obs):
        self.calls += 1
        return np.asarray([0.75, 0.25] + [0.0] * 8)


class FixtureEnv:
    def __init__(self, *, fail_second=False):
        self.fail_second = fail_second
        self.resets = 0
        self.closed = False

    def describe(self):
        return {"protocol": 1, "fixture": True}

    def reset(self, battle):
        self.resets += 1
        if self.fail_second and self.resets == 2:
            raise RuntimeError("fixture emulator disconnected")
        self.battle = battle
        return self._observation(False)

    def step(self, action):
        assert action in (0, 1)
        return self._observation(True)

    def screenshot(self, path):
        target = Path(path)
        target.write_bytes(b"fixture frame")
        return target

    def menu_screenshot(self, path):
        return self.screenshot(path)

    def _observation(self, terminal):
        return {
            "battle_id": self.battle["id"],
            "decision_id": 1 if terminal else 0,
            "turn": 1,
            "phase": "terminal" if terminal else "action",
            "outcome": ("win" if self.battle["id"] == "a" else "loss") if terminal else None,
            "legal_actions": [False] * 10 if terminal else [True, True] + [False] * 8,
            "player": {"hp_fraction": 0.75},
            "opponent": {"hp_fraction": 0.5},
            "party": [],
            "moves": [{"name": "Fixture 1"}, {"name": "Fixture 2"}],
        }

    def close(self):
        self.closed = True


def config_fixture(tmp_path):
    rom = tmp_path / "fixture.gba"
    rom.write_bytes(b"fixture ROM")
    rom_hash = sha256_file(rom)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"rom_sha256": rom_hash, "source_revision": "fixture"}))
    battles = []
    for i, name in enumerate(("a", "b")):
        state = tmp_path / f"{name}.ss0"
        state.write_bytes(f"state {name}".encode())
        battles.append(
            {
                "id": name,
                "split": "train",
                "scenario_group": name,
                "seed": i,
                "state_path": state.name,
                "state_sha256": sha256_file(state),
            }
        )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rom_sha256": rom_hash,
                "source_revision": "fixture",
                "battles": battles,
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "rom_path": str(rom),
                "corpus_path": str(corpus),
                "mgba": {"profile_path": str(profile)},
                "experiment": {"max_decisions": 5},
            }
        )
    )
    return config


def test_baseline_writes_real_episode_rows_trace_and_summary(tmp_path, monkeypatch, capsys):
    config = config_fixture(tmp_path)
    env = FixtureEnv()
    monkeypatch.setattr(cli, "_environment", lambda *_: env)
    monkeypatch.setattr(cli, "_laya", lambda *_: FixturePrior())
    output = tmp_path / "run"
    assert cli.main(["--config", str(config), "baseline", "--output", str(output)]) == 0
    assert env.closed
    assert json.loads(capsys.readouterr().out)["completed_episodes"] == 2
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert [row["outcome"] for row in rows] == ["win", "loss"]
    assert all(row["train_steps"] == 0 for row in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["episode_win_rate"] == 0.5
    assert summary["outcome_counts"] == {"win": 1, "loss": 1, "draw": 0, "truncated": 0}
    assert summary["scenario_bootstrap_ci95"] is not None
    assert summary["model_metrics"]["calls"] == 2
    trace = [json.loads(line) for line in (output / "first-battle-trace.jsonl").read_text().splitlines()]
    assert len(trace) == 1 and trace[0]["battle_id"] == "a"
    meta = json.loads((output / "metadata.json").read_text())
    assert meta["rom_sha256"] == sha256_file(tmp_path / "fixture.gba")
    assert meta["mgba_bridge"]["protocol"] == 1


def test_uniform_comparator_uses_same_battles_without_loading_laya(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    env = FixtureEnv()
    monkeypatch.setattr(cli, "_environment", lambda *_: env)

    def no_laya(*_):
        raise AssertionError("Uniform control must not load the model")

    monkeypatch.setattr(cli, "_laya", no_laya)
    output = tmp_path / "uniform"
    assert cli.main(["--config", str(config), "baseline", "--prior", "uniform", "--output", str(output)]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert metadata["mode"] == "uniform_baseline" and metadata["laya"] is None
    assert summary["prior"] == "uniform" and summary["model_metrics"]["calls"] == 0


def test_compare_baselines_requires_matching_rows_and_clusters_scenarios(tmp_path):
    def write_run(name, rows):
        path = tmp_path / name
        path.mkdir()
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "mode": name,
                    "rom_sha256": "r",
                    "source_revision": "s",
                    "corpus_sha256": "c",
                    "battle_ids": ["a", "b"],
                    "repeats": 1,
                    "max_decisions": 9,
                }
            )
        )
        (path / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    a = write_run(
        "a",
        [
            {"battle_id": "a", "repeat": 0, "scenario_group": "g1", "outcome": "win"},
            {"battle_id": "b", "repeat": 0, "scenario_group": "g2", "outcome": "loss"},
        ],
    )
    b = write_run(
        "b",
        [
            {"battle_id": "a", "repeat": 0, "scenario_group": "g1", "outcome": "loss"},
            {"battle_id": "b", "repeat": 0, "scenario_group": "g2", "outcome": "loss"},
        ],
    )
    result = compare_baselines(a, b)
    assert result["scenario_weighted_win_rate_difference_a_minus_b"] == 0.5
    assert result["scenario_bootstrap_ci95"] is not None


def test_baseline_failure_preserves_completed_rows_and_marks_partial(tmp_path, monkeypatch, capsys):
    config = config_fixture(tmp_path)
    env = FixtureEnv(fail_second=True)
    monkeypatch.setattr(cli, "_environment", lambda *_: env)
    monkeypatch.setattr(cli, "_laya", lambda *_: FixturePrior())
    output = tmp_path / "partial"
    assert cli.main(["--config", str(config), "baseline", "--output", str(output)]) == 2
    assert env.closed
    assert "1/2 episodes" in capsys.readouterr().err
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_episodes"] == 1
    assert summary["scenario_bootstrap_ci95"] is None
    assert "disconnected" in summary["failure"]


def test_baseline_limits_and_no_double_counting(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    env = FixtureEnv()
    monkeypatch.setattr(cli, "_environment", lambda *_: env)
    monkeypatch.setattr(cli, "_laya", lambda *_: FixturePrior())
    output = tmp_path / "limited"
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "baseline",
                "--max-battles",
                "1",
                "--repeats",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["planned_episodes"] == 2
    assert summary["scenario_groups_completed"] == 1
    assert summary["scenario_bootstrap_ci95"] is None


def test_visual_uniform_records_each_decision_and_captured_frames(tmp_path, monkeypatch, capsys):
    config = config_fixture(tmp_path)
    env = FixtureEnv()
    monkeypatch.setattr(cli, "_environment", lambda *_: env)

    def make_gif(frames, target, duration):
        assert len(frames) == 2 and duration == 900
        target.write_bytes(b"fixture gif")
        return target

    def annotate(_source, target, _obs, _probabilities, _selected):
        target.write_bytes(b"annotated fixture frame")
        return target

    monkeypatch.setattr(cli, "_make_gif", make_gif)
    monkeypatch.setattr(cli, "_annotate_frame", annotate)
    output = tmp_path / "visual"
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "visual",
                "--prior",
                "uniform",
                "--battle-id",
                "a",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete" and result["outcome"] == "win"
    assert result["frames"] == ["frames/0000-decision.png", "frames/terminal.png"]
    decision = json.loads((output / "decisions.jsonl").read_text())
    assert decision["action"] in (0, 1)
    assert decision["action_name"] == f"Fixture {decision['action'] + 1}"
    assert (output / "battle.gif").read_bytes() == b"fixture gif"
    assert env.closed


def test_baseline_summary_rejects_bad_outcomes_and_mismatched_completion():
    with pytest.raises(ValueError, match="Unknown battle outcome"):
        baseline_summary(
            [{"outcome": "unknown", "battle_id": "a", "repeat": 0, "scenario_group": "a"}],
            planned_episodes=1,
            status="complete",
        )
    with pytest.raises(ValueError, match="do not match"):
        baseline_summary([], planned_episodes=1, status="complete")
