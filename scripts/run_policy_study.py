#!/usr/bin/env python3
"""Run and resume the preregistered residual-policy study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--priors", nargs="+", choices=["laya", "prism", "jev"], default=["laya", "prism"])
    result.add_argument("--seeds", nargs="+", type=int, default=[0])
    result.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    result.add_argument("--runs-dir", type=Path, default=Path("runs"))
    result.add_argument("--host", type=Path, default=Path(".cache/mgba-runner"))
    result.add_argument("--rom", type=Path, default=Path("data/rogue-research.gba"))
    result.add_argument("--lua", type=Path, default=Path("data/rogue-profile.lua"))
    result.add_argument("--openrouter-model", default="typesafe/jev-1.13")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src/rogue_rl").glob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def acquire_pid(path: Path) -> None:
    if path.exists():
        try:
            previous = int(path.read_text().strip())
        except ValueError:
            previous = -1
        if previous > 0 and alive(previous):
            raise RuntimeError(f"policy study is already running as PID {previous}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{os.getpid()}\n")


def expected_config(settings: dict, seed: int) -> dict:
    from rogue_rl.experiment import ExperimentConfig
    from rogue_rl.ppo import PPOConfig

    experiment = dict(settings.get("experiment", {}))
    ppo = PPOConfig(**experiment.pop("ppo", {}))
    config = ExperimentConfig(
        **experiment, mode="residual", seed=seed, ppo=replace(ppo, seed=seed)
    )
    return asdict(config)


def expected_prior_config(settings: dict, prior: str, openrouter_model: str) -> dict:
    if prior == "laya":
        from rogue_rl.laya import LayaConfig

        return asdict(LayaConfig(**settings["laya"]))
    if prior == "prism":
        from rogue_rl.prism import PrismConfig

        return asdict(PrismConfig(**settings.get("prism", {})))
    from rogue_rl.openrouter import OpenRouterConfig

    return asdict(OpenRouterConfig(model=openrouter_model))


def validate_completed_run(
    output: Path, prior: str, seed: int, settings: dict, openrouter_model: str
) -> Path:
    metadata = read_json(output / "metadata.json")
    completed = read_json(output / "completed.json")
    expected = expected_config(settings, seed)
    if metadata.get("config") != expected:
        raise RuntimeError(f"completed run at {output} does not match the requested config and seed")
    provenance = metadata.get("provenance", {})
    if provenance.get("prior_name") != prior:
        raise RuntimeError(f"completed run at {output} was produced by another prior")
    prior_provenance = provenance.get("prior") or {}
    if prior_provenance.get("config") != expected_prior_config(settings, prior, openrouter_model):
        raise RuntimeError(f"completed run at {output} has different prior configuration")
    if prior == "laya":
        manifest = Path(settings["laya"]["model_dir"]) / "coreml_config.json"
        backend = prior_provenance.get("backend", {})
        if backend.get("manifest_sha256") != file_sha256(manifest):
            raise RuntimeError(f"completed run at {output} has a different Laya model artifact")
    corpus_path = Path(settings["corpus_path"])
    rom_path = Path(settings["rom_path"])
    profile_path = Path(settings["mgba"]["profile_path"])
    if metadata.get("corpus_sha256") != file_sha256(corpus_path):
        raise RuntimeError(f"completed run at {output} has a different corpus")
    if metadata.get("rom_sha256") != file_sha256(rom_path):
        raise RuntimeError(f"completed run at {output} has a different ROM")
    if provenance.get("mgba_profile_sha256") != file_sha256(profile_path):
        raise RuntimeError(f"completed run at {output} has a different mGBA profile")
    total_steps = expected["total_steps"]
    if completed.get("train_steps") != total_steps:
        raise RuntimeError(f"completed run at {output} did not reach the registered budget")
    checkpoint = output / completed.get("final_checkpoint", "")
    if not checkpoint.is_file() or checkpoint.name != f"checkpoint-{total_steps:09d}.pt":
        raise RuntimeError(f"completed run at {output} has no matching final checkpoint")
    return checkpoint


def bind_prior_provenance(path: Path, study: dict, prior: str, provenance: dict) -> None:
    manifest = read_json(path) if path.exists() else {"schema_version": 1, "study": study, "priors": {}}
    if not isinstance(manifest, dict) or manifest.get("study") != study:
        raise RuntimeError("existing policy-study manifest belongs to a different experiment")
    priors = manifest.get("priors")
    if not isinstance(priors, dict):
        raise TypeError("policy-study manifest has invalid prior provenance")
    if prior in priors and priors[prior] != provenance:
        raise RuntimeError(f"{prior} provenance differs from earlier seeds in this study")
    priors[prior] = provenance
    write_json(path, manifest)


def validate_test_evaluation(output: Path, checkpoint: Path, expected_episodes: int) -> bool:
    log = output / "test-evaluation.jsonl"
    manifest_path = output / "test-evaluation-metadata.json"
    if not log.exists() and not manifest_path.exists():
        return False
    if not log.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or not all(isinstance(row, dict) for row in rows):
        return False
    keys = [(row.get("battle_id"), row.get("repeat")) for row in rows]
    return (
        manifest.get("checkpoint") == checkpoint.name
        and manifest.get("checkpoint_sha256") == file_sha256(checkpoint)
        and manifest.get("episodes") == expected_episodes == len(rows)
        and manifest.get("result_sha256") == file_sha256(log)
        and len(set(keys)) == len(keys)
        and all(row.get("split") == "test" for row in rows)
    )


def quarantine_incomplete_test(output: Path) -> None:
    suffix = f".incomplete-{int(time.time())}"
    for name in (
        "test-evaluation.jsonl",
        "test-evaluation-metadata.json",
        "test-evaluation.jsonl.tmp",
        "test-evaluation-metadata.json.tmp",
    ):
        path = output / name
        if path.exists():
            path.replace(path.with_name(path.name + suffix))


def raise_if_interrupted(interrupted: dict) -> None:
    if interrupted.get("signal") is not None:
        raise InterruptedError(f"received signal {interrupted['signal']}")


def wait_for_host(
    host_log: Path, process: subprocess.Popen, interrupted: dict, timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raise_if_interrupted(interrupted)
        if process.poll() is not None:
            raise RuntimeError(f"mGBA host exited with code {process.returncode}; see {host_log}")
        if host_log.exists() and "bridge listening" in host_log.read_text(errors="replace"):
            return
        time.sleep(0.1)
    raise RuntimeError(f"mGBA host did not become ready within {timeout:g}s; see {host_log}")


def main() -> int:
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    pid_path = args.runs_dir / "policy-study.pid"
    status_path = args.runs_dir / "policy-study-status.json"
    study_path = args.runs_dir / "policy-study-manifest.json"
    host_log = args.runs_dir / "policy-study-mgba.log"
    acquire_pid(pid_path)

    planned = [{"prior": prior, "seed": seed} for prior in args.priors for seed in args.seeds]
    status: dict[str, Any] = {
        "state": "starting",
        "pid": os.getpid(),
        "started_unix_seconds": time.time(),
        "planned": planned,
        "completed": [],
        "current": None,
        "failure": None,
    }
    host: subprocess.Popen | None = None
    child: subprocess.Popen | None = None
    host_handle = None
    interrupted = {"signal": None}

    def handle_signal(signum, _frame) -> None:
        interrupted["signal"] = signum
        for process in (child, host):
            if process is not None and process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def run_logged(command: list[str], log_path: Path) -> None:
        nonlocal child
        raise_if_interrupted(interrupted)
        with log_path.open("a") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            returncode = child.wait()
            child = None
        raise_if_interrupted(interrupted)
        if returncode:
            raise RuntimeError(f"command exited with code {returncode}; see {log_path}")

    try:
        write_json(status_path, status)
        if "jev" in args.priors and not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("Jev requested but OPENROUTER_API_KEY is not set")
        settings = read_json(args.config)
        corpus = read_json(Path(settings["corpus_path"]))
        status["study"] = {
            "config_sha256": file_sha256(args.config),
            "corpus_sha256": file_sha256(Path(settings["corpus_path"])),
            "rom_sha256": file_sha256(Path(settings["rom_path"])),
            "mgba_profile_sha256": file_sha256(Path(settings["mgba"]["profile_path"])),
            "implementation_sha256": implementation_sha256(root),
        }
        if study_path.exists():
            existing_study = read_json(study_path)
            if not isinstance(existing_study, dict) or existing_study.get("study") != status["study"]:
                raise RuntimeError("existing policy-study manifest belongs to a different experiment")
        else:
            write_json(
                study_path,
                {"schema_version": 1, "study": status["study"], "priors": {}},
            )
        write_json(status_path, status)
        expected_test_episodes = sum(row["split"] == "test" for row in corpus["battles"])
        expected_test_episodes *= settings["experiment"].get("eval_repeats", 1)
        host_handle = host_log.open("w")
        host = subprocess.Popen(
            [str(args.host), str(args.rom), str(args.lua)],
            stdout=host_handle,
            stderr=subprocess.STDOUT,
        )
        wait_for_host(host_log, host, interrupted)
        raise_if_interrupted(interrupted)
        status["state"] = "running"
        write_json(status_path, status)
        for item in planned:
            raise_if_interrupted(interrupted)
            prior, seed = item["prior"], item["seed"]
            output = args.runs_dir / f"{prior}-residual-{seed}"
            log_path = args.runs_dir / f"{prior}-residual-{seed}.log"
            status["current"] = item
            write_json(status_path, status)
            if output.exists() and not (output / "completed.json").exists():
                raise RuntimeError(
                    f"incomplete run directory exists at {output}; preserve or remove it before resuming"
                )
            if not output.exists():
                command = [
                    sys.executable,
                    "-m",
                    "rogue_rl.cli",
                    "--config",
                    str(args.config),
                    "train",
                    "--mode",
                    "residual",
                    "--prior",
                    prior,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                ]
                if prior == "jev":
                    command += ["--openrouter-model", args.openrouter_model]
                run_logged(command, log_path)
            checkpoint = validate_completed_run(output, prior, seed, settings, args.openrouter_model)
            metadata = read_json(output / "metadata.json")
            bind_prior_provenance(
                study_path,
                status["study"],
                prior,
                metadata["provenance"]["prior"],
            )
            raise_if_interrupted(interrupted)
            if not validate_test_evaluation(output, checkpoint, expected_test_episodes):
                quarantine_incomplete_test(output)
                command = [
                    sys.executable,
                    "-m",
                    "rogue_rl.cli",
                    "--config",
                    str(args.config),
                    "evaluate",
                    "--checkpoint",
                    str(checkpoint),
                ]
                if prior == "jev":
                    command += ["--openrouter-model", args.openrouter_model]
                run_logged(command, log_path)
            if not validate_test_evaluation(output, checkpoint, expected_test_episodes):
                raise RuntimeError(f"test evaluation for {output} failed completeness validation")
            raise_if_interrupted(interrupted)
            status["completed"].append(item)
            write_json(status_path, status)
        raise_if_interrupted(interrupted)
        status["state"] = "complete"
        status["current"] = None
        status["finished_unix_seconds"] = time.time()
        write_json(status_path, status)
        return 0
    except BaseException as exc:
        interrupted_types = (InterruptedError, KeyboardInterrupt)
        status["state"] = "interrupted" if isinstance(exc, interrupted_types) else "failed"
        status["failure"] = f"{type(exc).__name__}: {exc}"
        status["finished_unix_seconds"] = time.time()
        write_json(status_path, status)
        raise
    finally:
        for process in (child, host):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if host_handle is not None:
            host_handle.close()
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
