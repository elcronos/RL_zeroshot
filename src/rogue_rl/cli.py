"""Command-line entry points. No ROM or synthetic-results fallback."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .experiment import (
    ExperimentConfig,
    UniformPrior,
    action_distribution,
    checked_prior,
    evaluate,
    load_checkpoint,
    sample_action,
    select_prior,
    train,
)
from .manifest import Corpus, sha256_file
from .observations import FeatureEncoder
from .ppo import PPOConfig
from .report import baseline_summary, compare_baselines, read_jsonl, summarize


def _settings(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _laya(settings: dict, mode: str):
    if mode in {"scratch", "uniform_residual"}:
        return None
    from .laya import LayaConfig, LayaPrior

    return LayaPrior(LayaConfig(**settings["laya"]))


def _prior(settings: dict, args: argparse.Namespace):
    if args.prior == "laya":
        prior = _laya(settings, "frozen")
        return prior, prior
    if args.prior == "uniform":
        return UniformPrior(), None
    if args.prior == "prism":
        from .prism import PrismConfig, PrismPrior

        return PrismPrior(PrismConfig(**settings.get("prism", {}))), None
    if not args.openrouter_model:
        raise ValueError("--openrouter-model is required for the Jev/OpenRouter baseline")
    from .openrouter import OpenRouterConfig, OpenRouterPrior

    return OpenRouterPrior(OpenRouterConfig(model=args.openrouter_model)), None


def _environment(settings: dict, corpus: Corpus):
    from .mgba import MGBAEnv

    corpus.verify_rom(settings["rom_path"])
    profile = json.loads(Path(settings["mgba"]["profile_path"]).read_text())
    if profile["rom_sha256"] != corpus.rom_sha256 or profile["source_revision"] != corpus.source_revision:
        raise ValueError("mGBA profile and corpus must match ROM hash and source revision")
    return MGBAEnv(**settings["mgba"])


def _config(settings: dict, mode: str, seed: int) -> ExperimentConfig:
    raw = dict(settings.get("experiment", {}))
    ppo = PPOConfig(**raw.pop("ppo", {}))
    return ExperimentConfig(**raw, mode=mode, seed=seed, ppo=replace(ppo, seed=seed))


def doctor(settings: dict) -> dict:
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "checks": {},
    }
    for package in ("torch", "numpy", "laya-coreml"):
        try:
            report["checks"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["checks"][package] = "MISSING"
    paths = {
        "rom": settings.get("rom_path", ""),
        "corpus": settings.get("corpus_path", ""),
        "mgba_profile": settings.get("mgba", {}).get("profile_path", ""),
        "laya_model": settings.get("laya", {}).get("model_dir", ""),
        "mgba_app": "/Applications/mGBA.app/Contents/MacOS/mGBA",
    }
    for name, value in paths.items():
        report["checks"][name] = "present" if value and Path(value).exists() else "MISSING"
    report["ready_for_real_battles"] = all(value != "MISSING" for value in report["checks"].values())
    report["note"] = "Presence checks only; run validate-corpus and connect the Lua bridge before training."
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laya + PPO in mGBA Emerald Rogue trainer singles")
    parser.add_argument(
        "--config", default="configs/experiment.json", help="JSON settings; paths relative to cwd"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check local prerequisites without starting emulation")
    commands.add_parser("validate-corpus", help="Verify ROM, save-state hashes, and scenario split isolation")
    verify = commands.add_parser(
        "verify-game", help="Replay one training battle twice to check deterministic control"
    )
    verify.add_argument("--steps", type=int, default=100)
    run = commands.add_parser("train", help="Train one fixed-budget arm and learner seed")
    run.add_argument(
        "--mode",
        choices=["frozen", "scratch", "residual", "gated", "uniform_residual", "shuffled_residual"],
        required=True,
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--prior", choices=["laya", "uniform", "prism", "jev"], default="laya",
        help="Frozen decision prior for frozen/residual/gated arms",
    )
    run.add_argument("--openrouter-model", default="~typesafe/jev-latest")
    run = commands.add_parser("evaluate", help="Evaluate a final checkpoint on untouched test battles")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--openrouter-model", default="~typesafe/jev-latest")
    baseline = commands.add_parser("baseline", help="Play actual battle states with frozen Laya; no training")
    baseline.add_argument("--prior", choices=["laya", "uniform", "prism", "jev"], default="laya")
    baseline.add_argument(
        "--openrouter-model",
        default="~typesafe/jev-latest",
        help="OpenRouter TypeSafe model slug for --prior jev",
    )
    baseline.add_argument("--split", choices=["train", "validation", "test"], default="train")
    baseline.add_argument("--max-battles", type=int, help="Use the first N battles of this split")
    baseline.add_argument("--repeats", type=int, default=1)
    baseline.add_argument("--max-decisions", type=int, help="Decision cap per battle")
    baseline.add_argument("--seed", type=int, default=0, help="Paired action-sampling seed")
    baseline.add_argument("--output", type=Path, required=True)
    visual = commands.add_parser(
        "visual", help="Replay one battle with screenshots and a GIF for Laya or the uniform control"
    )
    visual.add_argument("--prior", choices=["laya", "uniform", "prism", "jev"], default="laya")
    visual.add_argument(
        "--openrouter-model",
        default="~typesafe/jev-latest",
        help="OpenRouter TypeSafe model slug for --prior jev",
    )
    visual.add_argument("--split", choices=["train", "validation", "test"], default="train")
    visual.add_argument("--battle-id", help="Battle ID from the selected split; defaults to its first state")
    visual.add_argument("--max-decisions", type=int, help="Stop after this many policy decisions")
    visual.add_argument("--seed", type=int, default=0, help="Action-sampling seed")
    visual.add_argument("--frame-duration-ms", type=int, default=900)
    visual.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser(
        "compare-baselines", help="Compare two completed matched baseline directories"
    )
    compare.add_argument("baseline_a", type=Path)
    compare.add_argument("baseline_b", type=Path)
    compare.add_argument("--output", type=Path)
    report = commands.add_parser("summarize", help="Paired comparison and sample-efficiency learning curves")
    report.add_argument("runs", nargs="+", type=Path)
    report.add_argument("--split", choices=["validation", "test"], default="validation")
    report.add_argument("--output", type=Path)
    return parser


def _baseline(settings: dict, corpus: Corpus, args: argparse.Namespace) -> dict:
    if args.max_battles is not None and args.max_battles < 1:
        raise ValueError("--max-battles must be positive")
    max_decisions = (
        args.max_decisions
        if args.max_decisions is not None
        else settings.get("experiment", {}).get("max_decisions", 500)
    )
    if args.repeats < 1 or max_decisions < 1:
        raise ValueError("--repeats and --max-decisions must be positive")
    battles = corpus.split(args.split)
    if args.max_battles is not None:
        battles = battles[: args.max_battles]
    if not battles:
        raise ValueError(f"No {args.split} battles in corpus")
    prior, laya = _prior(settings, args)
    env = _environment(settings, corpus)
    # The directory is created only after local prerequisites are validated.
    args.output.mkdir(parents=True, exist_ok=False)
    episodes_path = args.output / "episodes.jsonl"
    trace_path = args.output / "first-battle-trace.jsonl"
    episodes_path.touch(exist_ok=False)
    trace_path.touch(exist_ok=False)
    planned = len(battles) * args.repeats
    metadata = {
        "mode": f"{args.prior}_baseline",
        "split": args.split,
        "battle_ids": [battle["id"] for battle in battles],
        "repeats": args.repeats,
        "max_decisions": max_decisions,
        "agent_seed": args.seed,
        "rom_sha256": corpus.rom_sha256,
        "source_revision": corpus.source_revision,
        "corpus_sha256": sha256_file(corpus.path),
        "mgba_profile_sha256": sha256_file(Path(settings["mgba"]["profile_path"])),
        "laya": laya.provenance if laya is not None else None,
        "prior_provenance": getattr(prior, "provenance", None),
        "started_unix_seconds": time.time(),
        "dataset": {
            "training_battles_used": 0,
            "evaluation_split": args.split,
            "evaluation_battles": len(battles),
            "corpus_train_battles": len(corpus.split("train")),
            "corpus_validation_battles": len(corpus.split("validation")),
            "corpus_test_battles": len(corpus.split("test")),
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    failure = None
    try:
        metadata["mgba_bridge"] = env.describe()
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        evaluate(
            env,
            battles,
            prior,
            None,
            seed=args.seed,
            repeats=args.repeats,
            max_decisions=max_decisions,
            log_path=episodes_path,
            trace_path=trace_path,
        )
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - persist partial episodes
        failure = exc
    finally:
        env.close()
        rows = read_jsonl(episodes_path)
        model_metrics = {
            "calls": getattr(prior, "calls", 0),
            "cache_hits": getattr(prior, "cache_hits", 0),
            "inference_seconds": getattr(prior, "inference_seconds", 0.0),
        }
        summary = baseline_summary(
            rows,
            planned_episodes=planned,
            status="failed" if failure else "complete",
            model_metrics=model_metrics,
        )
        summary["split"] = args.split
        summary["prior"] = args.prior
        summary["training_battles_used"] = 0
        summary["evaluation_battles"] = len(battles)
        if failure:
            summary["failure"] = f"{type(failure).__name__}: {failure}"
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if failure:
        raise RuntimeError(
            f"Baseline stopped after {len(rows)}/{planned} episodes; see {args.output}/summary.json: {failure}"
        ) from failure
    return summary


def _make_gif(frames: list[Path], target: Path, duration_ms: int) -> Path:
    """Write a compact replay that preserves the original mGBA pixels."""
    from PIL import Image

    if not frames:
        raise ValueError("A visual replay needs at least one captured frame")
    images = []
    try:
        for frame in frames:
            with Image.open(frame) as image:
                images.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
        images[0].save(
            target,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
    return target


def _action_label(obs: dict, action: int) -> str:
    if action < 4:
        move = obs.get("moves", [])[action]
        return (
            f"{action + 1}. {move.get('name', 'move')}"
            f"  type {move.get('type_name', move.get('type', '?'))}"
            f"  P{move.get('power', 0)} PP{move.get('pp', 0)}/{move.get('max_pp', 0)}"
        )
    mon = obs.get("party", [])[action - 4]
    return f"{action + 1}. Switch to {mon.get('species_name', mon.get('species', '?'))}"


def _annotate_frame(
    source: Path, target: Path, obs: dict, probabilities: np.ndarray | None, selected: int | None
) -> Path:
    """Make the action set visible without altering the original emulator PNG."""
    from PIL import Image, ImageDraw

    target = target.resolve()
    with Image.open(source) as raw:
        game = raw.convert("RGB")
    game = game.resize((480, 320), Image.Resampling.NEAREST)
    lines = [f"Decision {obs['decision_id']}  turn {obs['turn']}  {obs['phase']}"]
    if probabilities is None:
        lines.append(f"Outcome: {obs['outcome']}")
    else:
        for action, legal in enumerate(obs["legal_actions"]):
            if legal:
                marker = ">" if action == selected else " "
                lines.append(f"{marker} {_action_label(obs, action)}  {100 * probabilities[action]:.1f}%")
    panel_height = 18 + 18 * len(lines)
    canvas = Image.new("RGB", (480, 320 + panel_height), "#101820")
    canvas.paste(game, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        colour = "#ffe082" if line.startswith(">") else "#f5f7fa"
        draw.text((8, 326 + 18 * index), line, fill=colour)
    canvas.save(target)
    return target


def _visual(settings: dict, corpus: Corpus, args: argparse.Namespace) -> dict:
    if args.max_decisions is not None and args.max_decisions < 1:
        raise ValueError("--max-decisions must be positive")
    if args.frame_duration_ms < 1:
        raise ValueError("--frame-duration-ms must be positive")
    candidates = corpus.split(args.split)
    if args.battle_id:
        candidates = [battle for battle in candidates if battle["id"] == args.battle_id]
    if not candidates:
        raise ValueError("No matching battle state in the selected split")
    battle = candidates[0]
    max_decisions = args.max_decisions or settings.get("experiment", {}).get("max_decisions", 500)
    prior, laya = _prior(settings, args)
    env = _environment(settings, corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    frames_dir = args.output / "frames"
    frames_dir.mkdir()
    decisions_path = args.output / "decisions.jsonl"
    decisions_path.touch(exist_ok=False)
    metadata = {
        "mode": f"visual_{args.prior}",
        "battle_id": battle["id"],
        "split": args.split,
        "agent_seed": args.seed,
        "max_decisions": max_decisions,
        "frame_duration_ms": args.frame_duration_ms,
        "rom_sha256": corpus.rom_sha256,
        "source_revision": corpus.source_revision,
        "corpus_sha256": sha256_file(corpus.path),
        "mgba_profile_sha256": sha256_file(Path(settings["mgba"]["profile_path"])),
        "laya": laya.provenance if laya is not None else None,
        "prior_provenance": getattr(prior, "provenance", None),
        "started_unix_seconds": time.time(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    frames: list[Path] = []
    failure = None
    decision_count = 0
    obs: dict | None = None
    try:
        metadata["mgba_bridge"] = env.describe()
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        obs = env.reset(battle)
        encoder = FeatureEncoder()
        key = hashlib.sha256(f"{args.seed}:{battle['id']}:visual".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        while obs["phase"] != "terminal" and decision_count < max_decisions:
            capture = env.menu_screenshot if obs["phase"] == "action" else env.screenshot
            raw_frame = capture(frames_dir / f"raw-{decision_count:04d}.png")
            features = encoder.encode(obs)
            prior_probs = checked_prior(prior, obs)
            probs, _value, _kl = action_distribution(None, features, prior_probs, obs["legal_actions"])
            action = sample_action(probs, rng)
            moves = obs.get("moves", [])
            action_name = (
                str(moves[action].get("name", f"move_{action + 1}"))
                if action < 4 and action < len(moves)
                else f"action_{action}"
            )
            frame = _annotate_frame(
                raw_frame,
                frames_dir / f"{decision_count:04d}-decision.png",
                obs,
                probs,
                action,
            )
            frames.append(frame)
            row = {
                "frame": frame.name,
                "battle_id": obs["battle_id"],
                "decision_id": obs["decision_id"],
                "turn": obs["turn"],
                "phase": obs["phase"],
                "legal_actions": obs["legal_actions"],
                "action": action,
                "action_name": action_name,
                "probabilities": [float(value) for value in probs],
                "observation": obs,
            }
            with decisions_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            encoder.record_action(obs, action)
            obs = env.step(action)
            decision_count += 1
        terminal_raw = env.screenshot(frames_dir / "raw-terminal.png")
        terminal_frame = _annotate_frame(terminal_raw, frames_dir / "terminal.png", obs, None, None)
        frames.append(terminal_frame)
        replay = _make_gif(frames, args.output / "battle.gif", args.frame_duration_ms)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - retain captured evidence
        failure = exc
        replay = None
    finally:
        env.close()
    result = {
        "status": "failed" if failure else "complete",
        "prior": args.prior,
        "battle_id": battle["id"],
        "decisions": decision_count,
        "outcome": None if obs is None else obs["outcome"],
        "truncated": bool(obs is not None and obs["phase"] != "terminal"),
        "frames": [str(frame.relative_to(args.output.resolve())) for frame in frames],
        "replay": None if replay is None else replay.name,
        "model_metrics": {
            "calls": getattr(prior, "calls", 0),
            "cache_hits": getattr(prior, "cache_hits", 0),
            "inference_seconds": getattr(prior, "inference_seconds", 0.0),
        },
    }
    if failure:
        result["failure"] = f"{type(failure).__name__}: {failure}"
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if failure:
        raise RuntimeError(f"Visual replay stopped; see {args.output}/result.json: {failure}") from failure
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            result = summarize(args.runs, split=args.split)
            text = json.dumps(result, indent=2)
            if args.output:
                args.output.write_text(text + "\n")
            print(text)
            return 0
        if args.command == "compare-baselines":
            result = compare_baselines(args.baseline_a, args.baseline_b)
            text = json.dumps(result, indent=2)
            if args.output:
                args.output.write_text(text + "\n")
            print(text)
            return 0
        settings = _settings(args.config)
        if args.command == "doctor":
            result = doctor(settings)
            print(json.dumps(result, indent=2))
            return 0 if result["ready_for_real_battles"] else 1
        corpus = Corpus.load(settings["corpus_path"])
        corpus.verify_rom(settings["rom_path"])
        if args.command == "validate-corpus":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "rom_sha256": corpus.rom_sha256,
                        "splits": {name: len(corpus.split(name)) for name in ("train", "validation", "test")},
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "verify-game":
            battles = corpus.split("train")
            if not battles or args.steps < 1:
                raise ValueError("A training battle and positive --steps are required")
            env = _environment(settings, corpus)
            transcripts = []
            try:
                for delayed in (False, True):
                    obs = env.reset(battles[0])
                    transcript = [obs]
                    for _ in range(args.steps):
                        if obs["phase"] == "terminal":
                            break
                        if delayed:
                            time.sleep(0.05)
                        action = obs["legal_actions"].index(True)
                        obs = env.step(action)
                        transcript.append(obs)
                    transcripts.append(transcript)
            finally:
                env.close()
            if transcripts[0] != transcripts[1]:
                raise ValueError(
                    "Replay differs with inference delay; do not train until RNG/control is corrected"
                )
            print(
                json.dumps(
                    {
                        "deterministic_replay": True,
                        "decisions": len(transcripts[0]) - 1,
                        "reached_terminal": transcripts[0][-1]["phase"] == "terminal",
                    }
                )
            )
            return 0
        if args.command == "baseline":
            summary = _baseline(settings, corpus, args)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "visual":
            result = _visual(settings, corpus, args)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "train":
            config = _config(settings, args.mode, args.seed)
            prior, _ = _prior(settings, args)
            prior_name = args.prior
            if config.mode in {"scratch", "uniform_residual"}:
                prior, prior_name = None, "uniform"
            env = _environment(settings, corpus)
            try:
                checkpoint = train(
                    env,
                    corpus,
                    config,
                    args.output,
                    prior,
                    provenance={
                        "prior_name": prior_name,
                        "prior": None if prior is None else getattr(prior, "provenance", None),
                        "mgba_profile_sha256": sha256_file(Path(settings["mgba"]["profile_path"])),
                    },
                )
            finally:
                env.close()
            print(f"Final fixed-budget checkpoint: {checkpoint}")
            return 0
        policy, config, saved = load_checkpoint(args.checkpoint)
        if saved["steps"] != config.total_steps:
            raise ValueError("Test evaluation requires the final preregistered-budget checkpoint")
        if saved["corpus_sha256"] != sha256_file(corpus.path) or saved["rom_sha256"] != corpus.rom_sha256:
            raise ValueError("Evaluation corpus/ROM differs from training provenance")
        if saved["provenance"].get("mgba_profile_sha256") != sha256_file(
            Path(settings["mgba"]["profile_path"])
        ):
            raise ValueError("Evaluation mGBA profile differs from training")
        log = args.checkpoint.parent / "test-evaluation.jsonl"
        if log.exists():
            raise ValueError(
                "Test results already exist; inspect them instead of repeatedly tuning on test data"
            )
        prior_name = saved["provenance"].get("prior_name")
        if prior_name is None:
            raise ValueError("Checkpoint lacks frozen-prior provenance")
        prior_args = argparse.Namespace(prior=prior_name, openrouter_model=args.openrouter_model)
        prior, _ = _prior(settings, prior_args)
        if saved["provenance"].get("prior") != getattr(prior, "provenance", None):
            raise ValueError("Evaluation frozen-prior configuration/artifact differs from training")
        prior = select_prior(config, prior)
        env = _environment(settings, corpus)
        try:
            rows = evaluate(
                env,
                corpus.split("test"),
                prior,
                policy,
                seed=config.seed,
                repeats=config.eval_repeats,
                max_decisions=config.max_decisions,
                train_steps=saved["steps"],
            )
            with log.open("x") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        finally:
            env.close()
        print(f"Test evaluation written to {log}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"rogue-rl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
