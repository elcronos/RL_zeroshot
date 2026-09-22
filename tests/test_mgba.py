"""Real loopback transport, fake emulator responses; no ROM/performance claims."""

import json
import socket
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from rogue_rl.manifest import sha256_file
from rogue_rl.mgba import BridgeError, BridgeTimeout, MGBAEnv, ProtocolError


def observation(decision=1, terminal=False):
    return {
        "battle_id": "battle",
        "decision_id": decision,
        "turn": decision,
        "phase": "terminal" if terminal else "action",
        "outcome": "win" if terminal else None,
        "player": {"species": 25, "hp": 50, "max_hp": 100},
        "opponent": {"hp_fraction": 0.5, "hp": 200, "max_hp": 400, "ability": "private"},
        "party": [],
        "moves": [{"id": 85}],
        "legal_actions": [not terminal] + [False] * 9,
    }


@contextmanager
def bridge(tmp_path, handler, timeout=1.0):
    profile = {
        "schema_version": 1,
        "abi": 1,
        "rom_sha256": "a" * 64,
        "rom_crc32": "12345678",
        "source_revision": "fixture",
        "species": {"25": "Pikachu"},
        "moves": {"85": "Thunderbolt"},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    state = tmp_path / "state with space.bin"
    state.write_bytes(b"not a real save state")
    spec = {
        "id": "battle",
        "state_path": str(state),
        "state_sha256": sha256_file(state),
        "rom_sha256": profile["rom_sha256"],
    }
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    commands, failures = [], []
    stopped = threading.Event()

    def serve():
        try:
            with server.accept()[0] as conn:  # noqa: SIM117 - connection needed to construct file context
                with conn.makefile("rb") as incoming:
                    for line in incoming:
                        fields = line.decode().strip().split("\t")
                        commands.append(fields)
                        if fields[0] == "HELLO":
                            response = {
                                "seq": int(fields[1]),
                                "result": {"protocol": 1, **profile},
                                "error": None,
                            }
                        else:
                            response = handler(fields)
                        if response is None:
                            stopped.wait(1)
                            return
                        payload = json.dumps(response).encode() + b"\n"
                        # Split JSON deliberately to exercise stream framing.
                        conn.sendall(payload[:7])
                        conn.sendall(payload[7:])
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - surface background fixture failures in the test thread
            failures.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    env = MGBAEnv(port=server.getsockname()[1], timeout=timeout, profile_path=path)
    try:
        yield env, spec, commands
    finally:
        env.close()
        stopped.set()
        server.close()
        thread.join(2)
        assert not failures


def test_transport_reset_action_and_public_projection(tmp_path):
    def handler(fields):
        return {
            "seq": int(fields[1]),
            "result": observation(1 if fields[0] == "RESET" else 2, terminal=fields[0] == "STEP"),
        }

    with bridge(tmp_path, handler) as (env, spec, commands):
        obs = env.reset(spec)
        assert bytes.fromhex(commands[1][3]).decode() == spec["state_path"]
        assert obs["player"]["species_name"] == "Pikachu"
        assert obs["moves"][0]["name"] == "Thunderbolt"
        assert "ability" not in obs["opponent"] and "hp" not in obs["opponent"]
        with pytest.raises(ValueError, match="not legal"):
            env.step(1)
        assert len(commands) == 2
        assert env.step(0)["outcome"] == "win"
        assert commands[-1][2:4] == ["1", "0"]
        with pytest.raises(ValueError, match="live battle"):
            env.step(0)
    assert commands[-1][0] == "CLOSE"


def test_screenshot_requires_a_live_battle_and_checks_the_created_png(tmp_path):
    def handler(fields):
        if fields[0] == "RESET":
            return {"seq": int(fields[1]), "result": observation()}
        if fields[0] == "SCREENSHOT":
            target = Path(bytes.fromhex(fields[2]).decode())
            target.write_bytes(b"png fixture")
            return {"seq": int(fields[1]), "result": {"path": str(target)}}
        if fields[0] == "CLOSE":
            return None
        raise AssertionError(fields)

    with bridge(tmp_path, handler) as (env, spec, commands):
        with pytest.raises(ValueError, match="Reset"):
            env.screenshot(tmp_path / "before.png")
        env.reset(spec)
        target = env.screenshot(tmp_path / "frame.png")
        assert target.read_bytes() == b"png fixture"
        assert commands[-1][0] == "SCREENSHOT"
        assert bytes.fromhex(commands[-1][2]).decode() == str(target)


def test_menu_screenshot_restores_the_action_boundary_and_removes_scratch_state(tmp_path):
    def handler(fields):
        if fields[0] == "RESET":
            return {"seq": int(fields[1]), "result": observation()}
        if fields[0] == "MENU_SCREENSHOT":
            target, scratch = (Path(bytes.fromhex(value).decode()) for value in fields[2:4])
            target.write_bytes(b"real menu fixture")
            scratch.write_bytes(b"temporary state")
            return {"seq": int(fields[1]), "result": {"path": str(target)}}
        if fields[0] == "CLOSE":
            return None
        raise AssertionError(fields)

    with bridge(tmp_path, handler) as (env, spec, commands):
        env.reset(spec)
        target = env.menu_screenshot(tmp_path / "menu.png")
        assert target.read_bytes() == b"real menu fixture"
        scratch = target.with_suffix(".menu-capture.ss0")
        assert not scratch.exists()
        assert commands[-1][0] == "MENU_SCREENSHOT"


@pytest.mark.parametrize(
    "kind,error", [("sequence", ProtocolError), ("decision", ProtocolError), ("game", BridgeError)]
)
def test_stale_or_game_errors_close_connection(tmp_path, kind, error):
    def handler(fields):
        if fields[0] == "RESET":
            return {"seq": int(fields[1]), "result": observation()}
        return {
            "seq": 0 if kind == "sequence" else int(fields[1]),
            "result": observation(),
            "error": "frame_limit_exceeded" if kind == "game" else None,
        }

    with bridge(tmp_path, handler) as (env, spec, _):
        env.reset(spec)
        with pytest.raises(error):
            env.step(0)
        assert env._socket is None


def test_response_deadline(tmp_path):
    with bridge(tmp_path, lambda _: None, timeout=0.05) as (env, spec, _):
        with pytest.raises(BridgeTimeout):
            env.reset(spec)
        assert env._socket is None


def test_wrong_rom_is_rejected_before_connection(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "abi": 1, "rom_sha256": "a" * 64}))
    state = tmp_path / "state"
    state.write_bytes(b"fixture")
    env = MGBAEnv(profile_path=profile)
    with pytest.raises(ValueError, match="ROM hash"):
        env.reset(
            {"id": "x", "state_path": str(state), "state_sha256": sha256_file(state), "rom_sha256": "b" * 64}
        )


def test_loopback_only(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        MGBAEnv(host="0.0.0.0")
