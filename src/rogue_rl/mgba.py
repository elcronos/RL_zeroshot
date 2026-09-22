"""Bounded local TCP client for the mGBA Lua research mailbox bridge."""

from __future__ import annotations

import json
import math
import socket
import time
from pathlib import Path
from typing import Self

from .manifest import sha256_file
from .observations import public_observation


class BridgeError(RuntimeError):
    pass


class BridgeTimeout(BridgeError):
    pass


class ProtocolError(BridgeError):
    pass


class MGBAEnv:
    """One emulator, one request in flight; any protocol failure closes connection.

    Runtime never guesses RAM addresses. The profile comes from the instrumented
    ROM's ELF symbols, and HELLO verifies its ROM CRC and pinned identity.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout: float = 30,
        max_frames: int = 18000,
        profile_path: str | Path | None = None,
    ):
        if host != "127.0.0.1":
            raise ValueError("mGBA bridge must bind to local IPv4 loopback")
        if (
            not 0 < port < 65536
            or not math.isfinite(timeout)
            or timeout <= 0
            or not 1 <= max_frames <= 10_000_000
        ):
            raise ValueError("Invalid bridge port, timeout, or frame budget")
        if profile_path is None:
            raise ValueError("An instrumented-ROM profile_path is required")
        self.profile = json.loads(Path(profile_path).read_text())
        if self.profile.get("schema_version") != 1 or self.profile.get("abi") != 1:
            raise ValueError("Unsupported mGBA profile schema/ABI")
        self.host, self.port, self.timeout, self.max_frames = host, port, timeout, max_frames
        self._socket: socket.socket | None = None
        self._rx = b""
        self._seq = 0
        self._obs: dict | None = None
        self._hello: dict | None = None

    def _request(self, op: str, *fields: str | int) -> dict:
        assert self._socket is not None
        self._seq += 1
        line = "\t".join(map(str, (op, self._seq, *fields))) + "\n"
        deadline = time.monotonic() + self.timeout
        try:
            self._socket.settimeout(self.timeout)
            self._socket.sendall(line.encode("ascii"))
            while b"\n" not in self._rx:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("response deadline exceeded")
                self._socket.settimeout(remaining)
                block = self._socket.recv(65536)
                if not block:
                    raise ProtocolError("mGBA disconnected before replying")
                self._rx += block
                if len(self._rx) > 1_048_576:
                    raise ProtocolError("mGBA response exceeds 1 MiB")
            raw, self._rx = self._rx.split(b"\n", 1)
            try:
                response = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ProtocolError("Malformed mGBA JSON response") from exc
            if not isinstance(response, dict) or response.get("seq") != self._seq:
                raise ProtocolError("Stale or mismatched mGBA response sequence")
            if response.get("error"):
                raise BridgeError(f"mGBA: {response['error']}")
            if not isinstance(response.get("result"), dict):
                raise ProtocolError("mGBA response has no result object")
            return response["result"]
        except TimeoutError as exc:
            self.close()
            raise BridgeTimeout(
                "mGBA timed out; keep the emulator running with the generated Lua profile loaded"
            ) from exc
        except (OSError, BridgeError):
            self.close()
            raise

    def describe(self) -> dict:
        if self._socket is None:
            try:
                self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
                self._hello = self._request("HELLO")
                for key in ("abi", "rom_sha256", "rom_crc32", "source_revision"):
                    if self._hello.get(key) != self.profile.get(key):
                        raise ProtocolError(f"mGBA handshake differs from profile: {key}")
                if self._hello.get("protocol") != 1:
                    raise ProtocolError("Unsupported mGBA protocol")
            except (OSError, BridgeError):
                self.close()
                raise
        return dict(self._hello or {})

    def _decode(self, raw: dict) -> dict:
        for mon in [raw.get("player", {}), raw.get("opponent", {}), *raw.get("party", [])]:
            species = str(mon.get("species", 0))
            if species in self.profile.get("species", {}):
                mon["species_name"] = self.profile["species"][species]
        for move in raw.get("moves", []):
            move_id = str(move.get("id", 0))
            if move_id in self.profile.get("moves", {}):
                move["name"] = self.profile["moves"][move_id]
        try:
            return public_observation(raw)
        except (ValueError, KeyError, TypeError) as exc:
            self.close()
            raise ProtocolError(f"Invalid mGBA observation: {exc}") from exc

    def reset(self, battle: dict) -> dict:
        path = Path(battle["state_path"]).resolve(strict=True)
        if not battle.get("state_sha256") or sha256_file(path) != battle["state_sha256"]:
            raise ValueError("Battle state SHA256 mismatch or missing")
        if battle.get("rom_sha256") != self.profile["rom_sha256"]:
            raise ValueError("Battle ROM hash differs from mGBA profile")
        self.describe()
        result = self._request(
            "RESET", str(battle["id"]).encode().hex(), str(path).encode().hex(), self.max_frames
        )
        obs = self._decode(result)
        if obs["battle_id"] != battle["id"]:
            self.close()
            raise ProtocolError("Reset returned another battle's observation")
        self._obs = obs
        return obs

    def step(self, action: int) -> dict:
        if self._obs is None or self._obs["phase"] == "terminal":
            raise ValueError("Reset into a live battle before stepping")
        if type(action) is not int or not 0 <= action < 10 or not self._obs["legal_actions"][action]:
            raise ValueError("Action is not legal in this decision")
        previous = self._obs
        result = self._request("STEP", previous["decision_id"], action, self.max_frames)
        obs = self._decode(result)
        if obs["battle_id"] != previous["battle_id"] or obs["decision_id"] <= previous["decision_id"]:
            self.close()
            raise ProtocolError("STEP returned a stale decision or another battle")
        self._obs = obs
        return obs

    def screenshot(self, path: str | Path) -> Path:
        """Capture the current emulator frame for an active research battle."""
        if self._obs is None:
            raise ValueError("Reset into a battle before capturing a screenshot")
        target = Path(path).resolve()
        if target.suffix.lower() != ".png":
            raise ValueError("mGBA screenshots must use a .png path")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._request("SCREENSHOT", str(target).encode().hex())
        if not target.is_file() or target.stat().st_size == 0:
            raise ProtocolError("mGBA did not create the requested screenshot")
        return target

    def menu_screenshot(self, path: str | Path) -> Path:
        """Capture mGBA's real Fight move menu, then restore the same decision."""
        if self._obs is None or self._obs["phase"] != "action":
            raise ValueError("A normal action decision is required for a move-menu screenshot")
        target = Path(path).resolve()
        if target.suffix.lower() != ".png":
            raise ValueError("mGBA screenshots must use a .png path")
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.with_suffix(".menu-capture.ss0")
        try:
            self._request("MENU_SCREENSHOT", str(target).encode().hex(), str(scratch).encode().hex())
            if not target.is_file() or target.stat().st_size == 0:
                raise ProtocolError("mGBA did not create the requested move-menu screenshot")
            return target
        finally:
            scratch.unlink(missing_ok=True)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.settimeout(0.1)
                self._socket.sendall(f"CLOSE\t{self._seq + 1}\n".encode("ascii"))
            except OSError:
                pass
            self._socket.close()
        self._socket, self._obs, self._hello = None, None, None
        self._rx, self._seq = b"", 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
