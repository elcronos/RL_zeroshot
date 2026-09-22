"""Hash-locked battle corpus with scenario-level split isolation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Corpus:
    path: Path
    rom_sha256: str
    source_revision: str
    battles: tuple[dict, ...]

    @classmethod
    def load(cls, path: str | Path, *, verify_files: bool = True) -> Corpus:
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported corpus schema_version")
        rom_hash = raw.get("rom_sha256", "")
        if len(rom_hash) != 64 or any(c not in "0123456789abcdef" for c in rom_hash):
            raise ValueError("Corpus requires a lowercase ROM SHA256")
        if not raw.get("source_revision"):
            raise ValueError("Corpus requires the instrumented Rogue source revision")
        ids: set[str] = set()
        groups: dict[str, str] = {}
        hashes: dict[str, str] = {}
        battles = []
        for item in raw.get("battles", []):
            item = dict(item)
            if not isinstance(item.get("id"), str) or not item["id"] or item["id"] in ids:
                raise ValueError("Battle IDs must be nonempty and unique")
            ids.add(item["id"])
            split = item.get("split")
            if split not in {"train", "validation", "test"}:
                raise ValueError("Battle split must be train, validation, or test")
            group = item.get("scenario_group")
            if not isinstance(group, str) or not group:
                raise ValueError("scenario_group is required to keep related RNG variants together")
            if group in groups and groups[group] != split:
                raise ValueError(f"Scenario group leaks across splits: {group}")
            groups[group] = split
            state_hash = item.get("state_sha256", "")
            if len(state_hash) != 64 or any(c not in "0123456789abcdef" for c in state_hash):
                raise ValueError("Each battle requires a lowercase state SHA256")
            if state_hash in hashes:
                raise ValueError(
                    "Duplicate save-state bytes in corpus; RNG labels alone do not create new battles"
                )
            hashes[state_hash] = split
            seed = item.get("seed")
            if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
                raise ValueError("Battle seed must be a uint32 provenance value or null when unknown")
            state_path = (path.parent / item["state_path"]).resolve()
            if verify_files and sha256_file(state_path) != state_hash:
                raise ValueError(f"Save-state hash mismatch: {state_path}")
            item["state_path"] = str(state_path)
            item["rom_sha256"] = rom_hash
            battles.append(item)
        if not battles:
            raise ValueError("Corpus contains no battles")
        return cls(path, rom_hash, raw["source_revision"], tuple(battles))

    def split(self, name: str) -> list[dict]:
        return [dict(battle) for battle in self.battles if battle["split"] == name]

    def verify_rom(self, path: str | Path) -> None:
        if sha256_file(Path(path)) != self.rom_sha256:
            raise ValueError("ROM hash differs from the corpus; rebuild states for this exact ROM")
