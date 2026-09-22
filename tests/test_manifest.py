"""Corpus integrity fixtures contain arbitrary bytes, never game assets."""

import json

import pytest

from rogue_rl.manifest import Corpus, sha256_file


def write_manifest(tmp_path):
    rom = tmp_path / "placeholder.bin"
    rom.write_bytes(b"synthetic integrity fixture; not a ROM")
    battles = []
    for index, split in enumerate(("train", "validation", "test")):
        state = tmp_path / f"state-{index}.bin"
        state.write_bytes(f"synthetic state bytes {index}".encode())
        battles.append(
            {
                "id": f"case-{index}",
                "split": split,
                "scenario_group": f"group-{index}",
                "seed": index,
                "state_path": state.name,
                "state_sha256": sha256_file(state),
            }
        )
    raw = {
        "schema_version": 1,
        "rom_sha256": sha256_file(rom),
        "source_revision": "fixture-revision",
        "battles": battles,
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(raw))
    return path, rom, raw


def test_corpus_hashes_paths_and_split_copies(tmp_path):
    path, rom, _ = write_manifest(tmp_path)
    corpus = Corpus.load(path)
    corpus.verify_rom(rom)
    train = corpus.split("train")
    assert len(train) == 1
    assert train[0]["state_path"] == str(tmp_path / "state-0.bin")
    train[0]["id"] = "changed-copy"
    assert corpus.split("train")[0]["id"] == "case-0"


def test_corpus_allows_an_explicitly_unknown_legacy_state_seed(tmp_path):
    path, _, raw = write_manifest(tmp_path)
    raw["battles"][0]["seed"] = None
    path.write_text(json.dumps(raw))
    assert Corpus.load(path).split("train")[0]["seed"] is None


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda raw: raw["battles"][1].update(scenario_group="group-0"), "leaks across splits"),
        (lambda raw: raw["battles"][1].update(id="case-0"), "unique"),
        (
            lambda raw: raw["battles"][1].update(state_sha256=raw["battles"][0]["state_sha256"]),
            "Duplicate save-state",
        ),
        (lambda raw: raw["battles"][0].update(seed=True), "uint32"),
        (lambda raw: raw.update(rom_sha256="A" * 64), "lowercase ROM"),
    ],
)
def test_corpus_rejects_leakage_and_bad_identity(tmp_path, change, match):
    path, _, raw = write_manifest(tmp_path)
    change(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=match):
        Corpus.load(path, verify_files=False)


def test_corpus_detects_state_and_rom_mutation(tmp_path):
    path, rom, _ = write_manifest(tmp_path)
    corpus = Corpus.load(path)
    rom.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="ROM hash differs"):
        corpus.verify_rom(rom)
    (tmp_path / "state-0.bin").write_bytes(b"modified state")
    with pytest.raises(ValueError, match="Save-state hash mismatch"):
        Corpus.load(path)
