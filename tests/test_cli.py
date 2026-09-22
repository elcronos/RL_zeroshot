import json

from rogue_rl.cli import main


def test_doctor_reports_missing_game_assets_without_claiming_ready(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"rom_path": str(tmp_path / "absent.gba")}))
    assert main(["--config", str(path), "doctor"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready_for_real_battles"] is False


def test_missing_corpus_actionable_error(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"corpus_path": str(tmp_path / "missing.json")}))
    assert main(["--config", str(path), "validate-corpus"]) == 2
    assert "missing.json" in capsys.readouterr().err
