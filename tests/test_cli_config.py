"""CLI commands and settings."""
import json

import pytest

from crash_whatif.cli import EXAMPLE_CRASH, main
from crash_whatif.config import ConfigError, Settings


def test_settings_from_env(monkeypatch):
    assert Settings.from_env().resample == "class_weight"
    monkeypatch.setenv("CRASH_WHATIF_ACTIONABLE", "esc, airbags")
    monkeypatch.setenv("CRASH_WHATIF_N_QUERIES", "7")
    s = Settings.from_env()
    assert s.actionable == ("esc", "airbags") and s.n_queries == 7


@pytest.mark.parametrize(
    "name,value",
    [("CRASH_WHATIF_RESAMPLE", "smote"), ("CRASH_WHATIF_TARGET", "five"), ("CRASH_WHATIF_SEED", "x"), ("CRASH_WHATIF_N_QUERIES", "0")],
)
def test_bad_settings(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_generate_validate_train_counterfactual(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "crashes.csv"
    assert main(["generate", "--out", str(csv), "--rows", "1500"]) == 0
    assert main(["validate", "--csv", str(csv)]) == 0
    assert "crashes: 1500 rows in" in capsys.readouterr().out
    assert main(["train", "--csv", str(csv), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "models" / "random_forest.joblib").exists()
    capsys.readouterr()
    model = str(tmp_path / "out" / "models" / "logreg.joblib")
    assert main(["counterfactual", "--model", model, "--input", json.dumps(EXAMPLE_CRASH)]) == 0
    assert "P(severe)" in capsys.readouterr().out


def test_audit_command_on_make_rule_data(capsys):
    assert main(["audit", "--rows", "1500", "--mode", "make_rule"]) == 0
    out = capsys.readouterr().out
    start = out.index("\n{") + 1
    assert json.loads(out[start:])["make_dominates"] is True


def test_report_command_writes_files(tmp_path, capsys):
    assert main(["report", "--rows", "1500", "--queries", "3", "--repeats", "2", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert {"metrics", "audit", "importance", "counterfactuals", "three_class"} <= set(data)
    assert "SYNTHETIC" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_missing_csv_gives_error_code(capsys):
    assert main(["validate", "--csv", "does_not_exist.csv"]) == 1
    assert "error:" in capsys.readouterr().err
