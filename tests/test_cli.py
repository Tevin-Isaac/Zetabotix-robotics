import json

import pandas as pd
import pytest

from robo_lint.cli import main


@pytest.fixture
def dataset_dir(tmp_path):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    (root / "meta" / "info.json").write_text(json.dumps({"features": {"action": {"names": ["shoulder", "elbow"]}}}))

    pd.DataFrame(
        {
            "action": [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
            "episode_index": [0, 0, 1, 1],
        }
    ).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")

    return root


def test_cli_prints_markdown_and_flags_dead_joint(dataset_dir, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["robo-lint", str(dataset_dir)])
    exit_code = main()

    out = capsys.readouterr().out
    assert exit_code == 1  # elbow never moves -> dead joint -> critical issue
    assert "DEAD JOINT" in out
    assert "elbow" in out


def test_cli_json_output(dataset_dir, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["robo-lint", str(dataset_dir), "--json"])
    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["num_frames"] == 4
    assert payload["num_episodes"] == 2


def test_cli_reports_error_for_bad_source(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr("sys.argv", ["robo-lint", str(missing)])

    exit_code = main()

    assert exit_code == 2
    assert "failed to analyze" in capsys.readouterr().err
