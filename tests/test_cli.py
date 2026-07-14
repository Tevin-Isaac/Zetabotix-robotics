from robo_lint.cli import main


def test_main_returns_nonzero_when_unimplemented(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["robo-lint", "some/dataset"])
    exit_code = main()
    assert exit_code == 1
    assert "not implemented yet" in capsys.readouterr().err
