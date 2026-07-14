import json

import pandas as pd
import pytest

from robo_lint.loaders import load_frames


def _write_dataset(tmp_path, with_stats=False, with_success=False):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {"features": {"action": {"names": ["shoulder", "elbow"]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))

    data = {
        "action": [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
        "observation.state": [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        "episode_index": [0, 0, 1],
    }
    if with_success:
        data["success"] = [False, True, False]
    pd.DataFrame(data).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")

    if with_stats:
        stats = {"action": {"min": [-1.0, -1.0], "max": [1.0, 1.0]}}
        (root / "meta" / "stats.json").write_text(json.dumps(stats))

    return root


def test_load_frames_from_local_directory(tmp_path):
    root = _write_dataset(tmp_path)

    frames = load_frames(str(root))

    assert frames.actions.shape == (3, 2)
    assert frames.action_names == ["shoulder", "elbow"]
    assert frames.num_episodes == 2
    assert frames.action_limits is None
    assert frames.success is None


def test_load_frames_picks_up_stats_and_success(tmp_path):
    root = _write_dataset(tmp_path, with_stats=True, with_success=True)

    frames = load_frames(str(root))

    assert frames.action_limits is not None
    assert frames.success.tolist() == [False, True, False]


def test_load_frames_rejects_non_lerobot_directory(tmp_path):
    empty = tmp_path / "not-a-dataset"
    empty.mkdir()
    with pytest.raises(ValueError, match="does not look like a LeRobot dataset"):
        load_frames(str(empty))
