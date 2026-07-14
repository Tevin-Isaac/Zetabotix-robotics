"""Loads LeRobot-format datasets into the internal FrameData schema.

Both LeRobotDataset v2.1 (one parquet per episode) and v3.0 (many episodes per
parquet, relational metadata) store the same per-frame columns — action,
observation.state, episode_index — so reading every parquet file under
data/ and concatenating works for either layout without needing to special-case
the format version.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FrameData


def _resolve_local_root(source: str) -> Path:
    path = Path(source)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    # robo-lint only ever reads meta/ and data/ (parquet); skip videos/, which
    # dwarf the rest of the dataset in size and aren't used by any check.
    return Path(
        snapshot_download(
            repo_id=source,
            repo_type="dataset",
            allow_patterns=["meta/*", "data/**"],
        )
    )


def load_frames(source: str) -> FrameData:
    """Load a LeRobot dataset from a local directory or a Hugging Face repo id."""
    root = _resolve_local_root(source)

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(f"{source!r} does not look like a LeRobot dataset (missing meta/info.json)")
    info = json.loads(info_path.read_text())

    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError(f"No parquet data files found under {root / 'data'}")

    frames = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)

    if "action" not in frames.columns:
        raise ValueError(f"{source!r} has no 'action' column in its data files")
    if "episode_index" not in frames.columns:
        raise ValueError(f"{source!r} has no 'episode_index' column in its data files")

    actions = np.stack(frames["action"].to_numpy())
    action_dim = actions.shape[1]

    action_names = info.get("features", {}).get("action", {}).get("names")
    if not action_names or len(action_names) != action_dim:
        action_names = [f"action_{i}" for i in range(action_dim)]

    states = np.stack(frames["observation.state"].to_numpy()) if "observation.state" in frames.columns else None
    episode_index = frames["episode_index"].to_numpy()

    success = None
    for candidate in ("next.success", "success"):
        if candidate in frames.columns:
            success = frames[candidate].to_numpy().astype(bool)
            break

    action_limits = None
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        action_stats = stats.get("action")
        if action_stats and "min" in action_stats and "max" in action_stats:
            action_limits = (np.array(action_stats["min"]), np.array(action_stats["max"]))

    return FrameData(
        actions=actions,
        states=states,
        episode_index=episode_index,
        action_names=list(action_names),
        success=success,
        action_limits=action_limits,
        num_episodes=int(episode_index.max()) + 1 if episode_index.size else 0,
    )
