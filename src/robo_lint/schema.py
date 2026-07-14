"""Internal data schema that decouples checks.py from any specific dataset format."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    """Per-frame action/state data for one dataset, already unified across episodes.

    Loaders (see loaders.py) are responsible for producing this from whatever
    on-disk format a dataset uses; checks.py only ever sees this shape.
    """

    actions: np.ndarray  # (num_frames, action_dim)
    states: np.ndarray | None  # (num_frames, state_dim), optional
    episode_index: np.ndarray  # (num_frames,) int, which episode each frame belongs to
    action_names: list[str]
    success: np.ndarray | None = None  # (num_frames,) bool, optional per-frame success signal
    action_limits: tuple[np.ndarray, np.ndarray] | None = None  # (min, max) per action dim
    num_episodes: int = 0
