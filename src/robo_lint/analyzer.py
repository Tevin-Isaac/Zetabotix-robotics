"""Top-level entry point: dataset in, Report out."""

from .checks import action_feasibility, joint_coverage, readiness, success_ratio
from .loaders import load_frames
from .report import Report


def analyze_dataset(source: str, policy_type: str | None = None) -> Report:
    frames = load_frames(source)

    return Report(
        source=source,
        num_frames=int(frames.actions.shape[0]),
        num_episodes=frames.num_episodes,
        joint_coverage=joint_coverage(frames.actions, frames.action_names),
        action_feasibility=action_feasibility(frames.actions, frames.action_limits),
        success=success_ratio(frames.success, frames.episode_index),
        readiness=readiness(frames.num_episodes, policy_type),
    )
