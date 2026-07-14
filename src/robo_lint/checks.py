"""The individual diagnostic checks. Each check takes plain arrays and returns a
plain result dataclass — no I/O, no dataset-format knowledge. See loaders.py for
how raw datasets get turned into these arrays.
"""

from dataclasses import dataclass

import numpy as np

# Heuristic minimum episode counts per policy type, based on community-reported
# training experience (see docs/ROADMAP.md). These are starting points, not
# hard rules — expect to revise them as robo-lint is run against more datasets.
READINESS_MIN_EPISODES = {
    "act": 50,
    "diffusion": 100,
    "smolvla": 80,
    "vla": 150,
}


@dataclass
class JointCoverageResult:
    name: str
    range: float
    relative_range: float
    dead: bool


def joint_coverage(actions: np.ndarray, names: list[str], dead_threshold: float = 0.05) -> list[JointCoverageResult]:
    """Flag action dimensions that barely move across the whole dataset.

    A joint that never moves teaches the policy that "do nothing" is always
    correct for that joint — loss converges fine, and the failure only shows
    up at deployment when the joint is actually needed.
    """
    ranges = actions.max(axis=0) - actions.min(axis=0)
    max_range = float(ranges.max()) if ranges.size else 0.0

    results = []
    for i, name in enumerate(names):
        relative = float(ranges[i] / max_range) if max_range > 0 else 0.0
        results.append(
            JointCoverageResult(
                name=name,
                range=float(ranges[i]),
                relative_range=relative,
                dead=relative < dead_threshold,
            )
        )
    return results


@dataclass
class ActionFeasibilityResult:
    checked: bool
    fraction_out_of_bounds: float | None
    per_dim_out_of_bounds: list[float] | None


def action_feasibility(actions: np.ndarray, limits: tuple[np.ndarray, np.ndarray] | None) -> ActionFeasibilityResult:
    """Flag actions outside the dataset's declared kinematic limits, if known.

    Skipped (not "0% infeasible") when no limits are available for the
    dataset, since we can't tell feasible from infeasible without them.
    """
    if limits is None:
        return ActionFeasibilityResult(checked=False, fraction_out_of_bounds=None, per_dim_out_of_bounds=None)

    lo, hi = limits
    out_of_bounds = (actions < lo) | (actions > hi)
    per_dim = out_of_bounds.mean(axis=0).tolist()
    overall = float(out_of_bounds.any(axis=1).mean())
    return ActionFeasibilityResult(checked=True, fraction_out_of_bounds=overall, per_dim_out_of_bounds=per_dim)


@dataclass
class SuccessRatioResult:
    checked: bool
    num_episodes: int
    num_successful: int | None
    ratio: float | None


def success_ratio(success: np.ndarray | None, episode_index: np.ndarray) -> SuccessRatioResult:
    """Per-episode success rate, if the dataset records a success signal.

    Most public LeRobot datasets don't label success/failure explicitly, so
    this is best-effort and clearly marked as skipped when absent, rather
    than silently reporting a misleading 100%.
    """
    num_episodes = int(episode_index.max()) + 1 if episode_index.size else 0

    if success is None:
        return SuccessRatioResult(checked=False, num_episodes=num_episodes, num_successful=None, ratio=None)

    episodes = np.unique(episode_index)
    num_successful = sum(1 for ep in episodes if success[episode_index == ep].any())
    ratio = num_successful / len(episodes) if len(episodes) else None
    return SuccessRatioResult(checked=True, num_episodes=len(episodes), num_successful=num_successful, ratio=ratio)


@dataclass
class ReadinessResult:
    policy_type: str | None
    num_episodes: int
    minimum_recommended: int | None
    ready: bool | None
    note: str


def readiness(num_episodes: int, policy_type: str | None) -> ReadinessResult:
    """Compare episode count against a heuristic minimum for the target policy type."""
    if policy_type is None:
        return ReadinessResult(
            policy_type=None,
            num_episodes=num_episodes,
            minimum_recommended=None,
            ready=None,
            note="No --policy-type given; readiness check skipped.",
        )

    minimum = READINESS_MIN_EPISODES.get(policy_type.lower())
    if minimum is None:
        known = ", ".join(sorted(READINESS_MIN_EPISODES))
        return ReadinessResult(
            policy_type=policy_type,
            num_episodes=num_episodes,
            minimum_recommended=None,
            ready=None,
            note=f"No heuristic for policy type '{policy_type}'. Known types: {known}.",
        )

    return ReadinessResult(
        policy_type=policy_type,
        num_episodes=num_episodes,
        minimum_recommended=minimum,
        ready=num_episodes >= minimum,
        note="Heuristic minimum from community-reported training experience, not a hard rule.",
    )
