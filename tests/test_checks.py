import numpy as np

from robo_lint.checks import action_feasibility, joint_coverage, readiness, success_ratio


def test_joint_coverage_flags_dead_joint():
    actions = np.zeros((100, 3))
    actions[:, 0] = np.linspace(-1, 1, 100)
    actions[:, 1] = np.linspace(-1, 1, 100)
    actions[:, 2] = 0.001  # barely moves relative to the other two joints

    results = {r.name: r for r in joint_coverage(actions, ["shoulder", "elbow", "wrist"])}

    assert results["wrist"].dead is True
    assert results["shoulder"].dead is False
    assert results["elbow"].dead is False


def test_joint_coverage_handles_all_static_dataset():
    actions = np.zeros((10, 2))
    results = joint_coverage(actions, ["a", "b"])
    assert all(r.dead for r in results)


def test_action_feasibility_detects_out_of_bounds():
    actions = np.array([[0.0], [2.0], [-2.0], [0.5]])
    limits = (np.array([-1.0]), np.array([1.0]))

    result = action_feasibility(actions, limits)

    assert result.checked
    assert result.fraction_out_of_bounds == 0.5


def test_action_feasibility_skipped_without_limits():
    result = action_feasibility(np.zeros((10, 2)), None)
    assert not result.checked
    assert result.fraction_out_of_bounds is None


def test_success_ratio_counts_per_episode():
    episode_index = np.array([0, 0, 0, 1, 1, 1])
    success = np.array([False, False, True, False, False, False])

    result = success_ratio(success, episode_index)

    assert result.checked
    assert result.num_episodes == 2
    assert result.num_successful == 1
    assert result.ratio == 0.5


def test_success_ratio_skipped_without_signal():
    episode_index = np.array([0, 0, 1, 1])
    result = success_ratio(None, episode_index)
    assert not result.checked
    assert result.num_episodes == 2


def test_success_ratio_skipped_when_column_is_never_true():
    # Real-world case: lerobot/pusht has a next.success column that's always
    # False for every frame in the dataset — an unpopulated placeholder, not
    # a genuine "0% success rate" signal.
    episode_index = np.array([0, 0, 1, 1])
    success = np.array([False, False, False, False])

    result = success_ratio(success, episode_index)

    assert not result.checked
    assert result.num_successful is None
    assert "unpopulated placeholder" in result.note


def test_readiness_flags_under_resourced_dataset():
    assert readiness(num_episodes=30, policy_type="act").ready is False
    assert readiness(num_episodes=60, policy_type="act").ready is True


def test_readiness_skips_without_policy_type():
    result = readiness(num_episodes=30, policy_type=None)
    assert result.ready is None


def test_readiness_notes_unknown_policy_type():
    result = readiness(num_episodes=30, policy_type="made-up-policy")
    assert result.ready is None
    assert "made-up-policy" in result.note
