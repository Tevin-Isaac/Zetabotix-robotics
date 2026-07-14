"""Assembles check results into a single report, renderable as Markdown or JSON."""

from dataclasses import asdict, dataclass

from .checks import ActionFeasibilityResult, JointCoverageResult, ReadinessResult, SuccessRatioResult


@dataclass
class Report:
    source: str
    num_frames: int
    num_episodes: int
    joint_coverage: list[JointCoverageResult]
    action_feasibility: ActionFeasibilityResult
    success: SuccessRatioResult
    readiness: ReadinessResult

    def has_critical_issues(self) -> bool:
        """Whether this report should fail a CI check."""
        dead_joints = any(j.dead for j in self.joint_coverage)
        under_resourced = self.readiness.ready is False
        return dead_joints or under_resourced

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "num_frames": self.num_frames,
            "num_episodes": self.num_episodes,
            "joint_coverage": [asdict(j) for j in self.joint_coverage],
            "action_feasibility": asdict(self.action_feasibility),
            "success": asdict(self.success),
            "readiness": asdict(self.readiness),
        }

    def to_markdown(self) -> str:
        lines = [f"# robo-lint report: {self.source}", ""]
        lines.append(f"- Frames: {self.num_frames}")
        lines.append(f"- Episodes: {self.num_episodes}")
        lines.append("")

        lines.append("## Joint coverage")
        for j in self.joint_coverage:
            flag = "  ⚠️ DEAD JOINT" if j.dead else ""
            lines.append(f"- {j.name}: relative range {j.relative_range:.2f}{flag}")
        lines.append("")

        lines.append("## Action feasibility")
        af = self.action_feasibility
        if af.checked:
            lines.append(f"- {af.fraction_out_of_bounds:.1%} of actions outside declared kinematic limits")
        else:
            lines.append("- Skipped: no kinematic limits available for this dataset")
        lines.append("")

        lines.append("## Success ratio")
        s = self.success
        if s.checked:
            lines.append(f"- {s.num_successful}/{s.num_episodes} episodes successful ({s.ratio:.1%})")
        else:
            lines.append("- Skipped: no success signal found in this dataset")
        lines.append("")

        lines.append("## Model readiness")
        r = self.readiness
        if r.minimum_recommended is not None:
            status = "READY" if r.ready else "⚠️ UNDER-RESOURCED"
            lines.append(f"- {r.policy_type}: {self.num_episodes} episodes ({status}, heuristic minimum {r.minimum_recommended})")
        lines.append(f"  note: {r.note}")

        return "\n".join(lines)
