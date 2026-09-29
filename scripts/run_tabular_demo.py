"""Small example of deletion changing the preferred action."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_unlearning import EmpiricalMDP, Trajectory, Transition, delete_and_repair, value_iteration


def main() -> None:
    data = [
        Trajectory("retained-left", (Transition(0, 0, 0.2),)),
        Trajectory("forget-high-reward", (Transition(0, 0, 1.0),)),
        Trajectory("retained-right", (Transition(0, 1, 0.4),)),
    ]
    model = EmpiricalMDP(states=(0,), actions=(0, 1), gamma=0.9, trajectories=data)
    original = value_iteration(model)
    repaired = delete_and_repair(model, original.q, ["forget-high-reward"])
    reference = value_iteration(EmpiricalMDP((0,), (0, 1), 0.9, (data[0], data[2])))
    print(json.dumps({
        "policy_before": original.policy,
        "policy_after_deletion": repaired.policy,
        "policy_retrained_on_retained_data": reference.policy,
        "q_after_deletion": {str(k): v for k, v in repaired.q.items()},
        "q_retrained": {str(k): v for k, v in reference.q.items()},
        "certified_optimal_policy": repaired.certified_optimal_policy,
        "used_full_fallback": repaired.used_full_fallback,
        "local_backups": repaired.local_backups,
        "local_target_evaluations": repaired.local_target_evaluations,
        "full_residual_checks": repaired.residual_checks,
    }, indent=2))


if __name__ == "__main__":
    main()
