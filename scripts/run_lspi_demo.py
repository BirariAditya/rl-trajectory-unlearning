"""Reproduce a policy-switch counterexample and exact retained LSPI replay."""

import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_baselines import git
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, LSPIReplayState, scratch_lspi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/lspi_demo.json")
    args = parser.parse_args()
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    features = {pair: tuple(F(i == j) for j in range(4)) for i, pair in enumerate(pairs)}
    problem = LSPIProblem((0, 1), (0, 1), features, {0: 0, 1: 0}, iterations=3)
    logs = (Trajectory("xa", (Transition(0, 0, 0, 1), Transition(1, 0, 0))),
            Trajectory("xb", (Transition(0, 1, F(1, 16)),)),
            Trajectory("zb", (Transition(1, 1, F(1, 4)),)),
            Trajectory("forget", (Transition(1, 0, 1),)))
    full = scratch_lspi(problem, logs)
    replay = LSPIReplayState(problem, logs).delete_and_replay(["forget"])
    scratch = scratch_lspi(problem, logs[:-1])
    match = replay.iterates == scratch.iterates and replay.policy == scratch.policy
    one_shot = replay.iterates[0].policy_out
    report = dict(git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT / "src/rl_unlearning/lspi_control.py", Path(__file__))},
                  scope="Exact finite-schedule fixed-feature replay; assembly equivalence, no speedup/optimality/privacy/novelty claim.",
                  states=problem.states, actions=problem.actions, gamma=str(problem.gamma),
                  regularization=str(problem.regularization), iterations=problem.iterations,
                  initial_policy=dict(problem.initial_policy),
                  feature_order=pairs, features=[features[p] for p in pairs],
                  dataset=[asdict(t) for t in logs], deletion_ids=["forget"],
                  full=asdict(full), replay=asdict(replay), retained_scratch=asdict(scratch),
                  all_iterates_match=match, one_shot_policy=one_shot,
                  one_shot_disagrees=one_shot != scratch.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Full-data policy: {full.policy}; one-shot: {one_shot}; retained replay: {replay.policy}")
    print(f"All retained matrices/weights/policies match: {match}; correction count: {replay.work.target_rank_one_updates}")
    print(args.output)
    if not match or one_shot == scratch.policy or full.policy != (0, 0):
        raise SystemExit("counterexample/replay validation failed")


if __name__ == "__main__":
    main()
