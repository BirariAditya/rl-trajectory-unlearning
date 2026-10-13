from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning.lspi_control import LSPIProblem, scratch_lspi
from rl_unlearning.lspi_baselines import LSPIMemoizedState
from test_lspi_control import episode, switch_fixture
from fractions import Fraction as F


class MemoTests(unittest.TestCase):
    def compare(self, a, b):
        self.assertEqual(a.iterates, b.iterates)
        self.assertEqual((a.policy, a.weights, a.status), (b.policy, b.weights, b.status))

    def test_deletion_resets_memo_and_fixed_point_reuses_solves(self):
        problem, logs, retained = switch_fixture(iterations=6)
        state = LSPIMemoizedState(problem, logs)
        full = state.replay()
        self.compare(full, scratch_lspi(problem, logs))
        self.assertEqual(full.work.linear_solves, 1)
        after = state.delete_and_replay(["forget"])
        self.compare(after, scratch_lspi(problem, retained))
        self.assertEqual(after.work.linear_solves, 3)
        self.assertEqual(len(after.iterates), 6)

    def test_cycle_reuses_each_distinct_policy(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (3,), (0, 1): (1,)}, {0: 0},
                              gamma=F(9, 10), iterations=7)
        logs = (episode("cycle", (0, 1, 1, 0), (0, 1, 0, None)),)
        result = LSPIMemoizedState(problem, logs).replay()
        self.compare(result, scratch_lspi(problem, logs))
        self.assertEqual(result.status, "cycle")
        self.assertEqual(result.work.linear_solves, 2)

    def test_singularity_after_switch_remains_atomic(self):
        problem = LSPIProblem((0,), (0, 1), {(0, 0): (5,), (0, 1): (1,)}, {0: 1})
        logs = (episode("loop", (0, 1, 1, 0), (0, 1, 0, None)),
                episode("extra", (0, 1, 0, None)))
        state = LSPIMemoizedState(problem, logs)
        before = state.replay()
        result = state.delete_and_replay(["extra"])
        self.compare(result, scratch_lspi(problem, logs[:1]))
        self.assertEqual(result.status, "singular")
        self.assertEqual(state.trajectory_ids, frozenset(("loop", "extra")))
        self.compare(state.replay(), before)
