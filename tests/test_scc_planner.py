from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.local_policy_repair import LocalPolicyRepair
from rl_unlearning.policy_iteration import policy_iteration
from rl_unlearning.reference import exact_reference
from rl_unlearning.scc_planner import scc_policy_iteration


class SCCPlannerTests(unittest.TestCase):
    def test_sink_order_nongreedy_bridge_and_external_boundary(self):
        rows = {
            (0, 0): (F(1), {0: F(1)}),
            (0, 1): (F(0), {1: F(1)}),
            (1, 0): (F(3), {2: F(1)}),
            (1, 1): (F(0), {None: F(1)}),
            (2, 0): (F(0), {1: F(1, 2), 9: F(1, 2)}),
            (2, 1): (F(0), {None: F(1)}),
        }
        kwargs = dict(initial_policy={0: 0, 1: 1, 2: 1}, boundary_values={9: F(10)})
        dense = policy_iteration((0, 1, 2), (0, 1), .8, lambda s, a: rows[s, a], **kwargs)
        scc = scc_policy_iteration((0, 1, 2), (0, 1), .8, lambda s, a: rows[s, a], **kwargs)
        self.assertEqual((scc.policy, scc.values, scc.q), (dense.policy, dense.values, dense.q))
        self.assertEqual(scc.policy[0], 1)
        self.assertEqual(scc.component_sizes, (2, 1))
        self.assertEqual(scc.graph_rows_scanned, 6)

    def test_deleting_only_bridge_splits_component(self):
        logs = (Trajectory("forward", (Transition(0, 0, 1, 1), Transition(1, 1, 0))),
                Trajectory("back", (Transition(1, 0, 1, 0), Transition(0, 1, 0))))
        learner = LocalPolicyRepair((0, 1), (0, 1), .8, logs, solver="scc")
        self.assertEqual(learner.preparation_plan.component_sizes, (2,))
        result = learner.delete_and_repair(["back"], force_full=True)
        q, policy = exact_reference((0, 1), (0, 1), .8, logs[:1])
        self.assertEqual(dict(learner.policy), policy)
        self.assertEqual(dict(learner.values), {s: q[s, a] for s, a in policy.items()})
        self.assertEqual(result.component_sizes, (1, 1))

    def test_random_sequential_repair_with_pruning_matches_enumeration(self):
        rng = random.Random(947)
        for trial in range(24):
            logs = []
            for index in range(12):
                state, steps = rng.randrange(3), []
                for step in range(4):
                    successor = None if step == 3 or rng.random() < .4 else rng.randrange(3)
                    steps.append(Transition(state, rng.randrange(2), rng.randrange(-5, 6) / 10, successor))
                    if successor is None:
                        break
                    state = successor
                logs.append(Trajectory(str(index), tuple(steps)))
            learner = LocalPolicyRepair(range(3), (1, 0), .7, logs, solver="scc",
                                        anchor_policy=dict.fromkeys(range(3), 1) if trial % 2 else None)
            for removed in rng.sample(logs, 4):
                learner.delete_and_repair([removed.trajectory_id])
                logs = [t for t in logs if t != removed]
                q, policy = exact_reference(range(3), (1, 0), .7, logs)
                self.assertEqual(dict(learner.policy), policy)
                self.assertEqual(dict(learner.values), {s: q[s, a] for s, a in policy.items()})

    def test_later_component_failure_is_atomic_and_retry_updates_boundary(self):
        logs = (Trajectory("low", (Transition(1, 0, 0),)),
                Trajectory("high", (Transition(1, 0, 2),)),
                Trajectory("exit", (Transition(0, 0, 1),)),
                Trajectory("bridge", (Transition(0, 1, 0, 1), Transition(1, 1, 0))))
        learner = LocalPolicyRepair((0, 1), (0, 1), .9, logs, solver="scc")
        before = dict(learner.policy), dict(learner.values), repr(learner._rows), repr(learner._reverse)
        with self.assertRaises(RuntimeError):
            learner.delete_and_repair(["low"], max_iterations=1)
        self.assertEqual(before, (dict(learner.policy), dict(learner.values),
                                  repr(learner._rows), repr(learner._reverse)))
        learner.delete_and_repair(["low"])
        self.assertEqual(learner.policy[0], 1)
        self.assertEqual(learner.values[0], F(9, 5))

    def test_long_chain_does_not_use_python_recursion(self):
        size = 1100
        plan = scc_policy_iteration(range(size), (0,), .5,
                                    lambda s, a: (F(1), {s + 1 if s + 1 < size else None: F(1)}))
        self.assertEqual(plan.component_sizes, (1,) * size)
        self.assertEqual(plan.values[0], 2 - F(1, 2 ** (size - 1)))

    def test_ties_empty_block_and_missing_boundary(self):
        lookup = lambda s, a: (F(0), {None: F(1)})
        plan = scc_policy_iteration((0, 1), (1, 0), .9, lookup, initial_policy={0: 0, 1: 0})
        self.assertEqual(plan.policy, {0: 1, 1: 1})
        self.assertEqual(scc_policy_iteration((), (0,), .9, lookup).graph_rows_scanned, 0)
        with self.assertRaises(ValueError):
            scc_policy_iteration((0,), (0,), .9, lambda s, a: (F(0), {1: F(1)}))


if __name__ == "__main__":
    unittest.main()
