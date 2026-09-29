from pathlib import Path
from fractions import Fraction as F
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.local_policy_repair import LocalPolicyRepair
from rl_unlearning.reference import exact_reference


def trajectory(name, *steps):
    return Trajectory(name, tuple(steps))


class LocalPolicyRepairTests(unittest.TestCase):
    def test_dominated_bridge_pruning_keeps_repair_local(self):
        data = (trajectory("bonus", Transition(0, 0, 2)),
                trajectory("low", Transition(0, 0, 0)),
                trajectory("exit0", Transition(0, 1, .2)),
                trajectory("exit1", Transition(1, 1, .2)),
                trajectory("good1", Transition(1, 0, .4)),
                trajectory("bridge", Transition(1, 2, -100, 0), Transition(0, 1, .2)))
        plain = LocalPolicyRepair((0, 1), (0, 1, 2), .8, data)
        pruned = LocalPolicyRepair((0, 1), (0, 1, 2), .8, data, anchor_policy={0: 1, 1: 1})
        self.assertEqual(plain.delete_and_repair(["bonus"]).affected_states, (0, 1))
        result = pruned.delete_and_repair(["bonus"])
        self.assertEqual(result.affected_states, (0,))
        self.assertGreater(result.pruned_action_rows, 0)
        self.compare(pruned, data[1:])

    def test_newly_admissible_action_restores_reverse_edges(self):
        data = (trajectory("low", Transition(0, 0, -100, 1), Transition(1, 1, .2)),
                trajectory("high", Transition(0, 0, 2, 1), Transition(1, 1, .2)),
                trajectory("exit0", Transition(0, 1, .2)),
                trajectory("one", Transition(1, 0, 1)), trajectory("zero", Transition(1, 0, 0)))
        learner = LocalPolicyRepair((0, 1), (0, 1), .8, data, anchor_policy={0: 1, 1: 1})
        result = learner.delete_and_repair(["low"])
        self.assertGreater(result.index_edges_added, 0)
        self.compare(learner, data[1:])
        result = learner.delete_and_repair(["one"])
        self.assertEqual(result.affected_states, (0, 1))
        self.compare(learner, (data[1], data[2], data[4]))

    def test_anchor_invalidation_forces_full_replan_and_disables_pruning(self):
        data = (trajectory("a", Transition(0, 0, 2)), trajectory("b", Transition(0, 0, 0)),
                trajectory("exit-low", Transition(0, 1, .2)), trajectory("exit-high", Transition(0, 1, .4)),
                trajectory("s1", Transition(1, 1, .2)))
        learner = LocalPolicyRepair((0, 1), (0, 1), .8, data, anchor_policy={0: 1, 1: 1})
        result = learner.delete_and_repair(["exit-high"])
        self.assertTrue(result.pruning_disabled)
        self.assertTrue(result.used_full_fallback)
        self.assertEqual(result.pruned_action_rows, 0)
        self.assertGreater(result.index_rows_rebuilt, 0)
        retained = tuple(t for t in data if t.trajectory_id != "exit-high")
        self.compare(learner, retained)
        learner.delete_and_repair(["a"])
        self.compare(learner, retained[1:])

    def compare(self, learner, retained):
        q, policy = exact_reference(learner._states, learner._actions, learner._gamma, retained)
        self.assertEqual(dict(learner.policy), policy)
        self.assertEqual(dict(learner.values), {s: max(q[(s, a)] for a in learner._actions)
                                               for s in learner._states})

    def test_chosen_row_change_switches_only_one_cyclic_component(self):
        data = (
            trajectory("route", Transition(0, 0, 0, 1), Transition(1, 0, 0, 0), Transition(0, 1, .2)),
            trajectory("high", Transition(1, 0, 8)),
            trajectory("low", Transition(1, 0, -1)),
            trajectory("exit", Transition(1, 1, .3)),
            trajectory("other", Transition(2, 0, 1, 3), Transition(3, 0, 1, 2), Transition(2, 1, .1)),
            trajectory("other-exit", Transition(3, 1, .1)),
        )
        learner = LocalPolicyRepair(range(4), (0, 1), .8, data)
        outside = {s: learner.values[s] for s in (2, 3)}
        result = learner.delete_and_repair(["high"])
        self.assertEqual(result.affected_states, (0, 1))
        self.assertTrue(result.changed_actions)
        self.assertEqual({s: learner.values[s] for s in (2, 3)}, outside)
        self.assertTrue(all(n == 2 for n in result.linear_system_sizes))
        self.compare(learner, tuple(t for t in data if t.trajectory_id != "high"))

    def test_non_greedy_predecessor_is_included(self):
        data = (trajectory("direct", Transition(0, 0, 1)),
                trajectory("branch", Transition(0, 1, 0, 1), Transition(1, 1, 0)),
                trajectory("weak", Transition(1, 0, 0)),
                trajectory("strong", Transition(1, 0, 2)))
        learner = LocalPolicyRepair((0, 1), (0, 1), .9, data)
        self.assertEqual(learner.policy[0], 0)
        result = learner.delete_and_repair(["weak"])
        self.assertEqual(result.affected_states, (0, 1))
        self.assertEqual(learner.policy[0], 1)
        self.compare(learner, data[:2] + data[3:])

    def test_removed_support_edge_does_not_pollute_later_request(self):
        data = (trajectory("edge", Transition(0, 0, 0, 1), Transition(1, 1, 0)),
                trajectory("exit", Transition(0, 1, .2)),
                trajectory("high", Transition(1, 0, 2)),
                trajectory("low", Transition(1, 0, 1)))
        learner = LocalPolicyRepair((0, 1), (0, 1), .9, data)
        first = learner.delete_and_repair(["edge"])
        self.assertEqual(first.index_edges_removed, 1)
        second = learner.delete_and_repair(["high"])
        self.assertEqual(second.affected_states, (1,))
        self.compare(learner, (data[1], data[3]))

    def test_solver_failure_and_invalid_request_are_atomic(self):
        data = (trajectory("high", Transition(0, 0, 9)),
                trajectory("low", Transition(0, 0, 0)),
                trajectory("other", Transition(0, 1, 1)))
        learner = LocalPolicyRepair((0,), (0, 1), .9, data)
        with self.assertRaises(KeyError):
            learner.delete_and_repair(["high", "missing"])
        with self.assertRaises(RuntimeError):
            learner.delete_and_repair(["high"], max_iterations=1)
        self.compare(learner, data)
        learner.delete_and_repair(["high"])
        self.compare(learner, data[1:])

    def test_redundant_empty_and_exact_tie(self):
        data = (trajectory("a", Transition(0, 0, 1)), trajectory("b", Transition(0, 0, 1)),
                trajectory("c", Transition(0, 1, 1)))
        learner = LocalPolicyRepair((0,), (1, 0), .9, data)
        self.assertEqual(learner.policy[0], 1)
        for ids in ([], ["a"]):
            result = learner.delete_and_repair(ids)
            self.assertEqual(result.affected_states, ())
            self.assertEqual(result.linear_system_sizes, ())
        learner.delete_and_repair(["c"])
        self.assertEqual(learner.policy[0], 0)
        learner.delete_and_repair(["b"])
        self.assertEqual(learner.policy[0], 1)
        self.assertEqual(learner.values[0], F(0))

    def test_global_fallback_is_explicit(self):
        data = (trajectory("a", Transition(0, 0, 1)), trajectory("b", Transition(1, 0, 1)))
        learner = LocalPolicyRepair((0, 1), (0, 1), .9, data)
        result = learner.delete_and_repair(["a"], max_local_fraction=0)
        self.assertTrue(result.used_full_fallback)
        self.assertEqual(result.affected_states, (0,))
        self.assertTrue(all(n == 2 for n in result.linear_system_sizes))
        self.compare(learner, data[1:])

    def test_random_cyclic_sequential_deletions_match_enumeration(self):
        rng = random.Random(812)
        for trial in range(40):
            data = []
            for index in range(12):
                s, steps = rng.randrange(3), []
                while True:
                    a, reward = rng.randrange(2), rng.randrange(-5, 6) / 10
                    successor = None if rng.random() < .4 else rng.randrange(3)
                    steps.append(Transition(s, a, reward, successor))
                    if successor is None:
                        break
                    s = successor
                data.append(Trajectory(str(index), tuple(steps)))
            learner = LocalPolicyRepair(range(3), (0, 1), .7, data,
                                        anchor_policy=dict.fromkeys(range(3), 0) if trial % 2 else None)
            self.compare(learner, tuple(data))
            for t in rng.sample(data, 4):
                learner.delete_and_repair([t.trajectory_id], max_local_fraction=.8)
                data = [x for x in data if x.trajectory_id != t.trajectory_id]
                self.compare(learner, tuple(data))


if __name__ == "__main__":
    unittest.main()
