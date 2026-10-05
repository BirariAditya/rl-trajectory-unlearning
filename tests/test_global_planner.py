from fractions import Fraction as F
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.global_planner import ExactGlobalPlanner
from rl_unlearning.reference import exact_reference
from rl_unlearning.scc_planner import scc_policy_iteration


def trajectory(name, *steps):
    return Trajectory(name, tuple(steps))


class GlobalPlannerTests(unittest.TestCase):
    def compare(self, learner, logs):
        q, policy = exact_reference(learner._states, learner._actions, learner._gamma, logs)
        self.assertEqual(dict(learner.policy), policy)
        self.assertEqual(dict(learner.values), {s: q[s, a] for s, a in policy.items()})
        self.assertFalse(hasattr(learner, "_reverse"))

    def test_elimination_splits_graph_and_restores_newly_admissible_action(self):
        logs = (trajectory("low", Transition(0, 0, -100, 1), Transition(1, 1, .2)),
                trajectory("high", Transition(0, 0, 2, 1), Transition(1, 1, .2)),
                trajectory("exit", Transition(0, 1, .2)),
                trajectory("back", Transition(1, 0, -100, 0), Transition(0, 1, .2)))
        engine = ExactGlobalPlanner((0, 1), (0, 1), .8, logs, anchor_policy={0: 1, 1: 1})
        self.assertEqual(engine.preparation_plan.component_sizes, (1, 1))
        self.assertEqual(engine.preparation_result.pruned_action_rows, 2)
        result = engine.delete_and_replan(["low"])
        self.assertEqual(result.pruned_action_rows, 1)
        self.assertEqual(engine.policy[0], 0)
        self.assertFalse(result.anchor_refreshed)
        self.compare(engine, logs[1:])

    def test_anchor_refresh_keeps_pruning_active_after_deletion(self):
        logs = (trajectory("anchor-high", Transition(0, 1, 2)),
                trajectory("anchor-low", Transition(0, 1, .2)),
                trajectory("alternative", Transition(0, 0, .5)),
                trajectory("bridge", Transition(1, 0, -100, 0), Transition(0, 1, .2)),
                trajectory("exit", Transition(1, 1, .2)))
        engine = ExactGlobalPlanner((0, 1), (0, 1), .8, logs, anchor_policy={0: 1, 1: 1})
        result = engine.delete_and_replan(["anchor-high"])
        self.assertTrue(result.anchor_refreshed)
        self.assertIsNotNone(result.anchor_plan)
        self.assertGreater(result.pruned_action_rows, 0)
        self.assertEqual(result.pruning_rows_checked, 4)
        self.compare(engine, logs[1:])
        result = engine.delete_and_replan(["alternative"])
        self.assertFalse(result.anchor_refreshed)
        self.compare(engine, (logs[1], logs[3], logs[4]))

    def test_allowed_actions_preserve_ties_and_reset_disallowed_warm_action(self):
        lookup = lambda s, a: (F(0), {None: F(1)})
        result = scc_policy_iteration((0,), (1, 0, 2), .8, lookup,
                                      initial_policy={0: 2}, allowed_actions={0: (0, 1)})
        self.assertEqual(result.policy, {0: 1})
        self.assertEqual(set(result.q), {(0, 0), (0, 1)})
        for allowed in ((), (9,), (1, 1)):
            with self.assertRaises(ValueError):
                scc_policy_iteration((0,), (0, 1), .8, lookup, allowed_actions={0: allowed})
        logs = (trajectory("a", Transition(0, 0, 1)), trajectory("b", Transition(0, 1, 1)))
        engine = ExactGlobalPlanner((0,), (1, 0), .8, logs, anchor_policy={0: 0})
        self.assertEqual(engine.policy[0], 1)
        self.assertEqual(engine.preparation_result.pruned_action_rows, 0)

    def test_refresh_then_solver_failure_is_atomic(self):
        logs = (trajectory("high", Transition(0, 0, 2)),
                trajectory("low", Transition(0, 0, 0)), trajectory("other", Transition(0, 1, .4)))
        engine = ExactGlobalPlanner((0,), (0, 1), .8, logs, anchor_policy={0: 0})
        before = dict(engine.policy), dict(engine.values), dict(engine._anchor_values), repr(engine._rows)
        with self.assertRaises(RuntimeError):
            engine.delete_and_replan(["high"], max_iterations=1)
        self.assertEqual(before, (dict(engine.policy), dict(engine.values), dict(engine._anchor_values), repr(engine._rows)))
        engine.delete_and_replan(["high"])
        self.compare(engine, logs[1:])
        with self.assertRaises(KeyError):
            engine.delete_and_replan(["high"])

    def test_negative_rewards_empty_model_and_zero_discount(self):
        logs = (trajectory("negative", Transition(0, 0, -1)),)
        for gamma in (0, .9):
            engine = ExactGlobalPlanner((0,), (0, 1), gamma, logs, anchor_policy={0: 0})
            self.compare(engine, logs)
            engine.delete_and_replan(["negative"])
            self.compare(engine, ())

    def test_identical_anchor_sample_deletion_does_not_refresh_bound(self):
        logs = (trajectory("a", Transition(0, 0, .2)),
                trajectory("b", Transition(0, 0, .2)),
                trajectory("bad", Transition(0, 1, -1)))
        engine = ExactGlobalPlanner((0,), (0, 1), .8, logs, anchor_policy={0: 0})
        result = engine.delete_and_replan(["a"])
        self.assertFalse(result.anchor_refreshed)
        self.assertIsNone(result.anchor_plan)
        self.compare(engine, logs[1:])

    def test_random_sequential_requests_against_independent_enumeration(self):
        rng = random.Random(1071)
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
            engines = [ExactGlobalPlanner(range(3), (1, 0), .7, logs, solver=solver,
                                          anchor_policy=dict.fromkeys(range(3), 1) if prune else None)
                       for solver, prune in (("dense", False), ("scc", False), ("scc", True))]
            for removed in rng.sample(logs, 4):
                logs = [t for t in logs if t != removed]
                for engine in engines:
                    engine.delete_and_replan([removed.trajectory_id])
                    self.compare(engine, logs)


if __name__ == "__main__":
    unittest.main()
