"""Reproducible baseline experiment with request-time and work accounting."""

from dataclasses import asdict, dataclass
from time import perf_counter
import random

from .baselines import prioritized_sweeping
from .data import Trajectory, Transition
from .planner import value_iteration
from .reference import exact_reference
from .tabular import EmpiricalMDP
from .unlearning import delete_and_repair


@dataclass
class Work:
    bellman_rows: int = 0
    transition_entries_read: int = 0
    predecessor_edges_indexed: int = 0
    log_transitions_applied: int = 0


class MeteredMDP(EmpiricalMDP):
    def __init__(self, *args, **kwargs):
        self.work = Work()
        super().__init__(*args, **kwargs)

    def _apply(self, trajectory, sign):
        self.work.log_transitions_applied += len(trajectory.transitions)
        return super()._apply(trajectory, sign)

    def backup(self, q, state, action):
        self.work.bellman_rows += 1
        return super().backup(q, state, action)

    def transition_probabilities(self, state, action):
        probabilities = super().transition_probabilities(state, action)
        self.work.transition_entries_read += len(probabilities)
        return probabilities

    def predecessor_rows(self):
        reverse = super().predecessor_rows()
        self.work.predecessor_edges_indexed += sum(map(len, reverse.values()))
        return reverse


def make_case(name, seed):
    """Small complete episodes; seeds vary logged reward observations."""
    rng = random.Random(seed)
    data = []

    def add(identifier, steps):
        data.append(Trajectory(identifier, tuple(steps)))

    if name == "shared_state":
        add("retain-low", [Transition(0, 0, 0.2)])
        add("forget", [Transition(0, 0, round(rng.uniform(0.9, 1.1), 3))])
        add("retain-other", [Transition(0, 1, 0.4)])
        states = (0,)
    elif name == "nonpolicy_row":
        add("retain-best", [Transition(0, 0, 1.0)])
        add("retain-other", [Transition(0, 1, 0.2)])
        add("forget", [Transition(0, 1, round(rng.uniform(0.1, 0.3), 3))])
        states = (0,)
    elif name == "bottleneck":
        states = tuple(range(5))
        for state in states:
            add(f"exit-{state}", [Transition(state, 1, 0.3)])
            steps = [Transition(s, 0, 0.0, s + 1) for s in range(state, 4)]
            add(f"path-{state}", steps + [Transition(4, 0, 0.1)])
        add("forget", [Transition(4, 0, round(rng.uniform(10.0, 12.0), 3))])
    elif name == "coverage_loss":
        states = (0,)
        add("forget", [Transition(0, 0, 2.0)])
        add("retain", [Transition(0, 1, 0.5)])
    elif name == "tie":
        states = (0,)
        add("retain-a", [Transition(0, 0, 1.0)])
        add("retain-b", [Transition(0, 1, 1.0)])
        add("forget", [Transition(0, 1, 2.0)])
    else:
        raise ValueError(f"unknown case: {name}")
    return states, (0, 1), 0.9, tuple(data), ("forget",)


METHODS = ("noop", "scratch", "warm_start", "prioritized_sweeping", "certified_repair")
CASES = ("shared_state", "nonpolicy_row", "bottleneck", "coverage_loss", "tie")


def run_case(name, seed, *, tolerance=1e-10):
    states, actions, gamma, data, forgotten = make_case(name, seed)
    started = perf_counter()
    original = MeteredMDP(states, actions, gamma, data)
    old = value_iteration(original, tolerance=tolerance)
    training_seconds = perf_counter() - started
    training_work = asdict(original.work)
    retained = tuple(t for t in data if t.trajectory_id not in forgotten)
    exact_q, exact_policy = exact_reference(states, actions, gamma, retained)
    records = []
    for method in METHODS:
        started = perf_counter()
        model = MeteredMDP(states, actions, gamma, data) if method not in ("scratch", "noop") else None
        preparation_seconds = perf_counter() - started
        preparation_work = asdict(model.work) if model else asdict(Work())
        if model:
            model.work = Work()
        started = perf_counter()
        stages = {}
        certified, fallback = None, False
        if method == "noop":
            q, policy = old.q, old.policy
            work = Work()
            converged = None
        elif method == "scratch":
            stage = perf_counter()
            # Include retained-log selection and aggregation in request latency.
            model = MeteredMDP(states, actions, gamma,
                               (t for t in data if t.trajectory_id not in forgotten))
            stages["retained_rebuild_seconds"] = perf_counter() - stage
            stage = perf_counter()
            result = value_iteration(model, tolerance=tolerance)
            stages["planning_seconds"] = perf_counter() - stage
            q, policy, work, converged = result.q, result.policy, model.work, True
        elif method in ("warm_start", "prioritized_sweeping"):
            stage = perf_counter()
            changed = model.delete_trajectories(forgotten)
            stages["subtraction_seconds"] = perf_counter() - stage
            stage = perf_counter()
            result = (value_iteration(model, initial_q=old.q, tolerance=tolerance)
                      if method == "warm_start" else
                      prioritized_sweeping(model, old.q, changed, tolerance=tolerance))
            stages["planning_and_verification_seconds"] = perf_counter() - stage
            q, policy, work = result.q, result.policy, model.work
            converged = True if method == "warm_start" else result.converged
        else:
            result = delete_and_repair(model, old.q, forgotten, tolerance=tolerance)
            q, policy, work, converged = result.q, result.policy, model.work, result.residual <= tolerance
            certified, fallback = result.certified_optimal_policy, result.used_full_fallback
        request_seconds = perf_counter() - started
        # Independent validation is common offline experiment overhead, outside request timing.
        records.append(dict(method=method, request_seconds=request_seconds, stages=stages,
                            preparation_seconds=preparation_seconds, preparation_work=preparation_work,
                            work=asdict(work), converged=converged,
                            certified_optimal_policy=certified, fallback=fallback,
                            policy_disagreement=sum(policy[s] != exact_policy[s] for s in states) / len(states),
                            max_q_error=max(abs(q[row] - float(value)) for row, value in exact_q.items())))
    return dict(case=name, seed=seed, states=len(states), actions=len(actions), gamma=gamma,
                tolerance=tolerance, forget_ids=list(forgotten),
                trajectory_ids=[t.trajectory_id for t in data],
                dataset=[asdict(t) for t in data],
                original_training_seconds=training_seconds, original_training_work=training_work,
                exact_reference_policy=exact_policy, records=records)
