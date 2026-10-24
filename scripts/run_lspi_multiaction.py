"""Matched multi-action, higher-dimensional controls for exact LSPI deletion."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as F
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, scratch_lspi
from rl_unlearning.lspi_decision_repair import (
    LSPISelectiveRepairState,
    LSPISharedAnchorRepairState,
)
from rl_unlearning.lspi_inverse import LSPIInverseDeduplicatedState
from run_baselines import git


METHODS = (
    "scratch_retrain",
    "inverse_deduplicated",
    "selective_repair",
    "shared_anchor_repair",
)
CASES = tuple(
    (regime, states, dimension, 3)
    for regime in ("stable", "fragile")
    for states, dimension in ((32, 8), (128, 16))
)


def _dot(left, right):
    return sum((x * y for x, y in zip(left, right)), F(0))


def dataset(regime, states, dimension, actions, seed, requests=20):
    """Create fixed, unique, non-opposite action features and matched logs."""
    if regime not in ("stable", "fragile"):
        raise ValueError("regime must be stable or fragile")
    if states < 3 or dimension < 4 or actions < 3:
        raise ValueError("requires at least 3 states, 4 dimensions and 3 actions")
    if not 1 <= requests <= 40:
        raise ValueError("requests must be 1..40")

    state_space = tuple(range(states))
    action_space = tuple(range(actions))
    features = {}
    uniqueness_denominator = 64 * states * actions
    for state in state_space:
        preferred = state % actions
        for action in action_space:
            values = [
                F(((state + 1) * (j + 2) + (action + 2) * (j + 1)) % 9 - 4, 8)
                for j in range(dimension)
            ]
            if action == preferred:
                values[0] += 2
            values[(state + 2 * action + 1) % dimension] += F(3 + action, 8)
            values[-1] += F(state * actions + action + 1, uniqueness_denominator)
            features[state, action] = tuple(values)

    problem = LSPIProblem(
        state_space,
        action_space,
        features,
        dict.fromkeys(state_space, 0),
        gamma=F(1, 4),
        regularization=2,
        iterations=5,
    )
    teacher = tuple(
        F(1) if j == 0 else F(1 if j % 2 else -1, 16)
        for j in range(dimension)
    )
    rng = random.Random(seed)
    count = max(384, states * actions)
    logs = []
    for index in range(count):
        state = index % states
        action = (index // states) % actions
        successor = rng.randrange(states)
        next_action = rng.randrange(actions)

        def reward(candidate_state, candidate_action):
            noise = rng.randrange(-4, 5)
            if regime == "stable":
                return _dot(features[candidate_state, candidate_action], teacher) + F(noise, 100)
            return F(noise, 4) + F(
                features[candidate_state, candidate_action][0], 20
            )

        logs.append(
            Trajectory(
                str(index),
                (
                    Transition(state, action, reward(state, action), successor),
                    Transition(
                        successor,
                        next_action,
                        reward(successor, next_action),
                    ),
                ),
            )
        )
    deletion_order = list(range(count))
    rng.shuffle(deletion_order)
    return problem, tuple(logs), [[str(i)] for i in deletion_order[:requests]]


def feature_counts(problem):
    rows = {
        tuple(problem.features[state, action] for action in problem.actions)
        for state in problem.states
    }
    vectors = set(problem.features.values())
    opposite_pairs = sum(
        problem.features[state, left]
        == tuple(-value for value in problem.features[state, right])
        for state in problem.states
        for left in problem.actions
        for right in problem.actions
        if left < right
    )
    return {
        "state_action_vectors": len(problem.features),
        "unique_feature_vectors": len(vectors),
        "unique_ordered_action_rows": len(rows),
        "opposite_action_pairs": opposite_pairs,
    }


def signature(result):
    return (
        result.iterates,
        result.policy,
        result.weights,
        result.status,
        result.repeat_start,
        result.repeat_length,
    )


def initialize(method, problem, logs):
    if method == "scratch_retrain":
        return None, scratch_lspi(problem, logs)
    cls = {
        "inverse_deduplicated": LSPIInverseDeduplicatedState,
        "selective_repair": LSPISelectiveRepairState,
        "shared_anchor_repair": LSPISharedAnchorRepairState,
    }[method]
    state = cls(problem, logs)
    return state, state.replay()


def result_record(state, result, reference, seconds):
    matched = signature(result) == signature(reference)
    if not matched:
        raise AssertionError("retained-only full-trace mismatch")
    work = result.work if isinstance(result.work, dict) else asdict(result.work)
    return {
        "seconds": seconds,
        "full_trace_match": matched,
        "status": result.status,
        "policy_sha256": hashlib.sha256(repr(result.policy).encode()).hexdigest(),
        "work": work,
        "inverse_work": (
            dict(state.last_inverse_work)
            if state is not None and hasattr(state, "last_inverse_work")
            else None
        ),
        "decision_work": (
            dict(state.last_decision_work)
            if state is not None and hasattr(state, "last_decision_work")
            else None
        ),
    }


def run(case, seed, repeat, nrequests):
    problem, logs, batches = dataset(*case, seed, nrequests)
    rng = random.Random(seed * 1000 + repeat)
    order = list(METHODS)
    rng.shuffle(order)
    prep_results = {}
    engines = {}
    raw = {}
    times = {}
    for method in order:
        started = perf_counter()
        state, result = initialize(method, problem, logs)
        times[method] = perf_counter() - started
        engines[method] = state
        raw[method] = result
    reference = raw["scratch_retrain"]
    for method in METHODS:
        prep_results[method] = result_record(
            engines[method], raw[method], reference, times[method]
        )

    retained = logs
    previous = reference
    requests = []
    for batch in batches:
        removed = set(batch)
        candidate = tuple(
            trajectory
            for trajectory in retained
            if trajectory.trajectory_id not in removed
        )
        rng.shuffle(order)
        raw = {}
        times = {}
        for method in order:
            started = perf_counter()
            if method == "scratch_retrain":
                result = scratch_lspi(problem, candidate)
            else:
                result = engines[method].delete_and_replay(batch)
            times[method] = perf_counter() - started
            raw[method] = result
        reference = raw["scratch_retrain"]
        records = {
            method: result_record(
                engines[method], raw[method], reference, times[method]
            )
            for method in METHODS
        }
        changed = (
            sum(a != b for a, b in zip(previous.policy, reference.policy))
            if reference.status != "singular"
            else None
        )
        sequence_changed = signature(previous) != signature(reference)
        requests.append(
            {
                "deletion_ids": batch,
                "method_order": list(order),
                "methods": records,
                "changed_final_actions": changed,
                "trace_changed": sequence_changed,
            }
        )
        if reference.status != "singular":
            retained = candidate
            previous = reference

    return {
        "regime": case[0],
        "states": case[1],
        "dimension": case[2],
        "actions": case[3],
        "seed": seed,
        "repeat": repeat,
        "feature_counts": feature_counts(problem),
        "preparation_order": list(order),
        "preparation_results": prep_results,
        "preparation_seconds": {
            method: prep_results[method]["seconds"] for method in METHODS
        },
        "requests": requests,
        "gamma": str(problem.gamma),
        "regularization": str(problem.regularization),
        "iterations": problem.iterations,
        "dataset": [asdict(item) for item in logs] if repeat == 0 else None,
        "features": (
            [
                {"state": state, "action": action, "vector": vector}
                for (state, action), vector in problem.features.items()
            ]
            if repeat == 0
            else None
        ),
    }


def decision_totals(runs, method):
    records = [
        request["methods"][method]
        for run_record in runs
        for request in run_record["requests"]
    ]
    decision = [record["decision_work"] for record in records if record["decision_work"]]
    fields = (
        "margin_groups_built",
        "groups_certified",
        "groups_scored",
        "decisions_changed_from_anchor",
    )
    return {
        "action_scores": sum(record["work"]["action_scores"] for record in records),
        **{
            field: sum(item[field] for item in decision) if decision else None
            for field in fields
        },
    }


def memory_pass(case, method, nrequests):
    problem, logs, batches = dataset(*case, 0, nrequests)
    gc.collect()
    tracemalloc.start()
    try:
        state, result = initialize(method, problem, logs)
        retained = logs
        current, lifecycle_peak = tracemalloc.get_traced_memory()
        max_request_increment = 0
        for batch in batches:
            del result
            before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            if method == "scratch_retrain":
                removed = set(batch)
                retained = tuple(
                    item for item in retained if item.trajectory_id not in removed
                )
                result = scratch_lspi(problem, retained)
            else:
                result = state.delete_and_replay(batch)
            current, peak = tracemalloc.get_traced_memory()
            lifecycle_peak = max(lifecycle_peak, peak)
            max_request_increment = max(max_request_increment, peak - before)
        return {
            "regime": case[0],
            "states": case[1],
            "dimension": case[2],
            "actions": case[3],
            "method": method,
            "requests": len(batches),
            "lifecycle_peak_python_bytes": lifecycle_peak,
            "end_python_bytes": current,
            "max_request_increment_python_bytes": max_request_increment,
        }
    finally:
        tracemalloc.stop()


def summarize(report):
    runs = report["runs"]
    requests = [request for run in runs for request in run["requests"]]
    executions = len(requests) * len(METHODS)
    mismatches = sum(
        not record["full_trace_match"]
        for request in requests
        for record in request["methods"].values()
    )
    groups = sorted(
        {(run["regime"], run["states"], run["dimension"], run["actions"]) for run in runs}
    )
    lines = [
        "# Multi-action fixed-feature LSPI deletion controls",
        "",
        f"Source `{report['git_commit']}`; clean at launch: {not report['git_dirty']}.",
        "",
        f"**{len(runs)} runs, {len(requests)} requests, {executions} method executions; {mismatches} full-trace mismatches.**",
        "",
        "Every method is compared with direct retained-only LSPI. Lifecycle time includes initial learning, indexes, and the full sequential deletion workload. Reference comparison and serialization are outside competitor timings.",
        "",
        "## Lifecycle median milliseconds",
        "",
        "| Regime | States | Dimension | Actions | " + " | ".join(METHODS) + " |",
        "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in METHODS) + " |",
    ]
    for key in groups:
        selected = [
            run
            for run in runs
            if (run["regime"], run["states"], run["dimension"], run["actions"]) == key
        ]
        values = []
        for method in METHODS:
            per_run = [
                run["preparation_seconds"][method]
                + sum(request["methods"][method]["seconds"] for request in run["requests"])
                for run in selected
            ]
            values.append(1000 * median(per_run))
        lines.append(
            "| " + " | ".join(map(str, key)) + " | "
            + " | ".join(f"{value:.3f}" for value in values) + " |"
        )

    lines += [
        "",
        "## Changes and decision work",
        "",
        "| Regime | States | Dimension | Changed final policies / requests | Changed full traces | Method | Action scores | Certified groups | Scored groups |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key in groups:
        selected = [
            run
            for run in runs
            if (run["regime"], run["states"], run["dimension"], run["actions"]) == key
        ]
        selected_requests = [request for run in selected for request in run["requests"]]
        changes = sum(bool(request["changed_final_actions"]) for request in selected_requests)
        trace_changes = sum(request["trace_changed"] for request in selected_requests)
        for method in ("inverse_deduplicated", "selective_repair", "shared_anchor_repair"):
            totals = decision_totals(selected, method)
            certified = totals["groups_certified"]
            scored = totals["groups_scored"]
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {changes} / {len(selected_requests)} | "
                f"{trace_changes} | {method} | {totals['action_scores']} | "
                f"{certified if certified is not None else '—'} | "
                f"{scored if scored is not None else '—'} |"
            )

    if report.get("memory"):
        lines += [
            "",
            "## Separate Python allocation pass",
            "",
            "| Regime | States | Dimension | Method | Peak KiB | End KiB | Max request increment KiB |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for item in report["memory"]:
            lines.append(
                f"| {item['regime']} | {item['states']} | {item['dimension']} | {item['method']} | "
                f"{item['lifecycle_peak_python_bytes']/1024:.1f} | "
                f"{item['end_python_bytes']/1024:.1f} | "
                f"{item['max_request_increment_python_bytes']/1024:.1f} |"
            )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- These are synthetic fixed features with three actions, eight or sixteen dimensions, five scheduled LSPI iterations, and exact rational arithmetic.",
        "- Feature vectors and ordered action rows are unique and no within-state action pair is an exact sign opposite. This removes the earlier alias and opposite-action shortcuts but does not make the task realistic.",
        "- A full trace includes matrices, right-hand sides, weights, policies and status, so it can change after a deletion even when the final policy is unchanged.",
        "- Timing repetitions, when requested, reuse a generated dataset. The bounded archive uses one timing execution for each of two independently generated seeds; medians are descriptive.",
        "- Full-trace equality covers matrices, right-hand sides, weights, policies and status for the declared learner. It does not imply privacy or deletion of historical anchors.",
        "- No universal speedup, neural-RL result, floating-point stability result, or publication novelty follows from this experiment.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/lspi_multiaction.json.gz"
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.requests <= 40:
        parser.error("repeats must be positive and requests must be 1..40")

    if args.summarize:
        report = json.loads(gzip.decompress(args.summarize.read_bytes()))
        output = args.summarize
    else:
        cases = (CASES[0], CASES[-1]) if args.smoke else CASES
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version,
            "platform": platform.platform(),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "methods": METHODS,
            "runs": [],
            "memory": [],
            "source_sha256": {
                str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted((ROOT / "src").rglob("*.py"))
                + sorted((ROOT / "scripts").glob("*.py"))
            },
        }
        for case in cases:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    report["runs"].append(run(case, seed, repeat, args.requests))
                    print(f"{case} seed={seed} repeat={repeat}: done", flush=True)
        if args.memory:
            report["memory"] = [
                memory_pass(case, method, args.requests)
                for case in cases
                for method in METHODS
            ]
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(
            gzip.compress(json.dumps(report, indent=2, default=str).encode(), mtime=0)
        )
        print(f"Saved {len(report['runs'])} runs: {output}")

    summary = summarize(report)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary + "\n", encoding="utf-8")
        print(f"Saved summary: {args.summary}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
