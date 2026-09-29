"""Exact local repair versus scratch/warm exact PI on cyclic sequential requests."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_unlearning import EmpiricalMDP, Trajectory, Transition, value_iteration
from rl_unlearning.local_policy_repair import LocalPolicyRepair
from run_baselines import git


def make_logs(regime, size, seed, block=4):
    if size % block or regime not in ("modular", "irrelevant_bridges", "dense"):
        raise ValueError("size must be a multiple of block; invalid regime")
    rng = random.Random(seed)
    logs = []
    for state in range(size):
        start = state // block * block
        successors = range(size) if regime == "dense" else [start + (state + 1 - start) % block]
        for successor in successors:
            logs.append(Trajectory(f"edge-{state}-{successor}", (
                Transition(state, 0, 0, successor), Transition(successor, 1, .2))))
        if regime == "irrelevant_bridges":
            successor = (start + block) % size
            logs.append(Trajectory(f"bridge-{state}", (
                Transition(state, 2, -1000, successor), Transition(successor, 1, .2))))
        else:
            logs.append(Trajectory(f"bad-{state}", (Transition(state, 2, -100),)))
    requests = []
    for state in range(0, size, block):
        identifier = f"bonus-{state}"
        logs.append(Trajectory(identifier, (Transition(state, 0, round(3 * size + rng.random(), 3)),)))
        requests.append([identifier])
    rng.shuffle(requests)
    return tuple(range(size)), (0, 1, 2), .8, tuple(logs), requests


def run(regime, size, seed, repeat):
    states, actions, gamma, logs, requests = make_logs(regime, size, seed)
    engines, preparation = {}, {}
    for name in ("local_exact", "pruned_exact", "warm_exact"):
        started = perf_counter()
        engines[name] = LocalPolicyRepair(states, actions, gamma, logs,
                          anchor_policy=dict.fromkeys(states, 1) if name == "pruned_exact" else None)
        preparation[name] = perf_counter() - started
    started = perf_counter()
    float_model = EmpiricalMDP(states, actions, gamma, logs)
    float_plan = value_iteration(float_model, tolerance=1e-12)
    preparation["warm_float_vi"] = perf_counter() - started
    retained = logs
    records = []
    rng = random.Random(seed * 1000 + repeat)
    for step, request in enumerate(requests):
        methods = ["local_exact", "pruned_exact", "warm_exact", "scratch_exact", "warm_float_vi"]
        rng.shuffle(methods)
        results, policies, values = {}, {}, {}
        for name in methods:
            started = perf_counter()
            if name == "scratch_exact":
                scratch = LocalPolicyRepair(states, actions, gamma,
                            (t for t in retained if t.trajectory_id not in request))
                seconds = perf_counter() - started
                policies[name], values[name] = dict(scratch.policy), dict(scratch.values)
                results[name] = dict(seconds=seconds, linear_system_sizes=scratch.preparation_plan.linear_system_sizes,
                                     action_evaluations=scratch.preparation_plan.action_evaluations)
            elif name == "warm_float_vi":
                float_model.delete_trajectories(request)
                float_plan = value_iteration(float_model, initial_q=float_plan.q, tolerance=1e-12)
                seconds = perf_counter() - started
                policies[name] = float_plan.policy
                results[name] = dict(seconds=seconds, backups=float_plan.backups,
                                     residual_checks=float_plan.residual_checks,
                                     residual=float_plan.residual)
            else:
                result = engines[name].delete_and_repair(request, force_full=name == "warm_exact")
                seconds = perf_counter() - started
                policies[name], values[name] = dict(result.policy), dict(engines[name].values)
                results[name] = dict(seconds=seconds, affected_states=len(result.affected_states),
                                     changed_actions=len(result.changed_actions),
                                     linear_system_sizes=result.linear_system_sizes,
                                     reverse_edges_visited=result.reverse_edges_visited,
                                     index_edges_removed=result.index_edges_removed,
                                     action_evaluations=result.action_evaluations,
                                     transitions_read=result.transitions_read,
                                     successor_entries_copied=result.successor_entries_copied,
                                     pruned_action_rows=result.pruned_action_rows,
                                     pruning_rows_checked=result.pruning_rows_checked,
                                     pruning_disabled=result.pruning_disabled,
                                     index_rows_rebuilt=result.index_rows_rebuilt,
                                     index_edges_added=result.index_edges_added)
        reference = policies["scratch_exact"]
        for name, record in results.items():
            record["policy_disagreement"] = sum(policies[name][s] != reference[s] for s in states)
            if name in values:
                record["exact_value_match"] = values[name] == values["scratch_exact"]
        records.append(dict(step=step, deletion_ids=request, methods=results))
        retained = tuple(t for t in retained if t.trajectory_id not in request)
    return dict(regime=regime, size=size, seed=seed, repeat=repeat,
                preparation_seconds=preparation, requests=records,
                dataset=[asdict(t) for t in logs] if repeat == 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "local_checkpoint.json.gz")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), git_commit=git("rev-parse", "HEAD"),
                  git_dirty=bool(git("status", "--porcelain")), python=sys.version,
                  platform=platform.platform(), device="CPU",
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))},
                  scope="Rational structural closure baseline, not novel margin pruning. Dense exact arithmetic is expensive. Float VI is a separate numerical learner and included as a practical baseline. Policy snapshots and equality checks are offline evaluation overhead.", runs=[])
    failures = 0
    for regime in ("modular", "irrelevant_bridges", "dense"):
        for size in args.sizes:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    result = run(regime, size, seed, repeat)
                    report["runs"].append(result)
                    for request in result["requests"]:
                        for name, record in request["methods"].items():
                            failures += bool(record["policy_disagreement"] or record.get("exact_value_match") is False)
                    print(f"{regime} n={size} seed={seed} repeat={repeat}: done", flush=True)
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, indent=2).encode()
    args.output.write_bytes(gzip.compress(raw, mtime=0) if args.output.suffix == ".gz" else raw)
    print(f"Saved {len(report['runs'])} runs; failures={failures}: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
