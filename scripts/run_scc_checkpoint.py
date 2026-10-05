"""Matched dense/SCC exact solvers, including weak bounds and anchor invalidation."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

from run_local_checkpoint import ROOT, make_logs
from run_baselines import git
from rl_unlearning import Trajectory, Transition
from rl_unlearning.local_policy_repair import LocalPolicyRepair


METHODS = ("local_dense", "pruned_dense", "warm_dense", "scratch_dense",
           "local_scc", "pruned_scc", "warm_scc", "scratch_scc")
REGIMES = ("modular", "irrelevant_bridges", "weak_bridges", "anchor_invalidation", "block_deletion", "dense")


def dataset(regime, size, seed):
    base = "irrelevant_bridges" if regime in ("weak_bridges", "anchor_invalidation") else regime
    if regime == "block_deletion":
        base = "modular"
    states, actions, gamma, logs, requests = make_logs(base, size, seed)
    if regime == "weak_bridges":
        logs = tuple(replace(t, transitions=(replace(t.transitions[0], reward=-1),) + t.transitions[1:])
                     if t.trajectory_id.startswith("bridge-") else t for t in logs)
    elif regime == "anchor_invalidation":
        logs += (Trajectory("anchor-high", (Transition(0, 1, .6),)),)
        requests = [["anchor-high"]] + requests
    elif regime == "block_deletion":
        middle = max(1, len(requests) // 2)
        requests = [[r[0] for r in requests[:middle]], [r[0] for r in requests[middle:]]]
        requests = [r for r in requests if r]
    return states, actions, gamma, logs, requests


def plan_counts(plan):
    return dict(action_evaluations=plan.action_evaluations,
                linear_system_sizes=plan.linear_system_sizes,
                graph_rows_scanned=plan.graph_rows_scanned,
                graph_successors_scanned=plan.graph_successors_scanned,
                component_sizes=plan.component_sizes)


def run(regime, size, seed, repeat):
    states, actions, gamma, logs, requests = dataset(regime, size, seed)
    rng = random.Random(seed * 1000 + repeat)
    engines, preparation, preparation_work = {}, {}, {}
    order = [m for m in METHODS if not m.startswith("scratch")]
    rng.shuffle(order)
    preparation_order = list(order)
    for name in order:
        started = perf_counter()
        engine = LocalPolicyRepair(states, actions, gamma, logs, solver=name.split("_")[-1],
                                   anchor_policy=dict.fromkeys(states, 1) if name.startswith("pruned") else None)
        preparation[name] = perf_counter() - started
        engines[name] = engine
        preparation_work[name] = plan_counts(engine.preparation_plan)
    retained, records = logs, []
    for step, request in enumerate(requests):
        order = list(METHODS)
        rng.shuffle(order)
        results, policies, values = {}, {}, {}
        for name in order:
            started = perf_counter()
            if name.startswith("scratch"):
                engine = LocalPolicyRepair(states, actions, gamma,
                            (t for t in retained if t.trajectory_id not in request), solver=name.split("_")[-1])
                seconds = perf_counter() - started
                record = plan_counts(engine.preparation_plan)
            else:
                engine = engines[name]
                result = engine.delete_and_repair(request, force_full=name.startswith("warm"))
                seconds = perf_counter() - started
                record = {key: getattr(result, key) for key in (
                    "used_full_fallback", "transitions_read", "successor_entries_copied", "reverse_edges_visited",
                    "index_edges_removed", "index_edges_added", "index_rows_rebuilt", "action_evaluations",
                    "linear_system_sizes", "pruned_action_rows", "pruning_disabled", "pruning_rows_checked",
                    "graph_rows_scanned", "graph_successors_scanned", "component_sizes")}
                record.update(affected_states=len(result.affected_states), changed_actions=len(result.changed_actions))
            record["seconds"] = seconds
            # Common output materialization and correctness comparison are offline evaluation overhead.
            policies[name], values[name] = dict(engine.policy), dict(engine.values)
            results[name] = record
        for name, record in results.items():
            record["policy_disagreement"] = sum(policies[name][s] != policies["scratch_dense"][s] for s in states)
            record["exact_value_match"] = values[name] == values["scratch_dense"]
        records.append(dict(step=step, deletion_ids=request, method_order=order, methods=results))
        retained = tuple(t for t in retained if t.trajectory_id not in request)
    return dict(regime=regime, size=size, seed=seed, repeat=repeat,
                preparation_seconds=preparation, preparation_work=preparation_work,
                preparation_order=preparation_order, requests=records,
                dataset=[asdict(t) for t in logs] if repeat == 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--output", type=Path, default=ROOT / "results/scc_checkpoint.json.gz")
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 4 or n % 4 for n in args.sizes):
        parser.error("positive repeats and state sizes divisible by four required")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), git_commit=git("rev-parse", "HEAD"),
                  git_dirty=bool(git("status", "--porcelain")), python=sys.version,
                  platform=platform.platform(), device="CPU", arguments={**vars(args), "output": str(args.output)},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))},
                  scope="Same rational learner and dense inner solver. SCC graph rebuilding is timed. Shared LocalPolicyRepair scaffold includes reverse indexing for warm/scratch controls. Pruned SCC uses full action edges inside the selected block, not action-filtered SCCs. Not a reproduction of TVI/FTVI or a mature optimized planner. Output snapshots/equality are offline evaluation overhead. Anchor preparation remains a dense exact solve.", runs=[])
    failures = 0
    for regime in args.regimes:
        for size in args.sizes:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    result = run(regime, size, seed, repeat)
                    report["runs"].append(result)
                    failures += sum(bool(m["policy_disagreement"] or not m["exact_value_match"])
                                    for r in result["requests"] for m in r["methods"].values())
                    print(f"{regime} n={size} seed={seed} repeat={repeat}: done", flush=True)
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))
    print(f"Saved {len(report['runs'])} runs; failures={failures}: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
