"""Lean global planning and classical bound-elimination controls for deletion."""

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

from run_scc_checkpoint import ROOT, REGIMES, dataset, plan_counts
from run_baselines import git
from rl_unlearning.global_planner import ExactGlobalPlanner
from rl_unlearning.local_policy_repair import LocalPolicyRepair


METHODS = ("local_scc", "pruned_local_scc", "lean_warm_dense", "lean_warm_scc",
           "eliminate_scc", "lean_scratch_scc", "eliminate_scratch_scc")
# A lifecycle begins with original full-data training for every method, even
# when the deletion handler later discards that policy and retrains from scratch.
PREPARED = METHODS


def make_engine(name, states, actions, gamma, logs):
    anchor = dict.fromkeys(states, 1) if name.startswith(("pruned", "eliminate")) else None
    if "local" in name:
        return LocalPolicyRepair(states, actions, gamma, logs, solver="scc", anchor_policy=anchor)
    return ExactGlobalPlanner(states, actions, gamma, logs,
                              solver="dense" if name.endswith("dense") else "scc", anchor_policy=anchor)


def global_counts(result):
    return dict(**plan_counts(result.plan), anchor_plan=plan_counts(result.anchor_plan) if result.anchor_plan else None,
                transitions_read=result.transitions_read, successor_entries_copied=result.successor_entries_copied,
                pruning_rows_checked=result.pruning_rows_checked, pruned_action_rows=result.pruned_action_rows,
                anchor_refreshed=result.anchor_refreshed)


def run(regime, size, seed, repeat):
    states, actions, gamma, logs, requests = dataset(regime, size, seed)
    rng = random.Random(seed * 1000 + repeat)
    engines, preparation, preparation_work = {}, {}, {}
    order = list(PREPARED)
    rng.shuffle(order)
    preparation_order = list(order)
    for name in order:
        started = perf_counter()
        engine = make_engine(name, states, actions, gamma, logs)
        preparation[name] = perf_counter() - started
        if "scratch" not in name:
            engines[name] = engine
        preparation_work[name] = (plan_counts(engine.preparation_plan) if "local" in name
                                  else global_counts(engine.preparation_result))
    retained, records = logs, []
    for step, request in enumerate(requests):
        order = list(METHODS)
        rng.shuffle(order)
        results, policies, values = {}, {}, {}
        for name in order:
            started = perf_counter()
            if "scratch" in name:
                engine = make_engine(name, states, actions, gamma,
                                     (t for t in retained if t.trajectory_id not in request))
                seconds = perf_counter() - started
                record = global_counts(engine.preparation_result)
            elif "local" in name:
                engine = engines[name]
                result = engine.delete_and_repair(request)
                seconds = perf_counter() - started
                record = {key: getattr(result, key) for key in (
                    "used_full_fallback", "transitions_read", "successor_entries_copied", "reverse_edges_visited",
                    "index_edges_removed", "index_edges_added", "index_rows_rebuilt", "action_evaluations",
                    "linear_system_sizes", "pruned_action_rows", "pruning_disabled", "pruning_rows_checked",
                    "graph_rows_scanned", "graph_successors_scanned", "component_sizes")}
                record.update(affected_states=len(result.affected_states), changed_actions=len(result.changed_actions))
            else:
                engine = engines[name]
                result = engine.delete_and_replan(request)
                seconds = perf_counter() - started
                record = global_counts(result)
            record["seconds"] = seconds
            policies[name], values[name] = dict(engine.policy), dict(engine.values)
            results[name] = record
        for name, record in results.items():
            record["policy_disagreement"] = sum(policies[name][s] != policies["lean_scratch_scc"][s] for s in states)
            record["exact_value_match"] = values[name] == values["lean_scratch_scc"]
        records.append(dict(step=step, deletion_ids=request, method_order=order, methods=results))
        retained = tuple(t for t in retained if t.trajectory_id not in request)
    return dict(regime=regime, size=size, seed=seed, repeat=repeat,
                preparation_seconds=preparation, preparation_work=preparation_work,
                preparation_order=preparation_order, requests=records,
                dataset=[asdict(t) for t in logs] if repeat == 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--output", type=Path, default=ROOT / "results/global_checkpoint.json.gz")
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 4 or n % 4 for n in args.sizes):
        parser.error("positive repeats and state sizes divisible by four required")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), git_commit=git("rev-parse", "HEAD"),
                  git_dirty=bool(git("status", "--porcelain")), python=sys.version, platform=platform.platform(),
                  device="CPU", arguments={**vars(args), "output": str(args.output)},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))},
                  scope="Exact rational learner. Lean global controls construct no repair reverse index. Global elimination scans all rows, uses SCC anchor evaluation, restores actions and refreshes changed anchors. Legacy local pruning has dense anchor preparation and disables pruning on invalidation; solver preparation also differs. This compares implemented methods, not a single-variable ablation. Not FTVI reproduction. Snapshots/equality excluded equally as offline evaluation overhead.", runs=[])
    failures = 0
    for regime in args.regimes:
        for size in args.sizes:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    result = run(regime, size, seed, repeat)
                    report["runs"].append(result)
                    failures += sum(bool(m["policy_disagreement"] or not m["exact_value_match"])
                                    for q in result["requests"] for m in q["methods"].values())
                    print(f"{regime} n={size} seed={seed} repeat={repeat}: done", flush=True)
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))
    print(f"Saved {len(report['runs'])} runs; failures={failures}: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
