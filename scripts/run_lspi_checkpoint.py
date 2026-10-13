"""Complete-cost LSPI replay versus scratch and cached-statistic reconstruction."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_baselines import git
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, LSPIReplayState, scratch_lspi
from rl_unlearning.lspi_baselines import LSPICachedState, LSPIMemoizedState

METHODS = ("scratch", "cached", "cached_memo", "target_replay")
CASES = (("switch", 1, 4), ("switch", 32, 4), ("switch", 128, 4),
         ("shared", 16, 4), ("shared", 64, 4), ("shared", 256, 4), ("shared", 64, 8))


def dataset(regime, count, dimension, seed):
    if regime == "switch":
        states, actions = (0, 1), (0, 1)
        pairs = [(s, a) for s in states for a in actions]
        features = {pair: tuple(F(i == j) for j in range(4)) for i, pair in enumerate(pairs)}
        problem = LSPIProblem(states, actions, features, dict.fromkeys(states, 0), iterations=4)
        logs = []
        for i in range(count):
            logs.extend((Trajectory(f"xa-{i}", (Transition(0, 0, 0, 1), Transition(1, 0, 0))),
                         Trajectory(f"xb-{i}", (Transition(0, 1, F(1, 16)),)),
                         Trajectory(f"zb-{i}", (Transition(1, 1, F(1, 4)),)),
                         Trajectory(f"forget-{i}", (Transition(1, 0, 1),))))
        # This fixture is deterministic: different seeds shuffle deletion order
        # within a batch only, not create independent scientific datasets.
        forget = [f"forget-{i}" for i in range(count)]
        random.Random(seed).shuffle(forget)
        requests = [forget, [f"xb-{i}" for i in range(max(1, count // 4))]]
    else:
        states, actions = tuple(range(6)), (0, 1)
        feature_rng = random.Random(981 + dimension)  # Independent of data seed/log.
        features = {(s, a): tuple(F(feature_rng.randrange(-2, 3), 2) for _ in range(dimension))
                    for s in states for a in actions}
        problem = LSPIProblem(states, actions, features, dict.fromkeys(states, 0),
                              gamma=F(1, 2), regularization=F(1), iterations=4)
        rng, logs = random.Random(seed), []
        for i in range(count):
            state, successor = rng.randrange(6), rng.randrange(6)
            logs.append(Trajectory(str(i), (
                Transition(state, rng.randrange(2), F(rng.randrange(-4, 5), 4), successor),
                Transition(successor, rng.randrange(2), F(rng.randrange(-4, 5), 4)))))
        ids = list(range(count))
        rng.shuffle(ids)
        batch = max(1, count // 16)
        requests = [[str(i) for i in ids[start:start + batch]] for start in (0, batch, 2 * batch)]
    return problem, tuple(logs), requests


def signature(result):
    return result.iterates, result.policy, result.weights, result.status, result.repeat_start, result.repeat_length


def record(result):
    switches = sum(sum(a != b for a, b in zip(t.policy_in, t.policy_out))
                   for t in result.iterates if t.policy_out is not None)
    return dict(status=result.status, policy=result.policy, work=asdict(result.work),
                policy_switches=switches, iterations=len(result.iterates),
                trace_sha256=hashlib.sha256(repr(signature(result)).encode()).hexdigest())


def run(regime, count, dimension, seed, repeat):
    problem, logs, requests = dataset(regime, count, dimension, seed)
    rng, engines, preparation = random.Random(seed * 1000 + repeat), {}, {}
    order = list(METHODS)
    rng.shuffle(order)
    initial = {}
    for method in order:
        started = perf_counter()
        if method == "scratch":
            result = scratch_lspi(problem, logs)
        else:
            cls = {"cached": LSPICachedState, "cached_memo": LSPIMemoizedState,
                   "target_replay": LSPIReplayState}[method]
            engines[method] = cls(problem, logs)
            result = engines[method].replay()
        preparation[method] = perf_counter() - started
        initial[method] = result
    if any(signature(result) != signature(initial["scratch"]) for result in initial.values()):
        raise AssertionError("initial learners disagree")
    retained, records = logs, []
    for step, request in enumerate(requests):
        rng.shuffle(order)
        results, timings = {}, {}
        for method in order:
            started = perf_counter()
            if method == "scratch":
                removed = set(request)
                result = scratch_lspi(problem, (t for t in retained if t.trajectory_id not in removed))
            else:
                result = engines[method].delete_and_replay(request)
            timings[method] = perf_counter() - started
            results[method] = result
        methods = {m: dict(**record(r), seconds=timings[m],
                           exact_trace_match=signature(r) == signature(results["scratch"]))
                   for m, r in results.items()}
        records.append(dict(step=step, deletion_ids=request, method_order=list(order), methods=methods))
        if results["scratch"].status != "singular":
            removed = set(request)
            retained = tuple(t for t in retained if t.trajectory_id not in removed)
    return dict(regime=regime, count=count, dimension=dimension, seed=seed, repeat=repeat,
                trajectories=len(logs), transitions=sum(len(t.transitions) for t in logs),
                preparation_seconds=preparation, preparation_results={m: record(r) for m, r in initial.items()},
                requests=records, states=problem.states, actions=problem.actions,
                gamma=str(problem.gamma), regularization=str(problem.regularization), iterations=problem.iterations,
                initial_policy=dict(problem.initial_policy),
                features=[dict(state=s, action=a, vector=problem.features[s, a]) for s in problem.states for a in problem.actions],
                dataset=[asdict(t) for t in logs] if repeat == 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lspi_checkpoint.json.gz")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    cases = (CASES[0], CASES[3]) if args.smoke else CASES
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), git_commit=git("rev-parse", "HEAD"),
                  git_dirty=bool(git("status", "--porcelain")), python=sys.version, platform=platform.platform(),
                  arguments={**vars(args), "output": str(args.output)},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))},
                  scope="Exact finite-schedule LSPI; scratch vs cached C/b/G reconstruction vs within-request policy-evaluation memoization vs target corrections. Dense exact solves (memo reuses repeated policies) and full policy scoring. Timed methods include trace creation; equality/hash/serialization excluded equally. Preparation includes initial full-data learning for every method. No inverse-update, measured memory or numerical GPU benchmark. Switch fixture seeds only change within-batch order.", runs=[])
    for case in cases:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                report["runs"].append(run(*case, seed, repeat))
                print(f"{case} seed={seed} repeat={repeat}: done", flush=True)
    report["failures"] = sum(not m["exact_trace_match"] for r in report["runs"] for q in r["requests"] for m in q["methods"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(report, indent=2, default=str).encode(), mtime=0))
    print(f"Saved {len(report['runs'])} runs; trace mismatches={report['failures']}: {args.output}")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
