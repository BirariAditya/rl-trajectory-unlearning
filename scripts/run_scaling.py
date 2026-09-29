"""Repeated sparse/dense baseline runs; memory is measured in separate passes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_unlearning.experiments import METHODS, generated_case, run_dataset
from run_baselines import git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "scaling.json")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two timing repetitions required")
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                  python=sys.version, platform=platform.platform(), device="CPU",
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "src").rglob("*.py")) +
                                 sorted((ROOT / "scripts").glob("*.py"))},
                  timing_note="Randomized method order; medians describe this instrumented synthetic run, not general speedups. Memory measured separately; Python request allocation excludes resident model and native allocations.",
                  runs=[])
    failures = 0
    for graph in ("sparse", "dense"):
        for size in args.sizes:
            for seed in args.seeds:
                arguments = generated_case(graph, size, seed)
                # Untimed warm-up; no checkpoint from this run is reused.
                run_dataset(graph, seed, *arguments, order_seed=seed)
                repetitions = []
                for repeat in range(args.repeats):
                    result = run_dataset(graph, seed, *arguments, order_seed=seed * 100 + repeat)
                    dataset = result.pop("dataset")
                    repetitions.append(result)
                memory = run_dataset(graph, seed, *arguments, profile_memory=True)
                for result in repetitions + [memory]:
                    failures += sum(r["policy_disagreement"] != 0 or not r["converged"]
                                    for r in result["records"] if r["method"] != "noop")
                summary = {}
                for method in METHODS:
                    times = [next(r["request_seconds"] for r in run["records"] if r["method"] == method)
                             for run in repetitions]
                    summary[method] = dict(median_seconds=statistics.median(times),
                                           min_seconds=min(times), max_seconds=max(times))
                report["runs"].append(dict(graph=graph, size=size, seed=seed, dataset=dataset,
                                           timing_runs=repetitions, memory_records=memory["records"],
                                           summary=summary))
                print(f"{graph} size={size} seed={seed}: complete", flush=True)
    report["nonnoop_policy_or_convergence_failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(report['runs'])} settings; failures={failures}: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
