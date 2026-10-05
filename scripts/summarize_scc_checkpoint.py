"""Summarize matched SCC/dense planning and sequential deletion costs."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from run_scc_checkpoint import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/scc_checkpoint_summary.md"))
    args = parser.parse_args()
    report = json.loads(gzip.decompress(args.report.read_bytes()))
    runs = report["runs"]
    requests = sum(len(r["requests"]) for r in runs)
    executions = sum(len(q["methods"]) for r in runs for q in r["requests"])
    groups = sorted({(r["regime"], r["size"]) for r in runs})
    lines = ["# Matched SCC planning checkpoint", "",
             f"Source: `{report['git_commit']}`; clean at execution: `{not report['git_dirty']}`.", "",
             f"**{len(runs)} runs, {requests} sequential requests, {executions} method executions, {report['failures']} failures.**",
             "All methods checked for exact policy and value equality against scratch dense rational PI.", "",
             "## Request cost (milliseconds)", "",
             "Median across runs of each run's median request time. Scratch includes rebuilding retained statistics and its index. Whole-method timings include graph construction and index maintenance.", "",
             "| Regime | States | " + " | ".join(METHODS) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in METHODS) + " |"]
    for regime, size in groups:
        group = [r for r in runs if (r["regime"], r["size"]) == (regime, size)]
        times = [1000 * median(median(q["methods"][m]["seconds"] for q in r["requests"]) for r in group)
                 for m in METHODS]
        lines.append(f"| {regime} | {size} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
    prepared = [m for m in METHODS if not m.startswith("scratch")]
    lines += ["", "## Preparation plus full deletion sequence (milliseconds)", "",
              "Median of preparation plus every request, including the anchor-invalidating request where applicable. Preparation includes original training and retained indexes; pruned variants also include a dense exact anchor evaluation. Scratch is reported only as a per-request rebuild above.", "",
              "| Regime | States | Requests | " + " | ".join(prepared) + " |",
              "| --- | ---: | ---: | " + " | ".join("---:" for _ in prepared) + " |"]
    for regime, size in groups:
        group = [r for r in runs if (r["regime"], r["size"]) == (regime, size)]
        totals = [1000 * median(r["preparation_seconds"][m] + sum(q["methods"][m]["seconds"] for q in r["requests"]) for r in group)
                  for m in prepared]
        lines.append(f"| {regime} | {size} | {len(group[0]['requests'])} | " + " | ".join(f"{t:.3f}" for t in totals) + " |")
    lines += ["", "## Mechanism checks", "",
              "Affected sizes are reported closure sizes before any forced global fallback; an anchor-invalidating request can report one affected source while solving the full model. Largest SCC describes actual component solves. Pruned methods retain every action inside the solve block; they do not apply FTVI action elimination to its internal SCC graph.", "",
              "| Regime | States | Local SCC affected sizes | Pruned SCC affected sizes | Warm SCC largest components | Pruned SCC largest components | Pruning invalidations |",
              "| --- | ---: | --- | --- | --- | --- | ---: |"]
    for regime, size in groups:
        group = [r for r in runs if (r["regime"], r["size"]) == (regime, size)]
        queries = [q for r in group for q in r["requests"]]
        affected = [sorted({q["methods"][m]["affected_states"] for q in queries}) for m in ("local_scc", "pruned_scc")]
        largest = [sorted({max(q["methods"][m]["component_sizes"], default=0) for q in queries}) for m in ("warm_scc", "pruned_scc")]
        invalid = sum(q["methods"]["pruned_scc"]["pruning_disabled"] for q in queries)
        lines.append(f"| {regime} | {size} | {affected[0]} | {affected[1]} | {largest[0]} | {largest[1]} | {invalid} |")
    lines += ["", "## Timed anchor invalidation", "",
              "Median first-request time in anchor_invalidation. Later requests are included in the full-sequence table and run with pruning disabled.", "",
              "| States | Pruned dense | Warm dense | Pruned SCC | Warm SCC |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for size in sorted({r["size"] for r in runs if r["regime"] == "anchor_invalidation"}):
        group = [r for r in runs if r["regime"] == "anchor_invalidation" and r["size"] == size]
        times = [1000 * median(r["requests"][0]["methods"][m]["seconds"] for r in group)
                 for m in ("pruned_dense", "warm_dense", "pruned_scc", "warm_scc")]
        lines.append(f"| {size} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
    lines += ["", "## Scope", "",
              f"- Seeds: {sorted({r['seed'] for r in runs})}; repeats: {sorted({r['repeat'] for r in runs})}. Repeats reuse each seed's dataset and sequence, so they are timing replicates. Synthetic models only; medians are descriptive, without statistical significance claims.",
              "- Matched arithmetic, inner exact solver, data, and tie order isolate a graph-decomposition ablation. This is our implementation of classical SCC planning, not an optimized external solver or a TVI/FTVI reproduction.",
              "- All warm/scratch methods share the reverse-index scaffolding used by local repair. A standalone global planner could avoid that extra overhead; this limits speedup interpretation.",
              "- Whole-call wall times include preparation, rational arithmetic, SCC reconstruction, invalidation, and index work as applicable. Work counters are partial diagnostics; anchor evaluation and every Python operation are not individually counted.",
              "- Policy/value snapshots and equality checks are common offline evaluation overhead, excluded from request latency. No neural, privacy, universal locality, or novelty claim is made.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
