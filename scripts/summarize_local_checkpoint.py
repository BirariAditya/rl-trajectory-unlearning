"""Summarize exact cyclic repair with preparation-inclusive sequence costs."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/local_checkpoint_summary.md"))
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(gzip.decompress(raw) if args.report.suffix == ".gz" else raw)
    runs = report["runs"]
    requests = sum(len(run["requests"]) for run in runs)
    executions = sum(len(r["methods"]) for run in runs for r in run["requests"])
    lines = ["# Exact local repair checkpoint", "",
             f"Source: `{report['git_commit']}`; clean at execution: `{not report['git_dirty']}`.", "",
             f"**{len(runs)} runs, {requests} sequential requests, {executions} method executions, {report['failures']} failures.** "
             "Rational methods were checked for exact policy and value equality against scratch rational PI; warm binary64 VI was checked for policy agreement.", "",
             "## Request medians (milliseconds)", "",
             "| Regime | States | Local exact | Pruned exact | Warm exact PI | Scratch exact PI | Warm float VI |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    groups = sorted({(r["regime"], r["size"]) for r in runs})
    for regime, size in groups:
        group = [r for r in runs if r["regime"] == regime and r["size"] == size]
        times = [1000 * median(q["methods"][m]["seconds"] for r in group for q in r["requests"])
                 for m in ("local_exact", "pruned_exact", "warm_exact", "scratch_exact", "warm_float_vi")]
        lines.append(f"| {regime} | {size} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
    lines += ["", "## Full sequence including preparation (milliseconds)", "",
              "Median of (preparation + all requests) for each repeated run. Preparation includes original full rational training, reverse indexing, and anchor evaluation/pruning where used. This avoids treating the pruning index or anchor as free.", "",
              "| Regime | States | Requests/sequence | Local exact | Pruned exact | Warm exact PI | Warm float VI |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for regime, size in groups:
        group = [r for r in runs if r["regime"] == regime and r["size"] == size]
        totals = [1000 * median(r["preparation_seconds"][m] + sum(q["methods"][m]["seconds"] for q in r["requests"])
                               for r in group) for m in ("local_exact", "pruned_exact", "warm_exact", "warm_float_vi")]
        lines.append(f"| {regime} | {size} | {len(group[0]['requests'])} | " + " | ".join(f"{t:.3f}" for t in totals) + " |")
    lines += ["", "## Affected block sizes", "",
              "| Regime | States | Structural closure sizes | Pruned closure sizes |",
              "| --- | ---: | --- | --- |"]
    for regime, size in groups:
        group = [r for r in runs if r["regime"] == regime and r["size"] == size]
        sizes = [sorted({q["methods"][m]["affected_states"] for r in group for q in r["requests"]})
                 for m in ("local_exact", "pruned_exact")]
        lines.append(f"| {regime} | {size} | {sizes[0]} | {sizes[1]} |")
    lines += ["", "## Interpretation", "",
              "- Dominated bridge actions make structural closure global. Safe action pruning can recover component locality while still allowing the chosen optimal action to change.",
              "- Dense admissible influence still forces full blocks. This is an explicit failure of universal locality, not a hidden excluded setting.",
              "- These are constructed 8/16-state cases with three dataset seeds and three repeats. Request samples within a run are correlated. Medians are descriptive; no significance or broad speedup claim is made.",
              "- Exact rational Gaussian elimination is a correctness-oriented solver. Compare with mature sparse/SCC-aware and action-elimination planners before attributing speed gains to unlearning novelty. Binary64 VI solves a different numerical learner and is included for practical context.",
              "- This benchmark preserves the anchor policy. Tests cover invalidation and global fallback; a timed anchor-invalidation/deletion-shift study is still needed.",
              "- No GPU, learned representation, full-checkpoint deletion, or privacy indistinguishability claim follows from this checkpoint.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
