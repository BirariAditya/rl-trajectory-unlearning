"""Describe complete costs for lean global and local deletion planning."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from run_global_checkpoint import METHODS, PREPARED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/global_checkpoint_summary.md"))
    args = parser.parse_args()
    report = json.loads(gzip.decompress(args.report.read_bytes()))
    runs = report["runs"]
    requests = sum(len(r["requests"]) for r in runs)
    executions = sum(len(q["methods"]) for r in runs for q in r["requests"])
    groups = sorted({(r["regime"], r["size"]) for r in runs})
    lines = ["# Lean global planning and elimination checkpoint", "",
             f"Source: `{report['git_commit']}`; clean at launch: `{not report['git_dirty']}`.", "",
             f"**{len(runs)} runs, {requests} sequential requests, {executions} method executions, {report['failures']} failures.**",
             "Every method checked for exact policy and value agreement with lean scratch rational SCC planning.", "",
             "## Request cost (milliseconds)", "",
             "Median across runs of per-run median request time. Scratch reconstructs retained statistics; other methods subtract deleted records. All graph reconstruction, pruning scans and anchor refresh are charged. Snapshots/equality checks are excluded equally as offline evaluation overhead.", "",
             "| Regime | States | " + " | ".join(METHODS) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in METHODS) + " |"]
    for regime, size in groups:
        group = [r for r in runs if (r["regime"], r["size"]) == (regime, size)]
        times = [1000 * median(median(q["methods"][m]["seconds"] for q in r["requests"]) for r in group) for m in METHODS]
        lines.append(f"| {regime} | {size} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
    lines += ["", "## Preparation plus all requests (milliseconds)", "",
              "Median full-run cost. Every method is charged original full-data training once; scratch controls additionally retrain on retained logs at every request. Includes preparation, every request and anchor invalidation. Legacy pruned local and global elimination use different anchor-evaluation/preparation algorithms; this is a method comparison, not a pure locality ablation.", "",
              "| Regime | States | Requests | " + " | ".join(PREPARED) + " |",
              "| --- | ---: | ---: | " + " | ".join("---:" for _ in PREPARED) + " |"]
    for regime, size in groups:
        group = [r for r in runs if (r["regime"], r["size"]) == (regime, size)]
        times = [1000 * median(r["preparation_seconds"][m] + sum(q["methods"][m]["seconds"] for q in r["requests"]) for r in group) for m in PREPARED]
        lines.append(f"| {regime} | {size} | {len(group[0]['requests'])} | " + " | ".join(f"{t:.3f}" for t in times) + " |")
    lines += ["", "## Elimination mechanism", "",
              "Sizes are the largest SCC in each actual request solve. Anchor refreshes count warm eliminate_scc only (initial preparation excluded).", "",
              "| Regime | States | Lean warm SCC sizes | Eliminated SCC sizes | Pruned action counts | Anchor refreshes |",
              "| --- | ---: | --- | --- | --- | ---: |"]
    for regime, size in groups:
        group = [r for r in runs if (r["regime"], r["size"]) == (regime, size)]
        queries = [q for r in group for q in r["requests"]]
        sizes = [sorted({max(q["methods"][m]["component_sizes"], default=0) for q in queries})
                 for m in ("lean_warm_scc", "eliminate_scc")]
        pruned = sorted({q["methods"]["eliminate_scc"]["pruned_action_rows"] for q in queries})
        refresh = sum(q["methods"]["eliminate_scc"]["anchor_refreshed"] for q in queries)
        lines.append(f"| {regime} | {size} | {sizes[0]} | {sizes[1]} | {pruned} | {refresh} |")
    lines += ["", "## Interpretation limits", "",
              f"- Dataset seeds: {sorted({r['seed'] for r in runs})}; timing repeats: {sorted({r['repeat'] for r in runs})}. Repeats reuse datasets. Synthetic medians only, no significance or universal-speedup claim.",
              "- Lean global controls have no repair reverse index. Global elimination is classical strict-bound pruning plus SCC planning, not a new algorithm or a reproduction of FTVI's full heuristic search.",
              "- Global pruning rescans all action rows after every request; its initial reward upper bound stays conservative after deletion. Scratch elimination can get a tighter bound from retained rewards. Anchor evaluation uses a single-action SCC graph; refreshed anchor values are charged.",
              "- Legacy local pruning uses dense anchor preparation, keeps all actions inside the selected solve block, and disables pruning on anchor changes. Whole-method differences combine these choices with locality.",
              "- Dense/SCC solvers and statistics helpers share code. Tiny enumeration tests supply an independent reference; this benchmark's large-model equality is not an independent theorem.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
