"""Render a compact, reproducible Markdown summary of scaling JSON (or gzip)."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/benchmarks/scaling_summary.md"))
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(gzip.decompress(raw) if args.report.suffix == ".gz" else raw)
    runs = report["runs"]
    timed = [r for setting in runs for run in setting["timing_runs"] for r in run["records"]]
    memory = [r for setting in runs for r in setting["memory_records"]]
    max_error = max(r["max_q_error"] for r in timed if r["method"] != "noop")
    lines = ["# Sparse/dense scaling pilot", "",
             f"Source revision: `{report['git_commit']}`; dirty at execution: `{report['git_dirty']}`.", "",
             f"{len(runs)} settings; {len(timed)} timed method runs and {len(memory)} separate memory-pass runs. "
             f"Non-noop policy/convergence failures: **{report['nonnoop_policy_or_convergence_failures']}**. "
             f"Maximum updated Q error versus the recorded reference: `{max_error:.3g}`.", "",
             "Timings below are milliseconds: median of per-seed medians. Each seed's repeat distribution is preserved in the report. "
             "These instrumented CPU results use acyclic synthetic logs and do not establish general speedups or statistical significance.", "",
             "| Graph | States | Scratch | Warm start | Prioritized sweeping | Global certified repair |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for graph, size in sorted({(s["graph"], s["size"]) for s in runs}):
        group = [s for s in runs if s["graph"] == graph and s["size"] == size]
        values = [median(s["summary"][method]["median_seconds"] for s in group) * 1000
                  for method in ("scratch", "warm_start", "prioritized_sweeping", "certified_repair")]
        lines.append(f"| {graph} | {size} | " + " | ".join(f"{x:.3f}" for x in values) + " |")
    lines += ["", "## Interpretation and scope", "",
              "- Compare local queue methods with warm-started planning in every graph regime. Dense successor support can make queue propagation slower even when the final policy is correct.",
              "- `certified_repair` is the existing global-residual method. The new rational restricted certificate is not part of this timing matrix; no performance claim for that prototype follows from this report.",
              "- Models beyond the exact enumeration limit use tightly converged binary64 value iteration; inspect recorded residuals and action gaps. This is numerical fidelity, not a rational proof for the larger MDPs.",
              "- Memory passes measure peak Python allocation during the request, excluding the resident model, original training, and native allocations. They are separate from the timing medians. Each timing run records preparation and original-training costs separately.",
              "- Further gates: cyclic MDPs, sequential requests, more seeds/repetitions, warm-start policy iteration, and restricted-certificate preparation amortization.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
