"""Summarize exact LSPI complete-cost comparisons without novelty claims."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import median

from run_lspi_checkpoint import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(gzip.decompress(args.report.read_bytes()))
    methods = report.get("methods", METHODS)
    matched = "target_replay_memo" in methods
    if args.output is None:
        args.output = Path("output/benchmarks/lspi_matched_checkpoint_summary.md" if matched
                           else "output/benchmarks/lspi_checkpoint_summary.md")
    runs = report["runs"]
    queries = [q for r in runs for q in r["requests"]]
    groups = sorted({(r["regime"], r["count"], r["dimension"]) for r in runs})
    lines = ["# Exact LSPI cost checkpoint", "",
             f"Source: `{report['git_commit']}`; clean at launch: `{not report['git_dirty']}`.", "",
             f"**{len(runs)} runs, {len(queries)} requests, {sum(len(q['methods']) for q in queries)} method executions, {report['failures']} trace mismatches.**",
             f"Reference request statuses: {dict(Counter(q['methods']['scratch']['status'] for q in queries))}. Singular outcomes are rejected requests, not successful unlearning.", "",
             "## Request cost (milliseconds)", "",
             "Median across runs of each run's median request time. Exact matrix/RHS/weight/policy traces and failure statuses are compared; costs deliberately differ. Trace creation is timed; equality/hashing/JSON are common evaluation overhead.", "",
             "| Regime | Count parameter | Initial transitions | Features | " + " | ".join(methods) + " |",
             "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in methods) + " |"]
    for key in groups:
        group = [r for r in runs if (r['regime'], r['count'], r['dimension']) == key]
        times = [1000 * median(median(q['methods'][m]['seconds'] for q in r['requests']) for r in group) for m in methods]
        lines.append(f"| {key[0]} | {key[1]} | {group[0]['transitions']} | {key[2]} | " + " | ".join(f"{x:.3f}" for x in times) + " |")
    lines += ["", "## Preparation plus every request (milliseconds)", "",
              "Includes initial full-data learning for every method, plus statistic preparation for cached/replay methods. All requests start from the same initial policy and run four declared iterations. Cached memoization preserves those four trace entries while reusing evaluations of repeated policies within a request.", "",
              "| Regime | Count parameter | Features | Requests/run | " + " | ".join(methods) + " |",
              "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in methods) + " |"]
    for key in groups:
        group = [r for r in runs if (r['regime'], r['count'], r['dimension']) == key]
        totals = [1000 * median(r['preparation_seconds'][m] + sum(q['methods'][m]['seconds'] for q in r['requests']) for r in group) for m in methods]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(group[0]['requests'])} | " + " | ".join(f"{x:.3f}" for x in totals) + " |")
    lines += ["", "## Work diagnostics: median per request", "",
              "Terms are dense outer-product operations, not constant-time work. These counters omit copies, validation, bit complexity and cache allocation. Cached memoization stores up to K matrices/weights per request; memory is not measured here.", "",
              "| Regime | Count | Features | Cached target terms | Replay corrections | Memo solves | Replay solves |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in groups:
        selected = [q for r in runs if (r['regime'], r['count'], r['dimension']) == key for q in r['requests']]
        counters = [median(q['methods'][m]['work'][field] for q in selected) for m, field in (
            ('cached', 'target_rank_one_updates'), ('target_replay', 'target_rank_one_updates'),
            ('cached_memo', 'linear_solves'), ('target_replay', 'linear_solves'))]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(f"{x:g}" for x in counters) + " |")
    if matched:
        lines += ["", "## Matched reuse diagnostics", "",
                  "Both memoized methods use the same evaluation cache policy, dense solver and full action scoring. Caches reset for each deletion. All scheduled trace entries remain. Preparation/anchor assembly and zero-group handling still differ; this is not a pure instruction-level assembly ablation.", "",
                  "| Regime | Count | Features | Cached memo terms | Replay memo corrections | Cached memo solves | Replay memo solves |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key in groups:
            selected = [q for r in runs if (r['regime'], r['count'], r['dimension']) == key for q in r['requests']]
            counters = [median(q['methods'][m]['work'][field] for q in selected) for m, field in (
                ('cached_memo', 'target_rank_one_updates'), ('target_replay_memo', 'target_rank_one_updates'),
                ('cached_memo', 'linear_solves'), ('target_replay_memo', 'linear_solves'))]
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(f"{x:g}" for x in counters) + " |")
        unequal = sum(any(q['methods']['cached_memo']['work'][field] != q['methods']['target_replay_memo']['work'][field]
                          for field in ('linear_solves', 'action_scores')) for q in queries)
        lines += ["", f"Requests with unequal memoized solve/action-score counts: **{unequal}**."]
    lines += ["", "## Interpretation boundary", "",
              f"- Seeds {sorted({r['seed'] for r in runs})}; repeats {sorted({r['repeat'] for r in runs})}. Shared-feature seeds generate different logs. Switch seeds only shuffle deletion order inside an otherwise identical batch; these are not independent datasets.",
              "- Count is repetitions of the four-trajectory switch fixture, or the number of two-transition shared-feature trajectories. Shared cases use six states and a fixed feature map independent of log seed.",
              "- Cached reconstruction is established sufficient-statistic algebra. Cached memoization is ordinary reuse of an identical policy's evaluation within a request. Neither is a new algorithm. No incremental-inverse, measured-memory, large-dimensional floating-point or neural comparison is included.",
              "- Cached methods omit exact zero successor groups, restoring them on deletion when needed; target replay retains zero groups. This small engineering difference is charged in total time.",
              "- Synthetic exact-rational timings and partial work counts do not establish practical-scale speedups, an asymptotic theorem, privacy or novelty. No statistical significance claim is made.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
