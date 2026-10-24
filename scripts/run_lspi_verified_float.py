"""Benchmark verified binary64 LSPI replay against exact policy controls."""

import argparse
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

from rl_unlearning.lspi_control import scratch_lspi
from rl_unlearning.lspi_inverse import LSPIInverseDeduplicatedState
from rl_unlearning.lspi_verified_float import (
    LSPIIntervalVerifiedFloatState,
    LSPIVerifiedFloatState,
)
from run_baselines import git
from run_lspi_multiaction import CASES, dataset


METHODS = (
    "scratch_retrain", "inverse_policy", "verified_rational", "verified_interval"
)


def contract(result):
    sequence = (
        result.policy_sequence
        if hasattr(result, "policy_sequence")
        else tuple(item.policy_out for item in result.iterates)
    )
    return (
        result.policy,
        sequence,
        result.status,
        result.repeat_start,
        result.repeat_length,
    )


def initialize(method, problem, logs):
    if method == "scratch_retrain":
        return None, scratch_lspi(problem, logs)
    if method == "inverse_policy":
        state = LSPIInverseDeduplicatedState(problem, logs, policy_only=True)
    elif method == "verified_rational":
        state = LSPIVerifiedFloatState(problem, logs)
    else:
        state = LSPIIntervalVerifiedFloatState(problem, logs)
    return state, state.replay()


def record(result, reference, seconds):
    matched = contract(result) == contract(reference)
    if not matched:
        raise AssertionError("retained-only policy-sequence mismatch")
    certificates = getattr(result, "certificates", ())
    return {
        "seconds": seconds,
        "policy_sequence_match": matched,
        "status": result.status,
        "certified": getattr(result, "certified", None),
        "used_fallback": getattr(result, "used_fallback", None),
        "certified_evaluations": len(certificates),
        "max_inverse_defect": (
            str(max((item.inverse_defect for item in certificates), default=0))
            if certificates else None
        ),
        "max_weight_error_bound": (
            str(max((item.weight_error_bound for item in certificates), default=0))
            if certificates else None
        ),
        "policy_sha256": hashlib.sha256(repr(result.policy).encode()).hexdigest(),
    }


def run(case, seed, repeat, requests):
    problem, logs, batches = dataset(*case, seed, requests=requests)
    rng = random.Random(31000 + seed * 100 + repeat)
    order = list(METHODS)
    rng.shuffle(order)
    preparation_order = list(order)
    states, initial, times = {}, {}, {}
    for method in order:
        started = perf_counter()
        states[method], initial[method] = initialize(method, problem, logs)
        times[method] = perf_counter() - started
    reference = initial["scratch_retrain"]
    preparation = {
        method: record(initial[method], reference, times[method]) for method in METHODS
    }

    retained = logs
    request_records = []
    for batch in batches:
        removed = set(batch)
        candidate = tuple(
            trajectory for trajectory in retained
            if trajectory.trajectory_id not in removed
        )
        rng.shuffle(order)
        outputs, request_times = {}, {}
        for method in order:
            started = perf_counter()
            outputs[method] = (
                scratch_lspi(problem, candidate)
                if method == "scratch_retrain"
                else states[method].delete_and_replay(batch)
            )
            request_times[method] = perf_counter() - started
        reference = outputs["scratch_retrain"]
        request_records.append({
            "deletion_ids": batch,
            "method_order": list(order),
            "methods": {
                method: record(outputs[method], reference, request_times[method])
                for method in METHODS
            },
        })
        if reference.status != "singular":
            retained = candidate
    return {
        "regime": case[0], "states": case[1], "dimension": case[2],
        "actions": case[3], "seed": seed, "repeat": repeat,
        "preparation_order": preparation_order, "preparation": preparation,
        "requests": request_records,
    }


def summarize(report):
    runs = report["runs"]
    requests = [request for run_record in runs for request in run_record["requests"]]
    mismatches = sum(
        not item["policy_sequence_match"]
        for request in requests for item in request["methods"].values()
    )
    groups = sorted({
        (run["regime"], run["states"], run["dimension"], run["actions"])
        for run in runs
    })
    lines = [
        "# Verified binary64 LSPI policy-sequence control",
        "",
        f"Source `{report['git_commit']}`; clean at launch: {not report['git_dirty']}.",
        "",
        f"**{len(runs)} runs, {len(requests)} requests, "
        f"{len(requests) * len(METHODS)} executions; {mismatches} mismatches.**",
        "",
        "Lifecycle time includes construction, initial learning, verification, exact fallback, and every sequential request. Direct retained-only scratch LSPI is the reference.",
        "",
        "## Lifecycle median milliseconds",
        "",
        "| Regime | States | Dimension | " + " | ".join(METHODS) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in METHODS) + " |",
    ]
    for key in groups:
        selected = [run for run in runs if (
            run["regime"], run["states"], run["dimension"], run["actions"]
        ) == key]
        values = []
        for method in METHODS:
            lifecycles = [
                run["preparation"][method]["seconds"]
                + sum(request["methods"][method]["seconds"] for request in run["requests"])
                for run in selected
            ]
            values.append(1000 * median(lifecycles))
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            + " | ".join(f"{value:.3f}" for value in values) + " |"
        )

    lines += [
        "",
        "## Verification outcomes",
        "",
        "| Regime | States | Dimension | Verifier | Certified requests | Exact fallbacks | Maximum inverse defect (decimal) |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key in groups:
        selected = [run for run in runs if (
            run["regime"], run["states"], run["dimension"], run["actions"]
        ) == key]
        for method in ("verified_rational", "verified_interval"):
            records = [
                request["methods"][method]
                for run in selected for request in run["requests"]
            ]
            defects = [float(F(item["max_inverse_defect"])) for item in records
                       if item["max_inverse_defect"] is not None]
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {method} | "
                f"{sum(item['certified'] for item in records)} / {len(records)} | "
                f"{sum(item['used_fallback'] for item in records)} | "
                f"{max(defects, default=0):.3e} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- The certificate proves equality of the finite LSPI policy sequence, not equality of floating weights or a full matrix trace.",
        "- The rational verifier treats computed binary64 numbers as exact dyadic rationals. The interval verifier instead encloses every conversion, product, and sum with directed `nextafter` expansion. Both apply the same inverse-defect theorem; inconclusive cases use exact replay.",
        "- The rational verifier performs exact matrix products; the interval verifier performs cubic outward-rounded binary64 products. Any speed result is measured rather than inferred.",
        "- These synthetic fixed-feature tasks do not establish neural-RL unlearning, privacy, historical-state erasure, universal speedup, or publication novelty.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/lspi_verified_float.json.gz")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.requests <= 40:
        parser.error("repeats must be positive and requests must be 1..40")
    if args.summarize:
        report = json.loads(gzip.decompress(args.summarize.read_bytes()))
    else:
        cases = (CASES[0], CASES[-1]) if args.smoke else CASES
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version,
            "platform": platform.platform(),
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "methods": METHODS,
            "runs": [],
            "source_sha256": {
                str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted((ROOT / "src").rglob("*.py"))
                + sorted((ROOT / "scripts").glob("*.py"))
            },
        }
        for case in cases:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    report["runs"].append(run(case, seed, repeat, args.requests))
                    print(f"{case} seed={seed} repeat={repeat}: done", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(
            gzip.compress(json.dumps(report, indent=2).encode(), mtime=0)
        )
        print(f"Saved {len(report['runs'])} runs: {args.output}")
    summary = summarize(report)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary + "\n", encoding="utf-8")
        print(f"Saved summary: {args.summary}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
