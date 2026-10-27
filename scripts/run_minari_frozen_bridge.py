"""Run retained-only LSPI deletion on frozen features from Minari FourRooms."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_unlearning.frozen_bridge import build_minigrid_frozen_lspi
from rl_unlearning.lspi_control import scratch_lspi
from run_baselines import git
from run_lspi_verified_float import initialize, record


METHODS = ("scratch_retrain", "inverse_policy", "verified_interval")
DEFAULT_DATASET = "D4RL/minigrid/fourrooms-v0"


def run_window(episodes, window, requests, dimension, feature_seed):
    started = perf_counter()
    bridge = build_minigrid_frozen_lspi(
        episodes,
        dimension=dimension,
        feature_seed=feature_seed,
    )
    preprocessing_seconds = perf_counter() - started
    problem, logs = bridge.problem, bridge.trajectories
    rng = random.Random(88000 + window)
    deletion_ids = list(bridge.episode_ids)
    rng.shuffle(deletion_ids)
    batches = [(identifier,) for identifier in deletion_ids[:requests]]

    order = list(METHODS)
    rng.shuffle(order)
    preparation_order = list(order)
    states, initial, preparation_seconds = {}, {}, {}
    for method in order:
        started = perf_counter()
        states[method], initial[method] = initialize(method, problem, logs)
        preparation_seconds[method] = perf_counter() - started
    reference = initial["scratch_retrain"]
    preparation = {
        method: record(initial[method], reference, preparation_seconds[method])
        for method in METHODS
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
        outputs, seconds = {}, {}
        for method in order:
            started = perf_counter()
            outputs[method] = (
                scratch_lspi(problem, candidate)
                if method == "scratch_retrain"
                else states[method].delete_and_replay(batch)
            )
            seconds[method] = perf_counter() - started
        reference = outputs["scratch_retrain"]
        request_records.append({
            "deletion_ids": batch,
            "method_order": list(order),
            "methods": {
                method: record(outputs[method], reference, seconds[method])
                for method in METHODS
            },
        })
        if reference.status != "singular":
            retained = candidate

    return {
        "window": window,
        "episode_ids": bridge.episode_ids,
        "episodes": len(logs),
        "transitions": bridge.transitions,
        "unique_states": bridge.unique_states,
        "actions": list(problem.actions),
        "dimension": problem.dimension,
        "feature_seed": feature_seed,
        "preprocessing_seconds": preprocessing_seconds,
        "preparation_order": preparation_order,
        "preparation": preparation,
        "requests": request_records,
    }


def _lifecycle(run, method):
    return run["preparation"][method]["seconds"] + sum(
        request["methods"][method]["seconds"] for request in run["requests"]
    )


def summarize(report):
    runs = report["runs"]
    requests = [request for run in runs for request in run["requests"]]
    mismatches = sum(
        not item["policy_sequence_match"]
        for request in requests for item in request["methods"].values()
    )
    lines = [
        "# Minari FourRooms frozen-feature deletion bridge",
        "",
        f"Source `{report['git_commit']}`; clean at launch: {not report['git_dirty']}.",
        "",
        f"Dataset `{report['dataset']['id']}`: {report['dataset']['total_episodes']} episodes / {report['dataset']['total_steps']} transitions in the source; local load took {report['dataset']['load_seconds']:.3f} seconds.",
        "",
        f"**{len(runs)} disjoint windows, {len(requests)} deletion requests, {len(requests) * len(METHODS)} request executions; {mismatches} policy-sequence mismatches.**",
        "",
        "The state dictionary and frozen features are constructed once from each original pre-deletion window and never rebuilt after a request. Preprocessing is charged in every end-to-end method column. Remote download time is excluded; local load time is reported separately.",
        "",
        "## Window results",
        "",
        "| Window | Episodes | Transitions | States | Preprocess s | Scratch lifecycle s | Exact inverse lifecycle s | Verified interval lifecycle s | Exact inverse end-to-end s | Verified interval end-to-end s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        scratch = _lifecycle(run, "scratch_retrain")
        inverse = _lifecycle(run, "inverse_policy")
        interval = _lifecycle(run, "verified_interval")
        preprocessing = run["preprocessing_seconds"]
        lines.append(
            f"| {run['window']} | {run['episodes']} | {run['transitions']} | "
            f"{run['unique_states']} | {preprocessing:.3f} | {scratch:.3f} | "
            f"{inverse:.3f} | {interval:.3f} | {preprocessing + inverse:.3f} | "
            f"{preprocessing + interval:.3f} |"
        )

    inverse_total = [run["preprocessing_seconds"] + _lifecycle(run, "inverse_policy")
                     for run in runs]
    interval_total = [run["preprocessing_seconds"] + _lifecycle(run, "verified_interval")
                      for run in runs]
    verifier = [request["methods"]["verified_interval"] for request in requests]
    lines += [
        "",
        "## Aggregate and verification",
        "",
        f"Median end-to-end exact-inverse / verified-interval ratio: **{median(inverse_total) / median(interval_total):.3f}x**.",
        "",
        f"Certified deletion requests: **{sum(bool(item['certified']) for item in verifier)} / {len(verifier)}**; exact fallbacks: **{sum(bool(item['used_fallback']) for item in verifier)}**.",
        "",
        "## Scope",
        "",
        "- The source is an official external offline-RL trajectory dataset rather than a synthetic transition generator.",
        "- The representation is a deterministic CountSketch-style map over the 7x7x3 image, direction, and action, with explicit action indicators. It is not learned and uses no reward or retain/forget information.",
        "- Actions are restricted a priori to MiniGrid left, right, and forward (0, 1, 2); the official FourRooms environment documents pickup, drop, toggle, and done as unused.",
        "- The representation covers each selected window's original observed state support. Deletion does not remove state entries or change features.",
        "- Equality is for the declared finite LSPI policy sequence, not exact weights, neural offline RL, environment return, privacy erasure, or publication novelty.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--windows", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--selection-seed", type=int, default=1907)
    parser.add_argument("--feature-seed", default="fourrooms-countsketch-v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/minari_frozen_bridge.json.gz")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.episodes < 4 or not 1 <= args.requests < args.episodes:
        parser.error("episodes must be at least 4 and requests must be 1..episodes-1")
    if len(set(args.windows)) != len(args.windows) or min(args.windows) < 0:
        parser.error("windows must be unique nonnegative integers")

    if args.summarize:
        report = json.loads(gzip.decompress(args.summarize.read_bytes()))
    else:
        try:
            import minari
        except ImportError as error:
            parser.error(f"Minari is required for dataset ingestion: {error}")
        started = perf_counter()
        dataset = minari.load_dataset(args.dataset)
        all_episodes = tuple(dataset.iterate_episodes())
        load_seconds = perf_counter() - started
        indices = list(range(len(all_episodes)))
        random.Random(args.selection_seed).shuffle(indices)
        required = (max(args.windows) + 1) * args.episodes
        if required > len(indices):
            parser.error("requested disjoint windows exceed the dataset")
        data_path = Path(dataset.storage.data_path)
        dataset_files = {
            str(path.relative_to(data_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data_path.rglob("*")) if path.is_file()
        }
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                name: version(name) for name in ("minari", "numpy", "h5py")
            },
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "dataset": {
                "id": dataset.id,
                "minari_version": dataset.minari_version,
                "total_episodes": dataset.total_episodes,
                "total_steps": dataset.total_steps,
                "load_seconds": load_seconds,
                "files_sha256": dataset_files,
            },
            "methods": METHODS,
            "runs": [],
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((ROOT / "src").rglob("*.py"))
                + sorted((ROOT / "scripts").glob("*.py"))
            },
        }
        for window in args.windows:
            start = window * args.episodes
            selected = tuple(all_episodes[index] for index in indices[start:start + args.episodes])
            report["runs"].append(
                run_window(
                    selected, window, args.requests, args.dimension,
                    args.feature_seed,
                )
            )
            print(f"window={window}: done", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(
            gzip.compress(json.dumps(report, indent=2).encode(), mtime=0)
        )
        print(f"Saved {len(report['runs'])} windows: {args.output}")
    text = summarize(report)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
        print(f"Saved summary: {args.summary}")
    else:
        print(text)


if __name__ == "__main__":
    main()
