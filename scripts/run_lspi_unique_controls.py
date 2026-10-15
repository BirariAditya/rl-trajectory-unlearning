"""Feature-alias controls for exact LSPI deletion, with complete lifecycle costs.

This is a separate experiment; it does not replace earlier archived timings.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as F
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, scratch_lspi
from rl_unlearning.lspi_inverse import LSPIInversePolicyState, LSPIInverseDeduplicatedState
from rl_unlearning.lspi_decision_repair import LSPISelectiveRepairState, LSPISharedAnchorRepairState
from rl_unlearning.lspi_policy_certificate import LSPIPolicyCertificateState
from run_baselines import git
from run_lspi_checkpoint import dataset as switch_dataset, signature
from run_lspi_certificate import policy_signature

METHODS = ('inverse_reuse', 'inverse_deduplicated', 'inverse_deduplicated_policy_only',
           'policy_certificate', 'selective_repair', 'selective_repair_policy_only', 'shared_anchor_repair')
CASES = tuple((regime, alias, states, 4)
              for regime in ('stable', 'fragile')
              for states in (16, 128)
              for alias in ('aliased', 'unique')) + (('switch', 'one_hot', 2, 4),)


def dataset(regime, alias, states, dimension, seed, requests=8):
    """Paired fixed features and matched state/action/deletion draws.

    The stable teacher rewards also depend on the feature map, so these are
    matched generation protocols rather than identical reward datasets.
    """
    if not 1 <= requests <= 32:
        raise ValueError('requests must be 1..32')
    if regime == 'switch':
        if (alias, states, dimension) != ('one_hot', 2, 4):
            raise ValueError('switch fixture requires one_hot / 2 states / 4 features')
        problem, logs, batches = switch_dataset('switch', 32, 4, seed)
        return problem, logs, batches[:1] + [[f'xb-{i}'] for i in range(requests-1)]
    if regime not in ('stable', 'fragile') or alias not in ('aliased', 'unique'):
        raise ValueError('unknown fixture')
    if states < 2 or dimension < 2:
        raise ValueError('at least two states and dimensions required')
    features = {}
    for s in range(states):
        orientation = 1 if (s // dimension) % 2 == 0 else -1
        base = [F(1, 4) if j == 0 else F(1 if (s % dimension + j) % 2 else -1, 8)
                for j in range(dimension)]
        base[s % dimension] += 1
        if alias == 'unique':
            # Positive, distinct, and smaller than 1/16; primary-coordinate
            # differences remain separated. The map is independent of logs.
            base[0] += F(s + 1, 16 * (states + 1))
        for a in range(2):
            features[s, a] = tuple(orientation * (1 if a == 0 else -1) * x for x in base)
    problem = LSPIProblem(tuple(range(states)), (0, 1), features, dict.fromkeys(range(states), 1),
                          gamma=F(1, 4), regularization=1, iterations=4)
    rng, logs = random.Random(seed), []
    for i in range(256):
        s, z, a, b = rng.randrange(states), rng.randrange(states), rng.randrange(2), rng.randrange(2)
        def reward(state, action):
            noise = rng.randrange(-4, 5)
            return (features[state, action][0] + F(noise, 100)
                    if regime == 'stable' else F(noise, 4))
        logs.append(Trajectory(str(i), (Transition(s, a, reward(s, a), z),
                                          Transition(z, b, reward(z, b)))))
    ids = list(range(len(logs)))
    rng.shuffle(ids)
    return problem, tuple(logs), [[str(i)] for i in ids[:requests]]


def feature_counts(problem):
    return dict(state_action_vectors=len(problem.features),
                unique_feature_vectors=len(set(problem.features.values())),
                unique_ordered_action_rows=len({tuple(problem.features[s, a] for a in problem.actions)
                                                for s in problem.states}))


def initialize(method, problem, logs):
    if method == 'policy_certificate':
        state = LSPIPolicyCertificateState(problem, logs)
        return state, state.initial_result
    if method == 'inverse_deduplicated_policy_only':
        state = LSPIInverseDeduplicatedState(problem, logs, policy_only=True)
    elif method == 'selective_repair_policy_only':
        state = LSPISelectiveRepairState(problem, logs, policy_only=True)
    else:
        cls = {'inverse_reuse': LSPIInversePolicyState,
               'inverse_deduplicated': LSPIInverseDeduplicatedState,
               'selective_repair': LSPISelectiveRepairState,
               'shared_anchor_repair': LSPISharedAnchorRepairState}[method]
        state = cls(problem, logs)
    return state, state.replay()


def result_record(state, result, reference, seconds):
    full = signature(result) == signature(reference) if hasattr(result, 'iterates') else None
    shared = policy_signature(result) == policy_signature(reference)
    if not shared or full is False:
        raise AssertionError('retained-only counterfactual mismatch')
    return dict(seconds=seconds, policy_match=shared, full_trace_match=full,
                status=result.status, policy_sha256=hashlib.sha256(repr(policy_signature(result)).encode()).hexdigest(),
                certified=getattr(result, 'certified', False), fallback=getattr(result, 'used_fallback', False),
                work=result.work if isinstance(result.work, dict) else asdict(result.work),
                inverse_work=dict(state.last_inverse_work) if hasattr(state, 'last_inverse_work') else None,
                decision_work=dict(state.last_decision_work) if hasattr(state, 'last_decision_work') else None)


def run(case, seed, repeat, nrequests):
    problem, logs, batches = dataset(*case, seed, nrequests)
    reference = scratch_lspi(problem, logs)
    if reference.status == 'singular':
        raise AssertionError('initial fixture is singular')
    rng, order, engines, preparation = random.Random(seed * 1000 + repeat), list(METHODS), {}, {}
    rng.shuffle(order)
    prep_order = list(order)
    for method in order:
        started = perf_counter()
        state, result = initialize(method, problem, logs)
        elapsed = perf_counter() - started
        engines[method] = state
        preparation[method] = result_record(state, result, reference, elapsed)
    retained, previous, records = logs, reference, []
    for batch in batches:
        removed = set(batch)
        candidate = tuple(t for t in retained if t.trajectory_id not in removed)
        reference = scratch_lspi(problem, candidate)
        rng.shuffle(order)
        method_results = {}
        for method in order:
            started = perf_counter()
            result = engines[method].delete_and_replay(batch)
            elapsed = perf_counter() - started
            method_results[method] = result_record(engines[method], result, reference, elapsed)
        changed = sum(a != b for a, b in zip(previous.policy, reference.policy)) if reference.status != 'singular' else None
        sequence_changed = policy_signature(previous) != policy_signature(reference)
        records.append(dict(deletion_ids=batch, method_order=list(order), methods=method_results,
                            changed_final_actions=changed, policy_sequence_changed=sequence_changed))
        if reference.status != 'singular':
            retained, previous = candidate, reference
    return dict(regime=case[0], alias=case[1], states=case[2], dimension=case[3], seed=seed, repeat=repeat,
                feature_counts=feature_counts(problem), preparation_order=prep_order,
                preparation_results=preparation,
                preparation_seconds={m: r['seconds'] for m, r in preparation.items()}, requests=records,
                dataset=[asdict(t) for t in logs] if repeat == 0 else None,
                features=[dict(state=s, action=a, vector=v) for (s, a), v in problem.features.items()] if repeat == 0 else None,
                gamma=str(problem.gamma), regularization=str(problem.regularization),
                initial_policy=dict(problem.initial_policy), iterations=problem.iterations)


def decision_work_totals(runs, method):
    """Request-only counts; preparation is intentionally excluded here."""
    records = [q['methods'][method] for r in runs for q in r['requests']]
    fields = ('margin_groups_built', 'groups_certified', 'groups_scored',
              'decisions_changed_from_anchor')
    decision_records = [r['decision_work'] for r in records if r['decision_work'] is not None]
    return dict(requests=len(records),
                action_scores=sum(r['work']['action_scores'] for r in records),
                **{field: sum(d[field] for d in decision_records) if decision_records else None
                   for field in fields})


def memory_pass(case, method, nrequests):
    """Separate seed-zero allocations; input generation is outside tracing."""
    problem, logs, batches = dataset(*case, 0, nrequests)
    gc.collect()
    tracemalloc.start()
    try:
        state, result = initialize(method, problem, logs)
        current, lifecycle_peak = tracemalloc.get_traced_memory()
        max_request_increment = 0
        for batch in batches:
            # Do not retain all historical returned traces in the measurement.
            del result
            before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            result = state.delete_and_replay(batch)
            current, peak = tracemalloc.get_traced_memory()
            lifecycle_peak = max(lifecycle_peak, peak)
            max_request_increment = max(max_request_increment, peak - before)
        return dict(regime=case[0], alias=case[1], states=case[2], dimension=case[3],
                    method=method, seed=0, requests=len(batches),
                    lifecycle_peak_python_bytes=lifecycle_peak, end_python_bytes=current,
                    max_request_increment_python_bytes=max_request_increment)
    finally:
        tracemalloc.stop()


def summarize(report):
    runs, methods = report['runs'], report['methods']
    requests = [q for r in runs for q in r['requests']]
    mismatches = sum(not x['policy_match'] or x['full_trace_match'] is False
                     for q in requests for x in q['methods'].values())
    groups = sorted({(r['regime'], r['alias'], r['states'], r['dimension']) for r in runs})
    lines = ['# LSPI feature-alias and selective-repair controls', '',
             f"Source `{report['git_commit']}`; clean at launch: {not report['git_dirty']}.", '',
             f'**{len(runs)} runs, {len(requests)} requests, {len(requests)*len(methods)} method executions; {mismatches} mismatches.**', '',
             'Every method is checked against an independent per-transition retained-only LSPI reference. Full traces are additionally compared when exposed. Preparation includes feature indexing, initial training and certificate construction; request costs include fallbacks. Reference evaluation and report serialization are outside timings.', '']
    for scope in ('request', 'lifecycle'):
        lines += [f'## {scope.title()} median milliseconds', '',
                  '| Regime | Features | States | Dimension | ' + ' | '.join(methods) + ' |',
                  '| --- | --- | ---: | ---: | ' + ' | '.join('---:' for _ in methods) + ' |']
        for key in groups:
            selected = [r for r in runs if (r['regime'], r['alias'], r['states'], r['dimension']) == key]
            values = [1000 * median(median(q['methods'][m]['seconds'] for q in r['requests'])
                                   if scope == 'request' else r['preparation_seconds'][m] +
                                   sum(q['methods'][m]['seconds'] for q in r['requests'])
                                   for r in selected) for m in methods]
            lines.append('| ' + ' | '.join(map(str, key)) + ' | ' + ' | '.join(f'{x:.3f}' for x in values) + ' |')
        lines.append('')
    lines += ['## Feature counts, changes and certificate coverage', '',
              '| Regime | Features | States | Unique vectors / rows | Changed final policies / requests | Changed sequences | Certified | Fallbacks | Median run-max certificate ms |',
              '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for key in groups:
        selected = [r for r in runs if (r['regime'], r['alias'], r['states'], r['dimension']) == key]
        queries = [q for r in selected for q in r['requests']]
        certs = [q['methods']['policy_certificate'] for q in queries]
        counts = selected[0]['feature_counts']
        tail = 1000 * median(max(q['methods']['policy_certificate']['seconds'] for q in r['requests']) for r in selected)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {counts['unique_feature_vectors']} / {counts['unique_ordered_action_rows']} | "
                     f"{sum(bool(q['changed_final_actions']) for q in queries)} / {len(queries)} | "
                     f"{sum(q['policy_sequence_changed'] for q in queries)} | {sum(x['certified'] for x in certs)} | "
                     f"{sum(x['fallback'] for x in certs)} | {tail:.3f} |")
    lines += ['', '## Request-only decision work', '',
              'Counts below sum deletion requests across all seeds and timing repeats; preparation is excluded from this table and included in lifecycle timings above. Scores count individual action evaluations. Groups are ordered action-feature rows, and the same group may be evaluated for multiple distinct input policies in one request. A dash means the baseline does not expose that screening counter.', '',
              '| Regime / features / states | Method | Action scores | New margin groups / request | Certified groups | Scored groups | Decisions changed from anchor |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    work_methods = [m for m in ('inverse_deduplicated', 'selective_repair', 'shared_anchor_repair') if m in methods]
    for key in groups:
        selected = [r for r in runs if (r['regime'], r['alias'], r['states'], r['dimension']) == key]
        for method in work_methods:
            counts = decision_work_totals(selected, method)
            margins = (f"{counts['margin_groups_built'] / counts['requests']:.2f}"
                       if counts['margin_groups_built'] is not None else '—')
            fields = [str(counts[k]) if counts[k] is not None else '—'
                      for k in ('groups_certified', 'groups_scored', 'decisions_changed_from_anchor')]
            lines.append(f"| {key[0]} / {key[1]} / {key[2]} | {method} | {counts['action_scores']} | {margins} | "
                         + ' | '.join(fields) + ' |')
    lines += ['', '“Decisions changed from anchor” counts feature-group choices that differ from the selected historical margin anchor, summed across evaluations. It is **not** a count of new changes caused by the current deletion: the same persistent difference can be counted again on later requests, and a shared anchor can come from another input policy. The earlier coverage table compares each retained-only final policy/sequence with the previous successfully retained dataset. New-anchor choices are scored and indexed but are not counted as changes from an old anchor.', '']
    if report.get('memory'):
        lines += ['## Separate Python allocation pass', '',
                  'One seed-0 pass per case/method, outside the timing experiment, over the full request sequence. Input features/logs and deletion lists are allocated before tracing and excluded. Training, feature/index preparation, caches and requests are included. End allocations include the final returned result; earlier returned results are released. These are traced Python allocations, not process RSS or GPU memory.', '',
                  '| Regime / features / states | Method | Lifecycle peak KiB | End KiB | Max request incremental KiB |',
                  '| --- | --- | ---: | ---: | ---: |']
        for row in report['memory']:
            lines.append(f"| {row['regime']} / {row['alias']} / {row['states']} | {row['method']} | "
                         f"{row['lifecycle_peak_python_bytes']/1024:.1f} | {row['end_python_bytes']/1024:.1f} | "
                         f"{row['max_request_increment_python_bytes']/1024:.1f} |")
        lines.append('')
    lines += ['', '## Interpretation limits', '',
              '- Exact rational arithmetic, synthetic fixed features, two actions, finite LSPI schedule and small feature dimension. Feature uniqueness does not imply a difficult or realistic representation.',
              '- Aliased and unique cases share trajectory state/action draws and deletion IDs. Stable rewards use the corresponding feature-map teacher, so rewards differ slightly; fragile rewards are identical between the pair.',
              '- Stable rewards target a mixed-sign feature map whose preferred action varies by state. Unique maps add deterministic small perturbations; their arithmetic bit lengths also differ from the aliased maps.',
              '- The switch fixture repeats a deterministic two-state dataset; its seeds only reorder one deletion batch. Timing repeats are not independent scientific datasets.',
              '- Certificate and prediction-only controls expose policy sequences/status. Other methods return exact full traces; both contracts are checked explicitly. These in-process measurements do not charge serialization of a whole policy table.',
              '- No privacy, neural-RL, universal speedup, statistical significance or publication novelty claim follows from these controls. Optional allocation measurements exclude shared input objects and do not establish total process memory.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--requests', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--memory', action='store_true', help='Separate seed-0 Python allocation passes for all selected cases/methods')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/lspi_unique_controls.json.gz')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--summarize', type=Path, help='Regenerate summary from an existing .json.gz without timings')
    args = parser.parse_args()
    if not 1 <= args.requests <= 32 or args.repeats < 1:
        parser.error('requests must be 1..32 and repeats positive')
    if args.summarize:
        report = json.loads(gzip.decompress(args.summarize.read_bytes()))
        output = args.summarize
    else:
        cases = (CASES[0], CASES[1], CASES[-1]) if args.smoke else CASES
        report = dict(created_at=datetime.now(timezone.utc).isoformat(), git_commit=git('rev-parse', 'HEAD'),
                      git_dirty=bool(git('status', '--porcelain')), python=sys.version, platform=platform.platform(),
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      methods=METHODS, runs=[], memory=[],
                      source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))},
                      scope='Exact rational policy-sequence agreement; full traces checked when exposed. All preparation and deletion work timed; reference evaluation, serialization and input generation excluded. Optional separate Python allocation pass excludes preallocated inputs and includes preparation. No privacy or numerical claim.')
        for case in cases:
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    report['runs'].append(run(case, seed, repeat, args.requests))
                    print(case, seed, repeat, 'done', flush=True)
        if args.memory:
            for case in cases:
                for method in METHODS:
                    report['memory'].append(memory_pass(case, method, args.requests))
                    print('memory', case, method, 'done', flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(gzip.compress(json.dumps(report, indent=2, default=str).encode(), mtime=0))
        output = args.output
    summary = args.summary or output.with_name(output.name.removesuffix('.json.gz') + '_summary.md')
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summarize(report), encoding='utf-8')
    print('Saved', len(report['runs']), 'runs; summary', summary)


if __name__ == '__main__':
    main()
