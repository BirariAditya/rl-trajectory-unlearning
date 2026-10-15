"""Policy-output certificate versus cached replay and classical inverse updates."""
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
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rl_unlearning import Trajectory, Transition
from rl_unlearning.lspi_control import LSPIProblem, LSPIReplayMemoizedState, scratch_lspi
from rl_unlearning.lspi_baselines import LSPIMemoizedState
from rl_unlearning.lspi_inverse import LSPIInversePolicyState
from rl_unlearning.lspi_policy_certificate import LSPIPolicyCertificateState
from run_baselines import git
from run_lspi_checkpoint import dataset as switch_dataset, signature

METHODS = ('cached_memo', 'replay_memo', 'inverse_reuse', 'policy_certificate')
CLASSES = dict(zip(METHODS, (LSPIMemoizedState, LSPIReplayMemoizedState,
                           LSPIInversePolicyState, LSPIPolicyCertificateState)))
CASES = (('stable', 16, 4), ('stable', 128, 4), ('stable', 128, 8),
         ('fragile', 16, 4), ('switch', 2, 4))


def dataset(regime, states, dimension, seed, requests=20):
    if regime == 'switch':
        problem, logs, batches = switch_dataset('switch', 32, 4, seed)
        # First batch changes the policy; subsequent requests test renewed anchors.
        return problem, logs, batches[:1] + [[f'xb-{i}'] for i in range(requests-1)]
    features = {(s, a): tuple((1 if a == 0 else -1) *
                             (F(1, 8) + (F(1) + F(s % 5, 8) if j == s % dimension else 0))
                             for j in range(dimension))
                for s in range(states) for a in range(2)}
    problem = LSPIProblem(tuple(range(states)), (0, 1), features, dict.fromkeys(range(states), 1),
                          gamma=F(1, 4), regularization=1, iterations=4)
    rng = random.Random(seed)
    logs = []
    for i in range(256):
        s, z, a, b = rng.randrange(states), rng.randrange(states), rng.randrange(2), rng.randrange(2)
        def reward(action):
            return ((1 if action == 0 else -1) * (F(1) + F(rng.randrange(-2, 3), 20))
                    if regime == 'stable' else F(rng.randrange(-4, 5), 4))
        logs.append(Trajectory(str(i), (Transition(s, a, reward(a), z), Transition(z, b, reward(b)))))
    ids = list(range(len(logs)))
    rng.shuffle(ids)
    return problem, tuple(logs), [[str(i)] for i in ids[:requests]]


def policy_signature(result):
    sequence = (result.policy_sequence if hasattr(result, 'policy_sequence') else
                tuple(t.policy_out for t in result.iterates))
    return result.policy, sequence, result.status, result.repeat_start, result.repeat_length


def initialize(method, problem, logs):
    state = CLASSES[method](problem, logs)
    result = state.initial_result if method == 'policy_certificate' else state.replay()
    return state, result


def run(case, seed, repeat, nrequests):
    problem, logs, batches = dataset(*case, seed, nrequests)
    rng = random.Random(seed * 1000 + repeat)
    order, engines, preparation = list(METHODS), {}, {}
    rng.shuffle(order)
    reference = scratch_lspi(problem, logs)
    if reference.status == 'singular':
        raise AssertionError('benchmark initial schedule must be nonsingular')
    for method in order:
        start = perf_counter()
        state, result = initialize(method, problem, logs)
        preparation[method] = perf_counter()-start
        engines[method] = state
        assert policy_signature(result) == policy_signature(reference)
        if method != 'policy_certificate':
            assert signature(result) == signature(reference)
    retained, records = logs, []
    for batch in batches:
        removed = set(batch)
        candidate = tuple(t for t in retained if t.trajectory_id not in removed)
        # The independent per-transition reference is evaluation overhead, not a timed competitor.
        reference = scratch_lspi(problem, candidate)
        rng.shuffle(order)
        methods = {}
        for method in order:
            start = perf_counter()
            result = engines[method].delete_and_replay(batch)
            seconds = perf_counter()-start
            match = policy_signature(result) == policy_signature(reference)
            full_match = None if method == 'policy_certificate' else signature(result) == signature(reference)
            if not match or full_match is False:
                raise AssertionError(f'counterfactual mismatch: {case}/{seed}/{batch}/{method}')
            methods[method] = dict(seconds=seconds, policy_match=match, full_trace_match=full_match,
                status=result.status, policy_sha256=hashlib.sha256(repr(policy_signature(result)).encode()).hexdigest(),
                certified=getattr(result, 'certified', False), fallback=getattr(result, 'used_fallback', False),
                work=result.work if isinstance(result.work, dict) else asdict(result.work),
                inverse_work=dict(engines[method].last_inverse_work) if method == 'inverse_reuse' else None)
        records.append(dict(deletion_ids=batch, method_order=list(order), methods=methods))
        if reference.status != 'singular':
            retained = candidate
    return dict(regime=case[0], states=case[1], dimension=case[2], seed=seed, repeat=repeat,
                preparation_seconds=preparation, requests=records,
                dataset=[asdict(t) for t in logs] if repeat == 0 else None,
                features=[dict(state=s, action=a, vector=v) for (s,a),v in problem.features.items()] if repeat == 0 else None,
                gamma=str(problem.gamma), regularization=str(problem.regularization),
                initial_policy=dict(problem.initial_policy), iterations=problem.iterations)


def memory_pass(case, method, nrequests):
    problem, logs, batches = dataset(*case, 0, nrequests)
    gc.collect()
    tracemalloc.start()
    try:
        state, result = initialize(method, problem, logs)
        _, lifecycle_peak = tracemalloc.get_traced_memory()
        max_request_increment = 0
        for batch in batches:
            del result
            before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            result = state.delete_and_replay(batch)
            current, peak = tracemalloc.get_traced_memory()
            lifecycle_peak = max(lifecycle_peak, peak)
            max_request_increment = max(max_request_increment, peak-before)
        return dict(regime=case[0], states=case[1], dimension=case[2], method=method, seed=0,
                    lifecycle_peak_python_bytes=lifecycle_peak, end_python_bytes=current,
                    max_request_increment_python_bytes=max_request_increment)
    finally:
        tracemalloc.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--requests', type=int, default=20)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--memory', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'results/lspi_certificate.json.gz')
    args = p.parse_args()
    if not 1 <= args.requests <= 32 or args.repeats < 1:
        p.error('requests must be 1..32 and repeats positive')
    cases = (CASES[0], CASES[-1]) if args.smoke else CASES
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), git_commit=git('rev-parse','HEAD'),
                  git_dirty=bool(git('status','--porcelain')), python=sys.version, platform=platform.platform(),
                  arguments={**vars(args), 'output':str(args.output)}, methods=METHODS,
                  source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'scripts').glob('*.py'))},
                  scope='Policy-sequence output only for certificate; full traces checked for other methods. Original training, inverse/margin preparation and fallback charged. Input feature/log allocation excluded from separate Python tracemalloc pass. No numerical or privacy guarantee.', runs=[], memory=[])
    for case in cases:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                report['runs'].append(run(case,seed,repeat,args.requests))
                print(case,seed,repeat,'done',flush=True)
    if args.memory:
        for case in cases:
            for method in METHODS:
                report['memory'].append(memory_pass(case,method,args.requests))
                print('memory',case,method,'done',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(report,indent=2,default=str).encode(),mtime=0))
    print('Saved',len(report['runs']),'runs to',args.output)


if __name__ == '__main__':
    main()
