"""Describe policy-only LSPI certificate costs and fallback without novelty claims."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--output', type=Path, default=Path('output/benchmarks/lspi_certificate_summary.md'))
    args = p.parse_args()
    d = json.loads(gzip.decompress(args.report.read_bytes()))
    runs, methods = d['runs'], d['methods']
    groups = sorted({(r['regime'],r['states'],r['dimension']) for r in runs})
    queries = [q for r in runs for q in r['requests']]
    bad = sum(not m['policy_match'] or m['full_trace_match'] is False for q in queries for m in q['methods'].values())
    lines = ['# Policy-sequence certificate experiment', '',
             f"Source `{d['git_commit']}`; clean at launch: {not d['git_dirty']}.", '',
             f'**{len(runs)} runs, {len(queries)} requests, {len(queries)*len(methods)} method executions, {bad} mismatches.**', '',
             'The certificate promises the same policy sequence/status only. Other methods additionally reproduce weights/matrices. This is an explicit narrower output contract, not full-state unlearning. The inverse control reuses both evaluations and greedy outputs for repeated policies.', '']
    for scope in ('request', 'lifecycle'):
        lines += [f'## {scope.title()} median milliseconds', '',
                  '| Regime | States | Features | ' + ' | '.join(methods) + ' |',
                  '| --- | ---: | ---: | ' + ' | '.join('---:' for _ in methods) + ' |']
        for key in groups:
            selected = [r for r in runs if (r['regime'],r['states'],r['dimension']) == key]
            values = []
            for m in methods:
                values.append(1000*median((median(q['methods'][m]['seconds'] for q in r['requests'])
                    if scope == 'request' else r['preparation_seconds'][m]+sum(q['methods'][m]['seconds'] for q in r['requests']))
                    for r in selected))
            lines.append(f'| {key[0]} | {key[1]} | {key[2]} | ' + ' | '.join(f'{x:.3f}' for x in values) + ' |')
        lines.append('')
    lines += ['## Certificate coverage and tail cost', '',
              '| Regime | States | Features | Certified / requests | Fallbacks | Singular rejections | Median of run-max request ms |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in groups:
        selected = [r for r in runs if (r['regime'],r['states'],r['dimension']) == key]
        results = [q['methods']['policy_certificate'] for r in selected for q in r['requests']]
        tail = 1000*median(max(q['methods']['policy_certificate']['seconds'] for q in r['requests']) for r in selected)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {sum(x['certified'] for x in results)} / {len(results)} | {sum(x['fallback'] for x in results)} | {sum(x['status']=='singular' for x in results)} | {tail:.3f} |")
    if d['memory']:
        lines += ['', '## Separate Python allocation pass', '',
                  'Seed 0 per setting, separate from timings. Inputs/features are allocated before tracing and excluded. Peaks include training, stored caches and requests; these are traced Python allocations, not process RSS or GPU memory.', '',
                  '| Regime | States | Features | Method | Lifecycle peak KiB | End KiB | Max request incremental KiB |',
                  '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
        for row in d['memory']:
            lines.append(f"| {row['regime']} | {row['states']} | {row['dimension']} | {row['method']} | {row['lifecycle_peak_python_bytes']/1024:.1f} | {row['end_python_bytes']/1024:.1f} | {row['max_request_increment_python_bytes']/1024:.1f} |")
    lines += ['', '## Interpretation limits', '',
              '- Request medians are medians of within-run medians; lifecycle includes original learning, inverse/margin preparation, every certificate check and every fallback. Tail reports show why a low median can hide expensive rebuilding.',
              '- Stable/fragile cases use designed shared positive feature vectors with opposite action signs. Stable rewards favor one action; fragile rewards are random. Switch cases repeat the earlier two-state fixture. These are deliberately controlled synthetic regimes.',
              '- Seeds create different stable/fragile logs; switch seeds reorder the same batch. Timing repeats are not independent datasets. All methods run in randomized order; no statistical significance claim is made.',
              '- Exact rational arithmetic avoids numerical errors but does not establish numerical stability. Factorizations, rational bit growth, larger feature dimensions, realistic tasks and external optimized libraries remain gates.',
              '- Accepted certificates keep old internal anchors and return policy references. They do not erase internal influence or certify privacy. Serializing a whole policy table would cost at least its output size and is outside these in-process timings.',
              '- Sherman–Morrison, perturbation bounds and unchanged-prediction certificates have prior art. This experiment cannot establish publication-level novelty.', '']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
