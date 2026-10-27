# Checkpoint 10A: repeated high-dimensional timing and allocations

## Result

The clean three-seed high-dimensional experiment preserves the retained-only
policy-sequence contract in all 36 method executions. The outward-rounded
interval verifier certifies all 12 deletion requests without exact fallback.
Median lifecycle time is lower than exact policy-only inverse maintenance at
both tested dimensions, but the advantage is not dataset-wise uniform:

| Dimension | Exact inverse median | Verified interval median | Median ratio | Seed-wise exact / interval ratios |
| ---: | ---: | ---: | ---: | --- |
| 48 | 23.731 s | 21.498 s | 1.10x | 0.46x, 1.10x, 1.01x |
| 64 | 46.097 s | 34.517 s | 1.34x | 1.63x, 0.92x, 1.55x |

The d=48 seed-0 loss is a timing tail rather than verifier fallback: its second
certified request took 53.070 seconds while the other d=48 interval requests
took 2.641--6.032 seconds. Repeated paired timing on fixed datasets is required
before making a robust speed claim.

Separate `tracemalloc` passes show a clearer implementation-level memory
advantage:

| Dimension | Exact inverse peak | Verified interval peak | Peak ratio | Request-increment ratio |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 5.618 MiB | 1.466 MiB | 3.83x | 3.49x |
| 64 | 11.653 MiB | 2.373 MiB | 4.91x | 4.32x |

Inputs were allocated before tracing. These are Python allocations for
construction, initial replay, method state, verification/fallback machinery,
and two sequential requests; they are not process RSS or GPU memory. The
allocation pass uses seed 0 and is not yet evidence of a distributional memory
claim.

## Reproducibility

- Source: `7cf2a1989e4eaebde8522a16168de8c016d5a31a`, clean at launch.
- Validation: 127 local tests.
- Workload: fragile 128-state, three-action cases at feature dimensions 48 and
  64; three independent dataset seeds; one timing execution per dataset; two
  sequential deletion requests.
- Timed methods: direct retained-only scratch LSPI, exact policy-only inverse
  maintenance, and interval-verified binary64 replay.
- Allocation methods: exact policy-only inverse maintenance and interval-
  verified replay, measured in isolated passes.
- Raw archive: [gzip JSON](../output/benchmarks/lspi_verified_highdim_repeated.json.gz).
- Generated tables: [summary](../output/benchmarks/lspi_verified_highdim_repeated_summary.md).

Reproduce with:

```powershell
python scripts/run_lspi_verified_float.py --highdim --seeds 0 1 2 --repeats 1 --requests 2 --memory --memory-methods inverse_policy verified_interval --output results/lspi_verified_highdim_repeated.json.gz --summary results/lspi_verified_highdim_repeated_summary.md
```

## Claim boundary and next gate

This strengthens the evidence that outward-rounded verification can reduce the
time and Python-object footprint of exact rational LSPI replay at higher feature
dimensions. It does not establish a universal speedup, a neural-RL result,
privacy erasure, or publication novelty. The next CPU gate is a nontrivial
offline-RL trajectory adapter with frozen, data-independent features and the
same retained-only deletion contract. Paired retiming should also distinguish
dataset difficulty from system noise. GPU work remains staged until a credible
neural feature extractor or IQL learner is required.
