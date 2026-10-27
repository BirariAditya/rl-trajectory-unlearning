# Checkpoint 10: external offline-RL frozen-feature bridge

## Breakthrough and negative result

The verified deletion pipeline now runs on trajectories from the official
Minari `D4RL/minigrid/fourrooms-v0` dataset rather than only synthetic data.
Across three disjoint 64-episode windows, every one of nine sequential
whole-episode deletions changes the direct retained-only learner's final policy.
The interval verifier certifies all nine retained-only policy sequences without
fallback. All 27 request-level method executions match scratch retraining.

This is an external-data correctness breakthrough, not an efficiency win over
the strongest control. Exact policy-only inverse maintenance is faster than
interval verification on all three windows:

| Window | Episodes | Transitions | States | Feature preprocessing | Scratch lifecycle | Exact inverse lifecycle | Verified interval lifecycle | Exact / interval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 64 | 1,126 | 934 | 0.553 s | 75.724 s | 46.934 s | 49.620 s | 0.946x |
| 1 | 64 | 1,115 | 844 | 0.785 s | 72.639 s | 40.747 s | 42.313 s | 0.963x |
| 2 | 64 | 1,097 | 879 | 0.927 s | 74.128 s | 44.886 s | 46.733 s | 0.960x |

The median end-to-end exact / interval ratio, after charging feature
preprocessing to both methods, is 0.961x: interval verification is about 4.1%
slower. On lifecycle medians, exact inverse and interval verification remain
1.65x and 1.59x faster than direct scratch retraining, respectively. These are
single timing executions on each disjoint window and therefore descriptive.

## Dataset and representation contract

- Official source: `D4RL/minigrid/fourrooms-v0`, Minari dataset format version
  0.4.3; 590 episodes and 10,010 transitions in the complete source.
- Dataset HDF5 SHA-256:
  `273bf30752d5672ee1db4c4e6ab1e8b15266b06f83415e3407661f84f27669bb`.
- Metadata SHA-256:
  `bf6f313a38e42e623b84c0b1746501808b698374a81a6f16cf10d055d46d769d`.
- Windows are disjoint after one seed-1907 shuffle of the 590 episode indices.
- Each state is the raw direction plus 7x7x3 partial image. Mission text is
  constant and excluded.
- The 24-dimensional representation uses explicit action indicators and a
  deterministic BLAKE2b-seeded CountSketch of categorical observation tokens.
  It is fixed before deletion and never fitted to rewards or the retain/forget
  split.
- Actions are the meaningful FourRooms controls: left, right, and forward.
  Complete episodes, not individual transitions, are deleted.
- Feature construction is timed and charged to every end-to-end method total.
  Local dataset loading took 0.750 seconds and is reported separately. Network
  download is excluded from per-learner lifecycle cost.

The detailed [contract and limits](../audits/minari_frozen_bridge.md) define the
state support, terminal mask, comparison target, and prior-art boundary.

## Reproducibility

- Clean source: `e4978c808a006dae9346997bf7e82c6135572f92`.
- Dependencies: Python 3.11.15, Minari 0.5.4, NumPy 2.4.6, h5py 3.16.0.
- Validation: 132 local tests.
- Raw archive: [gzip JSON](../output/benchmarks/minari_frozen_bridge.json.gz).
- Generated tables: [summary](../output/benchmarks/minari_frozen_bridge_summary.md).

```powershell
uv venv .venv-minari --python 3.11
uv pip install --python .venv-minari\Scripts\python.exe -r requirements-minari.txt
$env:MINARI_DATASETS_PATH=(Resolve-Path results).Path + '\minari'
$env:PYTHONUTF8='1'
.\.venv-minari\Scripts\minari.exe download D4RL/minigrid/fourrooms-v0
.\.venv-minari\Scripts\python.exe scripts/run_minari_frozen_bridge.py --windows 0 1 2 --episodes 64 --requests 3 --dimension 24 --output results/minari_frozen_bridge.json.gz --summary results/minari_frozen_bridge_summary.md
```

## Claim boundary and next resource gate

The result establishes exact finite-policy-sequence deletion relative to the
declared retained-only frozen-feature learner on a nontrivial external offline-
RL dataset. It does not establish high environment return, neural unlearning,
privacy erasure, exact critic equality, universal speedup, or publication
novelty. Exact inverse maintenance remains the fastest tested control here.

CPU work can still add paired retiming and representation ablations. The next
credible *neural* external-validity step—extracting frozen learned features or
training/retraining IQL across multiple retain/forget splits—requires a GPU.
That stage must retain direct retraining, strong fine-tuning, multiple seeds,
utility, forget-set behavior, attacks, and complete preprocessing/training cost.
