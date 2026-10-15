# Checkpoint 06: cumulative policy-sequence certificates

15 October 2026. **98 tests pass. 45 runs cover 900 deletion requests and 3,600 method executions with zero mismatches under the declared output contracts.** The certificate accepts 819 requests and falls back on 81. It provides the strongest conditional computational result so far, but **publication-level novelty is not established**.

## What is new in this checkpoint?

Earlier checkpoints recomputed exact retained weights at every distinct LSPI policy. The new method instead asks whether a deletion can be proved to leave every action choice in the entire finite policy sequence unchanged. It maintains cumulative perturbation bounds relative to exact anchor evaluations and compares them with precomputed action margins. Passing requests need no fresh solves or state-action scans; inconclusive requests rebuild the retained learner and its anchors.

**The contract is deliberately narrower:** the certificate returns policy sequence and status, not retained-only matrices or weights. Old internal inverse/weight anchors remain. This is behavioral equality to a specified retained-only learner, not full-state erasure, distributional privacy or neural unlearning. The [proof, operation bound and prior-art audit](../audits/lspi_policy_certificate.md) make that boundary explicit.

The strongest new conventional control, `LSPIInversePolicyState`, applies exact Sherman–Morrison updates and reuses both evaluations and greedy outputs when an input policy repeats. It computes retained weights and preserves the full trace. The cached and replay memoized methods from earlier checkpoints remain in the comparison.

## Results, including original training and every request

Each run has 20 sequential deletion requests. Times below are descriptive medians in milliseconds; the [complete generated table](../output/benchmarks/lspi_certificate_summary.md) includes all four methods, fallback counts and allocation measurements.

| Setting | Inverse-control request | Certificate request | Inverse-control lifecycle | Certificate lifecycle | Inverse / certificate lifecycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable, 16 states, 4 features | 2.639 | 0.591 | 89.190 | 47.491 | 1.88x |
| Stable, 128 states, 4 features | 10.464 | 0.612 | 269.630 | 78.241 | **3.45x** |
| Stable, 128 states, 8 features | 22.825 | 1.583 | 693.163 | 426.610 | 1.62x |
| Fragile, 16 states, 4 features | 1.911 | 0.432 | **65.925** | 83.808 | 0.79x; certificate loses |
| Switch, 2 states, 4 features | 0.571 | 0.198 | 23.378 | 17.704 | 1.32x |

The three stable settings beat every measured comparator on lifecycle medians. The switch result is only slightly below the best noncertificate comparator: 17.704 ms versus 18.125 ms for cached reconstruction, so it is not a strong speed result. The fragile case loses to both inverse reuse and memoized replay even though its median individual request is cheap. A low request median alone would give a misleading conclusion.

Coverage is 180/180, 180/180, 171/180, 117/180 and 171/180 certified requests in the table's order. All timed requests have valid nonsingular outputs; singular rejection and cycles are tested separately. In the stable eight-feature case, the median of each run's most expensive certificate request is **184.620 ms**, compared with a typical request median of 1.583 ms. Rebuilding anchors produces large tail costs.

The certificate's policy sequence/status is checked against direct retained-only per-transition LSPI on every request. The three other methods additionally match every exact matrix, RHS, weight and iterate. The full reference is evaluation overhead outside the timed competitor calls. Correctness is tested against a separate assembly path, but the methods share exact arithmetic/solver components; zero mismatches are not a universal empirical proof.

## Why it can save work

Let J be the number of distinct anchor input policies, d the feature dimension and m the newly deleted transitions. Prepaid inverses and all-state margin minima allow an accepted request to use O(J m d^2 + J d + number_of_deleted_IDs) arithmetic/dictionary operations, without scanning retained samples or all state-actions. This follows from accumulating matrix-vector contributions and checking one bound per anchor. It excludes rational bit growth and serializing policy tables; the implementation returns existing immutable policy references.

The useful observation is not a new inverse formula. For a single deletion, classical inverse updates can already compute exact new weights efficiently. The candidate avoids action rescoring across the domain when a cumulative bound certifies that every policy-improvement step is unchanged. All inverse/margin preparation, bound maintenance and global fallback are included in lifecycle cost.

## Memory and limitations

Twenty separate allocation passes cover five settings and four methods at seed 0. They trace additional Python allocations after the original dataset and feature map exist. They do **not** measure process RSS, total dataset storage, GPU memory or multiple-seed memory variance.

- Stable 128-state/4-feature lifecycle peaks: certificate **98.5 KiB**, inverse control 109.7 KiB, cached reconstruction 91.5 KiB.
- Stable 128-state/8-feature peaks: certificate **210.5 KiB**, inverse control 204.8 KiB. Rebuilding produces a 143.6 KiB incremental request peak for the certificate.
- Fragile case peaks: certificate 68.3 KiB versus inverse control 59.4 KiB. Memory is not uniformly better.

The synthetic stable/fragile features are positive vectors with opposite signs for the two actions; stable rewards deliberately favor one action. Some states share feature vectors because of the generator's modular construction. Feature deduplication or a specialized action classifier could narrow the advantage. These are controlled mechanism tests, not representative deployment tasks. Three seeds produce different stable/fragile logs; switch seeds only reorder a batch. Timing repeats reuse data. No statistical significance claim is made.

The certificate also has an easier public output contract than the full-weight comparators. The inverse comparator already reuses repeated greedy decisions, but richer trace construction and other implementation differences remain. Stronger prediction-only and feature-deduplicated controls are required before claiming a general policy-only advantage. No floating-point verified certificate, optimized external library, real-environment return, privacy audit or GPU result exists yet.

## Novelty decision

This is a meaningful **project milestone**: a proved sufficient condition has survived exact tests, a classical inverse control and preparation-inclusive measurements in favorable cases, while exposing measurable failure regimes. It is not yet a confirmed novel research algorithm. Unchanged-prediction certification, matrix perturbation bounds and Sherman–Morrison updates are established. The [audit](../audits/lspi_policy_certificate.md) identifies Wicker et al. and Abstract Gradient Training as additional prior art.

The next claim to test is whether the finite-control-sequence specialization offers a useful deletion-specific guarantee beyond those methods. Prioritize unique/shared-feature ablations and matched policy-only controls, then verified numerical bounds and conditioning stress. Do not widen the claim to full-state unlearning just because the policy output matches.

## Provenance and reproduction

Clean source: `97db5085ecdad055fd94206199d07cf0dedc5bd7`. All **32 source/script hashes** were verified after execution. [Source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38674439858) passes all tests and smoke runners. The [150,161-byte archive](../output/benchmarks/lspi_certificate.json.gz) includes data/features for repeat-zero runs, request sequences, policy digests, comparison flags, timings, counters, allocation results and environment. Source code and data reconstruct the full comparisons.

```powershell
Set-Location C:\Users\adity\Research
python -m unittest discover -s tests -v
python scripts/run_lspi_certificate.py --seeds 0 1 2 --repeats 3 --requests 20 --memory --output results/lspi_certificate.json.gz
python scripts/summarize_lspi_certificate.py results/lspi_certificate.json.gz
```

To regenerate the archived table only:

```powershell
python scripts/summarize_lspi_certificate.py output/benchmarks/lspi_certificate.json.gz --output results/lspi_certificate_archived_summary.md
```
