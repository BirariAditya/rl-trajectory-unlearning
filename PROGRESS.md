# Research progress

Last updated: 2026-10-14. Repository: [BirariAditya/rl-trajectory-unlearning](https://github.com/BirariAditya/rl-trajectory-unlearning) (private).

## Current stage

**Checkpoint 04 reached: exact LSPI compared against cached and memoized controls.** Read [the checkpoint](checkpoints/04_lspi_cost.md): 63 runs, 162 requests, 648 method executions, zero trace mismatches, and 74 passing tests. Cached memoization has the lowest request median in all seven settings; replay has the lowest training-plus-request median in two. Neither a broad efficiency advantage nor paper-level novelty is established. The tabular and LSPI systems remain the verified baseline suite. A [comprehensive presentation outline](presentation/2026-10-18/TALK_TRACK.md) and [learning guide](docs/LEARNING_GUIDE.md) cover the full research journey for 18 October. GPU experiments have not started.

## Stage tracker

| Stage | Status | Evidence / next gate |
| --- | --- | --- |
| Literature and novelty audit | Initial review complete | `literature_review.md`, `novelty_reassessment.md`, three `audits/` notes; CURe full text remains unresolved |
| Tabular data subtraction and planning | Implemented and tested | Retained-only rebuild comparisons, empty rows, repeated requests, ties, non-greedy predecessor regression |
| Independent retained-data reference | Implemented and tested | Exact rational policy enumeration for tiny MDPs, independently aggregated from logs; 20 random-model comparisons |
| Baselines and cost ledger | Lean global and classical elimination checkpoint complete | 4,158 further method executions / 108 runs; full lifecycle and scratch costs charged; external optimized FTVI is not reproduced |
| Sparse decision certificate | Exact block repair and safe pruning implemented | Chosen-action changes, cyclic/sequential requests, exact ties, index repair, anchor invalidation; mature-planner novelty and general margin envelope remain open |
| Frozen-feature LSPI control | Correctness and first complete-cost comparison passed | 648 matching executions; cached memoization is a strong control; matched replay memoization, numerical factorization, memory and novelty gates remain |
| GPU IQL pilot | Not started | Dataset/learner identity, multiple retrains, strong fine-tune, matched controls, multi-attack audit |
| Full experiments and manuscript | Not started | Pass prior gates before expanding |

## Latest validation

- **13 October experiment; 14 October documentation: Checkpoint 04 complete.** 74 tests and [source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/38338893080) pass. Four methods across 63 runs / 162 requests / 648 executions have zero exact trace mismatches. [Tables](output/benchmarks/lspi_checkpoint_summary.md), [raw archive](output/benchmarks/lspi_checkpoint.json.gz), [interpretation](checkpoints/04_lspi_cost.md). All timed requests reach fixed points; cycles/singular rejection are unit-tested, not timed here.
- Source `475330cc3795e016d55babe089d398dded542aee` has 28 recorded source/script hashes, all verified unchanged after execution. The report correctly retains `git_dirty=true` because untracked presentation/guide documentation existed at launch. Memoized caching wins request medians in 7/7 settings; replay wins lifecycle medians in 2/7. Small synthetic rational timings are descriptive; memory, incremental inverse/factorization and GPU comparisons are not measured.

- **60 tests pass; Checkpoint 03 complete.** Global comparison: 108 runs / 594 requests / 4,158 method executions, zero policy/value mismatches. Six regimes, 16/32 states, three dataset seeds and three timing repeats. [Tables](output/benchmarks/global_checkpoint_summary.md), [raw archive](output/benchmarks/global_checkpoint.json.gz), [interpretation](checkpoints/03_global_controls_and_lspi.md).
- Global experiment source `51a999a2519ada3b95e08de9d380a1233a109bbf` was clean at launch. [Source CI](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/37545210432) passed 59 tests and all smoke runners; final local validation adds a passing later-iteration LSPI singularity regression for 60 tests total.
- At 32 states, dominated-bridge lifecycle cost is 32.969 ms for classical global elimination versus 96.257 ms for current pruned-local repair; anchor invalidation is 39.642 versus 835.267 ms. Local SCC still helps modular data (7.930 versus 24.282 ms lean warm SCC). Engineering choices differ; these descriptive synthetic results do not isolate locality alone.
- LSPI replay passes eleven tests, including 25 random shared-feature datasets with three deletion requests each, exact every-iterate comparisons, cycles, ties, and singular failure without mutation. [Worked counterexample](checkpoints/lspi_control_example.md), [machine-readable demo](output/benchmarks/lspi_demo.json), [contract](audits/lspi_control_contract.md). No LSPI performance or novelty claim yet.

- **SCC checkpoint: 42 tests passed; 162 runs / 711 requests / 5,688 method executions, zero failures.** All eight matched dense/SCC methods agree exactly with scratch dense policy/value output. Six regimes, 8/16/32 states, three dataset seeds and three timing repeats. [Report](checkpoints/02_scc_controls.md), [full tables](output/benchmarks/scc_checkpoint_summary.md), [compressed raw results](output/benchmarks/scc_checkpoint.json.gz).
- Source `94e7f45107346afb0b9a89d00122b007941e7abc` was clean at launch. [CI 37332934561](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/37332934561) passed 42 tests and all four smoke runners. An independent code review found no correctness defect; benchmark limitations remain documented.
- At 32 states, modular preparation-plus-sequence cost is 7.778 ms for local SCC versus 21.679 ms for warm SCC. Pruned SCC costs 47.027 ms there. Dominated bridges favor pruning (91.129 ms versus 695.543 ms warm SCC); weaker bridges (1026.023 versus 965.084), anchor invalidation (812.678 versus 766.089), and dense influence (627.802 versus 567.670) do not. These are synthetic descriptive medians, not a universal speedup or novelty result.

- **36 tests passed** after adding exact block policy iteration and certified action pruning. Forty random cyclic datasets × four deletions each match exact retained policies and values under enumeration. Deterministic tests cover restored admissible edges, anchor invalidation, and solver failure without mutation. [Proof and publication boundary](audits/local_repair_proof.md).
- **Full cyclic checkpoint completed:** 54 runs, 162 sequential deletion requests, **810 method executions**, zero failures. Exact local/pruned and warm policy iteration match scratch rational policies and values; binary64 warm VI matches policies in these cases. [Results](output/benchmarks/local_checkpoint_summary.md), [raw report](output/benchmarks/local_checkpoint.json.gz).
- Source `70f13773b8e467e99cb429115cc2b437c008708e` was clean at execution. [CI run 36603869847](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36603869847) passed 36 tests and tiny/scaling/cyclic smoke experiments.
- Constructed 16-state dominated-bridge case: pruned block size **4 versus 16**; median request **0.448 ms versus 9.847 ms** for warm exact PI. Preparation plus four requests: **12.502 ms versus 46.365 ms**. Dense influence still selects 16 states; full-sequence pruned cost **47.962 ms versus 42.633 ms**, so no universal benefit is claimed.

- Earlier scaling checkpoint: **26 tests passed**, including more than 100 accepted random deletion requests checked against exact retained-policy enumeration, generated-episode validation, memory-pass checks, and a 32-state case inspecting one affected action. [Special-case proof and contract](audits/nonpolicy_certificate_proof.md).
- [Scaling summary](output/benchmarks/scaling_summary.md) and [compressed raw report](output/benchmarks/scaling.json.gz): **12 settings**, sparse/dense × 16/64 states × 3 seeds, with 3 randomized-order timing repetitions and a separate memory pass. All 192 updating-method runs across timing/memory passes converged and matched the retained numerical reference. Maximum Q error: `1.67e-10` rounded upward. The reference minimum action gap across these runs was approximately `3.74e-4`.
- Scaling source: clean revision `c6ce7331a94a6f9484de0305bf1c2099dd058909`. [GitHub CI run 36600961815](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36600961815) passed all 26 tests plus tiny and scaling smoke runs.
- **Negative result retained:** median of per-seed median request times on the 64-state dense logs was 120.520 ms for global certified repair versus 29.086 ms for warm-started planning. On the 64-state sparse logs it was 32.378 ms versus 42.199 ms. These synthetic, instrumented results motivate testing graph dependence; they do not establish a general speedup or significance.

- Initial milestone: **17 tests passed**; superseded by the current 26-test suite above.
- [GitHub CI run 36522820517](https://github.com/BirariAditya/rl-trajectory-unlearning/actions/runs/36522820517) **passed** on revision `c229e7c`: all tests plus a one-seed baseline smoke run on Ubuntu/Python 3.11.
- `python scripts/run_baselines.py --output output/benchmarks/phase1.json`: **15 cases / 75 method runs**, zero non-noop convergence or policy-agreement failures. [Raw report](output/benchmarks/phase1.json), [interpretation](output/benchmarks/README.md).
- Pilot source revision: `bd37c7927c98494187ed9f20debadc8c2d6e76d4`, clean working tree at execution; the JSON also records source SHA-256 hashes, full datasets, deletion IDs, seeds, solver tolerance, and device.
- Maximum Q error for updating methods versus the rational reference: `5.56e-17` (rounded upward). At seed 0, no-op disagreed in shared-state, bottleneck, coverage-loss, and tie cases, and agreed in the non-policy-row case. The strict gap certificate rejected all three exact-tie requests and used fallback, as intended.
- Added five complete-episode scenarios: shared-state reward deletion, non-policy row deletion, bottleneck propagation, coverage loss, and exact tie.
- Work counters include every Bellman target evaluation (including convergence/certificate scans), transition entries read, predecessor edges indexed, and log transitions applied. Wall time includes remaining Python work and fallback. These counters are not a complete machine-instruction or memory profiler.
- Rational reference uses decimal-string interpretations of logged rewards and discount; the production planner uses binary64. Comparisons therefore use declared tolerances, with exact ties checked separately.

## Next implementation tasks

1. Use the [18 October preparation plan](presentation/2026-10-18/README.md) to study the equations, reproduce the demos, and rehearse all four checkpoints. Keep historical findings and later stronger comparisons together.
2. Match repeated-policy evaluation reuse on replay and cached reconstruction before attributing differences to target corrections. Then add stable numerical factorization controls, measured memory, long deletion sequences and conditioning/failure regimes. These are implementation gates, not novelty claims.
3. State and audit a deletion-specific theorem or advantage beyond established incremental LSPI before neural scale-up. If no such claim survives, reframe as a failure/evaluation study. User-identity batches, larger real tasks and GPU experiments remain staged.

## Limitations and open decisions

- Tiny synthetic results establish correctness checks, not benchmark superiority, novelty, or a deletion-time complexity theorem.
- The legacy `delete_and_repair` baseline rebuilds its predecessor index and scans all rows. The new exact block repair maintains an index and can avoid global verification under its structural/pruning conditions. Its worst case is global, and preparation/anchor invalidation are expensive.
- Policy equality does not certify deletion of released critic, metadata, optimizer, or neural encoder state.
- The exact oracle enumerates policies and intentionally refuses more than 4,096 policies. It is validation overhead, not a competitive planning baseline.
- The restricted no-change certificate remains available. The new block repair handles changed policies but uses rational arithmetic and potentially expensive global preparation. Action elimination is established prior art; this implementation is not a novelty claim. GPU experiments have not run.

## Update convention

At the end of each implementation session, update this file with the current stage, actual commands/results, output paths, completed tasks, next tasks, and blockers. Mark a gate complete only when its evidence exists. Keep failed experiments and scope limitations visible. Do not silently upgrade a hypothesis to a result.

## Activity log

- **2026-10-14:** archived Checkpoint 04 and its provenance, updated research decisions and professor summary, and prepared a comprehensive learning guide plus 26-slide outline with speaker notes, evidence links and anticipated questions for 18 October. This is Markdown presentation material, not a rendered PowerPoint deck.
- **2026-10-13:** implemented cached and memoized LSPI controls, expanded to 74 passing tests, pushed source `475330c`, and completed 648 matching method executions. Ordinary memoization removes much of the apparent advantage over scratch; retained both favorable and unfavorable replay results.

- **2026-10-07:** completed Checkpoint 03 with 4,158 failure-free global-control executions, 60 passing tests, and exact fixed-feature LSPI replay. Added later-iteration singularity atomicity regression and archived the one-shot-downdate counterexample. Strong classical controls undermine the current tabular headline; moved next research priority to LSPI efficiency/prior-art gates while retaining explicit novelty limits.

- **2026-10-05:** reached Checkpoint 02. Added exact SCC planner, matched dense/SCC runner, six correctness regressions, CI smoke coverage, and SCC/prior-art contract. Forty-two tests and CI pass; full matrix completed with 5,688 executions and zero failures. Archived full costs and timed failure regimes. FTVI already combines action elimination with recovered graph locality; that combination is not our novelty.

- **2026-09-29:** implemented and pushed exact block repair, warm full policy iteration, optional bound-certified action pruning, and atomic sequential requests. Added proof notes, 10 new tests, and cyclic checkpoint runner. All 36 tests and GitHub CI passed.
- **2026-09-29:** reached Checkpoint 01 after 810 failure-free cyclic method executions. Archived full preparation-inclusive results, retained dense failure regime, and recorded action-elimination prior art. Novelty remains unproven; checkpoint report specifies next gates.

- **2026-09-29:** implemented and pushed restricted rational certificate and its proof/contract, seven certificate tests, acyclic sparse/dense log generation, repeated timing runner and separate memory passes. All 26 tests passed locally and in GitHub CI.
- **2026-09-29:** completed scaling pilot, archived verified lossless gzip report (170,750 bytes), and added a reproducible summary script. Kept the raw 4 MB report locally under ignored `results/`. Recorded dense-case slowdown and next gates.

- **2026-09-29:** completed three-agent novelty reassessment and revised implementation phases.
- **2026-09-29:** implemented independent rational reference, ordinary prioritized-sweeping baseline, metered experiment harness, reproducible JSON manifest runner, and GitHub test workflow. All 17 tests passed.
- **2026-09-29:** created private GitHub repository and completed the three-seed correctness pilot. Archived 75 runs with source provenance and complete small datasets. GitHub workflow repeats the tests and a one-seed baseline smoke run on pushes and pull requests.
