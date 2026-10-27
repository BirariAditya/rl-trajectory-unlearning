# RL trajectory unlearning research

**Current stage: Checkpoint 10 external-validity gate in progress.** [Checkpoint 09](checkpoints/09_verified_numerical_replay.md) established exact policy-sequence certification and a descriptive high-dimensional crossover. The runner now supports separate Python-allocation passes and distinguishes independent dataset seeds from repeated timing on one dataset; 127 tests pass. The next clean archive repeats dimensions 48 and 64 across independently generated seeds before connecting the fixed-feature learner to nontrivial offline-RL trajectories. **No neural-RL, privacy-unlearning, universal-speedup, or publication-novelty claim is established, and GPU experiments have not started.**

For the comprehensive 18 October presentation, start with the [learning guide](docs/LEARNING_GUIDE.md), [33-slide outline and speaker notes](presentation/2026-10-18/TALK_TRACK.md), and [preparation plan](presentation/2026-10-18/README.md).

This project is located at `C:\Users\adity\Research`. Start with the [literature review](literature_review.md), then the [novelty reassessment](novelty_reassessment.md) and revised [implementation plan](implementation_plan.md). The [audits](audits/) cover prior art, proof contracts, theorem risks, and the GPU route. The implemented learners are explicit offline tabular baselines: empirical-model value iteration and rational exact policy iteration. Structural/pruned exact repair is implemented; the general margin-sensitive theorem and LSPI-control bridge remain research tasks.

Run the first tests from `C:\Users\adity\Research`:

```powershell
python -m unittest discover -s tests -v
python scripts/run_tabular_demo.py
python scripts/run_baselines.py --output results/phase1.json
python scripts/run_scaling.py --output results/scaling.json
python scripts/run_local_checkpoint.py --output results/local_checkpoint.json.gz
python scripts/run_scc_checkpoint.py --output results/scc_checkpoint.json.gz
python scripts/summarize_scc_checkpoint.py results/scc_checkpoint.json.gz
python scripts/run_global_checkpoint.py --output results/global_checkpoint.json.gz
python scripts/summarize_global_checkpoint.py results/global_checkpoint.json.gz
python scripts/run_lspi_demo.py --output results/lspi_demo.json
python scripts/run_lspi_checkpoint.py --seeds 0 1 2 --repeats 3 --output results/lspi_checkpoint.json.gz
python scripts/summarize_lspi_checkpoint.py results/lspi_checkpoint.json.gz
python scripts/run_lspi_checkpoint.py --matched-reuse --seeds 0 1 2 --repeats 3 --output results/lspi_matched_checkpoint.json.gz
python scripts/summarize_lspi_checkpoint.py results/lspi_matched_checkpoint.json.gz
python scripts/run_lspi_certificate.py --seeds 0 1 2 --repeats 3 --requests 20 --memory --output results/lspi_certificate.json.gz
python scripts/summarize_lspi_certificate.py results/lspi_certificate.json.gz
python scripts/run_lspi_unique_controls.py --seeds 0 1 2 --repeats 3 --requests 12 --memory --output results/lspi_unique_controls.json.gz
python scripts/run_lspi_multiaction.py --seeds 0 1 --repeats 1 --requests 10 --output results/lspi_multiaction.json.gz --summary results/lspi_multiaction_summary.md
python scripts/run_lspi_verified_float.py --seeds 0 1 --repeats 1 --requests 10 --output results/lspi_verified_float.json.gz --summary results/lspi_verified_float_summary.md
python scripts/run_lspi_verified_float.py --crossover --seeds 0 1 --repeats 1 --requests 5 --output results/lspi_verified_crossover.json.gz --summary results/lspi_verified_crossover_summary.md
python scripts/run_lspi_verified_float.py --highdim --seeds 0 1 --repeats 1 --requests 2 --output results/lspi_verified_highdim.json.gz --summary results/lspi_verified_highdim_summary.md
python scripts/run_lspi_verified_float.py --highdim --seeds 0 1 2 --repeats 1 --requests 2 --memory --memory-methods inverse_policy verified_interval --output results/lspi_verified_highdim_repeated.json.gz --summary results/lspi_verified_highdim_repeated_summary.md
```

The package is at `src/rl_unlearning`. It currently uses only the Python standard library. `delete_and_repair` removes specified trajectories, tries local Bellman backups, checks the retained-model residual and action gaps, and falls back to full warm-started value iteration when it cannot certify the policy. The returned `RepairResult` reports local backups and full residual checks separately.

This first-stage certificate applies to the unique optimal greedy policy of the explicitly defined retained empirical MDP when every state passes the strict action-gap check. The reference `value_iteration` routine stops at a numerical Bellman-residual tolerance; near a tie its finite-tolerance policy can differ from the mathematical optimum, and a warm-started fallback can differ from a zero-started scratch run. A fallback result without a passing action-gap check is a numerical approximation, not a certified policy match. None of these results establishes distributional exact unlearning of an arbitrary RL training process or deletion of independently released critic, optimizer, or metadata state.

The current certificate scans all state-action rows after a deletion. That cost is part of the method's latency and is the main reason the new research phase targets a sound sparse certificate. It must beat warm-started complete planning **after** verification and preprocessing costs are counted. GPU IQL experiments are staged only after the tabular/fixed-feature research gates and will make empirical, retraining-relative claims.

The baseline runner compares no-op, scratch retained-data retraining, warm-started planning, ordinary prioritized sweeping, and the existing certified repair on five tiny deletion scenarios and three seeds. It records dataset/deletion IDs, source hashes, Git state, Python/platform, request and preparation time, work counts, and agreement with an independent rational policy-enumeration oracle. These are correctness pilots; larger repeated experiments are required for speed claims. Python 3.10 or newer is required; there are no third-party runtime dependencies.

The scaling runner adds 16/64-state synthetic sparse/dense acyclic episode logs, randomized method order, repeated timings, and separate Python allocation measurements. Larger models use a tighter binary64 value-iteration reference rather than exponential rational enumeration. The new `NonPolicyCertificate` verifies requests affecting unchanged chosen-policy rows using exact rational statistics; its [proof and scope](audits/nonpolicy_certificate_proof.md) explain global preparation costs, tie handling, and rejection without mutation. It is a special-case baseline for the general sparse-certificate research, with no novelty or full-model privacy claim.

`LocalPolicyRepair` handles chosen-policy changes by solving the backward-closed affected block exactly. Optional anchor-policy bounds prune provably dominated edges; anchor changes trigger full repair and index rebuilding. The [proof and scope](audits/local_repair_proof.md) explain why outside values can remain fixed, the cost of dense rational solves, and the established action-elimination prior art. Its cyclic runner compares local/pruned repair with scratch exact PI, warm exact PI, and warm binary64 VI on identical repeated deletion requests.

Set `solver="scc"` on `LocalPolicyRepair` to use exact policy iteration component by component in sink-first order. The SCC runner compares matched dense/SCC variants on modular, dominated-bridge, weak-bridge, anchor-invalidation, batched-trajectory, and dense scenarios. It includes graph reconstruction and preparation costs. This is a classical planning ablation, not an optimized TVI/FTVI reproduction; [the contract](audits/scc_planning_checkpoint.md) identifies remaining comparator limitations and closer prior art.

`ExactGlobalPlanner` supplies lean full-planning controls without a local reverse index; optional strict-bound action elimination applies inside the SCC graph and refreshes invalidated anchor values. The global runner charges original training and all subsequent requests for every method, including scratch. [Its contract](audits/global_elimination_contract.md) explains differences from the legacy pruned-local implementation.

`LSPIReplayState` and `scratch_lspi` implement a declared finite fixed-feature LSPI schedule with exact rational arithmetic. Deletion replay corrects retained successor-policy targets and matches scratch matrices, weights and policies. Run the [worked example](checkpoints/lspi_control_example.md) to see a one-shot downdate produce the wrong final action. Singular requests do not commit; cycles remain explicit finite-schedule outputs. This is a correctness prototype with dense solves, not an efficiency or novelty result.

`LSPICachedState` reconstructs each matrix from retained C/b/G sufficient statistics. `LSPIMemoizedState` additionally reuses evaluations of repeated policies within one request, clearing its evaluation cache at each deletion. The LSPI runner compares all four methods with preparation and original training charged. Its [archived results](output/benchmarks/lspi_checkpoint_summary.md) cover small synthetic rational problems; incremental inverse/factorization, memory profiling and practical numerical scale remain future comparisons.

`LSPIReplayMemoizedState` gives target replay the same evaluation-cache policy. Use `--matched-reuse` to compare five methods and assert equal solve/action-score counts for both memoized variants. Its output and generated summary have distinct names so the four-method archive remains intact. See the [matched reuse contract](audits/lspi_matched_reuse.md).

`LSPIPolicyCertificateState` tests cumulative deletion bounds against precomputed action margins for every distinct input policy. An accepted request returns the exact retained-only **policy sequence**, without promising retained-only weights or erasure of internal anchors. Inconclusive requests rebuild exactly. `LSPIInversePolicyState` is the stronger classical inverse-update control, also reusing repeated greedy outputs. The [certificate proof and prior art](audits/lspi_policy_certificate.md) state this narrower output contract and its limits. The new runner measures preparation/fallback and optionally separate Python allocations.

`LSPIInverseDeduplicatedState` is a stronger conventional control that scores each
unique ordered action-feature row once. Its `policy_only=True` option avoids
constructing a matrix/weight output trace. `LSPISelectiveRepairState` maintains
immutable margin indexes and computes the exact retained weights before
rescoring only uncertain feature groups. It can repair changed decisions while
returning a full exact trace. `LSPISharedAnchorRepairState` tests reuse of the
initial anchor bank across new policy branches, paying every index query.
[Proof and prior-art audit](audits/decision_repair_review.md). These classes retain
historical internal anchors and do not certify full-state erasure or privacy.

The multi-action runner extends those controls to three actions, unique
non-opposite action features, dimensions 8 and 16, and 10 sequential deletion
requests. It compares the complete LSPI trace with direct retained-only
retraining and charges initial learning, preparation, and every request. The
archived checkpoint uses two independently generated seeds and one timing pass
per seed; its medians are descriptive rather than significance estimates.

`LSPIVerifiedFloatState` solves retained LSTD systems in binary64, then uses an
exact approximate-inverse defect and residual bound to certify each greedy
decision against the rational learner. `LSPIIntervalVerifiedFloatState` applies
the same theorem with outward-rounded binary64 intervals. Inconclusive requests
fall back atomically to exact cached replay. Their output contract is the
complete finite policy sequence rather than exact weights. The
[proof and numerical prior-art boundary](audits/verified_float_lspi.md) includes
an ill-conditioned regression where ordinary binary64 action scoring silently
chooses the wrong sequence.
Use `--memory` for separate `tracemalloc` passes. Generated problems, trajectory
logs, and deletion batches are excluded from those measurements; construction,
initial replay, verifier/fallback state, and sequential requests are included.
These values are Python allocations rather than process RSS or accelerator
memory. Different seeds are independent datasets, whereas `--repeats` re-times
one generated dataset with a newly randomized method order.
