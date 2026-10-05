# RL trajectory unlearning research

**Current stage:** [Checkpoint 01 reached](checkpoints/01_exact_local_repair.md): exact local policy repair with cyclic/sequential validation and measured cost limits. See [PROGRESS.md](PROGRESS.md) for the live research stage and next tasks.

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
```

The package is at `src/rl_unlearning`. It currently uses only the Python standard library. `delete_and_repair` removes specified trajectories, tries local Bellman backups, checks the retained-model residual and action gaps, and falls back to full warm-started value iteration when it cannot certify the policy. The returned `RepairResult` reports local backups and full residual checks separately.

This first-stage certificate applies to the unique optimal greedy policy of the explicitly defined retained empirical MDP when every state passes the strict action-gap check. The reference `value_iteration` routine stops at a numerical Bellman-residual tolerance; near a tie its finite-tolerance policy can differ from the mathematical optimum, and a warm-started fallback can differ from a zero-started scratch run. A fallback result without a passing action-gap check is a numerical approximation, not a certified policy match. None of these results establishes distributional exact unlearning of an arbitrary RL training process or deletion of independently released critic, optimizer, or metadata state.

The current certificate scans all state-action rows after a deletion. That cost is part of the method's latency and is the main reason the new research phase targets a sound sparse certificate. It must beat warm-started complete planning **after** verification and preprocessing costs are counted. GPU IQL experiments are staged only after the tabular/fixed-feature research gates and will make empirical, retraining-relative claims.

The baseline runner compares no-op, scratch retained-data retraining, warm-started planning, ordinary prioritized sweeping, and the existing certified repair on five tiny deletion scenarios and three seeds. It records dataset/deletion IDs, source hashes, Git state, Python/platform, request and preparation time, work counts, and agreement with an independent rational policy-enumeration oracle. These are correctness pilots; larger repeated experiments are required for speed claims. Python 3.10 or newer is required; there are no third-party runtime dependencies.

The scaling runner adds 16/64-state synthetic sparse/dense acyclic episode logs, randomized method order, repeated timings, and separate Python allocation measurements. Larger models use a tighter binary64 value-iteration reference rather than exponential rational enumeration. The new `NonPolicyCertificate` verifies requests affecting unchanged chosen-policy rows using exact rational statistics; its [proof and scope](audits/nonpolicy_certificate_proof.md) explain global preparation costs, tie handling, and rejection without mutation. It is a special-case baseline for the general sparse-certificate research, with no novelty or full-model privacy claim.

`LocalPolicyRepair` handles chosen-policy changes by solving the backward-closed affected block exactly. Optional anchor-policy bounds prune provably dominated edges; anchor changes trigger full repair and index rebuilding. The [proof and scope](audits/local_repair_proof.md) explain why outside values can remain fixed, the cost of dense rational solves, and the established action-elimination prior art. Its cyclic runner compares local/pruned repair with scratch exact PI, warm exact PI, and warm binary64 VI on identical repeated deletion requests.

Set `solver="scc"` on `LocalPolicyRepair` to use exact policy iteration component by component in sink-first order. The SCC runner compares matched dense/SCC variants on modular, dominated-bridge, weak-bridge, anchor-invalidation, batched-trajectory, and dense scenarios. It includes graph reconstruction and preparation costs. This is a classical planning ablation, not an optimized TVI/FTVI reproduction; [the contract](audits/scc_planning_checkpoint.md) identifies remaining comparator limitations and closer prior art.
