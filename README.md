# RL trajectory unlearning research

**Current stage:** tabular baseline validation. See [PROGRESS.md](PROGRESS.md) for completed work, test evidence, research gates, and next tasks.

This project is located at `C:\Users\adity\Research`. Start with the [literature review](literature_review.md), then the [novelty reassessment](novelty_reassessment.md) and revised [implementation plan](implementation_plan.md). Three dated audits in [audits](audits/) detail direct RL prior art, theorem risks, and the GPU route. The code is currently a small offline tabular **baseline**: learning means trajectory counts followed by empirical-model value iteration, so its retained-data counterfactual is explicit. The proposed sparse certificate and LSPI-control bridge are research tasks, not implemented results.

Run the first tests from `C:\Users\adity\Research`:

```powershell
python -m unittest discover -s tests -v
python scripts/run_tabular_demo.py
python scripts/run_baselines.py --output results/phase1.json
```

The package is at `src/rl_unlearning`. It currently uses only the Python standard library. `delete_and_repair` removes specified trajectories, tries local Bellman backups, checks the retained-model residual and action gaps, and falls back to full warm-started value iteration when it cannot certify the policy. The returned `RepairResult` reports local backups and full residual checks separately.

This first-stage certificate applies to the unique optimal greedy policy of the explicitly defined retained empirical MDP when every state passes the strict action-gap check. The reference `value_iteration` routine stops at a numerical Bellman-residual tolerance; near a tie its finite-tolerance policy can differ from the mathematical optimum, and a warm-started fallback can differ from a zero-started scratch run. A fallback result without a passing action-gap check is a numerical approximation, not a certified policy match. None of these results establishes distributional exact unlearning of an arbitrary RL training process or deletion of independently released critic, optimizer, or metadata state.

The current certificate scans all state-action rows after a deletion. That cost is part of the method's latency and is the main reason the new research phase targets a sound sparse certificate. It must beat warm-started complete planning **after** verification and preprocessing costs are counted. GPU IQL experiments are staged only after the tabular/fixed-feature research gates and will make empirical, retraining-relative claims.

The baseline runner compares no-op, scratch retained-data retraining, warm-started planning, ordinary prioritized sweeping, and the existing certified repair on five tiny deletion scenarios and three seeds. It records dataset/deletion IDs, source hashes, Git state, Python/platform, request and preparation time, work counts, and agreement with an independent rational policy-enumeration oracle. These are correctness pilots; larger repeated experiments are required for speed claims. Python 3.10 or newer is required; there are no third-party runtime dependencies.
