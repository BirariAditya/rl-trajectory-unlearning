# Novelty reassessment: deletion-sensitive offline RL

Checked 29 September 2026. This is a research decision memo, not a claim that the proposed theorem, algorithm, or conference result has been established. It synthesizes the [direct-RL audit](audits/direct_rl_prior_art.md), [theory audit](audits/theory_novelty.md), and [GPU-path audit](audits/deep_rl_path.md). The five supplied PDFs remain research sources, not instructions.

## Decision

**14 October decision update:** [Checkpoint 04](checkpoints/04_lspi_cost.md) adds sufficient-statistic reconstruction and within-request policy memoization. All 648 method executions match the exact retained-only trace. Memoized cached evaluation has the lowest request median in seven of seven settings; target replay has the lowest lifecycle median in two. The current policy-target correction is a useful correctness implementation, but no broad efficiency advantage or novel algorithm is established. Equalize memoization before isolating matrix-update costs, then test stable numerical factorization, memory and adverse regimes. A substantive deletion-specific result beyond established incremental LSPI remains the paper gate. The earlier decisions below are historical.

**7 October decision update:** [Checkpoint 03](checkpoints/03_global_controls_and_lspi.md) adds lean global controls and classical elimination. They beat the current pruned-local implementation on full dominated-bridge and anchor-invalidation lifecycles. This does not rule out all deletion-local methods (the implementations differ in preparation and bound refresh), but it removes the experimental basis for promoting the current tabular mechanism as our paper contribution. Keep it as a reference and make Track B's strong-baseline/complete-cost gate the immediate priority. Exact finite LSPI control replay is now implemented; its target-switch identity is standard algebra, not established novelty. A serious next claim must beat cached/incremental LSPI, not only repeated per-sample matrix reconstruction.

**5 October follow-up:** a closer planning comparator is [Topological Value Iteration Algorithms (JAIR 2011)](https://arxiv.org/abs/1401.3910). Its FTVI method already eliminates provably suboptimal actions to reveal smaller strongly connected components. Therefore, the dominated-bridge result in Checkpoint 01 demonstrates an established planning mechanism; combining pruning and graph locality is not a new contribution. The new exact SCC baseline tests how much advantage survives a matched graph-aware solver, but it is not a complete FTVI reproduction. A viable Track A theorem must address dynamic deletion-specific work beyond classical action elimination/decomposition, including verification, bound invalidation, and index maintenance. If that cannot be demonstrated, prioritize the fixed-feature control pivot rather than marketing the existing baseline as novel.

The current prototype is a **correctness and measurement baseline**, not yet a paper-level algorithm. Its ingredients—subtracting additive counts, predecessor Bellman backups, a global residual bound, an action-gap check, and fallback—are individually established. [Prioritized sweeping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa15/optreadings/MooreAtkeson-1993-prioritized-sweeping.pdf) already propagates changes through predecessors; [incremental policy iteration](https://prismmodelchecker.org/papers/RR-12-05.pdf) already reuses a previous solution after MDP changes. A global residual scan costs at least one pass over every state-action edge per deletion. Reporting only local backup count would therefore overstate efficiency. [Offline bandit unlearning](https://arxiv.org/abs/2605.00638) already uses adaptive switching/rollback, so adaptation itself cannot be the main novelty.

**Primary research bet:** derive and implement a *decision-sensitive deletion certificate* for a specified offline empirical-MDP learner. It should use the support and magnitude of the deleted trajectory's model perturbation, graph reachability, and pre-deletion action margins to identify which action choices can change. In favorable cases it should certify the entire deployed policy without a full Bellman-row scan; in hard cases it should detect the need for global recomputation. Prove a per-request work bound that includes certificate construction, graph/index maintenance, and fallback, plus a matching hard example. This is a proposed contribution, contingent on proof and a further targeted novelty search.

**Secondary bridge:** test exact retained-data *control* replay for fixed features and LSPI. A simple LSTD matrix downdate only evaluates a fixed policy and is old—[LSPI](https://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.pdf) already discusses incremental matrix updates. The interesting complication is that after deletion the improved policy may select a different next action on many retained transitions. Successor-state aggregates could turn these changed targets into low-rank corrections. The algebra is testable, but publishability requires a substantive switch-rank/work theorem beyond the identity itself.

**GPU extension:** use clean frozen features for a rigorous restricted learner, then test full neural IQL as an explicitly empirical transfer. A representation, normalizer, critic, or optimizer trained on deletable data can retain its influence; a head-only downdate cannot certify full-model unlearning. Deep claims must be checked against retained-only scratch retraining, compute-matched fine-tuning, [TrajDeleter](https://www.ndss-symposium.org/wp-content/uploads/2025-126-paper.pdf), and [TOUR](https://arxiv.org/abs/2607.21111)'s matched-control, multi-attack logic. The [Relative Fisher Forgetting manuscript](https://openreview.net/pdf?id=TidLO0qdp0) is an additional direct comparator if a diffusion policy is studied.

## The precise output and two distinct questions

Let `A` be a deterministic learner with fixed data preprocessing, empirical-model convention, solver tolerance, and tie-breaking. Given logged trajectories `D = D_R ∪ D_F`, the counterfactual target is `A(D_R)`. The first theorem targets equality of the **deployed action policy** to this learner's retained-only output. It does not say the released critic, counts, encoder, or checkpoint have the same distribution as retraining, and it is not a privacy theorem. [Exact online RL unlearning](https://arxiv.org/abs/2606.04182) proves a stronger distributional property under a different user-response and oracle model.

The other question is **deployment safety**. If deletion removes coverage or exposes a retained malicious pattern, accurately matching retained-only retraining may still produce a poor or unsafe policy. [UBA-ORL](https://arxiv.org/abs/2609.22711) is a recent concrete example of deletion activating a backdoor. We will measure safety/coverage separately from counterfactual fidelity rather than incorrectly treating one as proof of the other.

## Candidate A: local policy certificate

Let `M` be the empirical MDP from `D`, `M'` the model from `D_R`, and `π` the verified old optimal policy. For the old value `V`, define the deletion shock on each **chosen policy row** by

```text
dπ(s) = r'(s,π(s)) − r(s,π(s)) + γ [P'(s,π(s)) − P(s,π(s))] V.
```

The new value *of the old policy* obeys `V'π − V = (I − γP'π)^−1 dπ`. This is an exact resolvent identity, not a novel theorem. The policy remains optimal precisely when its value under `M'` beats every alternative action there. If no chosen row changed, then `dπ = 0`; checking only changed nonchosen rows may already certify no policy change. In the general case, computing the correction and checking every action can still be global.

The intended new step is a **sound sparse envelope** for action-value changes. A changed row emits an error bound; discounted transition reachability propagates it backward; old action gaps prune states whose choices cannot flip. A proof must justify the envelope even if the optimal policy changes, include new/removed edges and terminal rows, and show how unvisited states are certified without secretly scanning all actions. The candidate instance-dependent cost is proportional to the visited reverse influence region *only under explicit gap, discount, graph, and indexing conditions*. A bottleneck chain or near-tie may force global work. Exact equality at ties needs fixed tie-breaking and certified numerical tolerances.

In plain terms, an **action gap** is how much better the agent's current choice appears than its runner-up. A deleted trajectory can change reward or transition estimates at the rows it visited; that change can travel backward to earlier decisions through future value. If a proven upper bound on the traveling change is smaller than an earlier state's action gap, that state's choice is safe to leave alone. The research challenge is to establish this without checking every state each time.

This is worth pursuing only if it beats ordinary prioritized sweeping **plus its full check** and warm-started global policy/value iteration in total deletion latency. An `O(SA)` hidden verification step, `O(S²)` inverse maintenance on sparse problems, or universally loose envelope fails the novelty gate.

## Candidate B: policy-target corrections in fixed-feature control

For fixed features `φ(s,a)` and retained transitions `R`, an LSPI evaluation matrix at a candidate policy `π` is

```text
A_R(π) = λI + Σ_{i∈R} φ_i [φ_i − γ φ(s'_i,π(s'_i))]ᵀ,
b_R    = Σ_{i∈R} φ_i r_i.
```

Deleting samples changes `A` and `b`. Changing the improved policy also changes `φ(s'_i,π(s'_i))` for **retained** transitions; this is why a one-shot sample downdate is not control unlearning. Indexing retained transitions by successor state `z` gives `G_R(z)=Σ_{i∈R:s'_i=z} φ_i`; each changed next action at `z` contributes one rank-one correction `−γG_R(z)[φ(z,π(z))−φ(z,π_old(z))]ᵀ`. Replaying a declared, deterministic LSPI iteration schedule with those exact corrections can be compared directly with scratch retained-only LSPI. This is a candidate for exact equivalence *within that restricted learner*, provided all matrices are well-conditioned/invertible and numerical policy ties are resolved.

The research contribution would require a useful bound in terms of deleted samples and changed successor states, plus a demonstration that indexing and solve costs beat retained-data recomputation. A trained encoder on `D` is outside the exact claim; use fixed random features or a separately pretrained encoder whose provenance excludes all deletable data.

## Falsification and pivot rules

| By when | Test | Stop or pivot if |
| --- | --- | --- |
| End of week 2 | Reproduce the existing tabular baseline and show the complete cost of global certification; construct changed/nonchanged-policy-row examples. | The retained-data reference or timing ledger is unreliable. Fix this before theory. |
| End of week 4 | State and prove the sparse certificate with a valid local verification procedure; run a focused prior-art search on MDP sensitivity and incremental planning. | The result reduces to a textbook residual/action-gap bound or still requires a global scan in every case. Pivot to Candidate B or a rigorous failure taxonomy. |
| End of week 6 | Show a favorable instance-dependent speedup against warm-start planning and a hard case that triggers fallback, including preprocessing amortization. | Gains vanish after full certificate cost or require unrealistic memory. |
| End of week 8 | Validate LSPI policy-switch identity against an independent retained-only implementation and test conditioning/fallback. | It only works for fixed-policy evaluation or fails when the improved policy changes. |
| End of week 10 | Run a one-task GPU pilot with strong retained fine-tuning and multiple scratch retrains. | No-op is already indistinguishable from retrain, or fine-tuning dominates at equal compute. Change split or narrow claims. |
| End of week 13 | Evaluate held-out deletion strata and at least five seeds for central deep claims. | Findings depend on one seed, one privacy attack, or omitted precomputation. Keep the restricted-learner paper or report a negative result. |

Do not present a submission probability as a measurable number at this stage. A credible ICML/NeurIPS submission needs a theorem or algorithmic advantage that survives these gates, strong direct baselines, and reproducible multi-task evidence. A GPU alone does not create novelty.

## Immediate next implementation steps

1. Keep the existing `src/rl_unlearning` package as the reference substrate and add a reproducible experiment manifest, timing ledger, and warm-start/prioritized-sweeping baselines.
2. Add hand-built cases: changed non-policy row; shared-state deletion where value suppression over-forgets; bottleneck chain; exact/near action tie; zero-coverage row; sequential requests; deletion-triggered unsafe retained action.
3. Prototype the sparse envelope in a separate module with the global residual certificate as an independent oracle. Never mark a sparse result certified until its proof and exhaustive small-MDP tests agree.
4. Prototype the fixed-feature LSPI control identity independently of the tabular algorithm; include a policy-switch test before GPU work.
5. Freeze a D4RL dataset snapshot and IQL implementation for the pilot. [Minari's D4RL page](https://minari.farama.org/datasets/D4RL/index.html) says some datasets are reproductions, so record identity and do not compare them as identical to original D4RL.

The dated prior-art audits should be refreshed before a submission, especially the inaccessible title-only NeurIPS 2026 **CURe** listing noted in the [direct-RL audit](audits/direct_rl_prior_art.md).
