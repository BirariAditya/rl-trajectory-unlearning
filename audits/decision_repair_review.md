# Selective greedy repair for exact retained-data LSPI

16 October 2026. This is a proposed, sound extension of Checkpoint 06, not a claim of publication novelty or a record of completed experiments. The review covers the fixed-feature, finite-state, finite-schedule deterministic learner in this repository. It does not establish neural, online, privacy, or full-internal-state unlearning.

## Recommendation

Combine the existing exact inverse-maintenance control with a persistent index of state action margins. Compute the exact retained weights for every distinct input policy in the requested LSPI schedule. Use the index to find only the states whose greedy action might change, and recompute all actions at those states. This allows real decision changes while preserving the entire retained-only matrix/weight/policy trace. It removes the current certificate's requirement that the whole control sequence remain unchanged.

The important implementation choice is to **keep margin anchors fixed across requests**. Updating every margin after every request would pay the all-state work the index is supposed to avoid. The most recent exact inverse and the older margin anchor are separate objects with different invariants.

## Exact screening rule and proof

Let the feature map be fixed independently of the deleted data. For one input policy p, choose an anchor evaluation weight vector w0. Let g0(s) be its exact greedy action under the repository's fixed action-order tie rule. For every competing action a define

`v(s,a) = phi(s,g0(s)) - phi(s,a)`.

For states with at least one nonzero difference, store

`rho(s) = min over a with v(s,a) != 0 of [v(s,a)^T w0 / ||v(s,a)||_1]`.

These radii are nonnegative because g0 is greedy. A state with no nonzero differences has identical features for every action and has the same tie-selected action for every weight vector. Exclude those states from the sorted index.

At a later retained dataset, evaluate **the same input policy p** exactly to obtain w. Set `delta = ||w-w0||_infinity`. Sort the pairs `(rho(s), state_index)` once at anchor creation. A binary search returns the prefix with `rho(s) <= delta`. Score every action only for that prefix. Begin the output from g0 and replace the selected entries with the newly computed exact greedy choices.

For a state outside the prefix and a nonidentical competitor,

`v^T w = v^T w0 + v^T(w-w0) >= v^T w0 - ||v||_1 delta > 0`.

Thus g0 strictly beats every competitor with different features. Identical-feature competitors remain tied, and the fixed action order keeps g0. The skipped state is therefore exactly correct. States inside the prefix receive the same exact action scan as the direct learner. Equality at the radius must be included in the prefix: a new tie can change the deterministic action. If `delta == 0`, return g0 directly, including all exact ties.

This proof does not require the anchor to be recent, the deletion to be small, an inverse perturbation norm below one, or the current greedy output to equal the anchor output. It requires exact w, fixed features, and a correct original g0. In particular, measure delta from w0, **not from the previous request's weights** while continuing to use the old radii.

## Integration into the finite LSPI schedule

1. Store the current retained-data covariance, reward vector, successor aggregates and policy-specific exact inverse cache, as in `LSPIInversePolicyState`.
2. Maintain a separate immutable margin anchor `(w0, g0, sorted_radii)` for each cached input policy. At initial preparation, charge every score, difference norm, radius, sort and allocation.
3. For a deletion, make candidate statistics/caches. Restart at the declared data-independent initial policy. For each distinct encountered input policy, compute its exact retained matrix/inverse/weights using the existing downdates or final-matrix fallback.
4. If that input policy has an anchor, run the screening rule above. If it is new, compute a full exact greedy output and build its anchor. A newly encountered policy can require global matrix assembly and a fresh inverse: the algorithm does not magically avoid this branch cost.
5. Reuse evaluations and greedy outputs within the request, exactly as the strong baseline does. Emit all scheduled iterations, with their exact matrices, RHS, weights and policies. Preserve cycle, fixed-point and budget-exhaustion semantics.
6. Commit data, inverse cache and new margin anchors only if the entire schedule succeeds. On singular failure, discard all candidate mutations. Keeping only policies encountered in the latest successful schedule bounds the number of anchors by the iteration budget K. Losing an old policy's anchor is a performance cost if that policy later reappears, not a correctness problem.

Induction proves trace equality. The initial policies agree. For an equal input policy, exact retained sufficient statistics give the same matrix and weights as direct retained-only evaluation, or the same singular failure. The screening lemma gives the same next policy. Repeating this for all K steps gives identical full trace and status, even after a policy changes.

The internal margin anchors can retain deleted-data influence. Exact released matrices/weights/policies do not imply that every private cache has been erased. Keep the release contract explicit.

An optional improvement follows from the same proof: the greedy screen does not mathematically require the anchor and current evaluation to share an input policy. The function `argmax_a phi(s,a)^T w` depends only on w and the fixed features. Thus a bounded bank of valid anchors can screen any exact new evaluation weight. Compute each anchor's delta and candidate-prefix length and choose the smallest prefix, paying `O(J(d+log S))` for J anchors. This can avoid creating a fresh margin index at every new-policy branch, although the branch's retained matrix and weights still must be computed correctly. The per-input-policy organization above is a sufficient, simpler first implementation; an anchor-bank extension needs its own cost and refresh ablation.

## Cost accounting and refresh policy

Let S be the number of states, A the number of actions, d the feature dimension, and q the number of states selected by the margin index at one evaluation. Anchor construction uses `O(S A d + S log S)` arithmetic/comparison operations and `O(S+d)` stored margin/choice/weight entries, beyond the inverse cache. A query uses `O(d + log S + q A d)` arithmetic/comparison operations for the screening and scores. Python tuple copies, hashing complete policies and trace materialization still contribute `O(S)` work; the whole request is not state-independent. Rational bit complexity is separate from these operation counts.

Add the existing `O(m d^2)` inverse downdate work per previously cached input policy, dense inversion fallback, copying sufficient statistics, new-policy assembly, new anchor construction and serialization. If a changed branch creates many new policies, it can erase the gain. If most radii are small, q approaches S and screening adds overhead. Report q, actual decision changes, new anchors, matrix builds, inverse fallback, sort cost, tail latency and total lifecycle time.

Keep anchors fixed in the first implementation so behavior is simple to audit. A later refresh can rebuild the entire anchor after enough screened-query cost has accumulated, but that rebuild must be explicit and charged. Do not silently update g0 or w0 without rebuilding its radii. An adaptive refresh heuristic would need a matched ablation before being credited with gains.

## Required adversarial and fairness checks

- A request that actually changes at least one action, yet screens other states; compare every retained-only trace field.
- Distinct action-feature vectors per state, mixed-sign features, multiple greedy actions across states, and successor coupling. The earlier positive/opposite-action generator is insufficient.
- Exact new ties at `rho == delta`, original ties with `delta > 0`, identical features, and `delta == 0`.
- Multiple requests where anchor-relative drift is larger than the last-request drift; cancellation that moves weights back toward an old anchor.
- New input-policy branches, cycles, repeated input policies, final singularity, and an intermediate singular downdate whose final matrix is invertible.
- Comparisons against the same inverse engine and the same full trace interface, both with identical within-request reuse. Also compare an exact feature-deduplicated greedy engine: states with the same ordered action-feature tuples share their greedy result. A stronger canonical grouping can use action differences relative to one reference action, because a state-common score offset does not affect argmax.
- Include setup plus long sequential requests, request tails, memory and failing regimes. Fewer action dot products alone do not establish lower wall time.

## Prior-art boundary

The original LSPI paper already develops linear policy evaluation with data reuse and discusses incremental matrix computation. Inverse maintenance and reuse of a fixed dataset are conventional ingredients. Our addition is an execution strategy for a declared retained-data schedule, not a new policy-evaluation identity. [Lagoudakis and Parr, JMLR 2003](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf).

An especially relevant earlier result is **Quick sensitivity analysis for incremental data modification and its application to leave-one-out CV in linear classification problems**. It bounds linear scores after adding or removing examples without fully updating the classifier, and determines labels when those bounds fix their signs. Its assumptions are regularized convex classification, not policy-dependent LSTD control. The broad idea of certifying a post-deletion prediction with work tied to the modification is therefore established. The full paper's sections 2.3 and 3 explicitly discuss this distinction. [Okumura, Suzuki and Takeuchi, 2015 preprint / KDD 2015](https://arxiv.org/abs/1504.02870).

**Incremental Sensitivity Analysis for Kernelized Models** extends related reasoning to kernelized models and uses prediction bounds to avoid full leave-one-out retraining. The author-hosted search extract was inspected, but the PDF could not be opened by the browser in this session; treat it as an additional identified lead rather than a fully audited theorem match. [Sivan, Gabel and Schuster, author PDF](https://hadarsivan.cswp.cs.technion.ac.il/wp-content/uploads/sites/108/2020/09/Incremental_Sensitivity_Analysis_for_Kernelized_Models.pdf).

Safe screening has a substantial optimization literature. Gap Safe rules eliminate provably irrelevant optimization variables using safe regions and duality gaps. Those are different objects from greedy state decisions, so this is conceptual prior art rather than a claim that their algorithm directly solves LSPI deletion. It rules out presenting the general safe-region-plus-margin principle as new. [Ndiaye et al., JMLR 2017](https://jmlr.org/papers/v18/16-577.html).

AGT explicitly bounds reachable parameter sets under training-data removal and other perturbations, with unlearning among its applications. Our exact linear control trace differs in learner and output contract, but prediction robustness to deletion is already a recognized certification objective. [Sosnin et al., JMLR 2026](https://www.jmlr.org/papers/v27/25-2206.html).

**Novelty assessment:** the screening lemma follows directly from Holder's inequality, sorting radii is a standard index, and the inverse updates are classical. Their combination may be useful engineering and a rigorous specialization to finite LSPI control, but the current evidence does not justify a substantial algorithmic novelty claim. A stronger research contribution would need a deletion-specific complexity result that handles policy changes, a separation from equally optimized controls, and useful performance on non-aliased realistic problems. The present proposal is the next falsifiable experiment toward that question.
