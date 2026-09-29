# Checkpoint: exact block repair with certified action pruning

Status: implemented correctness checkpoint, 2026-09-29. Applies to a finite discounted **rational empirical MDP**, policy/value output, and trajectory deletion. It is not distributional unlearning for neural RL. The arbitrary margin/distance envelope in the research proposal remains open.

## Exact structural repair

Preparation learns the exact tie-ordered optimal policy and values by rational policy iteration and builds a reverse graph over all nonterminal state-action support edges. A request identifies rows whose **normalized model actually changes**; redundant sample deletion may change counts without changing the model. Let C be their source states. Let B be C and every predecessor reachable in the OLD all-action support graph.

Every state outside B has unchanged rewards/transitions and no edge into B. Its control problem is therefore unchanged, so its cached optimal value is still exact. Inside B, solve the retained Bellman control problem with cached outside values as boundary conditions. The resulting combined values satisfy the full retained Bellman optimality equations; uniqueness under discounting gives retained-only optimal values. Exact greedy action comparisons with the declared tie order give the retained-only learner's policy, including exact ties. This handles changed chosen-policy rows, policy switches, and cycles.

The crucial boundary condition is backward closure. Freezing outside values for a radius-truncated region without a separate error argument is invalid. The tests include a predecessor whose relevant action was previously non-greedy, and a cyclic component whose chosen action changes after deletion.

Deletion introduces no new nonterminal support edges in the unsmoothed count model. The old graph may conservatively include an edge removed by the request. After a successful solve, update the reverse index; later requests then use the current graph. Invalid requests and failed solves leave the state unchanged. Repeated requests preserve the invariant by induction. The returned policy mapping is a live view; historical snapshots require an explicitly charged copy.

## Safe action pruning using an unchanged anchor policy

Optionally supply an anchor policy b. Preparation computes its exact value L = V^b. Define `R_max = max(0, maximum original logged reward)` and `U = R_max / (1 − γ)`. Every retained-only empirical row has mean reward at most R_max, including the empty-row zero convention, so U remains an upper bound on every policy value after arbitrary **deletions only**.

For a row `(s,a)`, compute

```text
upper_Q(s,a) = r(s,a) + γ U Σ_{t nonterminal} P(t|s,a).
```

If `upper_Q(s,a) < L(s)`, the action is strictly suboptimal. Omit its edges from the reverse graph. Equality is never sufficient for pruning because of tie-breaking. The lower bound remains valid across a request only if all normalized anchor-policy rows are unchanged. Check this on the changed rows; there is no global request-time scan.

For an anchor-preserving request, recompute pruning status only for touched rows. Seed the closure with every changed row that was admissible before OR is admissible afterward. If both versions are certified dominated, the row's change cannot affect the policy or optimal values. Traverse the **old admissible** reverse graph. New admissible outgoing edges can arise only at seeded source states; this makes old-graph closure sufficient. Restore newly admissible reverse edges at commit so future requests remain sound.

Solve the selected block using **all actions**, with exact cached outside values. Why does freezing the outside still work despite ignored edges? Boundary values lie between L and U. Anchor rows are unchanged, so the block's optimal values remain at least L; U is a Bellman supersolution, so they remain at most U. Every omitted outside-to-inside action therefore remains strictly below L at its source. All non-omitted outside actions have unchanged targets entirely outside the block. Thus the combined value vector satisfies the full retained Bellman equations. This establishes exactness without assuming that the old optimal policy survives inside the block.

If any anchor row changes, the implementation **does a full exact solve, disables pruning, and rebuilds the unpruned reverse graph**. All this work is charged to the request. Subsequent requests use ordinary structural repair. It does not reuse an invalid lower bound. Refreshing a new anchor efficiently is future work.

## Cost and scope

Let m be deleted transitions, k touched rows, e their stored successor entries, b the number of selected states, E_B reverse edges visited, and I policy-iteration rounds. With dense exact elimination, the request performs O(m+k+e+E_B) index/statistic work plus block solution work approximately O(I(b³ + action-successor work + b²)). Sorting selected states and outputting action deltas are also charged. The exact bit complexity depends on rational numerator/denominator growth; these are arithmetic-operation descriptions, not constant-time machine guarantees.

Preparation includes full data aggregation, exact full policy iteration, optional anchor evaluation, all-action pruning checks, and graph storage. Report it separately and amortize it over actual requests. Dense matrix storage is quadratic in solved block size. The code reports system sizes, examined actions, visited reverse edges, copied row entries, index changes, and full-index rebuilds. Global or nearly global influence can make local work worse than warm-start full planning. The benchmark includes a force-full warm exact PI baseline that skips closure traversal, and a binary64 warm-start VI baseline using a different numerical learner.

## Prior art and publication gate

The structural argument is dynamic programming on an unaffected subproblem. It must be compared with incremental policy iteration and graph/SCC-aware planners, not sold as a first local planner. Pruning suboptimal actions with upper/lower value bounds is established at least by [Elimination of Suboptimal Actions in Markov Decision Problems (1973)](https://pubsonline.informs.org/doi/10.1287/opre.21.3.848) and the [action elimination algorithm (1975)](https://research.tue.nl/en/publications/the-action-elimination-algorithm-for-markov-decision-processes/). [Incremental MDP verification](https://prismmodelchecker.org/papers/RR-12-05.pdf) and [prioritized sweeping](https://publications.ri.cmu.edu/prioritized-sweeping-reinforcement-learning-with-less-data-and-less-real-time) are required planning references.

The implemented integration supplies a concrete exact deletion target, maintained indexes, sequential-request handling, action-pruning invalidation, and falsifiable measurements. **This is a substantial repository checkpoint, not a claim of publishable novelty.** A paper still needs a stronger deletion-specific theorem or advantage against mature incremental/SCC/action-elimination baselines, plus evidence beyond constructed tabular cases.

## Validation

The current suite includes 40 random cyclic datasets with four sequential deletions each, checking exact policies and values against rational policy enumeration; chosen-action switches; non-greedy predecessor propagation; removed/restored support edges; newly admissible actions; anchor invalidation; empty/redundant requests; ties; and solver-failure atomicity. The checkpoint runner adds modular cycles, dominated inter-component bridges, and dense cycles, with identical sequential requests for local, pruned local, scratch exact, warm exact, and warm binary64 VI methods.
