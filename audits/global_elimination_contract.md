# Lean global controls and classical action elimination

Implemented 5 October 2026. This is a stronger planning baseline and a correctness argument for a declared learner, not a new unlearning theorem or a reproduction of FTVI.

## Learner and state

`ExactGlobalPlanner` builds rational empirical counts/reward sums directly from complete trajectories. It shares subtraction/normalization helpers with the existing learner but constructs no reverse dependency index, local closure, or local-repair cache. It retains model statistics, trajectories, and an exact policy/value cache; an optional fixed anchor policy supplies lower bounds. Dense and SCC solvers use the existing terminal-zero empty-row convention and global action order at ties.

The solver's new `allowed_actions` argument accepts a nonempty subset for each state. Both its SCC graph and its improvement steps use that subset, ordered by the original action order. If a warm policy selects an eliminated action, the solver initializes that state to its first admissible action. Returned Q entries cover admissible actions only; equality claims concern optimal policy and state values, not a complete released Q table.

## Sound elimination under deletion

Let Rmax be the maximum of zero and every original observed reward. With discount gamma in [0,1), U = Rmax/(1-gamma) bounds every retained-model optimal state value. Deletion can raise empirical mean rewards but cannot exceed the original sample maximum; newly empty rows have reward zero. Thus the initial U remains valid across all requests, although it can get loose. A fresh scratch learner may derive a tighter U from its retained records.

For a fixed valid anchor policy b, compute its current retained-model values L = V_b exactly. Then L(s) <= V*(s). Every optimal action value satisfies:

```text
Q*(s,a) <= r(s,a) + gamma * P(nonterminal | s,a) * U = upper_Q(s,a).
```

If `upper_Q(s,a) < L(s)`, that action is strictly suboptimal. Removing only such actions preserves the full model's optimal values and tie-ordered policy. The anchor's own action cannot fail this strict test, so each state retains at least one admissible action. Equality is deliberately not pruned.

Anchor values are reused only while every normalized anchor row remains identical. Otherwise the anchor is re-evaluated on candidate retained rows, using SCC planning with one allowed action per state. This also accounts for upstream effects of the changed anchor row. Every deletion rescans all state-action rows against the current bound, allowing formerly pruned actions to re-enter before SCC construction. All candidate statistics, anchor values, and policy/value updates are committed only after successful exact solving. A solver failure leaves the old learner intact.

## What this comparison does and does not isolate

The lean warm and scratch controls remove the unnecessary reverse-index overhead identified in Checkpoint 02. All methods optimize the same rational empirical MDP with the same tie convention. Tiny randomized comparisons use independent retained-only policy enumeration; larger benchmarks compare exact policy and state values across solvers sharing an inner linear-system implementation.

The global elimination baseline is stronger than the current pruned-local implementation in several engineering choices: it applies pruning inside the full SCC graph, uses component-wise fixed-anchor evaluation, filters its initial solve, and refreshes an invalidated anchor instead of permanently disabling pruning. The legacy local method still uses dense anchor preparation and full-action internal SCCs. Consequently whole-method differences cannot be attributed solely to global versus local recomputation. We preserve this distinction in the report instead of claiming a pure locality ablation.

Request time includes subtraction, full bound scans, anchor refresh, SCC construction, exact arithmetic, and cache updates. Preparation time includes original training and anchor/index construction where applicable. Full lifecycle totals include original full-data training once for every method, including scratch controls; the latter then additionally rebuild retained data at every request. Partial work counts expose scans, action evaluations, and component solves but do not count every Python instruction. Policy/value snapshotting and correctness checks are common offline evaluation overhead. Timing repeats share datasets and must not be treated as independent scientific samples.

## Novelty boundary

[Topological Value Iteration Algorithms (JAIR 2011)](https://arxiv.org/abs/1401.3910) explicitly combines graph decomposition with removal of provably suboptimal actions. The global comparator here uses a simpler fixed-anchor bound and does not implement FTVI's heuristic search. Any gain it recovers is evidence that the earlier example can be explained by established planning. Remaining local gains must be assessed against dynamic planning and complete preprocessing/update cost, not claimed as a new RL-unlearning algorithm by themselves.
