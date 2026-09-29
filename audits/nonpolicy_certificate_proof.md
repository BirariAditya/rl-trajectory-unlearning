# Restricted no-policy-change certificate: implementation contract

Status: implemented special-case correctness argument, 2026-09-29. This is a direct Bellman-optimality consequence, **not a novelty claim or the general sparse-influence theorem**.

## Model and preparation

`NonPolicyCertificate` owns trajectory records and rational count/reward statistics. Rewards and the discount are interpreted as exact rationals from their decimal strings. Unseen state-action rows have reward zero and transition to termination. Its declared learner returns the exact optimal deterministic policy with a fixed action order for ties. This differs numerically from binary64 value iteration and its tolerance stopping rule.

Given a candidate policy π, preparation constructs and solves `(I − γPπ)V = rπ` exactly. For every action, it verifies `r(s,a) + γP(s,a)V ≤ V(s)` and requires strict inequality when `a` precedes `π(s)` in the tie order. Thus π is the tie-ordered optimal policy. Invalid starting policies are rejected. Preparation scans all data and actions and uses dense rational elimination; its arithmetic operation cost includes O(S³) elimination and its storage includes O(S²) temporary matrix space. Rational bit costs can grow substantially. This expensive stage is not charged as free preprocessing.

## Accepted deletion condition and proof

For a deletion request, construct candidate retained statistics only for affected rows. Require:

1. Every affected chosen-policy row has exactly the same normalized reward and transition distribution as before deletion. Unaffected chosen rows are unchanged automatically.
2. Every affected action remains below the cached value, with the same tie-order rule as preparation.

Condition 1 leaves `Pπ` and `rπ` unchanged, so the cached V is still the exact retained-model value of π. Unaffected actions retain their old verified inequalities; condition 2 verifies the remaining actions. Hence the retained Bellman optimality operator maps V to itself. Discount contraction makes this V the retained optimal value. The tie-order inequalities establish equality of the complete deployed policy to the declared exact learner's policy.

The argument composes over successive accepted deletions: each request preserves the same verified value and policy and commits the new retained statistics. If a request is inconclusive, no statistics or trajectory records are changed. Rejected requests must be sent to a general retained-data solver; rejection does not mean the old policy is wrong. In particular, deleting an identical redundant sample from a chosen row can pass, but a changed chosen transition/reward row cannot pass this special-case test.

## Request cost and output scope

Let m be the number of transitions in requested trajectories, k the number of affected rows, and e the sum of successor counts in the affected rows. Validation uses direct ID-dictionary lookup. Candidate copies, normalization, and action checks use O(m + k + e) arithmetic/dictionary operations, with no scan over all active IDs, states, or actions. Rational arithmetic has nonconstant bit complexity. The returned policy is the existing read-only mapping, not a fresh O(S) materialized table. Request allocation is O(k + e + number of requested IDs). Stored preparation memory includes all active logged transitions and their statistics.

The implementation reports affected rows, checked actions, transitions read, and copied successor entries. These counters do not claim to measure all rational operations or allocations. No wall-clock speedup follows from this special-case bound alone; amortization must include preparation. The unchanged policy value is valid, but action values of nonchosen actions may change, and the method returns no repaired Q table. This is a behavioral policy certificate, not a guarantee about a released arbitrary checkpoint or privacy indistinguishability.

## Validation and outstanding work

Tests cover exact tie order, deletion that makes an alternative superior, rejection atomicity, repeated requests, empty requests, duplicate/missing IDs, redundant chosen-row samples, and random accepted deletions checked by independent rational policy enumeration. A 32-state test checks that one affected non-policy row examines one action without materializing a new policy.

The general case where chosen-policy rows change remains open in this repository. Next work: measure preparation amortization and request cost across favorable and unfavorable deletions, implement a general fallback integration, and attempt a bounded influence envelope without incorrectly freezing changing boundary values.
