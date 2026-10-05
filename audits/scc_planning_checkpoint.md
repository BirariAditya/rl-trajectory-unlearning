# SCC planning baseline: contract and prior art

Checked 5 October 2026. This is a baseline implementation and correctness argument, not a new algorithmic theorem.

## Prior art changes the novelty boundary

[Dai and Goldsmith, TVI (IJCAI 2007)](https://www.ijcai.org/Abstract/07/300) decomposes an MDP into strongly connected components and solves them in dependency order. [Dai, Mausam, Weld and Goldsmith, Topological Value Iteration Algorithms (JAIR 2011)](https://arxiv.org/abs/1401.3910) additionally describes FTVI: provably suboptimal actions are eliminated to expose smaller components. The [FTVI conference paper](https://ojs.aaai.org/index.php/ICAPS/article/view/18138) appears in the ICAPS 2009 issue; the portal's 2021 publication field is a hosting date, not evidence of a new 2021 algorithm.

Thus the combination of safe action pruning and recovered graph locality is already established. This is a closer conceptual comparator to our dominated-bridge demonstration than action elimination alone. Any eventual novelty claim must survive comparison with this combination. The specific dynamic trajectory-deletion contract might support further results, but it does not by itself make the existing planning machinery new.

## Implemented solver and proof

`scc_policy_iteration` rebuilds the directed support graph of the supplied block using **every action's positive nonterminal transitions in the retained model**. Iterative Kosaraju decomposition avoids a Python recursion limit. It processes SCCs in sink-first order and uses the existing exact rational policy-iteration solver within each component. This is an exact-PI analogue of classical decomposition, not a reproduction of TVI's numerical value iteration or FTVI's heuristic search/action elimination.

For a component C, any outgoing transition either stays in C, enters a previously solved successor component, or exits the supplied block. The first kind is handled in C's linear systems; the second uses already established optimal values; the third uses the caller's boundary values. For a full solve there is no external boundary. For local repair, correctness of that boundary is inherited from the existing backward-closure/pruning contract. Discounted Bellman contraction gives a unique optimal value on C with these fixed boundary contributions. Induction over sink-first component order yields the full block's unique optimal value. Applying the same action order at exact ties yields the same deterministic policy as dense exact planning.

The all-action graph is essential: a previously unchosen action can become optimal after deletion. Using only the old greedy policy graph would miss dependencies. Candidate retained rows are used before committing, so deleting the last support edge can split an SCC. The pruned variant only prunes the **outer closure**; internal SCC construction keeps all actions, including dominated ones. Boundary values can overlap the block in the cache, but already-solved component values take precedence. Temporary maps accumulate results; no learner state is committed if a later component fails.

## Cost and comparison contract

- Graph rebuild scans every action row and successor entry in the supplied solve block, including rows that did not change. Counters expose these scans. SCC traversal is linear in the materialized graph size; exact rational policy iteration inside each component can be expensive. No bound on rational bit complexity is claimed.
- `max_iterations` is a **per-component** budget; returned iteration counts and linear-system sizes aggregate all component solves. This differs from dense PI's single-block budget.
- Dense and SCC variants use the same rational rewards, transition counts, discount, empty-row rule, and tie order. Within either solver, compare local/pruned repair against warm full and scratch full solves. Keep the dense solver as an ablation.
- All methods share the `LocalPolicyRepair` scaffold. Warm and scratch baselines therefore pay reverse-index overhead that a purpose-built global planner could omit. Results are not a comparison against the fastest available global planner.
- Timings include graph rebuilds, model/index updates, and full fallback. Preparation includes original training and, when used, dense anchor-policy evaluation. Policy/value snapshots and reference comparisons are excluded equally as offline evaluation overhead. Partial work counters are not a full instruction/allocation ledger.
- Weak-bridge cases change bridge rewards from -1000 to -1, exposing the looseness of the global reward upper bound. Anchor-invalidation cases start by deleting a high-reward sample from an anchor row, which triggers full repair and permanently disables pruning for that learner. Block deletion groups multiple trajectory requests; it is a batch proxy, not an implemented user-identity schema.

## Remaining gate

Passing this checkpoint does not complete the mature-planner comparison: FTVI-style internal action elimination, a standalone global planner without local-index overhead, dynamic SCC maintenance, and external optimized implementations remain untested. The next theoretical step must address deletion-specific sensitivity and complete request cost, or the project should pivot to the fixed-feature control question already described in the plan.
