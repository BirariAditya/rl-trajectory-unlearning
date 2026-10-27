# Minari FourRooms frozen-feature deletion bridge

Source `e4978c808a006dae9346997bf7e82c6135572f92`; clean at launch: True.

Dataset `D4RL/minigrid/fourrooms-v0`: 590 episodes / 10010 transitions in the source; local load took 0.750 seconds.

**3 disjoint windows, 9 deletion requests, 27 request executions; 0 policy-sequence mismatches.**

The state dictionary and frozen features are constructed once from each original pre-deletion window and never rebuilt after a request. Preprocessing is charged in every end-to-end method column. Remote download time is excluded; local load time is reported separately.

## Window results

| Window | Episodes | Transitions | States | Preprocess s | Scratch lifecycle s | Exact inverse lifecycle s | Verified interval lifecycle s | Exact inverse end-to-end s | Verified interval end-to-end s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 64 | 1126 | 934 | 0.553 | 75.724 | 46.934 | 49.620 | 47.486 | 50.172 |
| 1 | 64 | 1115 | 844 | 0.785 | 72.639 | 40.747 | 42.313 | 41.532 | 43.098 |
| 2 | 64 | 1097 | 879 | 0.927 | 74.128 | 44.886 | 46.733 | 45.813 | 47.660 |

## Aggregate and verification

Median end-to-end exact-inverse / verified-interval ratio: **0.961x**.

Certified deletion requests: **9 / 9**; exact fallbacks: **0**.

## Scope

- The source is an official external offline-RL trajectory dataset rather than a synthetic transition generator.
- The representation is a deterministic CountSketch-style map over the 7x7x3 image, direction, and action, with explicit action indicators. It is not learned and uses no reward or retain/forget information.
- Actions are restricted a priori to MiniGrid left, right, and forward (0, 1, 2); the official FourRooms environment documents pickup, drop, toggle, and done as unused.
- The representation covers each selected window's original observed state support. Deletion does not remove state entries or change features.
- Equality is for the declared finite LSPI policy sequence, not exact weights, neural offline RL, environment return, privacy erasure, or publication novelty.
