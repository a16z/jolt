# Akita high-basis digit-range fold result

Date: 2026-07-31 EDT

## Outcome

Accepted. Akita commit `a56b933c` parallelizes exact-prefix table folds in
dependency-safe geometric waves. Jolt commit `953196965` pins that revision.

At `T = 2^28`, the direct target fell from 5.800 to 0.583 seconds. The
optimization removes 6.20 seconds from all digit-range proofs and 5.09 seconds
from the complete evaluation proof.

| `2^28` span | Parent | Wavefront fold | Change |
|---|---:|---:|---:|
| `digit_range_fold_lanes` | 5.800429 s | 0.582825 s | -90.0% |
| Product substages | 11.362907 s | 6.178730 s | -45.6% |
| All digit-range proofs | 14.979955 s | 8.781831 s | -41.4% |
| Stage-1 sumchecks | 15.261617 s | 9.062252 s | -40.6% |
| Evaluation proof | 31.999698 s | 26.911621 s | -15.9% |

The candidate proof verified. Maximum RSS was 85,159,297,024 bytes
(79.311 GiB), versus 87,632,314,368 bytes (81.614 GiB) in the parent trace,
and the process reported zero swaps. The implementation allocates no new
table, so analytical peak memory is unchanged; the observed RSS decrease is
not attributed to the change.

Whole-prover time moved from 188.611 to 190.417 seconds because the unchanged
commitment span was 3.813 seconds slower in the candidate run. The localized
evaluation result is the causal measurement; the whole-run movement is
ordinary cross-run variation.

## Focused signal

A standalone basis-64 prover with a `2^26` domain and three-quarter live
prefix gave three stable runs per revision:

| Variant | Runs | Mean |
|---|---|---:|
| Serial parent | 1.173109, 1.172334, 1.172611 s | 1.172685 s |
| Wavefront candidate | 0.671124, 0.672413, 0.671960 s | 0.671832 s |

This is a 42.7% end-to-end reduction before the full target run.

## Why the schedule is safe

The old compaction visits pair `i`, reads input rows `2i` and `2i + 1`, and
writes the folded value to row `i`. A fully parallel in-place loop is invalid:
one worker could overwrite a row that another worker has not read.

The accepted schedule first folds `[0, a)` in order. It can then fold output
rows `[a, 2a)` in parallel because their inputs are `[2a, 4a)`, disjoint from
their outputs. Rows `[a, 2a)` are inputs only to pairs in `[a/2, a)`, which
the preceding wave has already consumed. Repeating with a doubled `a`
preserves the invariant. The first 4,096 outputs stay serial to avoid Rayon
overhead on small tables.

Every pair uses the same `fold_pair(left, right)` callback and lands at the
same index as before. The change is prover scheduling only: claims, challenges,
round messages, transcript bytes, verifier behavior, and protocol geometry
are unchanged.

## Scope

The payoff is concentrated at high-basis, high-`T` schedules. The `2^28`
D128 root uses basis 64 and materializes a 157,290,624-row eight-lane table.
The current `2^26` schedule is largely on the low-basis direct-leaf path, so
this optimization is not expected to move its headline by the same amount.

## Validation

- exact-prefix dense-reference tests crossing multiple waves;
- the same tests with and without Akita's `parallel` feature;
- all 268 parallel and 267 sequential `akita-prover` tests;
- both repository-wide Akita warning-denying Clippy configurations;
- Akita muldiv, forced-K256 muldiv, and committed-program muldiv in Jolt;
- standard and ZK Jolt muldiv suites;
- workspace Clippy for `host` and `host,zk`, plus scoped Akita Clippy;
- `cargo fmt`, lockfile validation, and `git diff --check`;
- exact `2^28` proof verification with zero swaps.

The Akita line-cap preflight still reports the two pre-existing baseline files
`backend/onehot/tests.rs` and `compute/cpu.rs`; this change does not touch
either file. The local `taplo` executable panicked in macOS system
configuration before checking files.

## Trace

`benchmark-runs/perfetto_traces/akita_28_fold_waves.json`

SHA-256:
`bd8f1970c81b02596aebc5cbe2a3ebf7e7eee865022cde9eae20820f8dc9ca2f`
