# Upstream Akita as protocol and prover stacks

Akita will be upstreamed as two stacked Jolt changes. `perf/akita-protocol-opts`
owns the proof statement, verifier behavior, and the generic prefix-packed
opening API. `perf/akita-prover-opt` adds the modular prover and preserves the
CPU and memory behavior of `perf/packed-onehot`. The split is a review boundary:
the prover branch may be larger, but it must not introduce new transcript,
claim, or verification semantics.

## Boundaries and invariants

`jolt-openings` owns one packing primitive: `PrefixPackedLayout`. It places an
ordered set of equal-point logical polynomials in a fixed power-of-two prefix
capacity and reduces them to one physical opening. It binds the logical point,
ordered evaluations, capacity, and semantic layout digest before sampling the
selector point. Unused slots are zero. The removed arbitrary-point packed API
is not retained as a second path.

`jolt-claims` owns Jolt meaning: column order, capacities (64 slots for K16 and
32 for K256), logical arities, layout digests, and zero-prefix embedding for a
shorter suffix-compatible claim. For example, embedding `P(x)` beneath a
two-variable zero prefix changes its logical evaluation to
`eq(s_prefix, 0) * P(x)`; `jolt-openings` then combines that value with the
other ordered slots. `jolt-akita` only maps this statement to Akita commitment,
setup, proof, and transcript objects.

The protocol branch also owns reconstruction and verifier wiring. Its required
invariants are:

- prover and verifier absorb identical layout data and claims before drawing
  the prefix selector;
- every omitted lane is public zero or is reconstructed by a checked relation;
- standard and ZK claim/constraint formulas remain synchronized;
- commitment metadata binds K and ring dimension; K256 uses D64 through
  `num_vars = 40` and D128 only from `num_vars = 41`;
- old packed proofs and setups need not verify under the new layout. There is
  no mixed-version compatibility promise.

## Prover stack

The prover branch targets `jolt-prover`, not `jolt-prover-legacy`. It starts
from the modular parity kernel stack and reuses the shared stage recipes; the
Akita flag changes commitment, reconstruction, and opening orchestration while
the Dory path remains intact.

The retained implementation includes the performance mechanisms already
validated on the reference branch: compact proof rows and exact padding,
streamed 29-lane commitment input, virtualized zero/selector prefixes, lazy RA
materialization, compact sparse entries and instruction rows, deferred Fp128
accumulators, stage-local ownership, and early release of setup, NTT, RAM, and
opening state. The external Akita dependency is the corresponding performance
stack at `8c2560586741ef08ce6c3619455bd96e1a0c1c34`.

Akita's `perf/akita-protocol-opts` currently equals Akita `origin/main` because
this work does not change Akita proof bytes, transcript order, or verifier
equations. The optimized Akita branch does contain shared setup-cache plumbing
and one verifier setup-read call site. That stays with the prover stack: it
re-derives the same public matrix coefficients and preserves serialized setup
bytes. Any later change to a verifier equation or proof container must move to
the protocol branch before review.

## Delivery and compatibility

The intended review order is:

1. Jolt `perf/akita-protocol-opts` against Jolt `origin/main`.
2. Akita `perf/akita-prover-opt` against Akita `origin/main`.
3. Jolt `perf/akita-prover-opt` against the merged Jolt protocol commit, pinned
   to the merged Akita prover commit.

The current Jolt prover worktree is logically, but not yet ancestrally, stacked
on the fifth protocol commit: its D128 change was replayed after the modular
port because those files had diverged. Rebase it once the modular parity and
protocol heads settle; do not resolve that mechanical rebase by changing proof
bytes. Benchmark artifacts remain ignored. Durable design and acceptance notes
belong in `specs/`.

## Acceptance

Protocol acceptance requires prefix-layout unit tests, Jolt claim-semantics
tests, Akita adapter tests, verifier tampering tests, and standard/ZK e2e tests.
Prover acceptance requires the full optimized-kernel parity suite, modular
Akita byte-diff tests, committed-program tests, and both host clippy modes.

Performance is checked on `sha2-chain`, K256, against the frozen reference
branch with the same Akita revision. At 2^20, 2^22, and 2^24, compare three
warm samples and reject a repeatable prover-time regression above 3%. At 2^26,
retain a verified Perfetto trace and compare commitment, PIOP, opening, wall
time, and peak RSS. At 2^28, one verified run must remain below 90 GiB without
swap growth; its component scaling and D128 selection must match the analytical
ledger. A memory change may not add repeatable prover time, and a speed change
may not increase the analytical peak. Dory is reported from the same harness
but is not changed by this stack.

Still unverified on the distilled prover head: the final 2^26 and 2^28 runs
after the latest RAM lifetime and collection changes. The protocol branch has
passed its targeted suite; full workspace host and host+zk clippy remain final
pre-push gates.
