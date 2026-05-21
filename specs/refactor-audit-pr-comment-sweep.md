# Refactor Audit Stack PR Comment Sweep

Fetched with `gh` on 2026-05-21. Updated after Markos decisions on
2026-05-21.

Scope: refactor audit stack PRs `#1540` through `#1551`. Excluded `#1552`
because it is the separate bytecode stack.

## Decisions

- `#1541` `Point` endianness const generic: deferred by Markos. No foundation
  redesign in this sweep.
- `#1545` Eq-cycle taxonomy: fixed by making instruction RA virtualization
  `EqCycle` a public source, matching the RAM RA convention.
- `#1546` sparse rows with duplicate variable indices: fixed by deduping in
  `LinearCombination::into_sparse_row`.
- `#1546` `R1csBuilder::multiply`: fixed by returning `LinearCombination<F>`
  directly.
- `#1550` `jolt-prover` spec comments: deferred/ignored by Markos for now.
- `#1551` extended Jolt / field-inline / wrapper spec comments:
  deferred/ignored by Markos for now.
- `#1540` stack automation: no action requested.

## Fixed

- `jolt-poly` MLE helpers now allow `range_end == domain_size`, treat
  full-domain less-than masks as all-one, and return `DomainTooLarge` rather
  than panicking on oversized block shifts.
- `try_eq_mle` and `eq_index_msb` moved into `jolt-poly::eq`, and `MleError`
  now derives `thiserror::Error`.
- Identity polynomial evaluation no longer has separate inherent helper methods
  that duplicate the `MultilinearEvaluation` trait implementation.
- BN254 scalar challenge byte reversal is documented as the legacy transcript
  big-endian convention.
- `HomomorphicCommitment` now uses `Default` plus explicit `add`/`linear_combine`
  APIs instead of an ambiguous `identity()` method, and `combine_commitments`
  avoids the unnecessary multiply-by-one path.
- Row-opening zero-padding expectations are documented in `jolt-crypto`.
- Transitional `ProverClaim`/`VerifierClaim` aliases were removed from
  `jolt-openings`.
- Dory commitment/proof traits were tightened where the upstream types permit
  it, including `Default` for commitments and `Eq` for the proof wrapper.
- `ClaimExpression` no longer exposes mutable expression internals directly;
  callers use `expression()`.
- Zero-coefficient claim terms preserve non-constant metadata instead of
  collapsing source requirements away.
- Jolt dimension power-of-two checks are enforced in release builds.
- R1CS variables have a private index with `Variable::new`/`index`, product
  assertions validate variable bounds before matrix conversion, and duplicate
  source-table inserts now panic instead of silently shadowing.
- `public_column_contributions` returns a named `MatrixColumnContributions`
  struct instead of a raw tuple.
- RV64 outer remainder docs/tests were hardened: corrected index ranges, C-column
  contribution guard, full eq-plus-product row parity test, hand-computed public
  claim roundtrip, and named challenge inputs to avoid swapped slices.
- Missing opening/public lowering errors now have direct tests.
- BlindFold verifier rejects degenerate zero-variable outer/inner Spartan checks.
- BlindFold field order and test helpers were adjusted to match the actual row
  layout and nondegenerate verification paths.

## Deferred

- `#1544` serde feature-gating for `jolt-crypto` and `jolt-dory` remains
  deferred. The current `Commitment` trait requires serde bounds, so doing this
  correctly needs a broader trait/API split rather than a local feature toggle.
- `#1544` deriving HyperKZG `Debug`/`PartialEq` remains deferred. The generic
  associated types on `PairingGroup` still need manual implementations unless
  the trait bounds are broadened.

## Validation

- `semgrep --config .semgrep/jolt-verifier-boundaries.yml --error`
- `git diff --check`
- `cargo nextest run -p jolt-r1cs --cargo-quiet`
- `cargo nextest run -p jolt-verifier --cargo-quiet --features core-fixtures,zk`
- `cargo clippy --all --features host -q --all-targets -- -D warnings`
- `cargo clippy --all --features host,zk -q --all-targets -- -D warnings`
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host`
- `cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk`
