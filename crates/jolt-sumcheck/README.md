# jolt-sumcheck

Sumcheck protocol verification for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides the core sumcheck protocol types and verification logic. Given a claim

```
sum_{x in {0,1}^n} g(x) = C
```

the sumcheck protocol reduces it to a single evaluation query `g(r_1, ..., r_n) = v` via `n` rounds of interaction. In each round the prover sends a univariate polynomial `s_i(X)` and the verifier checks `s_i(0) + s_i(1)` against the running sum, then derives a challenge `r_i` via Fiat-Shamir.

This crate is **verifier-only and backend-agnostic**: any field and transcript can be plugged in. Proving is handled by `jolt-zkvm`'s runtime, which drives sumcheck rounds via `ComputeBackend` primitives.

## Public API

### Types

- **`SumcheckClaim<F>`** -- The public statement: `num_vars`, `degree`, and `claimed_sum`.
- **`SumcheckProof<F>`** -- A sequence of univariate round polynomials, one per variable.
- **`SumcheckError`** -- Error variants: `RoundCheckFailed`, `FinalEvalMismatch`, `DegreeBoundExceeded`, `WrongNumberOfRounds`, `EmptyClaims`.

### Verifiers

- **`SumcheckVerifier`** -- Single-instance verifier. Replays the Fiat-Shamir transcript and checks each round.
- **`BatchedSumcheckVerifier`** -- Batched verification via random linear combination. Supports claims with different `num_vars` and `degree` bounds via front-loaded padding.

### Round Verification Strategy

- **`RoundVerifier<F>`** -- Trait controlling how round data is absorbed into the transcript and checked. Enables both clear and committed (ZK) verification modes.
- **`ClearRoundVerifier`** -- Cleartext implementation: checks `poly(0) + poly(1) == running_sum` and absorbs coefficients directly.

## Dependency Position

```
jolt-field ─────┐
jolt-poly  ─────┼─> jolt-sumcheck
jolt-transcript ┘
```

## License

MIT
