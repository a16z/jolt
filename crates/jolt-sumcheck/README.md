# jolt-sumcheck

Sumcheck protocol engine for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate implements the sumcheck interactive proof protocol for multilinear polynomials. Given a claim `sum_{x in {0,1}^n} g(x) = S`, the prover convinces the verifier of the sum's correctness in `n` rounds, each reducing one variable. The crate supports both single-polynomial and batched (random linear combination) sumcheck.

## Public API

### Claims

- **`SumcheckClaim<F>`** — A sumcheck claim specifying `num_vars`, `degree`, and `claimed_sum`.

### Proofs

- **`SumcheckProof<F>`** — A sequence of univariate round polynomials, one per variable. Also used as the proof type for the batched (RLC) variant.

### Prover

- **`SumcheckCompute<F>`** — Trait for witness polynomials. Implementors provide `round_polynomial` (the univariate restriction for the current round) and `bind` (fix a variable to the verifier's challenge).
- **`SumcheckProver`** — Drives the prover side: iterates rounds, queries the witness, and produces a `SumcheckProof`.
- **`BatchedSumcheckProver`** — Batches multiple sumcheck instances via random linear combination.
- **`StreamingSumcheckProver`** — Memory-efficient streaming variant for large witnesses.

### Verifier

- **`SumcheckVerifier`** — Verifies a `SumcheckProof` against a claim. Returns the final evaluation and the vector of challenges on success.
- **`BatchedSumcheckVerifier`** — Verifies a batched sumcheck proof.

### Reduction

- **`SumcheckReduction<F>`** — Trait for sumcheck-based claim reductions. Transforms `ProverClaim`s into sumcheck instances, then extracts reduced claims from the resulting challenges. The sumcheck analogue of `OpeningReduction` in jolt-openings.
- **`SumcheckWitnessBatch<F>`** — Type alias for `(Vec<SumcheckClaim<F>>, Vec<Box<dyn SumcheckCompute<F>>>)`.

### Round Handlers

- **`RoundHandler<F>`** — Prover-side strategy for absorbing round polynomials into the transcript.
- **`RoundVerifier<F>`** — Verifier-side strategy for checking round data.
- **`ClearRoundHandler`** / **`ClearRoundVerifier`** — Cleartext implementations (coefficients appended directly). The committed-mode implementations live in `jolt-blindfold`.

### Errors

- **`SumcheckError`** — Error type with variants: `RoundCheckFailed`, `FinalEvalMismatch`, `DegreeBoundExceeded`, `WrongNumberOfRounds`.

## Dependency Position

```
jolt-field ─────┐
jolt-poly  ─────┼─► jolt-sumcheck ─► jolt-blindfold, jolt-spartan, jolt-zkvm
jolt-transcript ┘
jolt-openings ──┘    (for ProverClaim/VerifierClaim used by SumcheckReduction)
```

## Feature Flags

This crate has no feature flags.

## License

MIT
