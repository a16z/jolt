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

- **`SumcheckWitness<F>`** — Trait for witness polynomials. Implementors provide `round_polynomial` (the univariate restriction for the current round) and `bind` (fix a variable to the verifier's challenge).
- **`SumcheckProver`** — Drives the prover side: iterates rounds, queries the witness, and produces a `SumcheckProof`.
- **`BatchedSumcheckProver`** — Batches multiple sumcheck instances via random linear combination.
- **`StreamingSumcheckProver`** — Memory-efficient streaming variant for large witnesses.

### Verifier

- **`SumcheckVerifier`** — Verifies a `SumcheckProof` against a claim. Returns the final evaluation and the vector of challenges on success.
- **`BatchedSumcheckVerifier`** — Verifies a batched sumcheck proof.

### Errors

- **`SumcheckError`** — Error type with variants: `RoundCheckFailed`, `FinalEvalMismatch`, `DegreeBoundExceeded`, `WrongNumberOfRounds`.

## License

MIT
