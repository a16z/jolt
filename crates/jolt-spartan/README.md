# jolt-spartan

Spartan R1CS prover and verifier for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate implements the [Spartan](https://eprint.iacr.org/2019/550) SNARK for R1CS constraint systems. Given an R1CS instance `Az * Bz = Cz`, the prover produces a succinct proof via a sumcheck reduction over the multilinear extensions of the constraint matrices, then opens the witness polynomial at the resulting evaluation point using a polynomial commitment scheme.

The implementation is generic over the commitment scheme and transcript, allowing it to be composed with different PCS backends (Dory, HyperKZG, etc.).

## Public API

### R1CS Interface

- **`R1CS<F>`** — Trait for R1CS constraint systems. Methods: `num_constraints`, `num_variables`, `multiply_witness` (matrix-vector product).
- **`SimpleR1CS<F>`** — Sparse triple-based R1CS representation. Useful for testing and small circuits.

### Key

- **`SpartanKey<F>`** — Precomputed key containing multilinear extensions of the A, B, C matrices. Built from an `R1CS` instance via `from_r1cs`. Provides `num_sumcheck_vars` and `num_witness_vars` for protocol dimensioning.

### Prover & Verifier

- **`SpartanProver`** — Produces a `SpartanProof` from an R1CS witness, key, and PCS setup.
- **`SpartanVerifier`** — Verifies a `SpartanProof` against a key and transcript.

### Proof

- **`SpartanProof<F, PCS>`** — Contains the witness commitment, sumcheck proof, matrix evaluations (`az_eval`, `bz_eval`, `cz_eval`), witness evaluation, and opening proof.

### Univariate Skip

- **`FirstRoundStrategy`** — Optimization for the first sumcheck round using univariate polynomial evaluation instead of the standard multilinear approach.

### Errors

- **`SpartanError`** — Error type with variants: `ConstraintViolation`, `Sumcheck`, `Opening`, `EvaluationMismatch`.

## License

MIT
