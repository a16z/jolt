# jolt-openings

Commitment scheme traits and opening proof accumulators for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the abstract interface for polynomial commitment schemes (PCS) and provides accumulators that batch opening claims before producing proofs. By separating the commitment abstraction from concrete implementations (like Dory), protocol code can be written generically over the PCS.

## Public API

### Commitment Scheme Traits

- **`CommitmentScheme`** — Base trait for any PCS. Associated types for `Field`, `Commitment`, `Proof`, `ProverSetup`, and `VerifierSetup`. Methods: `setup_prover`, `setup_verifier`, `commit`, `prove`, `verify`.

- **`HomomorphicCommitmentScheme`** — Extends `CommitmentScheme` with homomorphic operations: `combine_commitments` (linear combination of commitments) and `batch_prove` / `batch_verify` for batched opening proofs.

- **`StreamingCommitmentScheme`** — Extends `CommitmentScheme` with incremental commitment: `begin_streaming`, `stream_chunk`, `finalize_streaming`. Enables committing to polynomials larger than memory by processing chunks.

### Accumulators

- **`ProverOpeningAccumulator<F>`** — Collects opening claims `(polynomial, point, evaluation)` during proving, then reduces them via random linear combination and produces batched proofs. Methods: `new`, `accumulate`, `reduce_and_prove`, `len`, `is_empty`.

- **`VerifierOpeningAccumulator<F>`** — Collects opening claims `(commitment, point, evaluation)` during verification, then batch-verifies them. Methods: `new`, `accumulate`, `reduce_and_verify`, `len`, `is_empty`.

### Utilities

- **`rlc_combine`** — Random linear combination of field elements: computes `sum_i values[i] * rho^i`.
- **`rlc_combine_scalars`** — Same as above, operating on raw scalar slices.

### Errors

- **`OpeningsError`** — Error type with variants: `VerificationFailed`, `CommitmentMismatch`, `InvalidSetup`, `PolynomialTooLarge`.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `test-utils` | No | Enable `mock` module with a hash-based mock commitment scheme for testing |

## License

MIT
