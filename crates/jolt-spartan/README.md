# jolt-spartan

Spartan R1CS prover and verifier for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate implements the [Spartan](https://eprint.iacr.org/2019/550) SNARK for R1CS constraint systems. Given an R1CS instance `Az * Bz = Cz`, the prover produces a succinct proof via a sumcheck reduction over the multilinear extensions of the constraint matrices, then opens the witness polynomial at the resulting evaluation point using a polynomial commitment scheme.

The implementation is generic over the commitment scheme and transcript.

## Public API

### R1CS Interface

- **`R1CS<F>`** -- Trait for R1CS constraint systems. Methods: `num_constraints`, `num_variables`, `multiply_witness`.
- **`SimpleR1CS<F>`** -- Sparse triple-based R1CS representation.

### Keys

- **`SpartanKey<F>`** -- Precomputed key with multilinear extensions of A, B, C matrices.
- **`UniformSpartanKey<F>`** -- Key for uniform (repeated per-cycle) R1CS structure. Avoids materializing the full matrix.

### Prover & Verifier

- **`SpartanProver`** -- Produces a `SpartanProof` from an R1CS witness and key.
- **`SpartanVerifier`** -- Verifies a `SpartanProof`.
- **`UniformSpartanProver`** -- Prover for uniform R1CS (per-cycle witnesses).
- **`UniformSpartanVerifier`** -- Verifier for uniform Spartan proofs.

### Proofs

- **`SpartanProof<F>`** -- Standard Spartan proof (sumcheck proofs, matrix evaluations, witness evaluation). Witness commitment and opening proof handled externally.
- **`UniformSpartanProof<F>`** -- Proof for uniform R1CS.
- **`RelaxedSpartanProof<F, PCS>`** -- Proof for relaxed R1CS (used by Nova folding). Includes PCS opening proofs.

### jolt-ir Bridge

- **`ir_r1cs`** -- Implements `R1CS` for `jolt_ir::R1csEmission`.
- **`build_witness`** -- Assembles a witness vector from an `R1csEmission` and concrete opening values.

### Univariate Skip

- **`FirstRoundStrategy`** -- Optimization for the first sumcheck round using univariate polynomial evaluation.

### Errors

- **`SpartanError`** -- Variants: `ConstraintViolation`, `Sumcheck`, `Opening`, `OuterEvaluationMismatch`, `InnerEvaluationMismatch`, `RelaxedConstraintViolation`.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | **Yes** | Enable rayon parallelism |

## Dependency Position

```
jolt-field ─┐
jolt-poly  ─┤
jolt-transcript ─┼─> jolt-spartan
jolt-sumcheck ─┤
jolt-openings ─┤
jolt-ir ───────┘
```

Used by `jolt-blindfold` and `jolt-zkvm`.

## License

MIT
