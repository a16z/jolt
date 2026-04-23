# jolt-openings

Polynomial commitment scheme traits and opening reduction for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines abstract interfaces for polynomial commitment schemes (PCS) and provides a reduction framework for batching opening claims. By separating the PCS abstraction from concrete implementations (Dory, KZG, lattice, hash), protocol code is written generically over the PCS with zero implementation leakage.

### Design Principles

- **Stateless.** No accumulators. Claims are plain data (`ProverClaim`, `VerifierClaim`) collected by the caller in `Vec`s.
- **Reduction is separate from proving.** The `OpeningReduction` trait transforms claims (many -> fewer). The PCS opens the reduced claims.
- **No batching in PCS traits.** Batching is a reduction concern (`OpeningReduction`), not a PCS property.

## Trait Hierarchy

```
                    Commitment              (jolt-crypto: just Output type)
                        |
                CommitmentScheme            (+ Field, Proof, commit/open/verify)
                        |
            AdditivelyHomomorphic           (+ combine)
                        |
              StreamingCommitment           (+ begin/feed/finish)
```

## Public API

### PCS Traits

- **`CommitmentScheme: Commitment`** -- Base PCS trait. Methods: `commit`, `open`, `verify`.
- **`AdditivelyHomomorphic: CommitmentScheme`** -- Commitments can be linearly combined. Method: `combine`.
- **`StreamingCommitment: CommitmentScheme`** -- Incremental commitment: `begin`, `feed`, `finish`.
- **`ZkOpeningScheme`** -- PCS that supports zero-knowledge evaluation proofs (committed evaluation output).
### Claim Types

- **`ProverClaim<F>`** -- Leaf claim with polynomial, evaluation point, and claimed value.
- **`VerifierClaim<F, C>`** -- Leaf claim with commitment, point, and claimed value.

### Reduction

- **`OpeningReduction: CommitmentScheme`** -- Trait for claim transformations. Each PCS provides its own impl, since the natural batching strategy is scheme-specific (RLC for homomorphic schemes, FRI/DEEP-ALI for hash-based, etc.). Homomorphic schemes (Dory, HyperKZG, Mock) delegate to `homomorphic_reduce_prover` / `homomorphic_reduce_verifier`.

### Utilities

- **`homomorphic_reduce_prover`** -- RLC-based prover-side reduction; per-PCS `OpeningReduction::reduce_prover` impls forward to this for homomorphic schemes.
- **`homomorphic_reduce_verifier`** -- RLC-based verifier-side reduction (uses `PCS::combine`); the corresponding helper for verifier impls.
- **`rlc_combine`** -- Random linear combination of polynomial evaluation tables.
- **`rlc_combine_scalars`** -- Random linear combination of scalar evaluations.

### Errors

- **`OpeningsError`** -- Variants: `VerificationFailed`, `CommitmentMismatch`, `InvalidSetup`, `PolynomialTooLarge`.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `test-utils` | No | Enables the `mock` module with a homomorphic mock commitment scheme |

## Dependency Position

```
jolt-openings
  +-- jolt-crypto    (Commitment base trait, group abstractions)
  +-- jolt-field     (scalar field trait)
  +-- jolt-poly      (multilinear polynomial types)
  +-- jolt-transcript (Fiat-Shamir transcript trait)
```

Consumed by `jolt-dory`, `jolt-spartan`, and `jolt-zkvm`.

## License

MIT
