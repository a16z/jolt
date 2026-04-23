# jolt-openings

Polynomial commitment scheme traits and batched opening verification for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines abstract interfaces for polynomial commitment schemes (PCS) and provides a fused batched-opening framework. By separating the PCS abstraction from concrete implementations (Dory, KZG, lattice, hash), protocol code is written generically over the PCS with zero implementation leakage.

### Design Principles

- **Stateless.** No accumulators. Claims are plain data (`ProverClaim`, `VerifierClaim`, `OpeningClaim`) collected by the caller in `Vec`s.
- **Fused batching.** The `OpeningVerification` trait exposes a single `prove_batch` / `verify_batch_with_backend` pair that consumes a whole bag of claims and emits / accepts one `BatchProof`. Schemes that prefer a "reduce then open" decomposition (Mock, HyperKZG, Dory) implement it via `homomorphic_prove_batch` / `homomorphic_verify_batch_with_backend`; schemes with native fused batching (e.g. lattice-based Hachi) plug in their own `BatchedProof` directly.
- **No batching in PCS traits.** Batching is a verification concern (`OpeningVerification`), not a per-claim `CommitmentScheme` property.

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

- **`ProverClaim<F>`** -- Prover-side leaf claim: polynomial, evaluation point, claimed value. (Likely renamed to `ProverOpeningClaim<F>` in a follow-up; see TODO in `claims.rs`.)
- **`VerifierClaim<F, C>`** -- Native verifier-side leaf claim: commitment, point, claimed value. Used at API boundaries that have not yet been ported to the backend-aware `OpeningClaim`.
- **`OpeningClaim<B, PCS>`** -- Backend-aware verifier-side leaf claim consumed by `OpeningVerification::verify_batch_with_backend`. Holds `B::Commitment`, `Vec<B::Scalar>`, and `B::Scalar`.

### Verification

- **`OpeningVerification: CommitmentScheme`** -- Trait for fused batched proving / verification. Associated `BatchProof` is the scheme's natural batched proof object (e.g. `Vec<PCS::Proof>` for additively homomorphic schemes, `HachiCommitmentScheme::BatchedProof` for lattice schemes). Methods:
  - `prove_batch(claims, hints, pk, transcript) -> (BatchProof, Vec<F>)` — the trailing `Vec<F>` is per-group "binding evals" the runtime threads into transcript-binding ops downstream.
  - `verify_batch_with_backend(backend, vk, claims, batch_proof, transcript) -> Result<(), OpeningsError>` — fused: no separate "reduce then verify" step.

### Utilities

- **`homomorphic_prove_batch`** -- Group-by-point + RLC-combine + per-group `PCS::open` helper used by `OpeningVerification::prove_batch` impls for additively homomorphic schemes.
- **`homomorphic_verify_batch_with_backend`** -- Mirror helper used by additively homomorphic `verify_batch_with_backend` impls; calls `backend.verify_opening` once per group.
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
