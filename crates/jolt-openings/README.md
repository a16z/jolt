# jolt-openings

Polynomial commitment scheme traits and opening reduction for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines abstract interfaces for polynomial commitment schemes (PCS) and provides a reduction framework for batching opening claims. By separating the PCS abstraction from concrete implementations (Dory, KZG, lattice, hash), protocol code is written generically over the PCS with zero implementation leakage.

The crate is intentionally thin: PCS traits, claim types, a reduction trait, and RLC utilities. Protocol-level orchestration (DAG of sumcheck stages, claim accounting) belongs in `jolt-zkvm`.

### Design Principles

- **Stateless.** No accumulators. Claims are plain data (`ProverClaim`, `VerifierClaim`) collected by the caller in `Vec`s.
- **Reduction is separate from proving.** The `OpeningReduction` trait transforms claims (many → fewer). The PCS opens the reduced claims. Two distinct steps that compose independently.
- **No batching in PCS traits.** `batch_prove`/`batch_verify` are not part of the PCS interface. Batching is a reduction strategy (`RlcReduction`), not a PCS property.
- **Extends `jolt_crypto::Commitment`.** `CommitmentScheme` shares the `Output` associated type with the vector commitment hierarchy in `jolt-crypto`, avoiding duplicate bound definitions.

## Trait Hierarchy

```
                    Commitment              (jolt-crypto: just Output type)
                   /          \
    VectorCommitment            CommitmentScheme    (this crate: + open/verify)
          |                          |
  BlindableCommitment        AdditivelyHomomorphic  (+ combine)
                                     |
                             StreamingCommitment    (+ incremental)
```

## Public API

### PCS Traits

- **`CommitmentScheme: Commitment`** — Base PCS trait. Associated types for `Field`, `Proof`, `ProverSetup`, `VerifierSetup`. Uses `Commitment::Output` for the commitment type. Methods: `commit`, `open`, `verify`.

- **`AdditivelyHomomorphic: CommitmentScheme`** — Commitments can be linearly combined. Single method: `combine(commitments, scalars) -> Output`. Enables RLC-based reduction.

- **`StreamingCommitment: CommitmentScheme`** — Incremental commitment: `begin`, `feed`, `finish`. For polynomials that exceed memory.

### Claim Types

- **`ProverClaim<F>`** — Leaf claim with polynomial evaluations, point, and claimed value.

- **`VerifierClaim<F, C>`** — Leaf claim with commitment, point, and claimed value. Fully typed (no `dyn Any`).

### Reduction

- **`OpeningReduction<PCS>`** — Trait for claim transformations. Implementations define the batching strategy. Output type equals input type, so reductions compose. Associated type `ReductionProof` captures any proof artifact from the reduction itself (`()` for deterministic reductions like RLC).

- **`RlcReduction`** — Provided implementation. Requires `PCS: AdditivelyHomomorphic`. Groups claims by evaluation point, combines via random linear combination. Produces one reduced claim per distinct point.

### Utilities

- **`rlc_combine`** — Random linear combination of polynomial evaluation tables via Horner's method.
- **`rlc_combine_scalars`** — Random linear combination of scalar evaluations.

### Errors

- **`OpeningsError`** — Variants: `VerificationFailed`, `CommitmentMismatch`, `InvalidSetup`, `PolynomialTooLarge`.

## Usage

```rust
use jolt_openings::{CommitmentScheme, OpeningReduction, RlcReduction, ProverClaim};

// Protocol stages produce leaf claims
let leaves: Vec<ProverClaim<F>> = protocol_stages(&mut transcript);

// Reduce: group by point, combine via RLC
let (reduced, _) = RlcReduction::reduce_prover(leaves, &mut transcript);

// Open each reduced claim individually via PCS
let proofs: Vec<_> = reduced.iter().map(|claim| {
    MyPCS::open(&claim.evaluations, &claim.point, claim.eval, &setup, &mut transcript)
}).collect();
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `test-utils` | No | Enables the `mock` module with a hash-based mock commitment scheme for testing |

## Dependency Position

```
jolt-openings
  +-- jolt-crypto    (Commitment base trait, group abstractions)
  +-- jolt-field     (scalar field trait)
  +-- jolt-poly      (multilinear polynomial types, dev/test-utils only)
  +-- jolt-transcript (Fiat-Shamir transcript trait)
```

Consumed by `jolt-dory` (implements `CommitmentScheme`), `jolt-spartan`, and `jolt-zkvm`. Depends only on field, crypto, polynomial, and transcript abstractions — never references a concrete PCS implementation.

## License

MIT
