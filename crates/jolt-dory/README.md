# jolt-dory

Dory commitment scheme implementation for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate wraps the [Dory](https://eprint.iacr.org/2020/1274) polynomial commitment scheme for use in Jolt. Dory is a pairing-based PCS over BN254 with transparent setup, logarithmic proof size, and logarithmic verification time. It supports streaming commitment (for polynomials larger than memory) and additive homomorphism for batch opening reduction.

The crate implements `CommitmentScheme`, `AdditivelyHomomorphic`, `StreamingCommitment`, `ZkOpeningScheme`, and `VcSetupExtractable` from `jolt-openings`, making it a drop-in PCS backend for Spartan and other protocol components.

## Public API

### Commitment Scheme

- **`DoryScheme`** — Main entry point. Constructed from `DoryParams`, implements all five commitment scheme traits. Inherent methods: `setup_prover(num_vars)`, `setup_verifier(num_vars)`. Trait methods: `commit`, `open`, `verify`, `combine`, `begin`/`feed`/`finish`, `open_zk`/`verify_zk`, `extract_vc_setup`.

### Parameters

- **`DoryParams`** — Configuration for the Dory scheme: tier size `t`, `max_num_rows`, `num_columns`. Methods: `from_dimensions(k, t)` for deriving params from polynomial size, `sigma()` / `nu()` for log-dimensions, `total_vars()` for the total multilinear variable count.

### Types

- **`DoryCommitment`** — A commitment (BN254 pairing target element `GT`).
- **`DoryProof`** — A single opening proof.
- **`DoryProverSetup`** — Prover structured reference string (SRS).
- **`DoryVerifierSetup`** — Verifier SRS (serializable).
- **`DoryPartialCommitment`** — Intermediate state for streaming commitment.
- **`DoryHint`** — Auxiliary data (row commitments) reusable for opening proofs.

### Optimizations

The `optimizations` module provides BN254-specific elliptic curve operations:
- **GLV 2D** (G1) and **GLV 4D** (G2 via Frobenius endomorphism) scalar multiplication
- **Batch affine addition** via Montgomery's inversion trick
- **Vector-scalar operations** for Dory inner-product argument rounds

## Dependency Position

```
jolt-field ─┐
jolt-poly  ─┤
jolt-transcript ─┼─► jolt-dory
jolt-crypto ─┤
jolt-openings ─┘
```

Used by `jolt-zkvm`.

## Feature Flags

This crate has no feature flags.

## License

MIT
