# jolt-crypto

Backend-agnostic cryptographic group and commitment primitives for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the core group abstractions (`JoltGroup`, `PairingGroup`) and commitment trait hierarchy (`Commitment`, `VectorCommitment`, `BlindableCommitment`) used by the Jolt proving system. It provides a backend-agnostic interface — the BN254 implementation wraps arkworks internally, but no arkworks types appear in the public API.

Follows the same patterns as `jolt-field`: serde-based serialization, `#[repr(transparent)]` newtypes, zero backend leakage.

## Commitment Hierarchy

The commitment abstraction is designed to unify field-to-group commitments (Pedersen, Dory tier-1) and group-to-group commitments (Dory tier-2 / AFGHO) under a single trait hierarchy:

```
Commitment                   (base: just the Output type)
    |
VectorCommitment             (+ Message type, commit(&[Message]) -> Output)
    |
BlindableCommitment          (+ Blinding type, commit_blinded, verify_opening)
```

| Scheme | Message | Output | Blinding |
|--------|---------|--------|----------|
| Pedersen | `F` (field) | `G` (group) | `F` |
| Dory tier-1 | `F` (field) | `G1` | — |
| Dory tier-2 | `G1` (commitments) | `GT` | — |
| Hash-based | `F` (field) | `[u8; 32]` | — |

`Commitment` is the base trait shared with `jolt-openings::CommitmentScheme` — both extend it, sharing the `Output` associated type.

## Dependency Position

```
jolt-field <- jolt-crypto <- jolt-openings, jolt-sumcheck, jolt-dory, jolt-blindfold
              jolt-transcript ->
```

## Public API

### Core Traits

- **`Commitment`** — Base trait defining the commitment output type. Minimal: just `type Output` with bounds (`Clone`, `Debug`, `Eq`, `Send`, `Sync`, `Serialize`, `Deserialize`, `AppendToTranscript`). Extended by both `VectorCommitment` (this crate) and `CommitmentScheme` (`jolt-openings`).

- **`VectorCommitment: Commitment`** — Commit to a homogeneous vector of messages. Associates `Message` and `Setup` types. Methods: `capacity()`, `commit()`. Covers both field-element vectors and group-element vectors.

- **`BlindableCommitment: VectorCommitment`** — Extends vector commitment with blinding for ZK. Associates `Blinding` type. Methods: `commit_blinded()`, `verify_opening()`.

- **`JoltGroup`** — Cryptographic group suitable for commitments. Uses additive notation (`Add`/`Sub`/`Neg`). Provides `identity()`, `is_identity()`, `double()`, `scalar_mul()`, and `msm()`. All elements are `Copy`, thread-safe, serializable, and can be absorbed into Fiat-Shamir transcripts (`AppendToTranscript`).

- **`PairingGroup`** — Pairing-friendly group for schemes requiring bilinear maps (Dory, KZG). Associates a `ScalarField`, three group types (`G1`, `G2`, `GT` — all `JoltGroup`), and provides `pairing()` and `multi_pairing()`. Generators and randomness are inherent methods on concrete types, not on the trait.

### Pedersen Commitment

- **`Pedersen<G: JoltGroup>`** — Generic Pedersen implementation for any group. Implements `VectorCommitment<Message = Fr, Output = G>` and `BlindableCommitment<Blinding = Fr>`. Computes `C = sum_i values[i] * generators[i] + blinding * H`.

- **`PedersenSetup<G>`** — Setup parameters holding message generators and a blinding generator. Serializable. Constructed from externally-provided generators (e.g., extracted from a Dory URS).

### BN254 Concrete Types

- **`Bn254`** — BN254 pairing-friendly curve. Implements `PairingGroup` with `ScalarField = Fr`. Provides `g1_generator()`, `g2_generator()`, `random_g1()` as inherent methods.

- **`Bn254G1`** / **`Bn254G2`** — G1/G2 group elements wrapping arkworks projective points. Implement `JoltGroup` with optimized MSM via `msm_bigint`.

- **`Bn254GT`** — Target group element wrapping `Fq12`. Implements `JoltGroup` with additive notation (Add = Fq12 mul, Neg = Fq12 inverse, identity = Fq12::ONE). `Mul`/`MulAssign` also provided as convenience aliases.

### Internal Bridge

`field_to_fr` converts a generic `Field` element to arkworks `Fr` via little-endian byte serialization. This is `pub(crate)` and never appears in the public API.

## Testing

```bash
# Unit + integration tests
cargo nextest run -p jolt-crypto --cargo-quiet

# Benchmarks
cargo bench -p jolt-crypto

# Fuzzing (requires nightly)
cd crates/jolt-crypto/fuzz
cargo +nightly fuzz run deser_group     # deserialization safety
cargo +nightly fuzz run group_arith     # group law invariants
cargo +nightly fuzz run pedersen_commit # commitment properties
```

## License

MIT
