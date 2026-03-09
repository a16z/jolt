# jolt-crypto

Backend-agnostic cryptographic group and commitment primitives for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the core group abstractions (`JoltGroup`, `PairingGroup`) and commitment trait hierarchy (`Commitment`, `JoltCommitment`, `HomomorphicCommitment`) used by the Jolt proving system. It provides a backend-agnostic interface -- the BN254 implementation wraps arkworks internally, but no arkworks types appear in the public API.

## Commitment Hierarchy

```
Commitment                   (base: just the Output type)

JoltCommitment               (Setup, Commitment types; commit, verify)

HomomorphicCommitment<F>     (linear_combine for Nova folding)
```

| Scheme | Message | Output |
|--------|---------|--------|
| Pedersen | `F` (field) | `G` (group) |
| Dory tier-1 | `F` (field) | `G1` |
| Dory tier-2 | `G1` (commitments) | `GT` |

`Commitment` is the base trait shared with `jolt-openings::CommitmentScheme` -- both extend it, sharing the `Output` associated type.

## Public API

### Core Traits

- **`Commitment`** -- Base trait defining `type Output` with standard bounds.
- **`JoltCommitment`** -- Backend-agnostic vector commitment with `Setup`, `Commitment` associated types. Methods: `capacity()`, `commit()`, `verify()`. Commit takes a blinding factor.
- **`HomomorphicCommitment<F>`** -- Additive homomorphism: `linear_combine(c1, c2, scalar) = c1 + scalar * c2`. Blanket-implemented for `JoltGroup`.
- **`JoltGroup`** -- Cryptographic group with additive notation. Provides `identity()`, `is_identity()`, `double()`, `scalar_mul()`, `msm()`.
- **`PairingGroup`** -- Pairing-friendly group. Associates `ScalarField`, `G1`, `G2`, `GT` (all `JoltGroup`), provides `pairing()` and `multi_pairing()`.

### Pedersen Commitment

- **`Pedersen<G: JoltGroup>`** -- Generic Pedersen vector commitment. Implements `JoltCommitment`.
- **`PedersenSetup<G>`** -- Setup parameters (generators + blinding generator).

### BN254 Concrete Types

- **`Bn254`** -- BN254 pairing curve implementing `PairingGroup`.
- **`Bn254G1`** / **`Bn254G2`** -- G1/G2 group elements implementing `JoltGroup`.
- **`Bn254GT`** -- Target group element (additive notation over `Fq12`).

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `bn254` | **Yes** | Enable BN254 backend via arkworks |
| `dory-pcs` | No | Enable Dory PCS interop (implies `bn254`) |

## Dependency Position

```
jolt-field <- jolt-crypto <- jolt-openings, jolt-sumcheck, jolt-dory, jolt-blindfold
              jolt-transcript ->
```

## License

MIT
