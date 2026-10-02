# jolt-hyperkzg

HyperKZG multilinear polynomial commitment scheme for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

HyperKZG reduces multilinear polynomial commitments to univariate KZG using the Gemini transformation ([section 2.4.2](https://eprint.iacr.org/2022/420.pdf)), operating directly on evaluation-form polynomials (no FFT/interpolation).

This crate is generic over `PairingGroup` from `jolt-crypto` and implements `CommitmentScheme` and `AdditivelyHomomorphic` from `jolt-openings`.

The four-point basis uses a canonical BN254 Fr fourth root of unity; opening and
verification reject scalar fields in which that element is not a primitive fourth root.

### Protocol

1. **Commit** — MSM of evaluations against SRS G1 powers.
2. **Open** (Gemini reduction) — fold two variables per level, derive challenge `r`, batch KZG open at `[r, ir, -r, -ir, r⁴]`.
3. **Verify** — evaluation consistency check, then batch KZG pairing check.

## Public API

- **`HyperKZGScheme<P>`** — Main entry point. Implements `CommitmentScheme` and `AdditivelyHomomorphic`.
- **`HyperKZGCommitment<P>`** — A commitment (G1 point).
- **`HyperKZGProof<P>`** — Opening proof containing intermediate commitments and evaluations.
- **`HyperKZGProverSetup<P>`** / **`HyperKZGVerifierSetup<P>`** — Structured reference strings.

### Submodules

- **`kzg`** — Univariate KZG primitives (commit, open, batch verify).
- **`error`** — Error types.

## Dependency Position

```
jolt-field ─┐
jolt-crypto ─┤
jolt-poly  ─┼─► jolt-hyperkzg
jolt-transcript ─┤
jolt-openings ─┘
```

Used by `jolt-zkvm`.

## Feature Flags

This crate has no feature flags.

## License

MIT
