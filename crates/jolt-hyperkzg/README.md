# jolt-hyperkzg

HyperKZG multilinear polynomial commitment scheme for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

HyperKZG reduces multilinear polynomial commitments to univariate KZG using the Gemini transformation ([section 2.4.2](https://eprint.iacr.org/2022/420.pdf)), operating directly on evaluation-form polynomials (no FFT/interpolation).

This crate is generic over `PairingGroup` from `jolt-crypto` and implements `CommitmentScheme` and `AdditivelyHomomorphic` from `jolt-openings`.

### Protocol

1. **Commit** — MSM of evaluations against SRS G1 powers.
2. **Open** (Gemini reduction) — fold the multilinear polynomial `ℓ-1` times producing intermediate commitments, derive challenge `r`, batch KZG open at `[r, -r, r²]`.
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
