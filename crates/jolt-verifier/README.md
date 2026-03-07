# jolt-verifier

Lightweight Jolt proof verification — no prover dependencies.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate defines the proof types, verification key, verifier stage trait, and top-level `JoltVerifier`. External consumers import only this crate to verify Jolt proofs — no prover dependencies, no rayon, no compute backend.

## Public API

### Core Types

- **`JoltProof<PCS>`** — Complete Jolt proof (sumcheck proofs, opening proofs, public inputs).
- **`JoltVerifyingKey<PCS>`** — Verification key (Spartan key, PCS verifier setup, prover config).
- **`JoltPublicInput`** — Public input/output data for the proven computation.
- **`ProverConfig`** — Configuration parameters deserialized from the proof.

### Verifier

- **`JoltVerifier<PCS>`** — Top-level verification pipeline. Runs all verifier stages, collects opening claims, and checks PCS proofs.
- **`VerifierStage<PCS>`** — Trait for individual verification stages (sumcheck verify, claim extraction).

### Errors

- **`JoltError`** — Verification error type.

## Dependency Position

```
jolt-field ─┐
jolt-transcript ─┤
jolt-crypto ─┤
jolt-poly  ─┤
jolt-openings ─┼─► jolt-verifier
jolt-sumcheck ─┤
jolt-spartan ─┤
jolt-dory ─┘
```

Imported by external consumers for proof verification.

## Feature Flags

This crate has no feature flags.

## License

MIT
