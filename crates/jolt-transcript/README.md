# jolt-transcript

Fiat-Shamir transcript implementations for Jolt.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides hash-based Fiat-Shamir transcripts that convert interactive proof protocols into non-interactive ones. The transcript maintains a 256-bit running state, absorbs prover messages via hashing, and squeezes deterministic challenges for the verifier.

Two hash backends are provided. Both produce 128-bit challenges (drawn from `u128`) and use a `state || round_counter` domain separation scheme.

## Public API

### Core Traits

- **`Transcript`** -- The main transcript trait. Methods: `new(label)`, `append_bytes(bytes)`, `append(value)`, `challenge()`, `challenge_vector(len)`, `state()`.
- **`AppendToTranscript`** -- Trait for types that can be absorbed into a transcript.

### Implementations

- **`Blake2bTranscript`** -- Uses Blake2b-256. Default choice for Jolt proofs.
- **`KeccakTranscript`** -- Uses Keccak-256. EVM-compatible for on-chain verification.

## Dependency Position

`jolt-transcript` depends on `jolt-field` (for the blanket `AppendToTranscript` impl on `Field` types). It is used by `jolt-crypto`, `jolt-sumcheck`, `jolt-openings`, `jolt-dory`, `jolt-blindfold`, and `jolt-zkvm`.

## Feature Flags

This crate has no feature flags.

## License

MIT OR Apache-2.0
