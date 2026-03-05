# jolt-transcript

Fiat-Shamir transcript implementations for Jolt.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate provides hash-based Fiat-Shamir transcripts that convert interactive proof protocols into non-interactive ones. The transcript maintains a 256-bit running state, absorbs prover messages via hashing, and squeezes deterministic challenges for the verifier.

Two hash backends are provided. Both produce 128-bit challenges (drawn from `u128`) and use a `state || round_counter` domain separation scheme.

## Public API

### Core Traits

- **`Transcript`** — The main transcript trait. Methods:
  - `new(label)` — Initialize with a domain-separation label (< 33 bytes)
  - `append_bytes(bytes)` — Absorb raw bytes into the transcript state
  - `append(value)` — Absorb any `AppendToTranscript` value (field elements, commitments, etc.)
  - `challenge()` — Squeeze a 128-bit challenge
  - `challenge_vector(len)` — Squeeze multiple challenges
  - `state()` — Current 256-bit state (for transcript comparison in tests)

- **`AppendToTranscript`** — Trait for types that can be absorbed into a transcript. Implemented for field elements and commitment types.

### Implementations

- **`Blake2bTranscript`** — Uses Blake2b-256. Default choice for Jolt proofs.
- **`KeccakTranscript`** — Uses Keccak-256. EVM-compatible for on-chain verification.

## License

MIT OR Apache-2.0
