//! Fiat-Shamir transcript implementations for [Jolt](https://github.com/a16z/jolt).
//!
//! This crate provides hash-based Fiat-Shamir transcripts that convert
//! interactive proof protocols into non-interactive ones. The transcript
//! maintains a 256-bit running state, absorbs prover messages via hashing,
//! and squeezes deterministic challenges for the verifier.
//!
//! # Traits
//!
//! - [`Transcript`]: Main transcript trait — `new(label)`, `append_bytes(bytes)`,
//!   `append(value)`, `challenge()`, `challenge_vector(len)`, `state()`.
//! - [`AppendToTranscript`]: For types that can be absorbed into a transcript.
//!
//! # Implementations
//!
//! Three hash backends are provided. All produce 128-bit challenges (drawn
//! from `u128`) and use a `state || round_counter` domain separation scheme.
//!
//! - [`Blake2bTranscript`]: Uses Blake2b-256. Default choice for Jolt proofs.
//! - [`KeccakTranscript`]: Uses Keccak-256. EVM-compatible for on-chain verification.
//! - [`PoseidonTranscript`]: Uses Poseidon over BN254. SNARK-friendly for recursive verification.
//!
//! # Dependency position
//!
//! Depends on `jolt-field` (for the blanket [`AppendToTranscript`] impl on
//! [`Field`](jolt_field::Field) types). Used by `jolt-crypto`, `jolt-sumcheck`,
//! `jolt-openings`, `jolt-dory`, `jolt-blindfold`, and `jolt-zkvm`.
//!
//! # Example
//!
//! ```
//! use jolt_transcript::{Transcript, Blake2bTranscript};
//! use jolt_field::{Field, Fr};
//!
//! let mut transcript = Blake2bTranscript::<Fr>::new(b"my_protocol");
//!
//! // Absorb field elements using append (AppendToTranscript)
//! let value = Fr::from_u64(42);
//! transcript.append(&value);
//!
//! // Absorb raw bytes directly
//! transcript.append_bytes(b"raw bytes");
//!
//! // Squeeze a challenge — returns Fr directly
//! let challenge: Fr = transcript.challenge();
//! ```

#![deny(missing_docs)]

mod blake2b;
mod blanket;
mod digest;
mod keccak;
mod poseidon;
mod transcript;

pub use blake2b::Blake2bTranscript;
pub use digest::DigestTranscript;
pub use keccak::KeccakTranscript;
pub use poseidon::PoseidonTranscript;
pub use transcript::{AppendToTranscript, Transcript};
