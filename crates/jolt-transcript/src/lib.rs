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
//! Hash backends use a `state || round_counter` domain separation scheme and
//! delegate field-family challenge decoding to `TranscriptChallenge`.
//!
//! - [`Blake2bTranscript`]: Uses Blake2b-256. Default choice for Jolt proofs.
//! - [`KeccakTranscript`]: Uses Keccak-256. EVM-compatible for on-chain verification.
//! - [`PoseidonTranscript`]: Uses Poseidon over BN254 when the `poseidon` feature is enabled.
//!
//! # Dependency position
//!
//! Depends on `jolt-field` for field challenge decoding and transcript
//! absorption. The `poseidon` feature enables BN254/Poseidon dependencies.
//!
//! # Example
//!
//! ```ignore
//! use jolt_transcript::{Transcript, Blake2bTranscript};
//! use jolt_field::{FromPrimitiveInt, Fr};
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
pub mod domain;
mod keccak;
#[cfg(feature = "poseidon")]
mod poseidon;
mod transcript;

pub use blake2b::Blake2bTranscript;
pub use digest::DigestTranscript;
pub use domain::{Label, LabelWithCount, U64Word};
pub use keccak::KeccakTranscript;
#[cfg(feature = "poseidon")]
pub use poseidon::PoseidonTranscript;
pub use transcript::{AppendToTranscript, Transcript};
