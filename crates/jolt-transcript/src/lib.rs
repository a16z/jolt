//! Fiat-Shamir transcript implementations for Jolt.
//!
//! This crate provides the [`Transcript`] trait and implementations for
//! transforming interactive proofs into non-interactive ones via the
//! Fiat-Shamir heuristic.
//!
//! # Overview
//!
//! A Fiat-Shamir transcript absorbs data and produces deterministic challenges.
//! Both prover and verifier maintain identical transcripts, ensuring they
//! derive the same challenges.
//!
//! # Implementations
//!
//! - [`Blake2bTranscript`]: Uses Blake2b-256 hash function
//! - [`KeccakTranscript`]: Ethereum/EVM-compatible, uses Keccak-256
//! - [`PoseidonTranscript`]: SNARK-friendly, uses Poseidon over BN254
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
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

mod blake2b;
mod blanket;
mod impl_transcript;
mod keccak;
mod poseidon;
mod transcript;

pub use blake2b::Blake2bTranscript;
pub use keccak::KeccakTranscript;
pub use poseidon::PoseidonTranscript;
pub use transcript::{AppendToTranscript, Transcript};
