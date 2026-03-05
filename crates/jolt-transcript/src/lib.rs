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
//!
//! # Example
//!
//! ```
//! use jolt_transcript::{Transcript, Blake2bTranscript};
//!
//! let mut transcript = Blake2bTranscript::new(b"my_protocol");
//!
//! // Absorb data using append (for types implementing AppendToTranscript)
//! transcript.append(&42u64);
//! transcript.append(&[1u8, 2, 3, 4]);
//!
//! // Or use append_bytes directly
//! transcript.append_bytes(b"raw bytes");
//!
//! // Squeeze a challenge
//! let challenge: u128 = transcript.challenge();
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

mod blake2b;
mod blanket;
mod impl_transcript;
mod keccak;
mod transcript;

pub use blake2b::Blake2bTranscript;
pub use keccak::KeccakTranscript;
pub use transcript::{AppendToTranscript, Transcript};
