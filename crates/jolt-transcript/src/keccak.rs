//! Keccak-256 based Fiat-Shamir transcript (Ethereum/EVM compatible).

use sha3::Keccak256;

use crate::digest::DigestTranscript;

/// Fiat-Shamir transcript backed by Keccak-256.
#[cfg(feature = "poseidon")]
pub type KeccakTranscript<F = jolt_field::Fr> = DigestTranscript<Keccak256, F>;

/// Fiat-Shamir transcript backed by Keccak-256.
#[cfg(not(feature = "poseidon"))]
pub type KeccakTranscript<F> = DigestTranscript<Keccak256, F>;
