//! Keccak-256 based Fiat-Shamir transcript (Ethereum/EVM compatible).

use sha3::Keccak256;

use crate::digest::DigestTranscript;

/// Fiat-Shamir transcript backed by Keccak-256.
pub type KeccakTranscript<F = jolt_field::Fr> = DigestTranscript<Keccak256, F>;
