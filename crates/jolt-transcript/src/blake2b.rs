//! Blake2b-256 based Fiat-Shamir transcript.

use blake2::{digest::consts::U32, Blake2b};

use crate::digest::DigestTranscript;

/// Fiat-Shamir transcript backed by Blake2b-256.
#[cfg(feature = "poseidon")]
pub type Blake2bTranscript<F = jolt_field::Fr> = DigestTranscript<Blake2b<U32>, F>;

/// Fiat-Shamir transcript backed by Blake2b-256.
#[cfg(not(feature = "poseidon"))]
pub type Blake2bTranscript<F> = DigestTranscript<Blake2b<U32>, F>;
