//! Blake2b-256 based Fiat-Shamir transcript.

use blake2::{digest::consts::U32, Blake2b};

use crate::digest::DigestTranscript;

/// Fiat-Shamir transcript backed by Blake2b-256.
pub type Blake2bTranscript<F = jolt_field::Fr> = DigestTranscript<Blake2b<U32>, F>;
