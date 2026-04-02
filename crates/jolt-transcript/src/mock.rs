//! Deterministic mock transcript for testing.
//!
//! All absorb operations are no-ops. Challenges come from a seeded Blake2b
//! counter, producing the same sequence regardless of what is absorbed.
//! Use the same seed in both jolt-core's and jolt-transcript's mock
//! transcripts to get identical challenges for cross-system comparison.

use crate::transcript::{AppendToTranscript, Transcript};
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::Field;
use std::marker::PhantomData;

type Blake2b256 = Blake2b<U32>;

/// Mock transcript that ignores absorbs and produces deterministic challenges.
///
/// Challenges are derived from `H(seed || counter)` where counter increments
/// on each squeeze. Two mock transcripts with the same seed always produce
/// identical challenge sequences.
#[derive(Clone)]
pub struct MockTranscript<F: Field = jolt_field::Fr> {
    seed: [u8; 32],
    counter: u64,
    _field: PhantomData<F>,
}

impl<F: Field> Default for MockTranscript<F> {
    fn default() -> Self {
        Self {
            seed: [0u8; 32],
            counter: 0,
            _field: PhantomData,
        }
    }
}

impl<F: Field> MockTranscript<F> {
    /// Creates a mock transcript with the given seed bytes.
    #[must_use]
    pub fn with_seed(seed: &[u8]) -> Self {
        let hash: [u8; 32] = Blake2b256::new().chain_update(seed).finalize().into();
        Self {
            seed: hash,
            counter: 0,
            _field: PhantomData,
        }
    }

    fn next_u128(&mut self) -> u128 {
        let hash: [u8; 32] = Blake2b256::new()
            .chain_update(self.seed)
            .chain_update(self.counter.to_le_bytes())
            .finalize()
            .into();
        self.counter += 1;
        u128::from_le_bytes(hash[..16].try_into().unwrap())
    }
}

impl<F: Field> Transcript for MockTranscript<F> {
    type Challenge = F;

    fn new(_label: &'static [u8]) -> Self {
        Self::with_seed(b"mock_transcript_default_seed")
    }

    fn append_bytes(&mut self, _bytes: &[u8]) {}

    fn append<A: AppendToTranscript>(&mut self, _value: &A) {}

    fn challenge(&mut self) -> F {
        F::from_u128(self.next_u128())
    }

    fn state(&self) -> &[u8; 32] {
        &self.seed
    }

    #[cfg(test)]
    fn compare_to(&mut self, _other: &Self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    #[test]
    fn same_seed_same_challenges() {
        let mut t1 = MockTranscript::<Fr>::with_seed(b"test");
        let mut t2 = MockTranscript::<Fr>::with_seed(b"test");

        // Absorb different things — shouldn't matter
        t1.append_bytes(b"hello");
        t2.append_bytes(b"world");

        for _ in 0..100 {
            assert_eq!(t1.challenge(), t2.challenge());
        }
    }

    #[test]
    fn different_seed_different_challenges() {
        let mut t1 = MockTranscript::<Fr>::with_seed(b"seed_a");
        let mut t2 = MockTranscript::<Fr>::with_seed(b"seed_b");
        assert_ne!(t1.challenge(), t2.challenge());
    }
}
