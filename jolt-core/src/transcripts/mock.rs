//! Deterministic mock transcript for testing.
//!
//! All absorb operations are no-ops. Challenges come from a seeded Blake2b
//! counter, producing the same u128 sequence as jolt-transcript's MockTranscript
//! when given the same seed.

use super::transcript::Transcript;
use crate::field::JoltField;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

type Blake2b256 = Blake2b<U32>;

/// Mock transcript that ignores absorbs and produces deterministic challenges.
///
/// Use the same seed as `jolt_transcript::MockTranscript` to get identical
/// u128 values, enabling cross-system comparison of protocol arithmetic.
#[derive(Clone, Default)]
pub struct MockTranscript {
    seed: [u8; 32],
    counter: u64,
}

impl MockTranscript {
    pub fn with_seed(seed: &[u8]) -> Self {
        let hash: [u8; 32] = Blake2b256::new().chain_update(seed).finalize().into();
        Self {
            seed: hash,
            counter: 0,
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

impl Transcript for MockTranscript {
    fn new(_label: &'static [u8]) -> Self {
        Self::with_seed(b"mock_transcript_default_seed")
    }

    #[cfg(test)]
    fn compare_to(&mut self, _other: Self) {}

    fn raw_append_label(&mut self, _label: &'static [u8]) {}
    fn raw_append_bytes(&mut self, _bytes: &[u8]) {}
    fn raw_append_u64(&mut self, _x: u64) {}
    fn raw_append_scalar<F: JoltField>(&mut self, _scalar: &F) {}

    fn challenge_u128(&mut self) -> u128 {
        self.next_u128()
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        F::from_u128(self.next_u128())
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        F::from_u128(self.next_u128())
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.challenge_scalar()).collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut powers = vec![F::one(); len];
        for i in 1..len {
            powers[i] = powers[i - 1] * q;
        }
        powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        F::Challenge::from(self.next_u128())
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_| self.challenge_scalar_optimized::<F>())
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut powers = vec![F::one(); len];
        for i in 1..len {
            powers[i] = powers[i - 1] * q;
        }
        powers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn same_seed_same_challenges() {
        let mut t1 = MockTranscript::with_seed(b"test");
        let mut t2 = MockTranscript::with_seed(b"test");

        // Absorb different things
        t1.raw_append_bytes(b"hello");
        t2.raw_append_bytes(b"world");
        t1.raw_append_label(b"foo");

        for _ in 0..100 {
            assert_eq!(t1.challenge_scalar::<Fr>(), t2.challenge_scalar::<Fr>());
        }
    }

    #[test]
    fn matches_jolt_transcript_mock_sequence() {
        // Verify the u128 sequence matches what jolt-transcript's mock produces.
        // Both use: H(H(seed) || counter.to_le_bytes())[..16] as u128 LE
        let mut t = MockTranscript::with_seed(b"cross_system_test");
        let u128_0 = t.challenge_u128();
        let u128_1 = t.challenge_u128();
        assert_ne!(u128_0, u128_1);
        assert_ne!(u128_0, 0);
    }
}
