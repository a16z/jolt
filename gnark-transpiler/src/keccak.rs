//! Keccak transcript for symbolic execution (MleAst)
//!
//! This implementation mirrors jolt-core's KeccakTranscript but records
//! operations in MleAst instead of computing actual hashes.

use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use std::borrow::Borrow;
use zklean_extractor::mle_ast::MleAst;

/// Keccak transcript for symbolic execution.
///
/// Records Keccak256 hash operations in MleAst instead of computing actual hashes.
#[derive(Clone)]
pub struct KeccakMleTranscript {
    /// Current state (symbolic field element)
    state: MleAst,
}

impl Default for KeccakMleTranscript {
    fn default() -> Self {
        Self {
            state: MleAst::from_i128(0),
        }
    }
}

impl KeccakMleTranscript {
    /// Hash and update state using Keccak256.
    fn hash_and_update(&mut self, element: MleAst) {
        // Keccak256(state || element)
        self.state = MleAst::keccak256(&(self.state.clone() + element));
    }

    /// Derive a challenge as MleAst.
    fn challenge_mle(&mut self) -> MleAst {
        let challenge = MleAst::keccak256(&self.state);
        self.state = challenge.clone();
        challenge
    }
}

/// Implement Jolt's Transcript trait for KeccakMleTranscript.
impl Transcript for KeccakMleTranscript {
    fn new(_label: &'static [u8]) -> Self {
        Self {
            state: MleAst::keccak256(&MleAst::from_i128(0)),
        }
    }

    fn append_message(&mut self, _msg: &'static [u8]) {
        self.hash_and_update(MleAst::from_i128(0));
    }

    fn append_bytes(&mut self, _bytes: &[u8]) {
        self.hash_and_update(MleAst::from_i128(0));
    }

    fn append_u64(&mut self, x: u64) {
        self.hash_and_update(MleAst::from_i128(x as i128));
    }

    fn append_scalar<F: JoltField>(&mut self, _scalar: &F) {
        self.hash_and_update(MleAst::from_i128(0));
    }

    fn append_serializable<S: CanonicalSerialize>(&mut self, _scalar: &S) {
        self.hash_and_update(MleAst::from_i128(0));
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
        self.append_message(b"begin_append_vector");
        for _ in scalars.iter() {
            self.hash_and_update(MleAst::from_i128(0));
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, _point: &G) {
        self.hash_and_update(MleAst::from_i128(0));
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for _ in points.iter() {
            self.hash_and_update(MleAst::from_i128(0));
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        let _ = self.challenge_mle();
        0u128
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        let _ = self.challenge_mle();
        F::zero()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let _ = self.challenge_mle();
        F::zero()
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        for _ in 0..len {
            let _ = self.challenge_mle();
        }
        vec![F::zero(); len]
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let _ = self.challenge_mle();
        vec![F::zero(); len]
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        let _ = self.challenge_mle();
        F::Challenge::default()
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        for _ in 0..len {
            let _ = self.challenge_mle();
        }
        vec![F::Challenge::default(); len]
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let _ = self.challenge_mle();
        vec![F::zero(); len]
    }

    fn debug_state(&self, label: &str) {
        println!("TRANSCRIPT DEBUG [{}]: (keccak)", label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_creation() {
        let _transcript: KeccakMleTranscript = Transcript::new(b"test");
    }

    #[test]
    fn test_append_and_challenge() {
        let mut transcript: KeccakMleTranscript = Transcript::new(b"test");
        transcript.hash_and_update(MleAst::from_i128(42));
        let _challenge = transcript.challenge_mle();
    }
}
