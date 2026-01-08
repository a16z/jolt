//! Keccak256-based transcript for symbolic execution
//!
//! This module provides a Keccak256-based Fiat-Shamir transcript that works
//! with MleAst for transpilation to Gnark circuits.

use zklean_extractor::mle_ast::MleAst;
use jolt_core::field::JoltField;
use ark_serialize::CanonicalSerialize;
use ark_ec::CurveGroup;

/// Keccak256-based Fiat-Shamir transcript for symbolic execution
///
/// This transcript records Keccak256 hash operations in the AST instead of
/// computing actual hashes.
#[derive(Clone, Default)]
pub struct KeccakMleTranscript {
    /// Current state (accumulated hash)
    state: MleAst,
}

impl KeccakMleTranscript {
    /// Create a new Keccak transcript
    pub fn new(_label: &'static [u8]) -> Self {
        // Start with a dummy state (will be replaced on first append)
        Self {
            state: MleAst::from_i128(0),
        }
    }

    /// Append a field element to the transcript
    pub fn append_field(&mut self, element: MleAst) {
        // Hash current state with new element
        // Keccak256(state || element)
        self.state = MleAst::keccak256(&self.state) + element;
    }

    /// Append a message to the transcript
    pub fn append_message(&mut self, _msg: &'static [u8]) {
        // For symbolic execution, messages are ignored or treated as constants
        // In a real implementation, this would hash the message
        self.state = MleAst::keccak256(&self.state);
    }

    /// Derive a challenge from the current transcript state
    pub fn challenge(&mut self) -> MleAst {
        let challenge = MleAst::keccak256(&self.state);
        // Update state for domain separation
        self.state = challenge.clone();
        challenge
    }

    /// Challenge a scalar (same as challenge for MleAst)
    pub fn challenge_scalar<F>(&mut self) -> MleAst {
        self.challenge()
    }
}

/// Implement Jolt's Transcript trait for KeccakMleTranscript
impl jolt_core::transcripts::Transcript for KeccakMleTranscript {
    fn new(label: &'static [u8]) -> Self {
        Self::new(label)
    }

    fn append_message(&mut self, msg: &'static [u8]) {
        self.append_message(msg);
    }

    fn append_bytes(&mut self, _bytes: &[u8]) {
        // For symbolic execution, treat as hash update
        self.state = MleAst::keccak256(&self.state);
    }

    fn append_u64(&mut self, x: u64) {
        // Convert to field element and append
        let element = MleAst::from_i128(x as i128);
        self.append_field(element);
    }

    fn append_scalar<F: jolt_core::field::JoltField>(&mut self, _scalar: &F) {
        // For symbolic execution, treat as generic hash
        self.state = MleAst::keccak256(&self.state);
    }

    fn append_serializable<F: CanonicalSerialize>(&mut self, _scalar: &F) {
        // For symbolic execution, treat as generic hash
        self.state = MleAst::keccak256(&self.state);
    }

    fn append_scalars<F: jolt_core::field::JoltField>(
        &mut self,
        scalars: &[impl std::borrow::Borrow<F>],
    ) {
        self.append_message(b"begin_append_vector");
        for _ in scalars.iter() {
            self.state = MleAst::keccak256(&self.state);
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, _point: &G) {
        // For symbolic execution, treat as generic hash
        self.state = MleAst::keccak256(&self.state);
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for _ in points.iter() {
            self.state = MleAst::keccak256(&self.state);
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        // For symbolic execution, return a dummy value
        // The actual value doesn't matter since we're building AST
        0u128
    }

    fn challenge_scalar<F: jolt_core::field::JoltField>(&mut self) -> F {
        // This is tricky - we're building MleAst but need to return F
        // For symbolic execution, we can't actually return MleAst here
        // So we return zero and rely on the MleAst version of the verifier
        F::zero()
    }

    fn challenge_scalar_128_bits<F: jolt_core::field::JoltField>(&mut self) -> F {
        F::zero()
    }

    fn challenge_vector<F: jolt_core::field::JoltField>(&mut self, len: usize) -> Vec<F> {
        vec![F::zero(); len]
    }

    fn challenge_scalar_powers<F: jolt_core::field::JoltField>(&mut self, len: usize) -> Vec<F> {
        vec![F::zero(); len]
    }

    fn challenge_scalar_optimized<F: jolt_core::field::JoltField>(&mut self) -> F::Challenge {
        F::Challenge::default()
    }

    fn challenge_vector_optimized<F: jolt_core::field::JoltField>(
        &mut self,
        len: usize,
    ) -> Vec<F::Challenge> {
        vec![F::Challenge::default(); len]
    }

    fn challenge_scalar_powers_optimized<F: jolt_core::field::JoltField>(
        &mut self,
        len: usize,
    ) -> Vec<F> {
        vec![F::zero(); len]
    }
}
