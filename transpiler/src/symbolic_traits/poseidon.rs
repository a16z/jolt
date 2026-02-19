//! Poseidon transcript for symbolic execution (MleAst).
//!
//! # Overview
//!
//! This module implements the `Transcript` trait for symbolic execution. Instead of
//! computing actual Poseidon hashes, it records hash operations as AST nodes that
//! will be converted to Gnark code.
//!
//! # Why Poseidon?
//!
//! The Jolt proof must use Poseidon (not Blake2b/Keccak) because:
//! - Poseidon is SNARK-friendly: ~250 constraints per hash vs ~150,000 for Keccak
//! - The circuit recomputes all Fiat-Shamir challenges from the transcript
//! - Using Blake2b would make the circuit infeasibly large
//!
//! # Structure (Must Match jolt-core Exactly)
//!
//! - **Width-3 Poseidon**: `hash(state, n_rounds, data)` with 3 inputs
//! - **Domain separation**: `n_rounds` counter increments on each append operation
//! - **State update**: `new_state = poseidon(old_state, n_rounds, data)`
//! - **Challenge derivation**: `challenge = truncate_128(poseidon(state, n_rounds, 0))`
//!
//! # Fiat-Shamir Challenge Types
//!
//! Two challenge derivation methods exist:
//! - `challenge_scalar_128_bits()`: For batching coefficients. Uses `Truncate128`.
//! - `challenge_scalar_optimized()`: For sumcheck challenges. Uses `Truncate128Reverse`.
//!
//! The "Reverse" variant reverses bytes before truncation (matching jolt-core's byte order).
//!
//! # Critical: Transcript Matching
//!
//! The proof MUST be generated with `--features transcript-poseidon`. If generated
//! with Blake2b (the default), all challenge values will differ and verification
//! will fail silently (assertions won't be zero).

use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use zklean_extractor::mle_ast::{
    set_pending_challenge, take_pending_append, take_pending_commitment_chunks,
    take_pending_point_elements, MleAst,
};

/// Symbolic Poseidon transcript for AST-based transpilation.
#[derive(Clone)]
pub struct PoseidonAstTranscript {
    /// Current state (symbolic field element)
    state: MleAst,
    /// Round counter for domain separation
    n_rounds: u32,
}

impl PoseidonAstTranscript {
    /// Convert a label to a field element, matching jolt-core's behavior.
    ///
    /// jolt-core does: label_padded[..label.len()].copy_from_slice(label);
    ///                 Fr::from_le_bytes_mod_order(&label_padded)
    ///
    /// For symbolic execution, we compute the actual integer value that
    /// the label bytes represent in little-endian order.
    fn label_to_field(label: &[u8]) -> MleAst {
        assert!(label.len() <= 32, "Label must be <= 32 bytes");

        // Pad label to 32 bytes and convert to [u64; 4]
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let limbs = bytes_to_scalar(&padded);

        MleAst::from(limbs)
    }

    /// Hash a field element with domain separation.
    ///
    /// Mirrors jolt-core: poseidon(state, n_rounds, element)
    fn hash_and_update(&mut self, element: MleAst) {
        let round = MleAst::from_u64(self.n_rounds as u64);
        self.state = MleAst::poseidon(&self.state, &round, &element);
        self.n_rounds += 1;
    }

    /// Derive a challenge: `poseidon(state, n_rounds, 0)`, then update state.
    pub fn challenge_ast(&mut self) -> MleAst {
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);
        let challenge = MleAst::poseidon(&self.state, &round, &zero);
        self.state = challenge;
        self.n_rounds += 1;
        challenge
    }

    /// Append symbolic field elements with Poseidon chaining.
    ///
    /// First element: `poseidon(state, n_rounds, elem)` (domain separation).
    /// Remaining: `poseidon(prev, 0, elem)` (chained).
    /// Used by `raw_append_bytes` (concrete) and `append_serializable` (commitments).
    pub fn append_field_elements(&mut self, elements: &[MleAst]) {
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);

        let mut iter = elements.iter();

        // First element: includes n_rounds for domain separation
        let mut current = if let Some(first) = iter.next() {
            MleAst::poseidon(&self.state, &round, first)
        } else {
            // Empty: just hash state with n_rounds and zero
            MleAst::poseidon(&self.state, &round, &zero)
        };

        // Remaining elements: no n_rounds (already accounted for)
        for elem in iter {
            current = MleAst::poseidon(&current, &zero, elem);
        }

        self.state = current;
        self.n_rounds += 1;
    }
}

/// Implement Jolt's Transcript trait for PoseidonAstTranscript.
///
/// The challenge methods return MleAst when F = MleAst.
impl Transcript for PoseidonAstTranscript {
    fn new(label: &'static [u8]) -> Self {
        // Mirror jolt-core: initial_state = poseidon(label, 0, 0)
        let label_field = Self::label_to_field(label);
        let initial_state = MleAst::poseidon(
            &label_field,
            &MleAst::from_u64(0), // n_rounds = 0
            &MleAst::from_u64(0), // zero
        );
        Self {
            state: initial_state,
            n_rounds: 0,
        }
    }

    // === Internal raw_append_* methods ===

    fn raw_append_label(&mut self, label: &'static [u8]) {
        assert!(label.len() <= 32);
        let field = Self::label_to_field(label);
        self.hash_and_update(field);
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        let elements: Vec<MleAst> = bytes
            .chunks(32)
            .map(|chunk| {
                let mut padded = [0u8; 32];
                padded[..chunk.len()].copy_from_slice(chunk);
                MleAst::from(bytes_to_scalar(&padded))
            })
            .collect();
        self.append_field_elements(&elements);
    }

    fn raw_append_u64(&mut self, x: u64) {
        // PoseidonTranscript::raw_append_u64 packs x as LE in first 8 bytes of 32-byte word.
        // from_le_bytes_mod_order gives x directly — no transform needed.
        self.hash_and_update(MleAst::from_u64(x));
    }

    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F) {
        // Trigger serialization which stores MleAst in thread-local (if F = MleAst)
        let mut buf = vec![];
        let _ = scalar.serialize_uncompressed(&mut buf);

        if let Some(mle_ast) = take_pending_append() {
            // PoseidonTranscript hashes LE bytes directly (no byte-reversal).
            // from_le_bytes_mod_order(LE serialization) = the scalar itself.
            self.hash_and_update(mle_ast);
        } else {
            // Fallback for non-MleAst types (shouldn't happen in transpilation)
            self.hash_and_update(MleAst::from_u64(0));
        }
    }

    fn raw_append_point<G: CurveGroup>(&mut self, _point: &G) {
        // Symbolic path: check for pending point elements set via set_pending_point_elements()
        if let Some(elements) = take_pending_point_elements() {
            self.append_field_elements(&elements);
            return;
        }

        // No pending point elements — set_pending_point_elements() was never called.
        // The concrete fallback was removed because it contained byte-reversal (.rev())
        // that mismatches the concrete PoseidonTranscript (which no longer reverses).
        // If point support is needed, wire up set_pending_point_elements() properly.
        panic!(
            "PoseidonAstTranscript::raw_append_point: no pending point elements. \
             Call set_pending_point_elements() before raw_append_point() during symbolic execution."
        );
    }

    // === Override append_serializable to handle AstCommitment chunks ===

    fn append_serializable<T: CanonicalSerialize>(&mut self, label: &'static [u8], data: &T) {
        // For symbolic execution, serialization stores values in thread-local.
        let mut buf = vec![];
        let _ = data.serialize_uncompressed(&mut buf);

        // Check for commitment chunks first (12 MleAst for commitment hashing)
        // AstCommitment::serialize stores 12 chunks in PENDING_COMMITMENT_CHUNKS
        if let Some(chunks) = take_pending_commitment_chunks() {
            // CRITICAL: Match Transcript::append_serializable behavior:
            // 1. Use 384 bytes (12 chunks × 32 bytes) for the label length.
            //    Note: buf.len() is 0 because AstCommitment::serialize doesn't write bytes.
            let commitment_byte_len = chunks.len() * 32; // 12 * 32 = 384
            self.raw_append_label_with_len(label, commitment_byte_len as u64);

            // 2. Commitment bytes are LE (no reversal). Chunked into 12 × 32-byte pieces.
            //    Var(0)=chunk_0, Var(1)=chunk_1, ..., Var(11)=chunk_11.
            //    Hashed in order — no ByteReverse needed.

            self.append_field_elements(&chunks);
            return;
        }

        // Fallback: single MleAst (existing behavior for non-commitment types)
        if let Some(mle_ast) = take_pending_append() {
            self.raw_append_label_with_len(label, buf.len() as u64);
            // LE bytes directly — no byte-reversal needed.
            self.hash_and_update(mle_ast);
        } else {
            // Fallback: use default implementation for concrete types
            self.raw_append_label_with_len(label, buf.len() as u64);
            // LE bytes directly, no byte reversal (Groth16 circuit, not EVM)
            self.raw_append_bytes(&buf);
        }
    }

    // === Challenge generation methods ===

    fn challenge_u128(&mut self) -> u128 {
        let _ = self.challenge_ast();
        0u128
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        // Truncate to 128 bits (NO mask, NO shift) - plain truncation
        let hash = self.challenge_ast();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        F::from_bytes(&[0u8; 16])
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len).map(|_| self.challenge_scalar::<F>()).collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let base: F = self.challenge_scalar();
        let mut powers = Vec::with_capacity(len);
        let mut current = F::one();
        for _ in 0..len {
            powers.push(current);
            current *= base;
        }
        powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        // Truncate to 128 bits with reverse (optimized challenge format)
        let hash = self.challenge_ast();
        let challenge = MleAst::truncate_128_reverse(&hash);
        set_pending_challenge(challenge);
        let f_val: F = F::from_bytes(&[0u8; 16]);
        // Safe because for MleAst, F = F::Challenge = MleAst
        unsafe { std::mem::transmute_copy::<F, F::Challenge>(&f_val) }
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_| self.challenge_scalar_optimized::<F>())
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F::Challenge = self.challenge_scalar_optimized::<F>();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q * q_powers[i - 1];
        }
        q_powers
    }

    fn debug_state(&self, _label: &str) {
        // No-op for symbolic execution - debugging output not needed
    }
}

impl Default for PoseidonAstTranscript {
    fn default() -> Self {
        Self {
            state: MleAst::from_u64(0),
            n_rounds: 0,
        }
    }
}

/// Convert 32 little-endian bytes to `[u64; 4]` limbs (no mod reduction).
fn bytes_to_scalar(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        limbs[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    limbs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_to_field() {
        // "jolt" = [0x6a, 0x6f, 0x6c, 0x74] in ASCII
        // As little-endian u64 (padded to 8 bytes): 0x746c6f6a = 1953263466
        let label_field = PoseidonAstTranscript::label_to_field(b"jolt");

        // Check it's a scalar with the expected value
        let root = label_field.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        match node {
            zklean_extractor::mle_ast::Node::Atom(zklean_extractor::mle_ast::Atom::Scalar(v)) => {
                // "jolt" bytes: [0x6a, 0x6f, 0x6c, 0x74, 0, 0, 0, 0]
                // As little-endian u64: 0x00_00_00_00_74_6c_6f_6a = 1953263466
                // As [u64; 4]: [1953263466, 0, 0, 0]
                let expected: [u64; 4] = [1953263466, 0, 0, 0];
                assert_eq!(
                    v, expected,
                    "Label 'jolt' should be {expected:?} but got {v:?}"
                );
            }
            _ => panic!("Expected Scalar atom, got {node:?}"),
        }
    }

    #[test]
    fn test_transcript_creation() {
        let transcript: PoseidonAstTranscript = Transcript::new(b"test");
        assert_eq!(transcript.n_rounds, 0);
    }

    #[test]
    fn test_transcript_with_jolt_label() {
        // Verify that creating transcript with "Jolt" label produces a Poseidon node
        let transcript: PoseidonAstTranscript = Transcript::new(b"Jolt");
        assert_eq!(transcript.n_rounds, 0);

        // The initial state should be a Poseidon hash node
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        match node {
            zklean_extractor::mle_ast::Node::TranscriptHash(_, _, _) => {
                // Expected: poseidon(label, 0, 0)
            }
            _ => panic!("Expected TranscriptHash node for initial state, got {node:?}"),
        }
    }

    #[test]
    fn test_append_and_challenge() {
        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        transcript.hash_and_update(MleAst::from_u64(42));
        let _challenge = transcript.challenge_ast();
        assert_eq!(transcript.n_rounds, 2); // 1 append + 1 challenge
    }

    #[test]
    fn test_append_scalar_with_mle_ast() {
        use jolt_core::transcripts::Transcript as _;

        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");

        // Create a variable (not a constant)
        let var = MleAst::from_var(42);

        // Append it to transcript
        transcript.append_scalar(b"test_scalar", &var);

        // Check that the state now contains a Poseidon node with Var(42) directly
        // append_scalar does: hash(var) — no byte-reversal
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);

        match node {
            zklean_extractor::mle_ast::Node::TranscriptHash(
                zklean_extractor::mle_ast::TranscriptHashData::Poseidon(data_edge), _, _
            ) => {
                // Poseidon data element should be Var(42) directly (no ByteReverse)
                match data_edge {
                    zklean_extractor::mle_ast::Edge::Atom(
                        zklean_extractor::mle_ast::Atom::Var(idx),
                    ) => {
                        assert_eq!(idx, 42, "Expected Var(42), got Var({idx})");
                    }
                    other => panic!(
                        "Expected Atom(Var(42)) as Poseidon data arg, got {other:?}"
                    ),
                }
            }
            _ => panic!("Expected Poseidon node, got {node:?}"),
        }
    }
}
