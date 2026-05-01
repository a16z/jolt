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

use ark_serialize::CanonicalSerialize;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use zklean_extractor::mle_ast::{
    set_pending_challenge, take_pending_append, take_pending_commitment_chunks, MleAst,
};

use super::io_replay::pop_bytes_override;

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

    fn raw_append_label_with_len(&mut self, label: &'static [u8], len: u64) {
        // The default impl calls raw_append_bytes(&packed), which would check the
        // FIFO and steal an IO override meant for actual input/output data.
        // We do the same packing + hashing but call append_field_elements directly.
        assert!(label.len() <= 24);
        let mut packed = [0u8; 32];
        packed[..label.len()].copy_from_slice(label);
        packed[24..32].copy_from_slice(&len.to_be_bytes());
        let element = MleAst::from(bytes_to_scalar(&packed));
        self.append_field_elements(&[element]);
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        let elements: Vec<MleAst> = bytes
            .chunks(32)
            .map(|chunk| {
                // If symbolize_io_device pre-loaded a FIFO override for this chunk,
                // use the symbolic variable instead of the concrete bytes.
                if let Some(symbolic) = pop_bytes_override() {
                    symbolic
                } else {
                    let mut padded = [0u8; 32];
                    padded[..chunk.len()].copy_from_slice(chunk);
                    MleAst::from(bytes_to_scalar(&padded))
                }
            })
            .collect();
        self.append_field_elements(&elements);
    }

    fn raw_append_u64(&mut self, x: u64) {
        // PoseidonTranscript::raw_append_u64 packs x as LE in first 8 bytes of 32-byte word.
        // from_le_bytes_mod_order gives x directly. No transform needed.
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
            panic!("PoseidonAstTranscript::raw_append_scalar called but no pending MleAst found — serialize_uncompressed must store the symbolic value via set_pending_append()")
        }
    }

    // === Override append_serializable to handle AstCommitment chunks ===

    fn append_serializable<T: CanonicalSerialize>(&mut self, label: &'static [u8], data: &T) {
        // For symbolic execution, serialization stores values in thread-local.
        let mut buf = vec![];
        let _ = data.serialize_uncompressed(&mut buf);

        // Check for commitment chunks first (MleAst chunks for commitment hashing).
        // AstCommitment::serialize stores chunks in PENDING_COMMITMENT_CHUNKS.
        // Chunk count is PCS-dependent (e.g., Dory: 12 chunks for 384-byte G1Affine).
        if let Some(chunks) = take_pending_commitment_chunks() {
            // CRITICAL: Match Transcript::append_serializable behavior:
            // 1. Compute byte length from chunk count (each chunk = 32 bytes).
            //    Note: buf.len() is 0 because AstCommitment::serialize doesn't write bytes.
            let commitment_byte_len = chunks.len() * 32;
            self.raw_append_label_with_len(label, commitment_byte_len as u64);

            // 2. Commitment bytes are LE (no reversal). Chunked into N × 32-byte pieces.
            //    Hashed in order. No ByteReverse needed.
            self.append_field_elements(&chunks);
            return;
        }

        // Fallback: single MleAst (existing behavior for non-commitment types)
        if let Some(mle_ast) = take_pending_append() {
            self.raw_append_label_with_len(label, buf.len() as u64);
            // LE bytes directly. No byte-reversal needed.
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
        // Full Fr challenge. Hash output directly, no truncation.
        let hash = self.challenge_ast();
        set_pending_challenge(hash);
        F::from_bytes(&[0u8; 32])
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
        // Full Fr challenge. Hash output directly, no truncation.
        let hash = self.challenge_ast();
        set_pending_challenge(hash);
        let f_val: F = F::from_bytes(&[0u8; 32]);
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

    // =========================================================================
    // Helper Methods Tests
    // =========================================================================

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
    fn test_hash_and_update() {
        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        let initial_rounds = transcript.n_rounds;

        transcript.hash_and_update(MleAst::from_u64(42));

        // Verify n_rounds incremented
        assert_eq!(transcript.n_rounds, initial_rounds + 1);

        // Verify state is now a Poseidon hash node
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        assert!(
            matches!(
                node,
                zklean_extractor::mle_ast::Node::TranscriptHash(_, _, _)
            ),
            "Expected TranscriptHash node after hash_and_update"
        );
    }

    #[test]
    fn test_challenge_ast() {
        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        let initial_rounds = transcript.n_rounds;

        let challenge = transcript.challenge_ast();

        // Verify n_rounds incremented
        assert_eq!(transcript.n_rounds, initial_rounds + 1);

        // Verify challenge is a Poseidon hash node
        let root = challenge.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        assert!(
            matches!(
                node,
                zklean_extractor::mle_ast::Node::TranscriptHash(_, _, _)
            ),
            "Expected TranscriptHash node for challenge"
        );

        // Verify state was updated to challenge value
        assert_eq!(transcript.state.root(), challenge.root());
    }

    #[test]
    fn test_append_field_elements_chaining() {
        // CRITICAL: First element uses n_rounds for domain separation,
        // remaining elements use 0 for chaining (matching jolt-core)
        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        let initial_rounds = transcript.n_rounds;

        let elements = vec![
            MleAst::from_u64(10),
            MleAst::from_u64(20),
            MleAst::from_u64(30),
        ];

        transcript.append_field_elements(&elements);

        // Verify n_rounds incremented once (not per element)
        assert_eq!(transcript.n_rounds, initial_rounds + 1);

        // Verify state is a Poseidon hash node
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        assert!(
            matches!(
                node,
                zklean_extractor::mle_ast::Node::TranscriptHash(_, _, _)
            ),
            "Expected TranscriptHash node after append_field_elements"
        );
    }

    // =========================================================================
    // Transcript Trait Tests
    // =========================================================================

    #[test]
    fn test_new_with_label() {
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
    fn test_raw_append_u64() {
        use jolt_core::transcripts::Transcript as _;

        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        let initial_rounds = transcript.n_rounds;

        transcript.raw_append_u64(12345);

        // Verify n_rounds incremented
        assert_eq!(transcript.n_rounds, initial_rounds + 1);

        // Verify state contains a Poseidon hash with the u64 value
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        assert!(
            matches!(
                node,
                zklean_extractor::mle_ast::Node::TranscriptHash(_, _, _)
            ),
            "Expected TranscriptHash node after raw_append_u64"
        );
    }

    #[test]
    fn test_raw_append_scalar_with_mle_ast() {
        use jolt_core::transcripts::Transcript as _;

        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");

        // Create a variable (not a constant)
        let var = MleAst::from_var(42);

        // Append it to transcript via append_scalar (which calls raw_append_scalar)
        transcript.append_scalar(b"test_scalar", &var);

        // Check that the state now contains a Poseidon node with Var(42) directly
        // append_scalar does: hash(var). No byte-reversal.
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);

        match node {
            zklean_extractor::mle_ast::Node::TranscriptHash(
                zklean_extractor::mle_ast::TranscriptHashData::Poseidon(data_edge),
                _,
                _,
            ) => {
                // Poseidon data element should be Var(42) directly (no ByteReverse)
                match data_edge {
                    zklean_extractor::mle_ast::Edge::Atom(
                        zklean_extractor::mle_ast::Atom::Var(idx),
                    ) => {
                        assert_eq!(idx, 42, "Expected Var(42), got Var({idx})");
                    }
                    other => panic!("Expected Atom(Var(42)) as Poseidon data arg, got {other:?}"),
                }
            }
            _ => panic!("Expected Poseidon node, got {node:?}"),
        }
    }

    #[test]
    fn test_append_serializable_with_commitment_chunks() {
        use jolt_core::transcripts::Transcript as _;
        use zklean_extractor::mle_ast::{set_pending_commitment_chunks, AstCommitment};

        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        let initial_rounds = transcript.n_rounds;

        // Create mock commitment chunks (simulating what AstCommitment::serialize does)
        let chunks = vec![
            MleAst::from_u64(100),
            MleAst::from_u64(200),
            MleAst::from_u64(300),
        ];
        set_pending_commitment_chunks(chunks.clone());

        // Create a dummy AstCommitment (its serialize will have already set the chunks above)
        let commitment = AstCommitment::default();

        // Append the commitment
        transcript.append_serializable(b"commitment", &commitment);

        // Verify n_rounds incremented by 2:
        // 1. raw_append_label_with_len (label + length)
        // 2. append_field_elements (commitment chunks)
        assert_eq!(
            transcript.n_rounds,
            initial_rounds + 2,
            "n_rounds should increment by 2 after append_serializable (label+len + chunks)"
        );

        // Verify state is a Poseidon hash node (contains the commitment chunks)
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);
        assert!(
            matches!(
                node,
                zklean_extractor::mle_ast::Node::TranscriptHash(_, _, _)
            ),
            "Expected TranscriptHash node after appending commitment"
        );
    }
}
