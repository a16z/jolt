//! Poseidon transcript for symbolic execution (MleAst)
//!
//! This implementation mirrors jolt-core's PoseidonTranscript but records
//! operations in MleAst instead of computing actual hashes. The structure
//! matches exactly to ensure circuit compatibility:
//!
//! - Width-3 Poseidon: poseidon(state, n_rounds, data)
//! - Domain separation via n_rounds counter
//! - Same byte chunking and padding behavior

use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use std::borrow::Borrow;
use zklean_extractor::mle_ast::{set_pending_challenge, take_pending_append, take_pending_commitment_chunks, MleAst};

/// Convert 32 bytes (little-endian) to a [u64; 4] scalar.
/// This matches Fr::from_le_bytes_mod_order behavior for the full 256 bits.
fn bytes_to_scalar(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        limbs[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    limbs
}

/// Poseidon transcript for symbolic execution.
/// Mirrors jolt-core's PoseidonTranscript structure exactly:
/// - 32-byte state (represented as MleAst)
/// - n_rounds counter for domain separation
/// - Width-3 Poseidon: hash(state, n_rounds, data)
#[derive(Clone)]
pub struct PoseidonAstTranscript {
    /// Current state (symbolic field element)
    state: MleAst,
    /// Round counter for domain separation
    n_rounds: u32,
}

impl Default for PoseidonAstTranscript {
    fn default() -> Self {
        Self {
            state: MleAst::from_u64(0),
            n_rounds: 0,
        }
    }
}

impl PoseidonAstTranscript {
    /// Get the current round counter.
    pub fn n_rounds(&self) -> u32 {
        self.n_rounds
    }

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

    /// Create a new transcript with initial state.
    ///
    /// Mirrors jolt-core: initial_state = poseidon(label, 0, 0)
    pub fn new_mle(label: &'static [u8]) -> Self {
        let label_field = Self::label_to_field(label);
        let initial_state =
            MleAst::poseidon(&label_field, &MleAst::from_u64(0), &MleAst::from_u64(0));
        Self {
            state: initial_state,
            n_rounds: 0,
        }
    }

    /// Hash a field element with domain separation.
    ///
    /// Mirrors jolt-core: poseidon(state, n_rounds, element)
    fn hash_and_update(&mut self, element: MleAst) {
        let round = MleAst::from_u64(self.n_rounds as u64);
        self.state = MleAst::poseidon(&self.state, &round, &element);
        self.n_rounds += 1;
    }

    /// Derive a challenge as MleAst.
    ///
    /// Mirrors jolt-core: poseidon(state, n_rounds, 0)
    pub fn challenge_mle(&mut self) -> MleAst {
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);
        let challenge = MleAst::poseidon(&self.state, &round, &zero);
        self.state = challenge.clone();
        self.n_rounds += 1;
        challenge
    }

    /// Derive multiple challenges as MleAst.
    pub fn challenge_vector_mle(&mut self, len: usize) -> Vec<MleAst> {
        (0..len).map(|_| self.challenge_mle()).collect()
    }

    /// Append symbolic field elements (for commitments/preamble as circuit inputs).
    ///
    /// This mirrors append_bytes but with symbolic inputs instead of concrete bytes.
    /// Each field element corresponds to one 32-byte chunk.
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

    /// Append a single symbolic u64 (for preamble values as circuit inputs).
    ///
    /// This applies the same BE-padding transformation as PoseidonTranscript::append_u64:
    /// - u64 value is placed in bytes 24-31 of a 32-byte array (big-endian padding)
    /// - The 32 bytes are then interpreted as little-endian
    /// - Mathematically, this equals: value * 2^192
    ///
    /// Example: append_u64(4096) stores [0..0, 0x00, 0x00, 0x10, 0x00] (BE) in bytes 24-31
    /// Interpreted as LE 32-byte integer: 4096 * 2^192
    pub fn append_u64_symbolic(&mut self, value: MleAst) {
        // Apply the BE-padding transformation: value * 2^192
        let transformed = MleAst::mul_two_pow_192(&value);
        self.hash_and_update(transformed);
    }
}

/// Implement Jolt's Transcript trait for PoseidonAstTranscript.
///
/// This allows using PoseidonAstTranscript with verify_stage1_with_transcript.
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

    fn append_message(&mut self, msg: &'static [u8]) {
        // Same as append_bytes but for static messages
        // Pad to 32 bytes and convert to field element (LE)
        assert!(msg.len() <= 32);
        let mut padded = [0u8; 32];
        padded[..msg.len()].copy_from_slice(msg);
        let limbs = bytes_to_scalar(&padded);
        self.hash_and_update(MleAst::from(limbs));
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // Hash all bytes using Poseidon with domain separation via n_rounds.
        // First chunk: hash(state, n_rounds, chunk), includes domain separator.
        // Subsequent chunks: hash(prev, 0, chunk), chained but without redundant n_rounds.
        let round = MleAst::from_u64(self.n_rounds as u64);
        let zero = MleAst::from_u64(0);

        let mut chunks = bytes.chunks(32);

        // First chunk: includes n_rounds for domain separation
        let mut current = if let Some(first_chunk) = chunks.next() {
            let mut padded = [0u8; 32];
            padded[..first_chunk.len()].copy_from_slice(first_chunk);
            let chunk_field = MleAst::from(bytes_to_scalar(&padded));
            MleAst::poseidon(&self.state, &round, &chunk_field)
        } else {
            // Empty bytes: just hash state with n_rounds and zero
            MleAst::poseidon(&self.state, &round, &zero)
        };

        // Remaining chunks: no n_rounds (already accounted for)
        for chunk in chunks {
            let mut padded = [0u8; 32];
            padded[..chunk.len()].copy_from_slice(chunk);
            let chunk_field = MleAst::from(bytes_to_scalar(&padded));
            current = MleAst::poseidon(&current, &zero, &chunk_field);
        }

        self.state = current;
        self.n_rounds += 1;
    }

    fn append_u64(&mut self, x: u64) {
        // PoseidonTranscript::append_u64 does:
        //   1. Pack u64 into 32-byte array with BE-padding: packed[24..32] = x.to_be_bytes()
        //   2. Interpret packed as LE field element via from_le_bytes_mod_order
        //
        // The transformation is handled by the Go hint `appendU64TransformHint` which
        // computes the correct field element. Here we just create the AST node.
        let x_ast = MleAst::from_u64(x);
        let transformed = MleAst::mul_two_pow_192(&x_ast);
        self.hash_and_update(transformed);
    }

    fn append_scalar<F: JoltField>(&mut self, scalar: &F) {
        // Trigger serialization which stores MleAst in thread-local (if F = MleAst)
        let mut buf = vec![];
        let _ = scalar.serialize_uncompressed(&mut buf);

        // Retrieve the MleAst from thread-local (set by MleAst::serialize_with_mode)
        if let Some(mle_ast) = take_pending_append() {
            // Apply byte-reverse to match PoseidonTranscript::append_scalar behavior:
            // PoseidonTranscript does: serialize(LE) -> reverse -> from_le_bytes_mod_order -> hash
            // This transforms the value before hashing for EVM compatibility.
            let byte_reversed = MleAst::byte_reverse(&mle_ast);
            self.hash_and_update(byte_reversed);
        } else {
            // Fallback for non-MleAst types (shouldn't happen in transpilation)
            self.hash_and_update(MleAst::from_u64(0));
        }
    }

    fn append_serializable<S: CanonicalSerialize>(&mut self, scalar: &S) {
        // The real PoseidonTranscript::append_serializable does:
        // 1. serialize_uncompressed -> bytes (LE)
        // 2. reverse bytes (for EVM compat)
        // 3. append_bytes (which chunks into 32-byte pieces and hashes with chaining)
        //
        // For symbolic execution, serialization stores values in thread-local.
        let mut buf = vec![];
        let _ = scalar.serialize_uncompressed(&mut buf);

        // Check for commitment chunks first (12 MleAst for commitment hashing)
        // AstCommitment::serialize stores 12 chunks in PENDING_COMMITMENT_CHUNKS
        if let Some(chunks) = take_pending_commitment_chunks() {
            // Hash all 12 chunks with proper chaining (like append_bytes does)
            // This matches what PoseidonTranscript::append_serializable does:
            // - First chunk: poseidon(state, n_rounds, chunk_0)
            // - Remaining: poseidon(prev_hash, 0, chunk_i)
            // - Only increment n_rounds once at the end
            self.append_field_elements(&chunks);
            return;
        }

        // Fallback: single MleAst (existing behavior for non-commitment types)
        if let Some(mle_ast) = take_pending_append() {
            // Apply byte-reverse to match PoseidonTranscript::append_serializable behavior
            let byte_reversed = MleAst::byte_reverse(&mle_ast);
            self.hash_and_update(byte_reversed);
        } else {
            // Fallback for non-MleAst types (shouldn't happen in transpilation)
            self.hash_and_update(MleAst::from_u64(0));
        }
    }

    fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>]) {
        self.append_message(b"begin_append_vector");
        for scalar in scalars.iter() {
            self.append_scalar(scalar.borrow());
        }
        self.append_message(b"end_append_vector");
    }

    fn append_point<G: CurveGroup>(&mut self, _point: &G) {
        self.hash_and_update(MleAst::from_u64(0));
    }

    fn append_points<G: CurveGroup>(&mut self, points: &[G]) {
        self.append_message(b"begin_append_vector");
        for _ in points.iter() {
            self.hash_and_update(MleAst::from_u64(0));
        }
        self.append_message(b"end_append_vector");
    }

    fn challenge_u128(&mut self) -> u128 {
        let _ = self.challenge_mle();
        0u128
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        // Rust PoseidonTranscript::challenge_scalar calls challenge_scalar_128_bits
        // which truncates to 128 bits (NO mask, NO shift) - just plain truncation
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        F::from_bytes(&[0u8; 32])
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        // Truncate to 128 bits without mask or shift
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        F::from_bytes(&[0u8; 16])
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_| {
                let hash = self.challenge_mle();
                // challenge_vector uses challenge_scalar internally, so no mask/shift
                let challenge = MleAst::truncate_128(&hash);
                set_pending_challenge(challenge);
                F::from_bytes(&[0u8; 32])
            })
            .collect()
    }

    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        // Get base challenge - uses challenge_scalar which has no mask/shift
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128(&hash);
        set_pending_challenge(challenge);
        let base: F = F::from_bytes(&[0u8; 32]);

        // Compute powers: 1, base, base^2, ...
        let mut powers = Vec::with_capacity(len);
        let mut current = F::one();
        for _ in 0..len {
            powers.push(current);
            current = current * base;
        }
        powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        // For MleAst: F::Challenge = MleAst, so we need to use the pending challenge mechanism
        // Same as challenge_scalar_128_bits but returns F::Challenge
        //
        // Since F::Challenge doesn't have from_bytes in its trait bounds, but we know
        // this implementation is only used with F = MleAst where F::Challenge = MleAst,
        // we use the F::from_bytes and convert via Into.
        let hash = self.challenge_mle();
        let challenge = MleAst::truncate_128_reverse(&hash);
        set_pending_challenge(challenge);
        // F has from_bytes, and the pending challenge mechanism works via JoltField::from_bytes
        // For MleAst, this returns the pending challenge we just set
        let f_val: F = F::from_bytes(&[0u8; 16]);
        // Now convert F to F::Challenge. Since F::Challenge: Into<F>, but we need F -> F::Challenge,
        // and for MleAst where F = F::Challenge = MleAst, the Default is wrong.
        // The clean solution: return the challenge directly since we know F::Challenge = MleAst
        // Use unsafe transmute or just accept that this only works for MleAst
        //
        // Actually, the simplest fix: use Default but override with the pending challenge
        // in the JoltField implementation. But that's circular.
        //
        // Best approach: since we set pending_challenge, and F::from_bytes returns it,
        // we can return F::Challenge::default() but first convert f_val appropriately.
        // For MleAst, f_val IS the challenge, and we can transmute since they're the same type.
        //
        // Since this is MleAst-specific: use type punning via mem::transmute
        // This is safe because for MleAst, F = F::Challenge = MleAst
        unsafe { std::mem::transmute_copy::<F, F::Challenge>(&f_val) }
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        // Use the pending challenge mechanism for each element
        (0..len)
            .map(|_| {
                let hash = self.challenge_mle();
                let challenge = MleAst::truncate_128_reverse(&hash);
                set_pending_challenge(challenge);
                let f_val: F = F::from_bytes(&[0u8; 16]);
                // Same transmute as above - safe for MleAst where F = F::Challenge
                unsafe { std::mem::transmute_copy::<F, F::Challenge>(&f_val) }
            })
            .collect()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let _ = self.challenge_mle();
        vec![F::zero(); len]
    }

    fn debug_state(&self, label: &str) {
        println!("TRANSCRIPT DEBUG [{}]: n_rounds={}", label, self.n_rounds);
    }
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
                    "Label 'jolt' should be {:?} but got {:?}",
                    expected, v
                );
            }
            _ => panic!("Expected Scalar atom, got {:?}", node),
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
            zklean_extractor::mle_ast::Node::Poseidon(_, _, _) => {
                // Expected: poseidon(label, 0, 0)
            }
            _ => panic!("Expected Poseidon node for initial state, got {:?}", node),
        }
    }

    #[test]
    fn test_append_and_challenge() {
        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");
        transcript.hash_and_update(MleAst::from_u64(42));
        let _challenge = transcript.challenge_mle();
        assert_eq!(transcript.n_rounds, 2); // 1 append + 1 challenge
    }

    #[test]
    fn test_append_scalar_with_mle_ast() {
        use jolt_core::transcripts::Transcript as _;

        let mut transcript: PoseidonAstTranscript = Transcript::new(b"test");

        // Create a variable (not a constant)
        let var = MleAst::from_var(42);

        // Append it to transcript
        transcript.append_scalar(&var);

        // Check that the state now contains a Poseidon node with ByteReverse(variable)
        // append_scalar does: byte_reverse(var) -> hash
        let root = transcript.state.root();
        let node = zklean_extractor::mle_ast::get_node(root);

        match node {
            zklean_extractor::mle_ast::Node::Poseidon(_, _, e3) => {
                // The third argument should be a ByteReverse node containing the variable
                match e3 {
                    zklean_extractor::mle_ast::Edge::NodeRef(byte_rev_id) => {
                        let byte_rev_node = zklean_extractor::mle_ast::get_node(byte_rev_id);
                        match byte_rev_node {
                            zklean_extractor::mle_ast::Node::ByteReverse(inner) => {
                                match inner {
                                    zklean_extractor::mle_ast::Edge::Atom(
                                        zklean_extractor::mle_ast::Atom::Var(idx),
                                    ) => {
                                        assert_eq!(idx, 42, "Expected Var(42), got Var({})", idx);
                                    }
                                    other => panic!(
                                        "Expected Var(42) inside ByteReverse, got {:?}",
                                        other
                                    ),
                                }
                            }
                            other => panic!("Expected ByteReverse node, got {:?}", other),
                        }
                    }
                    other => panic!(
                        "Expected NodeRef to ByteReverse as third Poseidon arg, got {:?}",
                        other
                    ),
                }
            }
            _ => panic!("Expected Poseidon node, got {:?}", node),
        }
    }
}
