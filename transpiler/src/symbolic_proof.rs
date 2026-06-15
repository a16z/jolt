//! Symbolize a real `RV64IMACProof` for NARG-replay transpilation.
//!
//! # Overview
//!
//! Under the spongefish/NARG proof format, only the proof's *structural* parts are
//! symbolized here up front:
//!
//! - the non-ZK `opening_claims` (named `claim_{OpeningId:?}`), pre-seeded into the
//!   [`AstOpeningAccumulator`];
//! - the structural scalars (`trace_length`, `ram_K`, configs) carried as
//!   [`TranspilableProofData`].
//!
//! Everything else (witness commitments, advice presence, uni-skip coefficients,
//! sumcheck round polynomials) lives in the NARG byte-string: it is split into frames
//! by [`crate::narg_parser::parse_narg`] and symbolized *lazily during replay* by
//! `SymbolicVerifierFs::read_slice`, which allocates witness variables from the real
//! frame bytes at the exact protocol position the verifier reads them.
//!
//! The `VarAllocator` still guarantees the core invariant: symbolic variables and
//! their concrete witness values are recorded in the same call, making
//! witness/symbolization mismatches structurally impossible.
//!
//! # Commitment Serialization
//!
//! Dory commitments are 384-byte GT elements, split into 13 byte-rule chunks
//! (12 × 31 bytes + one 12-byte tail; each chunk < 2²⁴⁸ < r so the map to `Fr`
//! is injective — spec §4.2); `serialize_compressed` (== uncompressed for GT),
//! LE order, no reversal. MUST match `jolt_transcript`'s byte rule exactly.

#[cfg(not(feature = "zk"))]
use crate::narg_parser::parse_narg;
use crate::narg_parser::{NargParseError, ParsedNarg};
use crate::symbolic_traits::opening_accumulator::AstOpeningAccumulator;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use jolt_core::zkvm::transpilable_verifier::TranspilableProofData;
use jolt_core::zkvm::RV64IMACProof;
use zklean_extractor::mle_ast::{MleAst, TargetField};

/// Tracks variable index allocation and witness values during symbolization.
///
/// Each call to `alloc_with_value()` returns a fresh `MleAst::Var(index)` with a unique index,
/// while simultaneously recording the concrete witness value. This ensures witness values
/// are always in sync with symbolic variable allocation, making mismatch bugs structurally
/// impossible.
///
/// # Field Kind Tracking
///
/// Variables can be allocated for different target fields (Fr or Fq). The allocator
/// tracks field kind per variable to enable correct codegen. Currently only Fr is
/// supported at codegen time; Fq variables will panic with a clear error.
///
/// The allocator records:
/// - Human-readable descriptions for Go struct field names (e.g., `Stage1_Sumcheck_R0_0`)
/// - Concrete witness values as native `Fr` (rendered to decimal strings on demand
///   for JSON serialization to Go)
/// - Field kind per variable (Fr for native, Fq for emulated arithmetic)
pub struct VarAllocator {
    next_idx: u16,
    /// (index, name, target_field) tuples for each allocated variable.
    descriptions: Vec<(u16, String, TargetField)>,
    /// Witness values indexed by variable index, as native `Fr`: consumed by
    /// in-process AST evaluation (the challenge differential in
    /// `tests/symbolic_pipeline.rs`) and rendered to decimal by [`Self::witness_values`].
    witness_frs: Vec<ark_bn254::Fr>,
}

impl VarAllocator {
    pub fn new() -> Self {
        Self {
            next_idx: 0,
            descriptions: Vec::new(),
            witness_frs: Vec::new(),
        }
    }

    /// Allocate a single variable with its concrete value (Fr field, default).
    ///
    /// This is the primary allocation method for stages 1-7.
    pub fn alloc_with_value(&mut self, description: &str, value: &ark_bn254::Fr) -> MleAst {
        self.alloc_with_value_and_field(description, value, TargetField::Fr)
    }

    /// Allocate a single variable with explicit target field.
    ///
    /// # Arguments
    /// * `description`: Human-readable name for codegen
    /// * `value`: Concrete witness value (as Fr)
    /// * `target_field`: Target field (Fr for native, Fq for emulated)
    ///
    /// # Note
    /// The value is stored as `Fr` regardless of field.
    /// For Fq values, ensure the value fits in the Fq modulus.
    pub fn alloc_with_value_and_field(
        &mut self,
        description: &str,
        value: &ark_bn254::Fr,
        target_field: TargetField,
    ) -> MleAst {
        let idx = self.next_idx;
        self.descriptions
            .push((idx, description.to_string(), target_field));
        self.witness_frs.push(*value);
        // `MleAst::from_var` indexes witness variables with a `u16`. A NARG large
        // enough to need >65 535 variables (very large traces) would otherwise wrap
        // silently in release and reuse indices, corrupting the witness↔symbol map.
        // Fail loudly instead (code-review #2). Lifting the cap is an MleAst change.
        self.next_idx = self.next_idx.checked_add(1).unwrap_or_else(|| {
            panic!(
                "VarAllocator exceeded u16::MAX ({}) symbolic variables — NARG too large \
                 for the current MleAst u16 var-index width",
                u16::MAX
            )
        });
        MleAst::from_var(idx)
    }

    /// Allocate N variables with their concrete values (Fr field, default).
    ///
    /// Both symbolic variables and witness values are recorded in the same call,
    /// guaranteeing they stay in sync.
    pub fn alloc_n_with_values(&mut self, values: &[ark_bn254::Fr], prefix: &str) -> Vec<MleAst> {
        self.alloc_n_with_values_and_field(values, prefix, TargetField::Fr)
    }

    /// Allocate N variables with explicit field kind.
    pub fn alloc_n_with_values_and_field(
        &mut self,
        values: &[ark_bn254::Fr],
        prefix: &str,
        target_field: TargetField,
    ) -> Vec<MleAst> {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                self.alloc_with_value_and_field(&format!("{prefix}_{i}"), v, target_field)
            })
            .collect()
    }

    #[cfg(test)]
    pub fn next_idx(&self) -> u16 {
        self.next_idx
    }

    /// Get descriptions with target fields for AstBundle population.
    pub fn descriptions_with_fields(&self) -> &[(u16, String, TargetField)] {
        &self.descriptions
    }

    /// Witness values as native `Fr` keyed by variable index — the map
    /// `ast_evaluator::eval_root`/`eval_roots` consume (challenge differential,
    /// `tests/symbolic_pipeline.rs`).
    pub fn witness_fr_map(&self) -> std::collections::HashMap<u16, ark_bn254::Fr> {
        self.witness_frs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u16, *v))
            .collect()
    }

    /// Get witness values as a HashMap of decimal strings for JSON serialization.
    pub fn witness_values(&self) -> std::collections::HashMap<usize, String> {
        self.witness_frs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.into_bigint().to_string()))
            .collect()
    }

    /// Check if any variables with the specified field kind were allocated.
    #[cfg(test)]
    pub fn has_variables_for_field(&self, field: TargetField) -> bool {
        self.descriptions.iter().any(|(_, _, tf)| *tf == field)
    }

    /// Allocate variables for a commitment's byte-rule chunks (13 for a Dory
    /// GT) and record witness values (Fr field).
    ///
    /// Commitments are serialized as compressed LE bytes, then split into
    /// 31-byte chunks (12 × 31B + one 12-byte tail for the 384-byte GT; each
    /// chunk < 2²⁴⁸ < r) — the exact byte rule the field-aligned sponge
    /// absorbs them under.
    pub fn alloc_commitment<T: CanonicalSerialize>(
        &mut self,
        commitment: &T,
        prefix: &str,
    ) -> Vec<MleAst> {
        let chunks = commitment_to_field_chunks(commitment);
        self.alloc_n_with_values(&chunks, prefix)
    }
}

impl Default for VarAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Bytes per byte-rule chunk (spec §4.2: 31-byte LE chunks, each < 2²⁴⁸ < r
/// so `from_le_bytes_mod_order` is exact/injective). Mirrors
/// `jolt_transcript::BYTE_RULE_CHUNK` / `zklean_extractor::BYTES_PER_CHUNK`.
/// Only the chunking tests reference the width directly now that
/// `commitment_to_field_chunks` delegates to `jolt_transcript::commitment_to_chunks`.
#[cfg(test)]
const BYTES_PER_CHUNK: usize = jolt_transcript::BYTE_RULE_CHUNK;

/// Serialize a commitment to bytes in the format the field-aligned Poseidon
/// transcript absorbs: `serialize_compressed` (== uncompressed for Dory GT),
/// LE bytes directly — must match `jolt_transcript::CommitmentsMsg` /
/// `FsAbsorb::absorb_commitment` exactly.
fn commitment_to_bytes<T: CanonicalSerialize>(commitment: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    commitment
        .serialize_compressed(&mut bytes)
        .expect("serialization failed");
    bytes
}

/// Convert commitment bytes to byte-rule field element chunks
/// (`num_chunks = ceil(serialized_size / 31)`; Dory's 384 bytes → 13 chunks =
/// 12 × 31B + one 12-byte tail). Chunk values are exactly what the native
/// sponge's `push_byte_rule_units` absorbs for the same bytes.
fn commitment_to_field_chunks<T: CanonicalSerialize>(commitment: &T) -> Vec<ark_bn254::Fr> {
    jolt_transcript::commitment_to_chunks(&commitment_to_bytes(commitment))
}

/// The structural parts of a symbolized proof; the NARG frames are symbolized lazily
/// during replay (see module docs).
pub struct SymbolizedProof {
    /// The proof's NARG split into ordered frames (zk_mode already refused).
    pub parsed_narg: ParsedNarg,
    /// Accumulator pre-seeded with the symbolic opening claims, exactly as
    /// `JoltVerifier::new` seeds its accumulator from `proof.opening_claims`.
    pub accumulator: AstOpeningAccumulator,
    /// The structural proof fields `TranspilableVerifier::new` validates and uses.
    pub proof_data: TranspilableProofData,
}

/// Symbolize a real proof's structural parts for NARG-replay transpilation.
///
/// Allocates one named variable per non-ZK opening claim (`claim_{OpeningId:?}` —
/// the frozen Era-2 naming the Go side keys on) and splits the NARG into frames.
/// All other proof values become variables during replay, named by
/// `SymbolicVerifierFs`'s `FrameLabel` context (`commitment_{c}_{chunk}`,
/// `stage{n}_uni_skip_coeff_{i}`, `stage{n}_sumcheck_r{round}_{i}`, ...).
#[cfg(not(feature = "zk"))]
pub fn symbolize_proof(
    real_proof: &RV64IMACProof,
    alloc: &mut VarAllocator,
) -> Result<SymbolizedProof, NargParseError> {
    let parsed_narg = parse_narg(&real_proof.narg, real_proof.zk_mode)?;

    let claims: Vec<_> = real_proof
        .opening_claims
        .0
        .iter()
        .map(|(key, (_point, claim))| {
            (
                *key,
                alloc.alloc_with_value(&format!("claim_{key:?}"), claim),
            )
        })
        .collect();

    // Exact integer log2, matching the real verifier (`proof.trace_length.log_2()`,
    // verifier.rs) — trace_length is a validated power of two. (code-review #5)
    use jolt_core::utils::math::Math;
    #[expect(non_snake_case, reason = "matches VerifierOpeningAccumulator naming")]
    let log_T = real_proof.trace_length.log_2();
    let accumulator = AstOpeningAccumulator::new_with_claims(claims, log_T);

    let proof_data = TranspilableProofData::from_proof(real_proof);

    Ok(SymbolizedProof {
        parsed_narg,
        accumulator,
        proof_data,
    })
}

/// ZK proofs are out of scope (spec §16 guardrail 4 / §17): their NARG carries extra
/// frames that the non-ZK replay would silently mis-assign.
#[cfg(feature = "zk")]
pub fn symbolize_proof(
    _real_proof: &RV64IMACProof,
    _alloc: &mut VarAllocator,
) -> Result<SymbolizedProof, NargParseError> {
    Err(NargParseError::ZkProofUnsupported)
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::PrimeField;
    use zklean_extractor::mle_ast::{get_node, Atom, Node};

    // VarAllocator Tests

    #[test]
    fn test_var_allocator_index_increments() {
        // Each allocation must get a unique, incrementing index
        let mut alloc = VarAllocator::new();
        assert_eq!(alloc.next_idx(), 0);

        let var0 = alloc.alloc_with_value("v0", &Fr::from(100u64));
        assert_eq!(alloc.next_idx(), 1);

        let var1 = alloc.alloc_with_value("v1", &Fr::from(200u64));
        assert_eq!(alloc.next_idx(), 2);

        // Verify variables have different indices
        assert!(matches!(get_node(var0.root()), Node::Atom(Atom::Var(0))));
        assert!(matches!(get_node(var1.root()), Node::Atom(Atom::Var(1))));
    }

    #[test]
    fn test_var_allocator_witness_index_sync() {
        // CRITICAL: witness_values[i] MUST correspond to Var(i)
        // This is the core invariant - if this breaks, witness loading fails
        let mut alloc = VarAllocator::new();

        let val0 = Fr::from(111u64);
        let val1 = Fr::from(222u64);
        let val2 = Fr::from(333u64);

        let var0 = alloc.alloc_with_value("v0", &val0);
        let var1 = alloc.alloc_with_value("v1", &val1);
        let var2 = alloc.alloc_with_value("v2", &val2);

        let witness = alloc.witness_values();

        // Extract variable indices from MleAst nodes
        let idx0 = match get_node(var0.root()) {
            Node::Atom(Atom::Var(i)) => i,
            _ => panic!("Expected Var node"),
        };
        let idx1 = match get_node(var1.root()) {
            Node::Atom(Atom::Var(i)) => i,
            _ => panic!("Expected Var node"),
        };
        let idx2 = match get_node(var2.root()) {
            Node::Atom(Atom::Var(i)) => i,
            _ => panic!("Expected Var node"),
        };

        // CRITICAL: witness[idx] MUST match the value allocated with Var(idx)
        assert_eq!(
            witness.get(&(idx0 as usize)).unwrap(),
            &format!("{}", val0.into_bigint())
        );
        assert_eq!(
            witness.get(&(idx1 as usize)).unwrap(),
            &format!("{}", val1.into_bigint())
        );
        assert_eq!(
            witness.get(&(idx2 as usize)).unwrap(),
            &format!("{}", val2.into_bigint())
        );
    }

    #[test]
    fn test_var_allocator_field_type_fr() {
        // Verify Fr (native) field type tracking
        let mut alloc = VarAllocator::new();
        alloc.alloc_with_value_and_field("fr_var", &Fr::from(123u64), TargetField::Fr);

        assert!(alloc.has_variables_for_field(TargetField::Fr));
        assert!(!alloc.has_variables_for_field(TargetField::Fq));

        let descriptions = alloc.descriptions_with_fields();
        assert_eq!(descriptions[0].2, TargetField::Fr);
    }

    #[test]
    fn test_var_allocator_field_type_non_native() {
        // Verify non-native field type tracking (Fq in this case, but could be any non-Fr field)
        let mut alloc = VarAllocator::new();
        alloc.alloc_with_value_and_field("non_native_var", &Fr::from(456u64), TargetField::Fq);

        assert!(alloc.has_variables_for_field(TargetField::Fq));
        assert!(!alloc.has_variables_for_field(TargetField::Fr));

        let descriptions = alloc.descriptions_with_fields();
        assert_eq!(descriptions[0].2, TargetField::Fq);
    }

    #[test]
    fn test_var_allocator_has_variables_for_field() {
        // Verify field query works with mixed allocations (native + non-native)
        let mut alloc = VarAllocator::new();

        alloc.alloc_with_value("fr_default", &Fr::from(10u64)); // Default = Fr (native)
        alloc.alloc_with_value_and_field("non_native_explicit", &Fr::from(20u64), TargetField::Fq);

        assert!(alloc.has_variables_for_field(TargetField::Fr));
        assert!(alloc.has_variables_for_field(TargetField::Fq));
    }

    #[test]
    fn test_alloc_n_with_values() {
        // Test batch allocation with alloc_n_with_values
        let mut alloc = VarAllocator::new();

        let values = vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)];
        let vars = alloc.alloc_n_with_values(&values, "batch");

        // Verify correct number of variables allocated
        assert_eq!(vars.len(), 3);
        assert_eq!(alloc.next_idx(), 3);

        // Verify each variable has correct index
        for (i, var) in vars.iter().enumerate() {
            assert!(matches!(
                get_node(var.root()),
                Node::Atom(Atom::Var(idx)) if idx as usize == i
            ));
        }

        // Verify witness values
        let witness = alloc.witness_values();
        for (i, val) in values.iter().enumerate() {
            assert_eq!(witness.get(&i).unwrap(), &format!("{}", val.into_bigint()));
        }

        // Verify descriptions
        let descriptions = alloc.descriptions_with_fields();
        assert_eq!(descriptions[0].1, "batch_0");
        assert_eq!(descriptions[1].1, "batch_1");
        assert_eq!(descriptions[2].1, "batch_2");

        // Verify all have Fr field type (default)
        for (_, _, field) in descriptions {
            assert_eq!(*field, TargetField::Fr);
        }
    }

    #[test]
    fn test_alloc_commitment() {
        // Test alloc_commitment produces correct number of variables
        use ark_bn254::G1Affine;
        use ark_std::UniformRand;

        let mut alloc = VarAllocator::new();
        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);

        let vars = alloc.alloc_commitment(&point, "commitment");

        // Verify number of variables matches chunking
        let chunks = commitment_to_field_chunks(&point);
        assert_eq!(vars.len(), chunks.len());

        // Verify indices are sequential
        for (i, var) in vars.iter().enumerate() {
            assert!(matches!(
                get_node(var.root()),
                Node::Atom(Atom::Var(idx)) if idx as usize == i
            ));
        }

        // Verify witness values match chunks
        let witness = alloc.witness_values();
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                witness.get(&i).unwrap(),
                &format!("{}", chunk.into_bigint())
            );
        }

        // Verify descriptions
        let descriptions = alloc.descriptions_with_fields();
        assert_eq!(descriptions[0].1, "commitment_0");
        assert!(descriptions.len() == chunks.len());
    }

    // Commitment Chunking Tests

    #[test]
    fn test_commitment_to_field_chunks_g1() {
        // Verify G1Affine commitment produces correct number of chunks
        use ark_bn254::G1Affine;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);

        let chunks = commitment_to_field_chunks(&point);
        let bytes = commitment_to_bytes(&point);

        // Verify chunk count matches ceil(bytes.len() / 31) (field-aligned byte rule)
        let expected_chunks = bytes.len().div_ceil(BYTES_PER_CHUNK);
        assert_eq!(
            chunks.len(),
            expected_chunks,
            "G1Affine ({} bytes) should produce {} chunks",
            bytes.len(),
            expected_chunks
        );

        // Verify chunks are non-trivial (random point → non-zero chunks)
        for (i, chunk) in chunks.iter().enumerate() {
            assert_ne!(
                *chunk,
                Fr::from(0u64),
                "Chunk {i} should not be zero for random point"
            );
        }
    }

    #[test]
    fn test_commitment_chunking_matches_byte_rule() {
        // CRITICAL: Verify chunk values match the field-aligned byte rule.
        // This ensures commitment_to_field_chunks produces identical chunks
        // to what the transcript absorbs when hashing commitments.
        use ark_bn254::G1Affine;
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let point = G1Affine::rand(&mut rng);

        let chunks = commitment_to_field_chunks(&point);
        let bytes = commitment_to_bytes(&point);

        // Verify each chunk matches its byte slice
        for (i, chunk) in chunks.iter().enumerate() {
            let start = i * BYTES_PER_CHUNK;
            let end = std::cmp::min(start + BYTES_PER_CHUNK, bytes.len());
            let expected_chunk = Fr::from_le_bytes_mod_order(&bytes[start..end]);

            assert_eq!(
                *chunk, expected_chunk,
                "Chunk {i} should match bytes[{start}..{end}] (byte rule)"
            );
        }
    }
}
