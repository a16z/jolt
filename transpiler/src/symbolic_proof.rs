//! Convert a real JoltProof to a symbolic JoltProof for transpilation.
//!
//! # Overview
//!
//! This module creates symbolic versions of proof data structures. During symbolic
//! execution, we need a `JoltProof<MleAst>` where each field element is replaced
//! with an `MleAst::Var(index)`, a unique symbolic variable.
//!
//! # Key Function
//!
//! - [`symbolize_proof`]: Convert a concrete `RV64IMACProof` to symbolic form
//!
//! # How It Works
//!
//! The `VarAllocator` simultaneously:
//! 1. Allocates fresh symbolic variables (`MleAst::Var(index)`)
//! 2. Records the corresponding concrete witness values
//!
//! This single-pass approach makes witness/symbolization mismatches structurally
//! impossible - both are recorded in the same function call.
//!
//! # Commitment Serialization
//!
//! Dory commitments are 384-byte elliptic curve points. They're split into 12 chunks
//! of 32 bytes each (to fit in BN254 field elements). The serialization uses:
//! - `serialize_uncompressed` (not compressed)
//! - LE byte order (no reversal needed for circuit)
//! Dory will probably be replaced in future iterations,
//! the transpilation code will need to be updated in that case.
//!
//! This must match exactly how the Poseidon transcript hashes commitments.

use crate::symbolic_traits::ast_commitment_scheme::{AstCommitmentScheme, AstProof};
use crate::symbolic_traits::opening_accumulator::AstOpeningAccumulator;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use jolt_core::poly::opening_proof::OpeningPoint;
use jolt_core::poly::unipoly::CompressedUniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProof;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::RV64IMACProof;
use std::collections::BTreeMap;
use zklean_extractor::mle_ast::{MleAst, TargetField};
use zklean_extractor::AstCommitment;

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
/// - Concrete witness values as decimal strings (for JSON serialization to Go)
/// - Field kind per variable (Fr for native, Fq for emulated arithmetic)
pub struct VarAllocator {
    next_idx: u16,
    /// (index, name, target_field) tuples for each allocated variable.
    descriptions: Vec<(u16, String, TargetField)>,
    /// Witness values indexed by variable index, stored as decimal strings.
    witness_values: Vec<String>,
}

impl VarAllocator {
    pub fn new() -> Self {
        Self {
            next_idx: 0,
            descriptions: Vec::new(),
            witness_values: Vec::new(),
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
    /// * `value`: Concrete witness value (as Fr, converted to decimal string)
    /// * `target_field`: Target field (Fr for native, Fq for emulated)
    ///
    /// # Note
    /// The value is stored as a decimal string regardless of field.
    /// For Fq values, ensure the value fits in the Fq modulus.
    pub fn alloc_with_value_and_field(
        &mut self,
        description: &str,
        value: &ark_bn254::Fr,
        target_field: TargetField,
    ) -> MleAst {
        use ark_ff::PrimeField;
        let idx = self.next_idx;
        self.descriptions
            .push((idx, description.to_string(), target_field));
        self.witness_values.push(format!("{}", value.into_bigint()));
        self.next_idx += 1;
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

    pub fn next_idx(&self) -> u16 {
        self.next_idx
    }

    /// Get descriptions with target fields for AstBundle population.
    pub fn descriptions_with_fields(&self) -> &[(u16, String, TargetField)] {
        &self.descriptions
    }

    /// Get descriptions without field kinds (backward compatible iterator).
    pub fn descriptions(&self) -> impl Iterator<Item = (u16, &str)> + '_ {
        self.descriptions
            .iter()
            .map(|(idx, name, _)| (*idx, name.as_str()))
    }

    /// Get witness values as a HashMap for JSON serialization.
    pub fn witness_values(&self) -> std::collections::HashMap<usize, String> {
        self.witness_values
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect()
    }

    /// Check if any variables with the specified field kind were allocated.
    pub fn has_variables_for_field(&self, field: TargetField) -> bool {
        self.descriptions
            .iter()
            .any(|(_, _, tf)| *tf == field)
    }

    /// Allocate variables for a commitment's 12 chunks and record witness values (Fr field).
    ///
    /// Commitments are serialized as uncompressed LE bytes,
    /// then split into 12 × 32-byte chunks (each fits in a BN254 field element).
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

/// Number of bytes per chunk (one BN254 field element)
const BYTES_PER_CHUNK: usize = 32;

/// Serialize a commitment to bytes in the format used by Poseidon transcript.
/// MUST match the Poseidon transcript serialization exactly:
/// 1. Use serialize_uncompressed (not compressed)
/// 2. LE bytes directly (no byte reversal needed for circuit)
fn commitment_to_bytes<T: CanonicalSerialize>(commitment: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    commitment
        .serialize_uncompressed(&mut bytes)
        .expect("serialization failed");
    bytes
}

/// Convert commitment bytes to field element chunks.
///
/// The number of chunks is derived from the serialized size:
/// `num_chunks = ceil(serialized_size / 32)`
///
/// This is PCS-agnostic: Dory (384 bytes) produces 12 chunks,
/// other PCS types produce different chunk counts based on their commitment size.
fn commitment_to_field_chunks<T: CanonicalSerialize>(commitment: &T) -> Vec<ark_bn254::Fr> {
    let bytes = commitment_to_bytes(commitment);
    let num_chunks = (bytes.len() + BYTES_PER_CHUNK - 1) / BYTES_PER_CHUNK; // ceil division

    (0..num_chunks)
        .map(|i| {
            let start = i * BYTES_PER_CHUNK;
            let end = std::cmp::min(start + BYTES_PER_CHUNK, bytes.len());
            ark_bn254::Fr::from_le_bytes_mod_order(&bytes[start..end])
        })
        .collect()
}

/// Convert a real proof to a symbolic proof for transpilation.
///
/// This is the main entry point for proof symbolization. It creates symbolic
/// variables for every field element in the proof structure.
///
/// # Variable Naming Convention
///
/// Variables are named by their semantic role in the proof:
/// - `commitment_{n}_{chunk}` - Chunk (0-11) of commitment n
/// - `claim_{key:?}` - Opening claim for polynomial key
/// - `stage{n}_uni_skip_coeff_{i}` - Uni-skip polynomial coefficient i
/// - `stage{n}_sumcheck_r{round}_{coeff}` - Sumcheck round polynomial coefficient
/// - `untrusted_advice_commitment_{chunk}` - Advice commitment chunk (if present)
///
/// These names appear in the witness JSON and are transformed by `sanitize_go_name`
/// for Go struct field names.
///
/// # Returns
///
/// - `JoltProof<MleAst>`: The symbolic proof with variables instead of concrete values
/// - `AstOpeningAccumulator`: Accumulator pre-populated with symbolic opening claims
/// - `VarAllocator`: Tracks all allocated variables and their descriptions
///
/// # Variable Naming Convention
///
/// Variables are named by their semantic role:
/// - `commitment_N_M`: Chunk M of commitment N
/// - `stageX_sumcheck_rY_Z`: Stage X, round Y, coefficient Z
/// - `stageX_uni_skip_coeff_Y`: Univariate skip polynomial coefficient
/// - `claim_KEY`: Opening claim for polynomial KEY
///
/// # Type Parameter
///
/// `OutputTranscript` specifies the transcript type for the symbolic proof.
/// Use `PoseidonAstTranscript` for Poseidon-based proofs (current default).
pub fn symbolize_proof<OutputTranscript: Transcript>(
    real_proof: &RV64IMACProof,
) -> (
    JoltProof<MleAst, AstCommitmentScheme, OutputTranscript>,
    AstOpeningAccumulator,
    VarAllocator,
) {
    let mut alloc = VarAllocator::new();

    // === Symbolize commitments (with witness values) ===
    let commitments: Vec<AstCommitment> = real_proof
        .commitments
        .iter()
        .enumerate()
        .map(|(c, commitment)| {
            let chunks = alloc.alloc_commitment(commitment, &format!("commitment_{c}"));
            AstCommitment::new(chunks)
        })
        .collect();

    // === Symbolize opening claims (with witness values) ===
    let mut symbolic_claims = BTreeMap::new();
    for (key, (_point, claim)) in &real_proof.opening_claims.0 {
        let symbolic_claim = alloc.alloc_with_value(&format!("claim_{key:?}"), claim);
        symbolic_claims.insert(*key, (OpeningPoint::default(), symbolic_claim));
    }

    // === Symbolize stage 1 uni-skip proof ===
    let stage1_uni_skip = symbolize_uni_skip_proof::<_, OutputTranscript>(
        &real_proof.stage1_uni_skip_first_round_proof,
        &mut alloc,
        "stage1_uni_skip",
    );

    // === Symbolize stage 1 sumcheck proof ===
    let stage1_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage1_sumcheck_proof,
        &mut alloc,
        "stage1_sumcheck",
    );

    // === Symbolize stage 2 uni-skip proof ===
    let stage2_uni_skip = symbolize_uni_skip_proof::<_, OutputTranscript>(
        &real_proof.stage2_uni_skip_first_round_proof,
        &mut alloc,
        "stage2_uni_skip",
    );

    // === Symbolize stage 2 sumcheck proof ===
    let stage2_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage2_sumcheck_proof,
        &mut alloc,
        "stage2_sumcheck",
    );

    // === Symbolize stage 3 sumcheck proof ===
    let stage3_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage3_sumcheck_proof,
        &mut alloc,
        "stage3_sumcheck",
    );

    // === Symbolize stage 4 sumcheck proof ===
    let stage4_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage4_sumcheck_proof,
        &mut alloc,
        "stage4_sumcheck",
    );

    // === Symbolize stage 5 sumcheck proof ===
    let stage5_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage5_sumcheck_proof,
        &mut alloc,
        "stage5_sumcheck",
    );

    // === Symbolize stage 6 sumcheck proof ===
    let stage6_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage6_sumcheck_proof,
        &mut alloc,
        "stage6_sumcheck",
    );

    // === Symbolize stage 7 sumcheck proof ===
    let stage7_sumcheck = symbolize_sumcheck_proof::<_, OutputTranscript>(
        &real_proof.stage7_sumcheck_proof,
        &mut alloc,
        "stage7_sumcheck",
    );

    // === Symbolize advice commitment (if present, with witness values) ===
    let untrusted_advice_commitment =
        real_proof
            .untrusted_advice_commitment
            .as_ref()
            .map(|commitment| {
                let chunks = alloc.alloc_commitment(commitment, "untrusted_advice_commitment");
                AstCommitment::new(chunks)
            });

    // Build the symbolic proof
    let symbolic_proof = JoltProof {
        opening_claims: Claims(symbolic_claims),
        commitments,
        stage1_uni_skip_first_round_proof: stage1_uni_skip,
        stage1_sumcheck_proof: stage1_sumcheck,
        stage2_uni_skip_first_round_proof: stage2_uni_skip,
        stage2_sumcheck_proof: stage2_sumcheck,
        stage3_sumcheck_proof: stage3_sumcheck,
        stage4_sumcheck_proof: stage4_sumcheck,
        stage5_sumcheck_proof: stage5_sumcheck,
        stage6_sumcheck_proof: stage6_sumcheck,
        stage7_sumcheck_proof: stage7_sumcheck,
        joint_opening_proof: AstProof::default(),
        untrusted_advice_commitment,
        trace_length: real_proof.trace_length,
        ram_K: real_proof.ram_K,
        bytecode_K: real_proof.bytecode_K,
        rw_config: real_proof.rw_config.clone(),
        one_hot_config: real_proof.one_hot_config.clone(),
        dory_layout: real_proof.dory_layout,
    };

    // Build the opening accumulator with the symbolic claims we created
    #[allow(non_snake_case)] // Match VerifierOpeningAccumulator naming
    let log_T = (real_proof.trace_length as f64).log2().ceil() as usize;
    let mut accumulator = AstOpeningAccumulator::new(log_T);
    for (key, (_, claim)) in &symbolic_proof.opening_claims.0 {
        accumulator.openings.insert(*key, (vec![], *claim));
    }

    (symbolic_proof, accumulator, alloc)
}

// =============================================================================
// Symbolization Helpers
// =============================================================================
//
// These functions convert concrete proof components (Fr values) to symbolic form
// (MleAst variables) while simultaneously recording witness values in VarAllocator.

fn symbolize_uni_skip_proof<T: Transcript, OutT: Transcript>(
    real: &UniSkipFirstRoundProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> UniSkipFirstRoundProof<MleAst, OutT> {
    let coeffs = alloc.alloc_n_with_values(&real.uni_poly.coeffs, &format!("{prefix}_coeff"));
    UniSkipFirstRoundProof::new(jolt_core::poly::unipoly::UniPoly::from_coeff(coeffs))
}

fn symbolize_sumcheck_proof<T: Transcript, OutT: Transcript>(
    real: &SumcheckInstanceProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> SumcheckInstanceProof<MleAst, OutT> {
    let compressed_polys: Vec<CompressedUniPoly<MleAst>> = real
        .compressed_polys
        .iter()
        .enumerate()
        .map(|(round, poly)| {
            let coeffs = alloc.alloc_n_with_values(
                &poly.coeffs_except_linear_term,
                &format!("{prefix}_r{round}"),
            );
            CompressedUniPoly {
                coeffs_except_linear_term: coeffs,
            }
        })
        .collect();

    SumcheckInstanceProof::new(compressed_polys)
}
