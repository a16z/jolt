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
//! - Byte reversal for big-endian/EVM compatibility
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
use zklean_extractor::mle_ast::MleAst;
use zklean_extractor::AstCommitment;

/// Tracks variable index allocation and witness values during symbolization.
///
/// Each call to `alloc_with_value()` returns a fresh `MleAst::Var(index)` with a unique index,
/// while simultaneously recording the concrete witness value. This ensures witness values
/// are always in sync with symbolic variable allocation - making mismatch bugs structurally
/// impossible.
///
/// The allocator records:
/// - Human-readable descriptions for Go struct field names (e.g., `Stage1_Sumcheck_R0_0`)
/// - Concrete witness values as decimal strings (for JSON serialization to Go)
pub struct VarAllocator {
    next_idx: u16,
    descriptions: Vec<(u16, String)>,
    /// Witness values indexed by variable index, stored as decimal strings.
    witness_values: Vec<String>,
}

impl VarAllocator {
    // Public methods

    pub fn new() -> Self {
        Self {
            next_idx: 0,
            descriptions: Vec::new(),
            witness_values: Vec::new(),
        }
    }

    /// Allocate N variables with their concrete values.
    ///
    /// Both symbolic variables and witness values are recorded in the same call,
    /// guaranteeing they stay in sync.
    pub fn alloc_n_with_values(&mut self, values: &[ark_bn254::Fr], prefix: &str) -> Vec<MleAst> {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| self.alloc_with_value(&format!("{prefix}_{i}"), v))
            .collect()
    }

    /// Allocate a single variable with its concrete value.
    pub fn alloc_with_value(&mut self, description: &str, value: &ark_bn254::Fr) -> MleAst {
        use ark_ff::PrimeField;
        let idx = self.next_idx;
        self.descriptions.push((idx, description.to_string()));
        self.witness_values.push(format!("{}", value.into_bigint()));
        self.next_idx += 1;
        MleAst::from_var(idx)
    }

    pub fn next_idx(&self) -> u16 {
        self.next_idx
    }

    pub fn descriptions(&self) -> &[(u16, String)] {
        &self.descriptions
    }

    /// Get witness values as a HashMap for JSON serialization.
    pub fn witness_values(&self) -> std::collections::HashMap<usize, String> {
        self.witness_values
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect()
    }

    /// Allocate variables for a commitment's 12 chunks and record witness values.
    ///
    /// Commitments are serialized as uncompressed bytes, reversed for BE format,
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

/// Number of 32-byte chunks per commitment (384 bytes / 32 = 12)
const CHUNKS_PER_COMMITMENT: usize = 12;

/// Serialize a commitment to bytes in the format used by Poseidon transcript.
/// MUST match the Poseidon transcript serialization exactly:
/// 1. Use serialize_uncompressed (not compressed)
/// 2. Reverse bytes for BE/EVM format
fn commitment_to_bytes<T: CanonicalSerialize>(commitment: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    commitment
        .serialize_uncompressed(&mut bytes)
        .expect("serialization failed");
    // Reverse bytes to match Poseidon transcript format (BE for EVM compatibility)
    bytes.reverse();
    bytes
}

/// Convert commitment bytes to 12 field element chunks.
fn commitment_to_field_chunks<T: CanonicalSerialize>(commitment: &T) -> Vec<ark_bn254::Fr> {
    let bytes = commitment_to_bytes(commitment);
    (0..CHUNKS_PER_COMMITMENT)
        .map(|i| {
            let start = i * 32;
            let end = std::cmp::min(start + 32, bytes.len());
            if start >= bytes.len() {
                ark_bn254::Fr::from(0u64)
            } else {
                ark_bn254::Fr::from_le_bytes_mod_order(&bytes[start..end])
            }
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
