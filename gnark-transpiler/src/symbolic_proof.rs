//! Convert a real JoltProof to a symbolic JoltProof for transpilation.
//!
//! # Overview
//!
//! This module creates symbolic versions of proof data structures. During symbolic
//! execution, we need a `JoltProof<MleAst>` where each field element is replaced
//! with an `MleAst::Var(index)`, a unique symbolic variable.
//!
//! # Key Functions
//!
//! - [`symbolize_proof`]: Convert a concrete `RV64IMACProof` to symbolic form
//! - [`extract_witness_values`]: Extract concrete values for witness generation
//!
//! # How It Works
//!
//! 1. **Symbolization**: Each field in the proof (commitments, sumcheck coefficients,
//!    opening claims) gets a unique variable index. For example:
//!    - `commitment_0_0`, `commitment_0_1`, ... (12 chunks per commitment)
//!    - `stage1_sumcheck_r0_0`, `stage1_sumcheck_r0_1`, ... (coefficients per round)
//!
//! 2. **Witness Extraction**: The same traversal order is used to extract concrete
//!    values from the real proof. The indices match exactly, so `values[i]` corresponds
//!    to the variable allocated at position `i`.
//!
//! # Commitment Serialization
//!
//! Dory commitments are 384-byte elliptic curve points. They're split into 12 chunks
//! of 32 bytes each (to fit in BN254 field elements). The serialization uses:
//! - `serialize_uncompressed` (not compressed)
//! - Byte reversal for big-endian/EVM compatibility
//!
//! This must match exactly how the Poseidon transcript hashes commitments.

use crate::ast_commitment_scheme::{AstCommitmentScheme, AstProof};
use crate::MleOpeningAccumulator;
use crate::PoseidonAstTranscript;
use jolt_core::transcripts::Transcript;
use jolt_core::poly::opening_proof::OpeningPoint;
use jolt_core::poly::unipoly::CompressedUniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProof;
use jolt_core::zkvm::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::RV64IMACProof;
use std::collections::BTreeMap;
use zklean_extractor::mle_ast::MleAst;
use zklean_extractor::AstCommitment;

/// Tracks variable index allocation during symbolization.
///
/// Each call to `alloc()` returns a fresh `MleAst::Var(index)` with a unique index.
/// The allocator also records a human-readable description for each variable,
/// which is used for:
/// - Generating readable Go struct field names (e.g., `Stage1_Sumcheck_R0_0`)
/// - Mapping witness values back to their semantic meaning
pub struct VarAllocator {
    next_idx: u16,
    descriptions: Vec<(u16, String)>,
}

impl VarAllocator {
    pub fn new() -> Self {
        Self {
            next_idx: 0,
            descriptions: Vec::new(),
        }
    }

    pub fn alloc(&mut self, description: &str) -> MleAst {
        let idx = self.next_idx;
        self.descriptions.push((idx, description.to_string()));
        self.next_idx += 1;
        MleAst::from_var(idx)
    }

    pub fn alloc_n(&mut self, n: usize, prefix: &str) -> Vec<MleAst> {
        (0..n)
            .map(|i| self.alloc(&format!("{prefix}_{i}")))
            .collect()
    }

    pub fn next_idx(&self) -> u16 {
        self.next_idx
    }

    pub fn descriptions(&self) -> &[(u16, String)] {
        &self.descriptions
    }
}

impl Default for VarAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Number of 32-byte chunks per commitment (384 bytes / 32 = 12)
const CHUNKS_PER_COMMITMENT: usize = 12;

/// Convert a real proof to a symbolic proof for transpilation.
///
/// This is the main entry point for proof symbolization. It creates symbolic
/// variables for every field element in the proof structure.
///
/// # Returns
///
/// - `JoltProof<MleAst>`: The symbolic proof with variables instead of concrete values
/// - `MleOpeningAccumulator`: Accumulator pre-populated with symbolic opening claims
/// - `VarAllocator`: Tracks all allocated variables and their descriptions
///
/// # Variable Naming Convention
///
/// Variables are named by their semantic role:
/// - `commitment_N_M`: Chunk M of commitment N
/// - `stageX_sumcheck_rY_Z`: Stage X, round Y, coefficient Z
/// - `stageX_uni_skip_coeff_Y`: Univariate skip polynomial coefficient
/// - `claim_KEY`: Opening claim for polynomial KEY
pub fn symbolize_proof(
    real_proof: &RV64IMACProof,
) -> (
    JoltProof<MleAst, AstCommitmentScheme, PoseidonAstTranscript>,
    MleOpeningAccumulator,
    VarAllocator,
) {
    symbolize_proof_generic::<PoseidonAstTranscript>(real_proof)
}

/// Generic proof symbolization over any Transcript type
fn symbolize_proof_generic<ProofTranscript: Transcript>(
    real_proof: &RV64IMACProof,
) -> (
    JoltProof<MleAst, AstCommitmentScheme, ProofTranscript>,
    MleOpeningAccumulator,
    VarAllocator,
) {
    let mut alloc = VarAllocator::new();

    // === Symbolize commitments ===
    let commitments: Vec<AstCommitment> = (0..real_proof.commitments.len())
        .map(|c| {
            let chunks = alloc.alloc_n(CHUNKS_PER_COMMITMENT, &format!("commitment_{c}"));
            AstCommitment::new(chunks)
        })
        .collect();

    // === Symbolize opening claims ===
    let mut symbolic_claims = BTreeMap::new();
    for (key, (_point, _claim)) in &real_proof.opening_claims.0 {
        let symbolic_claim = alloc.alloc(&format!("claim_{key:?}"));
        symbolic_claims.insert(*key, (OpeningPoint::default(), symbolic_claim));
    }

    // === Symbolize stage 1 uni-skip proof ===
    let stage1_uni_skip = symbolize_uni_skip_proof::<_, ProofTranscript>(
        &real_proof.stage1_uni_skip_first_round_proof,
        &mut alloc,
        "stage1_uni_skip",
    );

    // === Symbolize stage 1 sumcheck proof ===
    let stage1_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage1_sumcheck_proof,
        &mut alloc,
        "stage1_sumcheck",
    );

    // === Symbolize stage 2 uni-skip proof ===
    let stage2_uni_skip = symbolize_uni_skip_proof::<_, ProofTranscript>(
        &real_proof.stage2_uni_skip_first_round_proof,
        &mut alloc,
        "stage2_uni_skip",
    );

    // === Symbolize stage 2 sumcheck proof ===
    let stage2_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage2_sumcheck_proof,
        &mut alloc,
        "stage2_sumcheck",
    );

    // === Symbolize stage 3 sumcheck proof ===
    let stage3_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage3_sumcheck_proof,
        &mut alloc,
        "stage3_sumcheck",
    );

    // === Symbolize stage 4 sumcheck proof ===
    let stage4_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage4_sumcheck_proof,
        &mut alloc,
        "stage4_sumcheck",
    );

    // === Symbolize stage 5 sumcheck proof ===
    let stage5_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage5_sumcheck_proof,
        &mut alloc,
        "stage5_sumcheck",
    );

    // === Symbolize stage 6 sumcheck proof ===
    let stage6_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage6_sumcheck_proof,
        &mut alloc,
        "stage6_sumcheck",
    );

    // === Symbolize stage 7 sumcheck proof ===
    let stage7_sumcheck = symbolize_sumcheck_proof::<_, ProofTranscript>(
        &real_proof.stage7_sumcheck_proof,
        &mut alloc,
        "stage7_sumcheck",
    );

    // === Symbolize advice commitment (if present) ===
    let untrusted_advice_commitment = real_proof.untrusted_advice_commitment.as_ref().map(|_| {
        let chunks = alloc.alloc_n(CHUNKS_PER_COMMITMENT, "untrusted_advice_commitment");
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
    let mut accumulator = MleOpeningAccumulator::new();
    for (key, (_, claim)) in &symbolic_proof.opening_claims.0 {
        accumulator
            .openings
            .insert(*key, (vec![], *claim));
    }

    (symbolic_proof, accumulator, alloc)
}

fn symbolize_uni_skip_proof<T: Transcript, OutT: Transcript>(
    real: &UniSkipFirstRoundProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> UniSkipFirstRoundProof<MleAst, OutT> {
    let coeffs = alloc.alloc_n(real.uni_poly.coeffs.len(), &format!("{prefix}_coeff"));
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
            let coeffs = alloc.alloc_n(
                poly.coeffs_except_linear_term.len(),
                &format!("{prefix}_r{round}"),
            );
            CompressedUniPoly {
                coeffs_except_linear_term: coeffs,
            }
        })
        .collect();

    SumcheckInstanceProof::new(compressed_polys)
}

/// Extract concrete witness values from a real proof.
///
/// This function traverses the proof in exactly the same order as `symbolize_proof`,
/// extracting the concrete field element values. The indices in the returned HashMap
/// correspond directly to the variable indices allocated during symbolization.
///
/// # Returns
///
/// A `HashMap<variable_index, decimal_string>` where:
/// - Key: Variable index (matches `VarAllocator` allocation order)
/// - Value: Field element as decimal string (for JSON serialization to Go)
///
/// # Important: Traversal Order
///
/// The traversal order MUST match `symbolize_proof` exactly:
/// 1. Commitments (12 chunks each)
/// 2. Opening claims
/// 3. Stage 1 uni-skip coefficients
/// 4. Stage 1 sumcheck coefficients (by round)
/// 5. ... repeat for stages 2-7 ...
/// 6. Untrusted advice commitment (if present)
///
/// Any mismatch will cause witness values to be assigned to wrong variables.
pub fn extract_witness_values(real_proof: &RV64IMACProof) -> std::collections::HashMap<usize, String> {
    use ark_ff::PrimeField;
    use ark_serialize::CanonicalSerialize;
    let mut values: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
    let mut idx: usize = 0;

    // Helper to convert bytes to field element chunks
    fn bytes_to_chunks(bytes: &[u8]) -> Vec<ark_bn254::Fr> {
        let num_chunks = 12; // Always 12 chunks per commitment
        (0..num_chunks)
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

    // Helper to serialize a commitment to bytes
    // MUST match the Poseidon transcript serialization:
    // 1. Use serialize_uncompressed (not compressed)
    // 2. Reverse bytes for BE/EVM format
    fn commitment_to_bytes<T: CanonicalSerialize>(commitment: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        commitment
            .serialize_uncompressed(&mut bytes)
            .expect("serialization failed");
        // Reverse bytes to match Poseidon transcript format (BE for EVM compatibility)
        bytes.reverse();
        bytes
    }

    // === Commitments (N commitments × 12 chunks each) ===
    for commitment in &real_proof.commitments {
        let chunks = bytes_to_chunks(&commitment_to_bytes(commitment));
        for chunk in chunks {
            values.insert(idx, format!("{}", chunk.into_bigint()));
            idx += 1;
        }
    }

    // === Opening claims ===
    for (_point, claim) in real_proof.opening_claims.0.values() {
        values.insert(idx, format!("{}", claim.into_bigint()));
        idx += 1;
    }

    // === Stage 1 uni-skip proof ===
    for coeff in &real_proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs {
        values.insert(idx, format!("{}", coeff.into_bigint()));
        idx += 1;
    }

    // === Stage 1 sumcheck proof ===
    for poly in &real_proof.stage1_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 2 uni-skip proof ===
    for coeff in &real_proof.stage2_uni_skip_first_round_proof.uni_poly.coeffs {
        values.insert(idx, format!("{}", coeff.into_bigint()));
        idx += 1;
    }

    // === Stage 2 sumcheck proof ===
    for poly in &real_proof.stage2_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 3 sumcheck proof ===
    for poly in &real_proof.stage3_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 4 sumcheck proof ===
    for poly in &real_proof.stage4_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 5 sumcheck proof ===
    for poly in &real_proof.stage5_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 6 sumcheck proof ===
    for poly in &real_proof.stage6_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Stage 7 sumcheck proof ===
    for poly in &real_proof.stage7_sumcheck_proof.compressed_polys {
        for coeff in &poly.coeffs_except_linear_term {
            values.insert(idx, format!("{}", coeff.into_bigint()));
            idx += 1;
        }
    }

    // === Untrusted advice commitment (if present) ===
    if let Some(ref commitment) = real_proof.untrusted_advice_commitment {
        let chunks = bytes_to_chunks(&commitment_to_bytes(commitment));
        for chunk in chunks {
            values.insert(idx, format!("{}", chunk.into_bigint()));
            idx += 1;
        }
    }

    values
}
