//! Convert a real JoltProof to a symbolic JoltProof for transpilation
//!
//! This module provides functions to create symbolic versions of proof data structures,
//! where concrete field elements are replaced with MleAst variables.

use crate::ast_commitment_scheme::{AstCommitmentScheme, AstProof};
use crate::MleOpeningAccumulator;
use crate::PoseidonAstTranscript;
use jolt_core::poly::opening_proof::OpeningPoint;
use jolt_core::poly::unipoly::CompressedUniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProof;
use jolt_core::zkvm::proof_serialization::{Claims, JoltProof};
use jolt_core::zkvm::RV64IMACProof;
use std::collections::BTreeMap;
use zklean_extractor::mle_ast::MleAst;
use zklean_extractor::AstCommitment;

/// Tracks variable index allocation during symbolization
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
            .map(|i| self.alloc(&format!("{}_{}", prefix, i)))
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

/// Convert a real proof to a symbolic proof
///
/// Returns:
/// - The symbolic JoltProof
/// - The MleOpeningAccumulator with symbolic claims
/// - The VarAllocator with all variable descriptions
pub fn symbolize_proof(
    real_proof: &RV64IMACProof,
) -> (
    JoltProof<MleAst, AstCommitmentScheme, PoseidonAstTranscript>,
    MleOpeningAccumulator,
    VarAllocator,
) {
    let mut alloc = VarAllocator::new();

    // === Symbolize commitments ===
    let commitments: Vec<AstCommitment> = (0..real_proof.commitments.len())
        .map(|c| {
            let chunks = alloc.alloc_n(CHUNKS_PER_COMMITMENT, &format!("commitment_{}", c));
            AstCommitment::new(chunks)
        })
        .collect();

    // === Symbolize opening claims ===
    let mut symbolic_claims = BTreeMap::new();
    for (key, (_point, _claim)) in &real_proof.opening_claims.0 {
        let symbolic_claim = alloc.alloc(&format!("claim_{:?}", key));
        symbolic_claims.insert(key.clone(), (OpeningPoint::default(), symbolic_claim));
    }

    // === Symbolize stage 1 uni-skip proof ===
    let stage1_uni_skip = symbolize_uni_skip_proof(
        &real_proof.stage1_uni_skip_first_round_proof,
        &mut alloc,
        "stage1_uni_skip",
    );

    // === Symbolize stage 1 sumcheck proof ===
    let stage1_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage1_sumcheck_proof,
        &mut alloc,
        "stage1_sumcheck",
    );

    // === Symbolize stage 2 uni-skip proof ===
    let stage2_uni_skip = symbolize_uni_skip_proof(
        &real_proof.stage2_uni_skip_first_round_proof,
        &mut alloc,
        "stage2_uni_skip",
    );

    // === Symbolize stage 2 sumcheck proof ===
    let stage2_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage2_sumcheck_proof,
        &mut alloc,
        "stage2_sumcheck",
    );

    // === Symbolize stage 3 sumcheck proof ===
    let stage3_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage3_sumcheck_proof,
        &mut alloc,
        "stage3_sumcheck",
    );

    // === Symbolize stage 4 sumcheck proof ===
    let stage4_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage4_sumcheck_proof,
        &mut alloc,
        "stage4_sumcheck",
    );

    // === Symbolize stage 5 sumcheck proof ===
    let stage5_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage5_sumcheck_proof,
        &mut alloc,
        "stage5_sumcheck",
    );

    // === Symbolize stage 6 sumcheck proof ===
    let stage6_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage6_sumcheck_proof,
        &mut alloc,
        "stage6_sumcheck",
    );

    // === Symbolize stage 7 sumcheck proof ===
    let stage7_sumcheck = symbolize_sumcheck_proof(
        &real_proof.stage7_sumcheck_proof,
        &mut alloc,
        "stage7_sumcheck",
    );

    // === Symbolize stage 7 claims ===
    let stage7_sumcheck_claims: Vec<MleAst> = (0..real_proof.stage7_sumcheck_claims.len())
        .map(|i| alloc.alloc(&format!("stage7_claim_{}", i)))
        .collect();

    // === Symbolize advice commitments ===
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
        stage7_sumcheck_claims,
        joint_opening_proof: AstProof::default(),
        #[cfg(test)]
        joint_commitment_for_test: None,
        trusted_advice_val_evaluation_proof: None,
        trusted_advice_val_final_proof: None,
        untrusted_advice_val_evaluation_proof: None,
        untrusted_advice_val_final_proof: None,
        untrusted_advice_commitment,
        trace_length: real_proof.trace_length,
        ram_K: real_proof.ram_K,
        bytecode_K: real_proof.bytecode_K,
        log_k_chunk: real_proof.log_k_chunk,
        lookups_ra_virtual_log_k_chunk: real_proof.lookups_ra_virtual_log_k_chunk,
    };

    // Build the opening accumulator with the symbolic claims we created
    let mut accumulator = MleOpeningAccumulator::new();
    for (key, (_, claim)) in &symbolic_proof.opening_claims.0 {
        accumulator.openings.insert(key.clone(), (vec![], claim.clone()));
    }

    (symbolic_proof, accumulator, alloc)
}

fn symbolize_uni_skip_proof<T: jolt_core::transcripts::Transcript>(
    real: &UniSkipFirstRoundProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> UniSkipFirstRoundProof<MleAst, PoseidonAstTranscript> {
    let coeffs = alloc.alloc_n(real.uni_poly.coeffs.len(), &format!("{}_coeff", prefix));
    UniSkipFirstRoundProof::new(jolt_core::poly::unipoly::UniPoly::from_coeff(coeffs))
}

fn symbolize_sumcheck_proof<T: jolt_core::transcripts::Transcript>(
    real: &SumcheckInstanceProof<ark_bn254::Fr, T>,
    alloc: &mut VarAllocator,
    prefix: &str,
) -> SumcheckInstanceProof<MleAst, PoseidonAstTranscript> {
    let compressed_polys: Vec<CompressedUniPoly<MleAst>> = real
        .compressed_polys
        .iter()
        .enumerate()
        .map(|(round, poly)| {
            let coeffs = alloc.alloc_n(
                poly.coeffs_except_linear_term.len(),
                &format!("{}_r{}", prefix, round),
            );
            CompressedUniPoly {
                coeffs_except_linear_term: coeffs,
            }
        })
        .collect();

    SumcheckInstanceProof::new(compressed_polys)
}
