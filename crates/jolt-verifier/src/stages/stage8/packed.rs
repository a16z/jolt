//! The packed (lattice) final opening: every committed column lives in one
//! packed one-hot witness `W`, and the whole batch discharges through
//! jolt-openings' `PackedBatch` reduction sumcheck plus a single PCS opening
//! — no homomorphism anywhere.
//!
//! The verifier's job is statement assembly only: walk the canonical
//! `proof_packing` columns and hand each its one leaf claim (named by
//! `lattice::final_opening`) **at its own point** — the reduction handles
//! claims at mutually independent points, so no point unification happens
//! here.

use jolt_claims::protocols::jolt::lattice::{
    final_opening, proof_packing, LatticeColumn, LatticeFinalOpening, ProofPackingShape,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::JoltFormulaDimensions, JoltCommittedPolynomial, LatticeJolt,
};
use jolt_field::Field;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, PackedBatch, PackedBatchProof,
    PrefixPackedStatement, PrefixPackedVerifierSetup, PrefixPacking,
};
use jolt_poly::Point;
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::stages::relations::OpeningClaim;
use crate::stages::stage6::outputs::Stage6ClearOutput;
use crate::stages::stage7::outputs::Stage7ClearOutput;
use crate::VerifierError;
use jolt_crypto::VectorCommitment;

/// A packed-mode proof: one commitment, the reduction-sumcheck batch proof,
/// lattice-shaped claims. Transparent only (zk x packed is rejected
/// fail-closed).
pub type PackedJoltProof<PCS, VC, ZkProof> = JoltProof<
    PCS,
    VC,
    ZkProof,
    PackedBatchProof<<PCS as CommitmentScheme>::Field, <PCS as CommitmentScheme>::Proof>,
    <PCS as jolt_crypto::Commitment>::Output,
    LatticeJolt,
>;

pub fn verify_packed<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &PackedJoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage6: &Stage6ClearOutput<F, LatticeJolt>,
    stage7: &Stage7ClearOutput<F, LatticeJolt>,
) -> Result<(), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let packing = packed_proof_packing(formula_dimensions, proof)?;
    let statement = packed_statement(&packing, proof.commitments.clone(), stage6, stage7)?;
    let setup = PrefixPackedVerifierSetup::<PCS, LatticeColumn> {
        pcs: preprocessing.pcs_setup.clone(),
        packing,
    };
    PackedBatch::<PCS, LatticeColumn>::verify_batch(
        &setup,
        statement,
        &proof.joint_opening_proof,
        transcript,
    )
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })
}

/// The canonical per-proof packing, derived from the (Fiat-Shamir-absorbed)
/// config and shape on both sides — nothing about it travels in the proof.
pub fn packed_proof_packing<PCS, VC, ZkProof>(
    formula_dimensions: &JoltFormulaDimensions,
    proof: &PackedJoltProof<PCS, VC, ZkProof>,
) -> Result<PrefixPacking<LatticeColumn>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    proof_packing(&ProofPackingShape {
        ra_layout: formula_dimensions.ra_layout,
        log_t: formula_dimensions.trace.log_t(),
        log_k_chunk: proof.one_hot_config.committed_chunk_bits(),
        // Phase-B scope: no advice under the packed axis
        // (`validate_packed_inputs`).
        untrusted_advice_word_vars: None,
    })
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })
}

/// Resolves each packed column's single leaf claim from the lattice stage
/// outputs, in the packing's canonical column order.
fn packed_statement<F, C>(
    packing: &PrefixPacking<LatticeColumn>,
    commitment: C,
    stage6: &Stage6ClearOutput<F, LatticeJolt>,
    stage7: &Stage7ClearOutput<F, LatticeJolt>,
) -> Result<PrefixPackedStatement<F, LatticeColumn, C>, VerifierError>
where
    F: Field,
{
    let mut claims = Vec::new();
    for (column, _slot) in packing {
        let LatticeColumn::Committed(polynomial) = column else {
            return Err(batch_failed(format!(
                "non-committed column {column:?} in the per-proof packing"
            )));
        };
        let LatticeFinalOpening::Packed { leaf: Some(_), .. } = final_opening(*polynomial) else {
            return Err(batch_failed(format!(
                "packed column {polynomial:?} has no relation leaf"
            )));
        };
        claims.push((*column, leaf_claim(*polynomial, stage6, stage7)?));
    }
    Ok(PrefixPackedStatement::new(commitment, claims))
}

/// The stage-6/7 output claim backing a packed column, at its own point (the
/// hamming point for `Ra` columns, the reconstruction point for the chunk
/// columns, the booleanity cycle point for the msb).
fn leaf_claim<F: Field>(
    polynomial: JoltCommittedPolynomial,
    stage6: &Stage6ClearOutput<F, LatticeJolt>,
    stage7: &Stage7ClearOutput<F, LatticeJolt>,
) -> Result<EvaluationClaim<F>, VerifierError> {
    let hamming = &stage7.output_claims.hamming_weight_claim_reduction;
    let claim = |cells: &[OpeningClaim<F>], index: usize| {
        cells
            .get(index)
            .map(|cell| EvaluationClaim::new(Point::high_to_low(cell.point.clone()), cell.value))
            .ok_or_else(|| batch_failed(format!("missing stage output claim for {polynomial:?}")))
    };
    match polynomial {
        JoltCommittedPolynomial::InstructionRa(index) => claim(&hamming.instruction_ra, index),
        JoltCommittedPolynomial::BytecodeRa(index) => claim(&hamming.bytecode_ra, index),
        JoltCommittedPolynomial::RamRa(index) => claim(&hamming.ram_ra, index),
        JoltCommittedPolynomial::UnsignedIncChunk(index) => {
            claim(&stage7.output_claims.chunk_reconstruction.chunks, index)
        }
        JoltCommittedPolynomial::UnsignedIncMsb => Ok(EvaluationClaim::new(
            Point::high_to_low(stage6.output_points.booleanity.unsigned_inc_msb.clone()),
            stage6.output_claims.booleanity.unsigned_inc_msb,
        )),
        other => Err(batch_failed(format!(
            "polynomial {other:?} does not open through the per-proof packed witness"
        ))),
    }
}

fn batch_failed(reason: String) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed { reason }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::lattice::relations::booleanity::LatticeBooleanityOutputClaims;
    use jolt_claims::protocols::jolt::lattice::relations::chunk_reconstruction::ChunkReconstructionOutputClaims;
    use jolt_claims::protocols::jolt::lattice::relations::inc_virtualization::IncVirtualizationOutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};

    use crate::stages::stage6::outputs as s6;
    use crate::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims;
    use crate::stages::stage7::outputs::Stage7OutputClaims;

    const LOG_T: usize = 4;
    const LOG_K_CHUNK: usize = 8;
    const CHUNKS: usize = 8;

    fn opening(arity: usize, value: u64) -> OpeningClaim<Fr> {
        OpeningClaim {
            point: vec![Fr::from_u64(1); arity],
            value: Fr::from_u64(value),
        }
    }

    fn stage6() -> Stage6ClearOutput<Fr, LatticeJolt> {
        let zero = Fr::from_u64(0);
        let claims = |msb: Fr| s6::Stage6OutputClaims::<Fr, LatticeJolt> {
            address_phase: s6::Stage6AddressPhaseClaims {
                bytecode_read_raf: zero,
                booleanity: zero,
                bytecode_val_stages: None,
            },
            bytecode_read_raf: s6::BytecodeReadRafOutputClaims {
                bytecode_ra: Vec::new(),
            },
            booleanity: LatticeBooleanityOutputClaims {
                instruction_ra: Vec::new(),
                bytecode_ra: Vec::new(),
                ram_ra: Vec::new(),
                unsigned_inc_chunks: Vec::new(),
                unsigned_inc_msb: msb,
            },
            ram_hamming_booleanity: s6::RamHammingBooleanityOutputClaims {
                ram_hamming_weight: zero,
            },
            ram_ra_virtualization: s6::RamRaVirtualizationOutputClaims { ram_ra: Vec::new() },
            instruction_ra_virtualization: s6::InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: Vec::new(),
            },
            inc_claim_reduction: IncVirtualizationOutputClaims {
                fused_inc: zero,
                store: zero,
            },
            advice_cycle_phase: s6::Stage6AdviceCyclePhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        };
        let mut output_points = s6::Stage6OutputClaims::<Vec<Fr>, LatticeJolt> {
            address_phase: s6::Stage6AddressPhaseClaims {
                bytecode_read_raf: Vec::new(),
                booleanity: Vec::new(),
                bytecode_val_stages: None,
            },
            bytecode_read_raf: s6::BytecodeReadRafOutputClaims {
                bytecode_ra: Vec::new(),
            },
            booleanity: LatticeBooleanityOutputClaims {
                instruction_ra: Vec::new(),
                bytecode_ra: Vec::new(),
                ram_ra: Vec::new(),
                unsigned_inc_chunks: Vec::new(),
                unsigned_inc_msb: Vec::new(),
            },
            ram_hamming_booleanity: s6::RamHammingBooleanityOutputClaims {
                ram_hamming_weight: Vec::new(),
            },
            ram_ra_virtualization: s6::RamRaVirtualizationOutputClaims { ram_ra: Vec::new() },
            instruction_ra_virtualization: s6::InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: Vec::new(),
            },
            inc_claim_reduction: IncVirtualizationOutputClaims {
                fused_inc: Vec::new(),
                store: Vec::new(),
            },
            advice_cycle_phase: s6::Stage6AdviceCyclePhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        };
        output_points.booleanity.unsigned_inc_msb = vec![Fr::from_u64(1); LOG_T];
        Stage6ClearOutput {
            output_claims: claims(Fr::from_u64(7)),
            output_points,
            bytecode_reduction_weights: None,
        }
    }

    fn stage7(layout: JoltRaPolynomialLayout) -> Stage7ClearOutput<Fr, LatticeJolt> {
        let one_hot_arity = LOG_K_CHUNK + LOG_T;
        Stage7ClearOutput {
            output_claims: Stage7OutputClaims {
                hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims {
                    instruction_ra: (0..layout.instruction())
                        .map(|i| opening(one_hot_arity, 100 + i as u64))
                        .collect(),
                    bytecode_ra: (0..layout.bytecode())
                        .map(|i| opening(one_hot_arity, 200 + i as u64))
                        .collect(),
                    ram_ra: (0..layout.ram())
                        .map(|i| opening(one_hot_arity, 300 + i as u64))
                        .collect(),
                },
                chunk_reconstruction: ChunkReconstructionOutputClaims {
                    chunks: (0..CHUNKS)
                        .map(|i| opening(one_hot_arity, 400 + i as u64))
                        .collect(),
                },
                advice_address_phase:
                    crate::stages::stage7::advice_address_phase::AdviceAddressPhaseOutputClaims {
                        trusted: None,
                        untrusted: None,
                    },
                bytecode_address_phase: None,
                program_image_address_phase: None,
            },
            hamming_weight_opening_point: Vec::new(),
            precommitted_final_openings: Vec::new(),
        }
    }

    #[test]
    fn packed_statement_covers_every_column_exactly_once_at_slot_arity() {
        let layout = JoltRaPolynomialLayout::new(2, 1, 1).unwrap();
        let packing = proof_packing(&ProofPackingShape {
            ra_layout: layout,
            log_t: LOG_T,
            log_k_chunk: LOG_K_CHUNK,
            untrusted_advice_word_vars: None,
        })
        .unwrap();

        let statement = packed_statement(&packing, (), &stage6(), &stage7(layout)).unwrap();

        let column_count = packing.iter().count();
        assert_eq!(statement.claims.len(), column_count);
        for (column, slot) in &packing {
            let matches: Vec<_> = statement
                .claims
                .iter()
                .filter(|(id, _)| id == column)
                .collect();
            assert_eq!(matches.len(), 1, "column {column:?} must have one claim");
            let (_, claim) = matches[0];
            assert_eq!(
                claim.point.len(),
                slot.num_vars,
                "claim arity must match slot arity for {column:?}"
            );
        }
    }
}
