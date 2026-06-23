use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::{bytecode as bytecode_reduction, increments},
        dimensions::TraceDimensions,
        lattice,
    },
    JoltRelationClaims, JoltRelationId, JoltSumcheckDomain,
};
use jolt_field::Field;

use super::inputs::{
    BooleanityOutputOpeningClaims, IncClaimReductionOutputOpeningClaims,
    UnsignedIncClaimReductionOutputOpeningClaims,
};
use crate::{
    config::{validate_protocol_config, PcsFamily},
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage6BatchInputClaims<F: Field> {
    pub(super) bytecode_read_raf_address: F,
    pub(super) booleanity_address: F,
    pub(super) bytecode_read_raf: F,
    pub(super) booleanity: F,
    pub(super) ram_hamming_booleanity: F,
    pub(super) ram_ra_virtualization: F,
    pub(super) instruction_ra_virtualization: F,
    pub(super) inc_claim_reduction: Option<F>,
    pub(super) unsigned_inc_claim_reduction: Option<F>,
    pub(super) unsigned_inc_msb_booleanity: Option<F>,
    #[cfg(feature = "field-inline")]
    pub(super) field_registers_inc_claim_reduction: F,
    pub(super) trusted_advice_cycle_phase: Option<F>,
    pub(super) untrusted_advice_cycle_phase: Option<F>,
    pub(super) bytecode_claim_reduction: Option<F>,
    pub(super) program_image_claim_reduction: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage6BatchExpectedOutputClaims<F: Field> {
    pub(super) bytecode_read_raf_address: F,
    pub(super) booleanity_address: F,
    pub(super) bytecode_read_raf: F,
    pub(super) booleanity: F,
    pub(super) ram_hamming_booleanity: F,
    pub(super) ram_ra_virtualization: F,
    pub(super) instruction_ra_virtualization: F,
    pub(super) inc_claim_reduction: Option<F>,
    pub(super) unsigned_inc_claim_reduction: Option<F>,
    pub(super) unsigned_inc_msb_booleanity: Option<F>,
    #[cfg(feature = "field-inline")]
    pub(super) field_registers_inc_claim_reduction: F,
    pub(super) trusted_advice_cycle_phase: Option<F>,
    pub(super) untrusted_advice_cycle_phase: Option<F>,
    pub(super) bytecode_claim_reduction: Option<F>,
    pub(super) program_image_claim_reduction: Option<F>,
}

pub(super) fn validate_compressed_stage_claim<F: Field>(
    claim: &JoltRelationClaims<F>,
) -> Result<(), VerifierError> {
    if claim.sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: claim.id,
            degree: claim.sumcheck.degree,
        });
    }
    if !matches!(claim.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain { stage: claim.id });
    }
    Ok(())
}

pub(super) fn unsigned_inc_claims_for_protocol<F: Field>(
    protocol: &crate::config::JoltProtocolConfig,
    trace_dimensions: TraceDimensions,
    has_unsigned_inc_claims: bool,
) -> Result<Option<JoltRelationClaims<F>>, VerifierError> {
    let lattice = validate_protocol_config(protocol)? == PcsFamily::Lattice;
    if has_unsigned_inc_claims && !lattice {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: lattice::unsigned_inc_opening(),
        });
    }
    if lattice && !has_unsigned_inc_claims {
        return Err(VerifierError::MissingOpeningClaim {
            id: lattice::unsigned_inc_opening(),
        });
    }

    Ok(if lattice {
        Some(lattice::unsigned_inc_claim_reduction_claim(
            trace_dimensions,
        ))
    } else {
        None
    })
}

pub(super) fn validate_lattice_increment_claim_shape<F: Field>(
    claim: Option<&JoltRelationClaims<F>>,
    output_claims: Option<&UnsignedIncClaimReductionOutputOpeningClaims<F>>,
    booleanity_claims: &BooleanityOutputOpeningClaims<F>,
    log_k_chunk: usize,
) -> Result<(), VerifierError> {
    let Some(_claim) = claim else {
        if !booleanity_claims.unsigned_inc_chunks.is_empty() {
            return Err(VerifierError::UnexpectedOpeningClaim {
                id: lattice::unsigned_inc_chunk_opening(0),
            });
        }
        return Ok(());
    };
    let _output_claims = output_claims.ok_or(VerifierError::MissingOpeningClaim {
        id: lattice::unsigned_inc_opening(),
    })?;
    let expected_chunks =
        lattice::unsigned_inc_lower_chunk_count(log_k_chunk).ok_or_else(|| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::UnsignedIncClaimReduction,
                reason: format!(
                    "unsigned increment chunk size must evenly divide 64 bits, got {log_k_chunk}"
                ),
            }
        })?;
    if booleanity_claims.unsigned_inc_chunks.len() != expected_chunks {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: format!(
                "unsigned increment chunk booleanity claim count mismatch: expected {expected_chunks}, got {}",
                booleanity_claims.unsigned_inc_chunks.len()
            ),
        });
    }
    Ok(())
}

pub(super) fn validate_dense_increment_claim_shape<F: Field>(
    lattice: bool,
    output_claims: Option<&IncClaimReductionOutputOpeningClaims<F>>,
) -> Result<(), VerifierError> {
    match (lattice, output_claims.is_some()) {
        (true, true) => Err(VerifierError::UnexpectedOpeningClaim {
            id: increments::claim_reduction_output_openings()[0],
        }),
        (false, false) => Err(VerifierError::MissingOpeningClaim {
            id: increments::claim_reduction_output_openings()[0],
        }),
        _ => Ok(()),
    }
}

pub(super) fn validate_bytecode_val_stage_claim_count<F: Field>(
    committed_program: bool,
    bind_store_bytecode: bool,
    stage_claims: Option<&[F]>,
) -> Result<(), VerifierError> {
    if committed_program != stage_claims.is_some() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode Val-stage claims presence ({}) does not match committed program mode ({committed_program})",
                stage_claims.is_some()
            ),
        });
    }
    if let Some(stage_claims) = stage_claims {
        let expected = bytecode_reduction::bytecode_val_stage_count(bind_store_bytecode);
        if stage_claims.len() != expected {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: format!(
                    "bytecode Val-stage claim count mismatch: expected {expected}, got {}",
                    stage_claims.len()
                ),
            });
        }
    }
    Ok(())
}
