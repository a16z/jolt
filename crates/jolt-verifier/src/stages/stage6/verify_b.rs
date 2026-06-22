//! Stage 6b verifier helpers.

use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::{
        advice,
        bytecode::{
            self as bytecode_reduction, BytecodeLaneWeightInputs, BytecodeOutputWeightInputs,
        },
        program_image,
    },
    AdviceClaimReductionLayout, AdviceClaimReductionPublic, BytecodeClaimReductionLayout,
    BytecodeClaimReductionPublic, JoltAdviceKind, JoltPublicId, JoltRelationClaims, JoltRelationId,
    PrecommittedClaimReduction, PrecommittedReductionLayout, ProgramImageClaimReductionLayout,
    ProgramImageClaimReductionPublic,
};
use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::{
    stages::{
        stage4::Stage4ClearOutput,
        stage6::{
            inputs::{
                AdviceCyclePhaseOutputClaim, BytecodeCyclePhaseOutputClaims,
                ProgramImageCyclePhaseOutputClaim, Stage6Claims,
            },
            outputs::{
                AdviceCyclePhasePublicOutput, BytecodeReductionWeights,
                CommittedReductionCyclePhasePublicOutput, VerifiedAdviceCyclePhaseSumcheck,
                VerifiedBytecodeCyclePhaseSumcheck, VerifiedProgramImageCyclePhaseSumcheck,
            },
        },
    },
    VerifierError,
};

pub(super) fn aliased_booleanity_bytecode_openings<F: Field>(
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> usize {
    bytecode_ra_opening_points
        .iter()
        .filter(|point| point.as_slice() == booleanity_opening_point)
        .count()
}

pub(super) fn advice_cycle_phase_input<F: Field>(
    claim: &JoltRelationClaims<F>,
    stage4: &Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let advice_input = advice::ram_val_check_advice_opening(kind);
    claim.input.expression().try_evaluate(
        |id| match *id {
            id if id == advice_input => stage4
                .ram_val_check_init
                .advice_contributions
                .iter()
                .find(|contribution| contribution.kind == kind)
                .map(|contribution| contribution.opening_claim)
                .ok_or(VerifierError::MissingOpeningClaim { id }),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )
}

pub(super) fn verify_advice_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    opening_claim: &AdviceCyclePhaseOutputClaim<F>,
    stage4: &Stage4ClearOutput<F>,
) -> Result<VerifiedAdviceCyclePhaseSumcheck<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let opening_point = layout
        .cycle_phase_opening_point(advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let cycle_phase_variables = layout
        .cycle_phase_variable_challenges(advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let contribution = stage4
        .ram_val_check_init
        .advice_contributions
        .iter()
        .find(|contribution| contribution.kind == kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    let output_openings = advice::cycle_phase_output_openings(kind, layout.dimensions());
    let expected_output_claim = claim.output.expression().try_evaluate(
        |id| {
            if output_openings.contains(id) {
                Ok(opening_claim.opening_claim)
            } else {
                Err(VerifierError::MissingOpeningClaim { id: *id })
            }
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| match id {
            JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                public_kind,
            )) if *public_kind == kind => layout
                .cycle_phase_final_output_scale(&contribution.opening_point, advice_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::AdviceClaimReductionCyclePhase,
                    reason: error.to_string(),
                }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    Ok(VerifiedAdviceCyclePhaseSumcheck {
        kind,
        input_claim: contribution.opening_claim,
        sumcheck_point: advice_point.to_vec(),
        opening_point,
        cycle_phase_variables,
        expected_output_claim,
    })
}

pub(super) fn advice_cycle_phase_public<F: Field, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
) -> Result<AdviceCyclePhasePublicOutput<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let opening_point = layout
        .cycle_phase_opening_point(&advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let cycle_phase_variables = layout
        .cycle_phase_variable_challenges(&advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;

    Ok(AdviceCyclePhasePublicOutput {
        kind,
        sumcheck_point: advice_point,
        opening_point,
        cycle_phase_variables,
    })
}

pub(crate) struct BytecodeReductionWeightInputs<'a, F: Field> {
    pub eta: F,
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    /// Full bytecode address point (the `BytecodeReadRafAddrClaim` opening).
    pub bytecode_r_address: &'a [F],
}

pub(crate) fn bytecode_reduction_weights<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    inputs: BytecodeReductionWeightInputs<'_, F>,
) -> Result<BytecodeReductionWeights<F>, VerifierError> {
    let address_point = layout
        .split_address_point(inputs.bytecode_r_address)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let lane_weights = bytecode_reduction::lane_weights(BytecodeLaneWeightInputs {
        eta: inputs.eta,
        stage1_gammas: inputs.stage1_gammas,
        stage2_gammas: inputs.stage2_gammas,
        stage3_gammas: inputs.stage3_gammas,
        stage4_gammas: inputs.stage4_gammas,
        stage5_gammas: inputs.stage5_gammas,
        register_read_write_point: inputs.register_read_write_point,
        register_val_evaluation_point: inputs.register_val_evaluation_point,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
        reason: error.to_string(),
    })?;
    Ok(BytecodeReductionWeights {
        r_bc: address_point.r_bc,
        chunk_rbc_weights: address_point.chunk_rbc_weights,
        lane_weights,
    })
}

pub(super) fn verify_bytecode_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &BytecodeClaimReductionLayout,
    output_claims: &BytecodeCyclePhaseOutputClaims<F>,
    weights: BytecodeReductionWeights<F>,
    input_claim: F,
) -> Result<VerifiedBytecodeCyclePhaseSumcheck<F>, VerifierError> {
    let stage = JoltRelationId::BytecodeClaimReductionCyclePhase;
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    let opening_point = layout.cycle_phase_opening_point(point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        }
    })?;
    let cycle_phase_variables = layout
        .cycle_phase_variable_challenges(point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;
    let has_address_phase = layout.dimensions().has_address_phase();
    if let BytecodeCyclePhaseOutputClaims::Chunks(chunks) = output_claims {
        if has_address_phase || chunks.len() != layout.chunk_count() {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: format!(
                    "bytecode chunk claim count mismatch: expected {}, got {} (address phase: {})",
                    layout.chunk_count(),
                    chunks.len(),
                    has_address_phase
                ),
            });
        }
    }
    let chunk_weights = (!has_address_phase)
        .then(|| {
            layout.cycle_phase_final_output_weights(
                BytecodeOutputWeightInputs {
                    r_bc: &weights.r_bc,
                    chunk_rbc_weights: &weights.chunk_rbc_weights,
                    lane_weights: &weights.lane_weights,
                },
                point,
            )
        })
        .transpose()
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;
    let expected_output_claim = claim.output.expression().try_evaluate(
        |id| {
            if *id == bytecode_reduction::cycle_phase_intermediate_opening() {
                return match output_claims {
                    BytecodeCyclePhaseOutputClaims::Intermediate(value) => Ok(*value),
                    BytecodeCyclePhaseOutputClaims::Chunks(_) => {
                        Err(VerifierError::MissingOpeningClaim { id: *id })
                    }
                };
            }
            for chunk_idx in 0..layout.chunk_count() {
                if *id == bytecode_reduction::final_bytecode_chunk_opening(chunk_idx) {
                    return match output_claims {
                        BytecodeCyclePhaseOutputClaims::Chunks(chunks) => chunks
                            .get(chunk_idx)
                            .copied()
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id }),
                        BytecodeCyclePhaseOutputClaims::Intermediate(_) => {
                            Err(VerifierError::MissingOpeningClaim { id: *id })
                        }
                    };
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| match id {
            JoltPublicId::BytecodeClaimReduction(
                BytecodeClaimReductionPublic::ChunkOutputWeight(chunk_idx),
            ) => chunk_weights
                .as_ref()
                .and_then(|chunk_weights| chunk_weights.get(*chunk_idx).copied())
                .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    Ok(VerifiedBytecodeCyclePhaseSumcheck {
        input_claim,
        sumcheck_point: point.to_vec(),
        opening_point,
        cycle_phase_variables,
        weights,
        expected_output_claim,
    })
}

pub(super) fn committed_reduction_cycle_phase_public<F: Field, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    precommitted: &PrecommittedClaimReduction,
    stage: JoltRelationId,
) -> Result<CommittedReductionCyclePhasePublicOutput<F>, VerifierError> {
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    let opening_point = precommitted
        .cycle_phase_opening_point(&point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;
    let cycle_phase_variables = precommitted
        .cycle_phase_variable_challenges(&point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;

    Ok(CommittedReductionCyclePhasePublicOutput {
        sumcheck_point: point,
        opening_point,
        cycle_phase_variables,
    })
}

pub(super) fn verify_program_image_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &ProgramImageClaimReductionLayout,
    output_claim: &ProgramImageCyclePhaseOutputClaim<F>,
    r_addr_rw: &[F],
    input_claim: F,
) -> Result<VerifiedProgramImageCyclePhaseSumcheck<F>, VerifierError> {
    let stage = JoltRelationId::ProgramImageClaimReductionCyclePhase;
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    let opening_point = layout.cycle_phase_opening_point(point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        }
    })?;
    let cycle_phase_variables = layout
        .cycle_phase_variable_challenges(point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;
    let has_address_phase = layout.dimensions().has_address_phase();
    let final_scale = (!has_address_phase)
        .then(|| layout.cycle_phase_final_output_scale(r_addr_rw, point))
        .transpose()
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;
    let expected_output_claim = claim.output.expression().try_evaluate(
        |id| {
            if *id == program_image::cycle_phase_program_image_opening()
                || *id == program_image::final_program_image_opening()
            {
                Ok(output_claim.opening_claim)
            } else {
                Err(VerifierError::MissingOpeningClaim { id: *id })
            }
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| match id {
            JoltPublicId::ProgramImageClaimReduction(
                ProgramImageClaimReductionPublic::FinalScale,
            ) => final_scale.ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    Ok(VerifiedProgramImageCyclePhaseSumcheck {
        input_claim,
        sumcheck_point: point.to_vec(),
        opening_point,
        cycle_phase_variables,
        expected_output_claim,
    })
}

pub(super) fn append_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6Claims<F>,
    bytecode_read_raf_points: &[Vec<F>],
    booleanity_point: &[F],
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.bytecode_read_raf.bytecode_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.ram_hamming_booleanity.ram_hamming_weight,
    );
    for opening_claim in &claims.ram_ra_virtualization.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims
        .instruction_ra_virtualization
        .committed_instruction_ra
    {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(b"opening_claim", &claims.inc_claim_reduction.ram_inc);
    transcript.append_labeled(b"opening_claim", &claims.inc_claim_reduction.rd_inc);
    if let Some(output_claims) = &claims.unsigned_inc_claim_reduction {
        transcript.append_labeled(b"opening_claim", &output_claims.unsigned_inc);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(output_claims) = &claims.bytecode_claim_reduction {
        match output_claims {
            BytecodeCyclePhaseOutputClaims::Intermediate(opening_claim) => {
                transcript.append_labeled(b"opening_claim", opening_claim);
            }
            BytecodeCyclePhaseOutputClaims::Chunks(chunks) => {
                for opening_claim in chunks {
                    transcript.append_labeled(b"opening_claim", opening_claim);
                }
            }
        }
    }
    if let Some(output_claim) = &claims.program_image_claim_reduction {
        transcript.append_labeled(b"opening_claim", &output_claim.opening_claim);
    }
}
