#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{bytecode as field_bytecode, claim_reductions::increments as field_increments},
    FieldInlineVirtualPolynomial, FieldRegistersTraceDimensions,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity::{self, BooleanityDimensions},
        bytecode::{self, BytecodeReadRafEvaluationInputs},
        claim_reductions::{advice, increments},
        dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram,
    },
    AdviceClaimReductionLayout, AdviceClaimReductionPublic, BooleanityChallenge, BooleanityPublic,
    BytecodeReadRafChallenge, IncClaimReductionChallenge, IncClaimReductionPublic,
    InstructionRaVirtualizationChallenge, JoltAdviceKind, JoltChallengeId, JoltPublicId,
    JoltRelationClaims, JoltRelationId, JoltSumcheckDomain, JoltVirtualPolynomial,
    RamHammingBooleanityChallenge, RamRaVirtualizationChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::try_eq_mle;
#[cfg(not(feature = "field-inline"))]
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::Transcript;
use num_traits::{One, Zero};

#[cfg(feature = "field-inline")]
use super::outputs::{FieldInlineStage6PublicOutput, FieldInlineStage6ZkOutput};
use super::{
    inputs::{AdviceCyclePhaseOutputClaim, Deps, Stage6Claims},
    outputs::{
        AdviceCyclePhasePublicOutput, BooleanityPublicOutput, BytecodeReadRafPublicOutput,
        InstructionRaVirtualizationPublicOutput, RamRaVirtualizationPublicOutput,
        Stage6ClearOutput, Stage6Output, Stage6PublicOutput, Stage6SumcheckPublicOutput,
        Stage6ZkOutput, VerifiedAdviceCyclePhaseSumcheck, VerifiedBooleanitySumcheck,
        VerifiedBytecodeReadRafSumcheck, VerifiedInstructionRaVirtualizationSumcheck,
        VerifiedRamRaVirtualizationSumcheck, VerifiedStage6Batch, VerifiedStage6Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{stage4::Stage4ClearOutput, zk::committed},
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage6BatchInputClaims<F: Field> {
    bytecode_read_raf: F,
    booleanity: F,
    ram_hamming_booleanity: F,
    ram_ra_virtualization: F,
    instruction_ra_virtualization: F,
    inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction: F,
    trusted_advice_cycle_phase: Option<F>,
    untrusted_advice_cycle_phase: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage6BatchExpectedOutputClaims<F: Field> {
    bytecode_read_raf: F,
    booleanity: F,
    ram_hamming_booleanity: F,
    ram_ra_virtualization: F,
    instruction_ra_virtualization: F,
    inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction: F,
    trusted_advice_cycle_phase: Option<F>,
    untrusted_advice_cycle_phase: Option<F>,
}

#[cfg(feature = "field-inline")]
const fn field_inline_stage6_output_claim_count() -> usize {
    1
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_stage6_output_claim_count() -> usize {
    0
}

#[cfg(feature = "field-inline")]
const fn stage1_gamma_count() -> usize {
    field_bytecode::FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT
}

#[cfg(not(feature = "field-inline"))]
const fn stage1_gamma_count() -> usize {
    2 + NUM_CIRCUIT_FLAGS
}

#[cfg(feature = "field-inline")]
const fn stage4_gamma_count() -> usize {
    field_bytecode::FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT
}

#[cfg(not(feature = "field-inline"))]
const fn stage4_gamma_count() -> usize {
    3
}

#[cfg(feature = "field-inline")]
const fn stage5_gamma_count() -> usize {
    2 + LookupTableKind::<RISCV_XLEN>::COUNT
        + field_bytecode::FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS
}

#[cfg(not(feature = "field-inline"))]
const fn stage5_gamma_count() -> usize {
    2 + LookupTableKind::<RISCV_XLEN>::COUNT
}

#[cfg(feature = "field-inline")]
fn field_inline_bytecode_rows<PCS, VC>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
) -> Result<&[field_bytecode::FieldInlineBytecodeRow], VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    preprocessing
        .field_inline_bytecode
        .as_deref()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "field-inline bytecode metadata is missing".to_string(),
        })
}

#[cfg(feature = "field-inline")]
#[expect(
    clippy::too_many_arguments,
    reason = "Field-inline bytecode evaluation depends on several stage points."
)]
fn add_field_inline_bytecode_public_values<F>(
    bytecode_public_values: &mut bytecode::BytecodeReadRafPublicValues<F>,
    field_inline_bytecode: &[field_bytecode::FieldInlineBytecodeRow],
    r_address: &[F],
    r_cycle: &[F],
    stage1_cycle: &[F],
    field_registers_read_write_opening_point: &[F],
    field_registers_val_evaluation_opening_point: &[F],
    field_log_k: usize,
    log_t: usize,
    stage1_gammas: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> Result<(), VerifierError>
where
    F: Field,
{
    if field_registers_read_write_opening_point.len() != field_log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "field-register read-write opening point length mismatch: expected {}, got {}",
                field_log_k + log_t,
                field_registers_read_write_opening_point.len()
            ),
        });
    }
    if field_registers_val_evaluation_opening_point.len() != field_log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "field-register val-evaluation opening point length mismatch: expected {}, got {}",
                field_log_k + log_t,
                field_registers_val_evaluation_opening_point.len()
            ),
        });
    }

    let (field_read_write_address, field_read_write_cycle) =
        field_registers_read_write_opening_point.split_at(field_log_k);
    let (field_val_evaluation_address, field_val_evaluation_cycle) =
        field_registers_val_evaluation_opening_point.split_at(field_log_k);
    let field_values = field_bytecode::read_raf_public_values(
        field_bytecode::FieldInlineBytecodeReadRafEvaluationInputs {
            bytecode: field_inline_bytecode,
            field_register_log_k: field_log_k,
            r_address,
            r_cycle,
            stage1_cycle_point: stage1_cycle,
            field_register_read_write_point: field_read_write_address,
            field_register_read_write_cycle_point: field_read_write_cycle,
            field_register_val_evaluation_point: field_val_evaluation_address,
            field_register_val_evaluation_cycle_point: field_val_evaluation_cycle,
            stage1_gammas,
            stage4_gammas,
            stage5_gammas,
        },
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    for (stage_value, field_value) in bytecode_public_values
        .stage_values
        .iter_mut()
        .zip(field_values.stage_values)
    {
        *stage_value += field_value;
    }
    Ok(())
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage6Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage5" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage5" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions =
        jolt_claims::protocols::jolt::formulas::dimensions::TraceDimensions::new(log_t);
    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode.code_size,
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_claims = bytecode::read_raf::<PCS::Field>(formula_dimensions.bytecode_read_raf);
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_claims = booleanity::booleanity::<PCS::Field>(booleanity_dimensions);
    let ram_hamming_claims = ram::hamming_booleanity::<PCS::Field>(trace_dimensions);
    let ram_ra_claims =
        ram::ra_virtualization::<PCS::Field>(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = instruction::ra_virtualization::<PCS::Field>(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = increments::claim_reduction::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_inc_claims =
        field_increments::claim_reduction::<PCS::Field>(FieldRegistersTraceDimensions::new(log_t));

    let trusted_advice_layout = checked.trusted_advice_commitment_present.then(|| {
        AdviceClaimReductionLayout::balanced(
            proof.trace_polynomial_order,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            checked.public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = proof.untrusted_advice_commitment.as_ref().map(|_| {
        AdviceClaimReductionLayout::balanced(
            proof.trace_polynomial_order,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            checked.public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });
    let trusted_advice_claims = trusted_advice_layout.as_ref().map(|layout| {
        advice::cycle_phase::<PCS::Field>(JoltAdviceKind::Trusted, layout.dimensions())
    });
    let untrusted_advice_claims = untrusted_advice_layout.as_ref().map(|layout| {
        advice::cycle_phase::<PCS::Field>(JoltAdviceKind::Untrusted, layout.dimensions())
    });

    for claim in [
        &bytecode_claims,
        &booleanity_claims,
        &ram_hamming_claims,
        &ram_ra_claims,
        &instruction_ra_claims,
        &inc_claims,
    ] {
        validate_compressed_stage_claim(claim)?;
    }
    if let Some(claim) = &trusted_advice_claims {
        validate_compressed_stage_claim(claim)?;
    }
    if let Some(claim) = &untrusted_advice_claims {
        validate_compressed_stage_claim(claim)?;
    }
    #[cfg(feature = "field-inline")]
    {
        if field_inc_claims.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: JoltRelationId::IncClaimReduction,
                degree: field_inc_claims.sumcheck.degree,
            });
        }
    }

    let bytecode_gamma_powers = transcript.challenge_scalar_powers(8);
    let bytecode_gamma = bytecode_gamma_powers[1];
    let stage1_gammas = transcript.challenge_scalar_powers(stage1_gamma_count());
    let stage2_gammas = transcript.challenge_scalar_powers(4);
    let stage3_gammas = transcript.challenge_scalar_powers(9);
    let stage4_gammas = transcript.challenge_scalar_powers(stage4_gamma_count());
    let stage5_gammas = transcript.challenge_scalar_powers(stage5_gamma_count());

    let (stage5_instruction_address, stage5_instruction_cycle) = match deps {
        Deps::Clear { stage5, .. } => (
            &stage5.batch.instruction_read_raf.r_address,
            &stage5.batch.instruction_read_raf.r_cycle,
        ),
        Deps::Zk { stage5 } => (
            &stage5.instruction_read_raf.r_address,
            &stage5.instruction_read_raf.r_cycle,
        ),
    };
    let mut booleanity_reference_address = stage5_instruction_address.clone();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < proof.one_hot_config.committed_chunk_bits() {
        let missing =
            proof.one_hot_config.committed_chunk_bits() - booleanity_reference_address.len();
        booleanity_reference_address.extend(transcript.challenge_vector(missing));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - proof.one_hot_config.committed_chunk_bits()..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5_instruction_cycle.clone();
    booleanity_reference_cycle.reverse();
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma.is_zero() {
        booleanity_gamma = PCS::Field::one();
    }

    let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
        formula_dimensions
            .instruction_ra_virtualization
            .num_virtual_ra_polys(),
    );
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(PCS::Field::one);
    let inc_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_inc_gamma = transcript.challenge_scalar();

    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage6PublicOutput {
            challenges,
            batching_coefficients,
            bytecode_gamma_powers: bytecode_gamma_powers.clone(),
            stage1_gammas: stage1_gammas.clone(),
            stage2_gammas: stage2_gammas.clone(),
            stage3_gammas: stage3_gammas.clone(),
            stage4_gammas: stage4_gammas.clone(),
            stage5_gammas: stage5_gammas.clone(),
            booleanity_reference_address: booleanity_reference_address.clone(),
            booleanity_reference_cycle: booleanity_reference_cycle.clone(),
            booleanity_gamma,
            instruction_ra_gamma_powers: instruction_ra_gamma_powers.clone(),
            inc_gamma,
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineStage6PublicOutput { field_inc_gamma },
        };

    if checked.zk {
        let Deps::Zk { stage5 } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage5" });
        };
        let mut statements = vec![
            SumcheckStatement::new(
                bytecode_claims.sumcheck.rounds,
                bytecode_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                booleanity_claims.sumcheck.rounds,
                booleanity_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                ram_hamming_claims.sumcheck.rounds,
                ram_hamming_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(ram_ra_claims.sumcheck.rounds, ram_ra_claims.sumcheck.degree),
            SumcheckStatement::new(
                instruction_ra_claims.sumcheck.rounds,
                instruction_ra_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(inc_claims.sumcheck.rounds, inc_claims.sumcheck.degree),
        ];
        #[cfg(feature = "field-inline")]
        statements.push(SumcheckStatement::new(
            field_inc_claims.sumcheck.rounds,
            field_inc_claims.sumcheck.degree,
        ));
        if let Some(claim) = &trusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &untrusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage6_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;

        let bytecode_output_openings =
            bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
        let booleanity_output_openings =
            booleanity::booleanity_output_openings(formula_dimensions.ra_layout);
        let ram_ra_output_openings =
            ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization);
        let instruction_ra_output_openings = instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        );
        let flat_instruction_ra_output_openings = instruction_ra_output_openings.all();
        let committed_output_claims = bytecode_output_openings.bytecode_ra.len()
            + booleanity_output_openings.len()
            + 1
            + ram_ra_output_openings.len()
            + flat_instruction_ra_output_openings.len()
            + 2
            + field_inline_stage6_output_claim_count()
            + usize::from(trusted_advice_claims.is_some())
            + usize::from(untrusted_advice_claims.is_some());
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage6_sumcheck_proof,
                proof_label: "stage6_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::BytecodeReadRaf,
            })?;

        let bytecode_point = consistency
            .try_instance_point(bytecode_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: error.to_string(),
            })?;
        let bytecode_opening_point = formula_dimensions
            .bytecode_read_raf
            .opening_point(&bytecode_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: error.to_string(),
            })?;
        let bytecode_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(&bytecode_opening_point.r_address)
            .into_iter()
            .map(|r_address_chunk| {
                [
                    r_address_chunk.as_slice(),
                    bytecode_opening_point.r_cycle.as_slice(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();

        let booleanity_point = consistency
            .try_instance_point(booleanity_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            })?;
        let booleanity_opening_point = booleanity_dimensions
            .opening_point(&booleanity_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            })?;

        let ram_hamming_point = consistency
            .try_instance_point(ram_hamming_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;
        let ram_hamming_opening_point = trace_dimensions
            .cycle_opening_point(&ram_hamming_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;

        let ram_ra_point = consistency
            .try_instance_point(ram_ra_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: error.to_string(),
            })?;
        let ram_ra_cycle = trace_dimensions
            .cycle_opening_point(&ram_ra_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: error.to_string(),
            })?;
        let ram_reduced_opening_point = &stage5.ram_ra_claim_reduction.opening_point;
        if ram_reduced_opening_point.len() != log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: format!(
                    "RAM RA reduction opening point length mismatch: expected {}, got {}",
                    log_k + log_t,
                    ram_reduced_opening_point.len()
                ),
            });
        }
        let (ram_reduced_address, _) = ram_reduced_opening_point.split_at(log_k);
        let ram_ra_opening_point = [ram_reduced_address, ram_ra_cycle.as_slice()].concat();
        let ram_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(ram_reduced_address)
            .into_iter()
            .map(|r_address_chunk| [r_address_chunk.as_slice(), ram_ra_cycle.as_slice()].concat())
            .collect::<Vec<_>>();

        let instruction_ra_point = consistency
            .try_instance_point(instruction_ra_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            })?;
        let instruction_ra_cycle = trace_dimensions
            .cycle_opening_point(&instruction_ra_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            })?;
        let instruction_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(&stage5.instruction_read_raf.r_address)
            .into_iter()
            .map(|r_address_chunk| {
                [r_address_chunk.as_slice(), instruction_ra_cycle.as_slice()].concat()
            })
            .collect::<Vec<_>>();
        let instruction_ra_opening_point = [
            stage5.instruction_read_raf.r_address.as_slice(),
            instruction_ra_cycle.as_slice(),
        ]
        .concat();

        let inc_point = consistency
            .try_instance_point(inc_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let inc_opening_point =
            trace_dimensions
                .cycle_opening_point(&inc_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;

        #[cfg(feature = "field-inline")]
        let field_registers_inc_claim_reduction = {
            let field_inc_point = consistency
                .try_instance_point(field_inc_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;
            let opening_point = trace_dimensions
                .cycle_opening_point(&field_inc_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;
            Stage6SumcheckPublicOutput {
                sumcheck_point: field_inc_point,
                opening_point,
            }
        };

        let trusted_advice = if let (Some(layout), Some(claim)) = (
            trusted_advice_layout.as_ref(),
            trusted_advice_claims.as_ref(),
        ) {
            Some(advice_cycle_phase_public(
                &consistency,
                claim,
                layout,
                JoltAdviceKind::Trusted,
            )?)
        } else {
            None
        };
        let untrusted_advice = if let (Some(layout), Some(claim)) = (
            untrusted_advice_layout.as_ref(),
            untrusted_advice_claims.as_ref(),
        ) {
            Some(advice_cycle_phase_public(
                &consistency,
                claim,
                layout,
                JoltAdviceKind::Untrusted,
            )?)
        } else {
            None
        };

        return Ok(Stage6Output::Zk(Stage6ZkOutput {
            public: public(
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
            bytecode_read_raf: BytecodeReadRafPublicOutput {
                sumcheck_point: bytecode_point,
                r_address: bytecode_opening_point.r_address,
                r_cycle: bytecode_opening_point.r_cycle,
                full_opening_point: bytecode_opening_point.opening_point,
                bytecode_ra_opening_points,
            },
            booleanity: BooleanityPublicOutput {
                sumcheck_point: booleanity_point,
                r_address: booleanity_opening_point.r_address,
                r_cycle: booleanity_opening_point.r_cycle,
                opening_point: booleanity_opening_point.opening_point,
                reference_address: booleanity_reference_address,
                reference_cycle: booleanity_reference_cycle,
            },
            ram_hamming_booleanity: Stage6SumcheckPublicOutput {
                sumcheck_point: ram_hamming_point,
                opening_point: ram_hamming_opening_point,
            },
            ram_ra_virtualization: RamRaVirtualizationPublicOutput {
                sumcheck_point: ram_ra_point,
                opening_point: ram_ra_opening_point,
                ram_ra_opening_points,
            },
            instruction_ra_virtualization: InstructionRaVirtualizationPublicOutput {
                sumcheck_point: instruction_ra_point,
                opening_point: instruction_ra_opening_point,
                instruction_ra_opening_points,
            },
            inc_claim_reduction: Stage6SumcheckPublicOutput {
                sumcheck_point: inc_point,
                opening_point: inc_opening_point,
            },
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineStage6ZkOutput {
                field_registers_inc_claim_reduction,
            },
            trusted_advice_cycle_phase: trusted_advice,
            untrusted_advice_cycle_phase: untrusted_advice,
        }));
    }

    let Deps::Clear {
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
    } = deps
    else {
        return Err(VerifierError::ExpectedClearProof { field: "stage5" });
    };
    let claims = &proof.clear_claims()?.stage6;

    let bytecode_input_openings = bytecode::read_raf_input_openings();
    let [(spartan_shift_unexpanded_pc, instruction_input_unexpanded_pc)] =
        bytecode::read_raf_consistency_openings();
    if stage3.output_claims.shift.unexpanded_pc
        != stage3.output_claims.instruction_input.unexpanded_pc
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
            left: spartan_shift_unexpanded_pc,
            right: instruction_input_unexpanded_pc,
        });
    }

    let [ram_ra_reduced] = ram::ra_virtualization_input_openings();
    let instruction_ra_input_openings = instruction::ra_virtualization_input_openings(
        formula_dimensions.instruction_ra_virtualization,
    );
    let [ram_inc_read_write, ram_inc_val_check, rd_inc_read_write, rd_inc_val_evaluation] =
        increments::claim_reduction_input_openings();
    let input_claims = Stage6BatchInputClaims {
        bytecode_read_raf: {
            let mut input_claim = bytecode_claims.input.expression().try_evaluate(
                |id| {
                    if *id == bytecode_input_openings.spartan_outer.unexpanded_pc {
                        return Ok(stage1.outer.unexpanded_pc);
                    }
                    if *id == bytecode_input_openings.spartan_outer.imm {
                        return Ok(stage1.outer.imm);
                    }
                    for (flag, opening) in &bytecode_input_openings.spartan_outer.op_flags {
                        if *id == *opening {
                            return stage1
                                .outer
                                .claim(JoltVirtualPolynomial::OpFlags(*flag))
                                .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                        }
                    }
                    if *id == bytecode_input_openings.spartan_product.jump {
                        return Ok(stage2.output_claims.product_remainder.jump_flag);
                    }
                    if *id == bytecode_input_openings.spartan_product.branch {
                        return Ok(stage2.output_claims.product_remainder.branch_flag);
                    }
                    if *id
                        == bytecode_input_openings
                            .spartan_product
                            .write_lookup_output_to_rd
                    {
                        return Ok(stage2
                            .output_claims
                            .product_remainder
                            .write_lookup_output_to_rd);
                    }
                    if *id == bytecode_input_openings.spartan_product.virtual_instruction {
                        return Ok(stage2.output_claims.product_remainder.virtual_instruction);
                    }
                    if *id == bytecode_input_openings.instruction_input.imm {
                        return Ok(stage3.output_claims.instruction_input.imm);
                    }
                    if *id
                        == bytecode_input_openings
                            .instruction_input
                            .unexpanded_pc_from_shift
                    {
                        return Ok(stage3.output_claims.shift.unexpanded_pc);
                    }
                    if *id
                        == bytecode_input_openings
                            .instruction_input
                            .left_operand_is_rs1_value
                    {
                        return Ok(stage3.output_claims.instruction_input.left_operand_is_rs1);
                    }
                    if *id == bytecode_input_openings.instruction_input.left_operand_is_pc {
                        return Ok(stage3.output_claims.instruction_input.left_operand_is_pc);
                    }
                    if *id
                        == bytecode_input_openings
                            .instruction_input
                            .right_operand_is_rs2_value
                    {
                        return Ok(stage3.output_claims.instruction_input.right_operand_is_rs2);
                    }
                    if *id
                        == bytecode_input_openings
                            .instruction_input
                            .right_operand_is_imm
                    {
                        return Ok(stage3.output_claims.instruction_input.right_operand_is_imm);
                    }
                    if *id == bytecode_input_openings.instruction_input.is_noop_from_shift {
                        return Ok(stage3.output_claims.shift.is_noop);
                    }
                    if *id
                        == bytecode_input_openings
                            .instruction_input
                            .virtual_instruction_from_shift
                    {
                        return Ok(stage3.output_claims.shift.is_virtual);
                    }
                    if *id
                        == bytecode_input_openings
                            .instruction_input
                            .is_first_in_sequence_from_shift
                    {
                        return Ok(stage3.output_claims.shift.is_first_in_sequence);
                    }
                    if *id == bytecode_input_openings.registers_read_write.rd_wa {
                        return Ok(stage4.output_claims.registers_read_write.rd_wa);
                    }
                    if *id == bytecode_input_openings.registers_read_write.rs1_ra {
                        return Ok(stage4.output_claims.registers_read_write.rs1_ra);
                    }
                    if *id == bytecode_input_openings.registers_read_write.rs2_ra {
                        return Ok(stage4.output_claims.registers_read_write.rs2_ra);
                    }
                    if *id == bytecode_input_openings.registers_val_evaluation.rd_wa {
                        return Ok(stage5.output_claims.registers_val_evaluation.rd_wa);
                    }
                    if *id
                        == bytecode_input_openings
                            .registers_val_evaluation
                            .instruction_raf_flag
                    {
                        return Ok(stage5
                            .output_claims
                            .instruction_read_raf
                            .instruction_raf_flag);
                    }
                    for (table, opening) in &bytecode_input_openings
                        .registers_val_evaluation
                        .lookup_table_flags
                    {
                        if *id == *opening {
                            return stage5
                                .output_claims
                                .instruction_read_raf
                                .lookup_table_flags
                                .get(table.index())
                                .copied()
                                .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                        }
                    }
                    if *id == bytecode_input_openings.spartan_outer_pc {
                        return Ok(stage1.outer.pc);
                    }
                    if *id == bytecode_input_openings.spartan_shift_pc {
                        return Ok(stage3.output_claims.shift.pc);
                    }
                    Err(VerifierError::MissingOpeningClaim { id: *id })
                },
                |id| match id {
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                        Ok(bytecode_gamma)
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                        Ok(stage1_gammas[1])
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage2Gamma) => {
                        Ok(stage2_gammas[1])
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage3Gamma) => {
                        Ok(stage3_gammas[1])
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                        Ok(stage4_gammas[1])
                    }
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                        Ok(stage5_gammas[1])
                    }
                    _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                },
                |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
            )?;

            #[cfg(feature = "field-inline")]
            {
                let field_openings = field_bytecode::read_raf_input_openings();
                input_claim += field_bytecode::read_raf_input_extension::<PCS::Field>()
                    .try_evaluate(
                        |id| {
                            for (index, flag) in field_bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS
                                .into_iter()
                                .enumerate()
                            {
                                if *id == field_openings[index] {
                                    return stage1
                                        .field_inline
                                        .claim(FieldInlineVirtualPolynomial::FieldOpFlag(flag))
                                        .ok_or(VerifierError::MissingFieldInlineOpeningClaim {
                                            id: *id,
                                        });
                                }
                            }
                            if *id == field_openings[8] {
                                return Ok(stage4
                                    .output_claims
                                    .field_inline
                                    .field_registers_read_write
                                    .field_rd_wa);
                            }
                            if *id == field_openings[9] {
                                return Ok(stage4
                                    .output_claims
                                    .field_inline
                                    .field_registers_read_write
                                    .field_rs1_ra);
                            }
                            if *id == field_openings[10] {
                                return Ok(stage4
                                    .output_claims
                                    .field_inline
                                    .field_registers_read_write
                                    .field_rs2_ra);
                            }
                            if *id == field_openings[11] {
                                return Ok(stage5
                                    .output_claims
                                    .field_inline
                                    .field_registers_val_evaluation
                                    .field_rd_wa);
                            }
                            Err(VerifierError::MissingFieldInlineOpeningClaim { id: *id })
                        },
                        |id| match id {
                            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                                Ok(bytecode_gamma)
                            }
                            JoltChallengeId::BytecodeReadRaf(
                                BytecodeReadRafChallenge::Stage1Gamma,
                            ) => Ok(stage1_gammas[1]),
                            JoltChallengeId::BytecodeReadRaf(
                                BytecodeReadRafChallenge::Stage4Gamma,
                            ) => Ok(stage4_gammas[1]),
                            JoltChallengeId::BytecodeReadRaf(
                                BytecodeReadRafChallenge::Stage5Gamma,
                            ) => Ok(stage5_gammas[1]),
                            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                        },
                        |()| Ok(PCS::Field::zero()),
                    )?;
            }

            input_claim
        },
        booleanity: PCS::Field::zero(),
        ram_hamming_booleanity: PCS::Field::zero(),
        ram_ra_virtualization: ram_ra_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_ra_reduced => {
                    Ok(stage5.output_claims.ram_ra_claim_reduction.ram_ra)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        instruction_ra_virtualization: instruction_ra_claims.input.expression().try_evaluate(
            |id| {
                for (index, opening) in instruction_ra_input_openings.iter().enumerate() {
                    if *id == *opening {
                        return stage5
                            .output_claims
                            .instruction_read_raf
                            .instruction_ra
                            .get(index)
                            .copied()
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                Err(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| match id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => Ok(instruction_ra_gamma),
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        inc_claim_reduction: inc_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_inc_read_write => Ok(stage2.output_claims.ram_read_write.inc),
                id if id == ram_inc_val_check => Ok(stage4.output_claims.ram_val_check.ram_inc),
                id if id == rd_inc_read_write => {
                    Ok(stage4.output_claims.registers_read_write.rd_inc)
                }
                id if id == rd_inc_val_evaluation => {
                    Ok(stage5.output_claims.registers_val_evaluation.rd_inc)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => {
                    Ok(inc_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: {
            let read_write_inc = stage4
                .output_claims
                .field_inline
                .field_registers_read_write
                .field_rd_inc;
            let val_evaluation_inc = stage5
                .output_claims
                .field_inline
                .field_registers_val_evaluation
                .field_rd_inc;
            read_write_inc + field_inc_gamma * val_evaluation_inc
        },
        trusted_advice_cycle_phase: trusted_advice_claims
            .as_ref()
            .map(|claim| {
                advice_cycle_phase_input::<PCS::Field>(claim, stage4, JoltAdviceKind::Trusted)
            })
            .transpose()?,
        untrusted_advice_cycle_phase: untrusted_advice_claims
            .as_ref()
            .map(|claim| {
                advice_cycle_phase_input::<PCS::Field>(claim, stage4, JoltAdviceKind::Untrusted)
            })
            .transpose()?,
    };

    if trusted_advice_claims.is_none() && claims.advice_cycle_phase.trusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_claims.is_none() && claims.advice_cycle_phase.untrusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted),
        });
    }

    let mut sumcheck_claims = vec![
        SumcheckClaim::new(
            bytecode_claims.sumcheck.rounds,
            bytecode_claims.sumcheck.degree,
            input_claims.bytecode_read_raf,
        ),
        SumcheckClaim::new(
            booleanity_claims.sumcheck.rounds,
            booleanity_claims.sumcheck.degree,
            input_claims.booleanity,
        ),
        SumcheckClaim::new(
            ram_hamming_claims.sumcheck.rounds,
            ram_hamming_claims.sumcheck.degree,
            input_claims.ram_hamming_booleanity,
        ),
        SumcheckClaim::new(
            ram_ra_claims.sumcheck.rounds,
            ram_ra_claims.sumcheck.degree,
            input_claims.ram_ra_virtualization,
        ),
        SumcheckClaim::new(
            instruction_ra_claims.sumcheck.rounds,
            instruction_ra_claims.sumcheck.degree,
            input_claims.instruction_ra_virtualization,
        ),
        SumcheckClaim::new(
            inc_claims.sumcheck.rounds,
            inc_claims.sumcheck.degree,
            input_claims.inc_claim_reduction,
        ),
    ];
    #[cfg(feature = "field-inline")]
    sumcheck_claims.push(SumcheckClaim::new(
        field_inc_claims.sumcheck.rounds,
        field_inc_claims.sumcheck.degree,
        input_claims.field_registers_inc_claim_reduction,
    ));
    if let (Some(claim), Some(input_claim)) = (
        &trusted_advice_claims,
        input_claims.trusted_advice_cycle_phase,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }
    if let (Some(claim), Some(input_claim)) = (
        &untrusted_advice_claims,
        input_claims.untrusted_advice_cycle_phase,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }

    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage6_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_point = batch
        .try_instance_point(bytecode_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    let bytecode_opening_point = formula_dimensions
        .bytecode_read_raf
        .opening_point(bytecode_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    if claims.bytecode_read_raf.bytecode_ra.len() != bytecode_output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                bytecode_output_openings.bytecode_ra.len(),
                claims.bytecode_read_raf.bytecode_ra.len()
            ),
        });
    }

    let stage1_point = stage1.remainder.sumcheck_point.as_slice();
    let (_, stage1_cycle_binding) =
        stage1_point
            .split_first()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 1 remainder point is empty".to_string(),
            })?;
    let stage1_cycle = stage1_cycle_binding
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage2_cycle = stage2.batch.product_remainder.opening_point.clone();
    let stage3_cycle = stage3.batch.shift.opening_point.clone();
    let (stage4_register_address, stage4_cycle) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let (stage5_register_address, stage5_cycle) = stage5
        .batch
        .registers_val_evaluation
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let entry_bytecode_index = preprocessing
        .program
        .bytecode
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    let mut bytecode_public_values =
        bytecode::read_raf_public_values::<PCS::Field>(BytecodeReadRafEvaluationInputs {
            bytecode: &preprocessing.program.bytecode.bytecode,
            r_address: &bytecode_opening_point.r_address,
            r_cycle: &bytecode_opening_point.r_cycle,
            stage_cycle_points: [
                &stage1_cycle,
                &stage2_cycle,
                &stage3_cycle,
                stage4_cycle,
                stage5_cycle,
            ],
            register_read_write_point: stage4_register_address,
            register_val_evaluation_point: stage5_register_address,
            entry_bytecode_index,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &stage2_gammas,
            stage3_gammas: &stage3_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        })
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    #[cfg(feature = "field-inline")]
    add_field_inline_bytecode_public_values(
        &mut bytecode_public_values,
        field_inline_bytecode_rows(preprocessing)?,
        &bytecode_opening_point.r_address,
        &bytecode_opening_point.r_cycle,
        &stage1_cycle,
        &stage4.batch.field_registers_read_write.opening_point,
        &stage5.batch.field_registers_val_evaluation.opening_point,
        proof.protocol.field_inline.field_register_log_k,
        log_t,
        &stage1_gammas,
        &stage4_gammas,
        &stage5_gammas,
    )?;
    let bytecode_output = bytecode_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in bytecode_output_openings.bytecode_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims.bytecode_read_raf.bytecode_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(bytecode_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::BytecodeReadRaf(public_id) => {
                Ok(bytecode_public_values.value(*public_id))
            }
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;
    let bytecode_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(&bytecode_opening_point.r_address)
        .into_iter()
        .map(|r_address_chunk| {
            [
                r_address_chunk.as_slice(),
                bytecode_opening_point.r_cycle.as_slice(),
            ]
            .concat()
        })
        .collect::<Vec<_>>();

    let booleanity_point = batch
        .try_instance_point(booleanity_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    let booleanity_opening_point = booleanity_dimensions
        .opening_point(booleanity_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    if claims.booleanity.instruction_ra.len() != formula_dimensions.ra_layout.instruction()
        || claims.booleanity.bytecode_ra.len() != formula_dimensions.ra_layout.bytecode()
        || claims.booleanity.ram_ra.len() != formula_dimensions.ra_layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            reason: format!(
                "booleanity RA claim count mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                formula_dimensions.ra_layout.instruction(),
                formula_dimensions.ra_layout.bytecode(),
                formula_dimensions.ra_layout.ram(),
                claims.booleanity.instruction_ra.len(),
                claims.booleanity.bytecode_ra.len(),
                claims.booleanity.ram_ra.len()
            ),
        });
    }
    let booleanity_reference_eq_point = booleanity_reference_address
        .iter()
        .rev()
        .chain(booleanity_reference_cycle.iter().rev())
        .copied()
        .collect::<Vec<_>>();
    let eq_address_cycle =
        try_eq_mle(booleanity_point, &booleanity_reference_eq_point).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            }
        })?;
    let booleanity_output_openings =
        booleanity::booleanity_output_openings(formula_dimensions.ra_layout);
    let booleanity_output = booleanity_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in booleanity_output_openings.iter().enumerate() {
                if *id == *opening {
                    if index < formula_dimensions.ra_layout.instruction() {
                        return Ok(claims.booleanity.instruction_ra[index]);
                    }
                    let bytecode_index = index - formula_dimensions.ra_layout.instruction();
                    if bytecode_index < formula_dimensions.ra_layout.bytecode() {
                        return Ok(claims.booleanity.bytecode_ra[bytecode_index]);
                    }
                    let ram_index = bytecode_index - formula_dimensions.ra_layout.bytecode();
                    return Ok(claims.booleanity.ram_ra[ram_index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => Ok(booleanity_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::Booleanity(BooleanityPublic::EqAddressCycle) => Ok(eq_address_cycle),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;
    let ram_hamming_point = batch
        .try_instance_point(ram_hamming_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamHammingBooleanity,
            reason: error.to_string(),
        })?;
    let ram_hamming_opening_point = trace_dimensions
        .cycle_opening_point(ram_hamming_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamHammingBooleanity,
            reason: error.to_string(),
        })?;
    let eq_spartan_outer_cycle =
        try_eq_mle(ram_hamming_point, stage1_cycle_binding).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            }
        })?;
    let [ram_hamming_weight] = ram::hamming_booleanity_output_openings();
    let ram_hamming_output = ram_hamming_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_hamming_weight => Ok(claims.ram_hamming_booleanity.ram_hamming_weight),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match *id {
            JoltChallengeId::RamHammingBooleanity(RamHammingBooleanityChallenge::EqCycle) => {
                Ok(eq_spartan_outer_cycle)
            }
            id => Err(VerifierError::MissingStageClaimChallenge { id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let ram_ra_point = batch
        .try_instance_point(ram_ra_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_cycle = trace_dimensions
        .cycle_opening_point(ram_ra_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_output_openings =
        ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization);
    if claims.ram_ra_virtualization.ram_ra.len() != ram_ra_output_openings.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: format!(
                "RAM RA virtualization claim count mismatch: expected {}, got {}",
                ram_ra_output_openings.len(),
                claims.ram_ra_virtualization.ram_ra.len()
            ),
        });
    }
    let ram_reduced_opening_point = &stage5.batch.ram_ra_claim_reduction.opening_point;
    if ram_reduced_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: format!(
                "RAM RA reduction opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_reduced_opening_point.len()
            ),
        });
    }
    let (ram_reduced_address, ram_reduced_cycle) = ram_reduced_opening_point.split_at(log_k);
    let eq_ram_ra_cycle = try_eq_mle(ram_reduced_cycle, &ram_ra_cycle).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        }
    })?;
    let ram_ra_output = ram_ra_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in ram_ra_output_openings.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims.ram_ra_virtualization.ram_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match *id {
            JoltChallengeId::RamRaVirtualization(RamRaVirtualizationChallenge::EqCycle) => {
                Ok(eq_ram_ra_cycle)
            }
            id => Err(VerifierError::MissingStageClaimChallenge { id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let ram_ra_opening_point = [ram_reduced_address, ram_ra_cycle.as_slice()].concat();
    let ram_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(ram_reduced_address)
        .into_iter()
        .map(|r_address_chunk| [r_address_chunk.as_slice(), ram_ra_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    let instruction_ra_point = batch
        .try_instance_point(instruction_ra_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_cycle = trace_dimensions
        .cycle_opening_point(instruction_ra_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_output_openings = instruction::ra_virtualization_output_openings(
        formula_dimensions.instruction_ra_virtualization,
    );
    let flat_instruction_ra_output_openings = instruction_ra_output_openings.all();
    if claims
        .instruction_ra_virtualization
        .committed_instruction_ra
        .len()
        != flat_instruction_ra_output_openings.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: format!(
                "instruction RA virtualization claim count mismatch: expected {}, got {}",
                flat_instruction_ra_output_openings.len(),
                claims
                    .instruction_ra_virtualization
                    .committed_instruction_ra
                    .len()
            ),
        });
    }
    let eq_instruction_ra_cycle = try_eq_mle(
        &stage5.batch.instruction_read_raf.r_cycle,
        &instruction_ra_cycle,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionRaVirtualization,
        reason: error.to_string(),
    })?;
    let instruction_ra_output = instruction_ra_claims.output.expression().try_evaluate(
        |id| {
            for (index, opening) in flat_instruction_ra_output_openings.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims
                        .instruction_ra_virtualization
                        .committed_instruction_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::Gamma,
            ) => Ok(instruction_ra_gamma),
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::EqCycle,
            ) => Ok(eq_instruction_ra_cycle),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let instruction_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(&stage5.batch.instruction_read_raf.r_address)
        .into_iter()
        .map(|r_address_chunk| {
            [r_address_chunk.as_slice(), instruction_ra_cycle.as_slice()].concat()
        })
        .collect::<Vec<_>>();
    let instruction_ra_opening_point = [
        stage5.batch.instruction_read_raf.r_address.as_slice(),
        instruction_ra_cycle.as_slice(),
    ]
    .concat();

    let inc_point = batch
        .try_instance_point(inc_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let inc_opening_point = trace_dimensions
        .cycle_opening_point(inc_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let (_, ram_read_write_cycle) = stage2.batch.ram_read_write.opening_point.split_at(log_k);
    let (_, ram_val_check_cycle) = stage4.batch.ram_val_check.opening_point.split_at(log_k);
    let (_, registers_read_write_cycle) = stage4
        .batch
        .registers_read_write
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let (_, registers_val_evaluation_cycle) = stage5
        .batch
        .registers_val_evaluation
        .opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let eq_ram_read_write =
        try_eq_mle(&inc_opening_point, ram_read_write_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let eq_ram_val_check =
        try_eq_mle(&inc_opening_point, ram_val_check_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let eq_registers_read_write = try_eq_mle(&inc_opening_point, registers_read_write_cycle)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let eq_registers_val_evaluation =
        try_eq_mle(&inc_opening_point, registers_val_evaluation_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let [ram_inc, rd_inc] = increments::claim_reduction_output_openings();
    let inc_output = inc_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_inc => Ok(claims.inc_claim_reduction.ram_inc),
            id if id == rd_inc => Ok(claims.inc_claim_reduction.rd_inc),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => Ok(inc_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamReadWrite) => {
                Ok(eq_ram_read_write)
            }
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamValCheck) => {
                Ok(eq_ram_val_check)
            }
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersReadWrite) => {
                Ok(eq_registers_read_write)
            }
            JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersValEvaluation) => {
                Ok(eq_registers_val_evaluation)
            }
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    #[cfg(feature = "field-inline")]
    let (field_inc_point, field_inc_opening_point, field_inc_output) = {
        let field_inc_point = batch
            .try_instance_point(field_inc_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let field_inc_opening_point = trace_dimensions
            .cycle_opening_point(field_inc_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let field_log_k = proof.protocol.field_inline.field_register_log_k;
        let field_read_write_opening_point = &stage4.batch.field_registers_read_write.opening_point;
        let field_val_evaluation_opening_point =
            &stage5.batch.field_registers_val_evaluation.opening_point;
        if field_read_write_opening_point.len() != field_log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: format!(
                    "field-register read-write opening point length mismatch: expected {}, got {}",
                    field_log_k + log_t,
                    field_read_write_opening_point.len()
                ),
            });
        }
        if field_val_evaluation_opening_point.len() != field_log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: format!(
                    "field-register val-evaluation opening point length mismatch: expected {}, got {}",
                    field_log_k + log_t,
                    field_val_evaluation_opening_point.len()
                ),
            });
        }
        let (_, field_read_write_cycle) = field_read_write_opening_point.split_at(field_log_k);
        let (_, field_val_evaluation_cycle) =
            field_val_evaluation_opening_point.split_at(field_log_k);
        let eq_read_write =
            try_eq_mle(&field_inc_opening_point, field_read_write_cycle).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                }
            })?;
        let eq_val_evaluation = try_eq_mle(&field_inc_opening_point, field_val_evaluation_cycle)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let field_rd_inc = claims
            .field_inline
            .field_registers_inc_claim_reduction
            .field_rd_inc;
        let output = (eq_read_write + field_inc_gamma * eq_val_evaluation) * field_rd_inc;
        (field_inc_point.to_vec(), field_inc_opening_point, output)
    };

    let trusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        trusted_advice_layout.as_ref(),
        trusted_advice_claims.as_ref(),
        claims.advice_cycle_phase.trusted.as_ref(),
    ) {
        Some(verify_advice_cycle_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Trusted,
            opening_claim,
            stage4,
        )?)
    } else {
        None
    };
    let untrusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        untrusted_advice_layout.as_ref(),
        untrusted_advice_claims.as_ref(),
        claims.advice_cycle_phase.untrusted.as_ref(),
    ) {
        Some(verify_advice_cycle_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Untrusted,
            opening_claim,
            stage4,
        )?)
    } else {
        None
    };

    if let (Some(layout), Some(_)) = (
        trusted_advice_layout.as_ref(),
        trusted_advice_claims.as_ref(),
    ) {
        if trusted_advice.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_output_openings(
                    JoltAdviceKind::Trusted,
                    layout.dimensions(),
                )[0],
            });
        }
    }
    if let (Some(layout), Some(_)) = (
        untrusted_advice_layout.as_ref(),
        untrusted_advice_claims.as_ref(),
    ) {
        if untrusted_advice.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_output_openings(
                    JoltAdviceKind::Untrusted,
                    layout.dimensions(),
                )[0],
            });
        }
    }

    let expected_outputs = Stage6BatchExpectedOutputClaims {
        bytecode_read_raf: bytecode_output,
        booleanity: booleanity_output,
        ram_hamming_booleanity: ram_hamming_output,
        ram_ra_virtualization: ram_ra_output,
        instruction_ra_virtualization: instruction_ra_output,
        inc_claim_reduction: inc_output,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: field_inc_output,
        trusted_advice_cycle_phase: trusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        untrusted_advice_cycle_phase: untrusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
    };
    let mut expected_outputs_in_order = vec![
        expected_outputs.bytecode_read_raf,
        expected_outputs.booleanity,
        expected_outputs.ram_hamming_booleanity,
        expected_outputs.ram_ra_virtualization,
        expected_outputs.instruction_ra_virtualization,
        expected_outputs.inc_claim_reduction,
    ];
    #[cfg(feature = "field-inline")]
    expected_outputs_in_order.push(expected_outputs.field_registers_inc_claim_reduction);
    if let Some(output_claim) = expected_outputs.trusted_advice_cycle_phase {
        expected_outputs_in_order.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.untrusted_advice_cycle_phase {
        expected_outputs_in_order.push(output_claim);
    }
    if batch.batching_coefficients.len() != expected_outputs_in_order.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 batch verifier returned {} coefficients for {} instances",
                batch.batching_coefficients.len(),
                expected_outputs_in_order.len()
            ),
        });
    }
    let expected_final_claim = batch
        .batching_coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum();
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
        });
    }

    append_stage6_opening_claims(transcript, claims);

    Ok(Stage6Output::Clear(Stage6ClearOutput {
        public: public(
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        batch: VerifiedStage6Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.clone(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            bytecode_read_raf: VerifiedBytecodeReadRafSumcheck {
                input_claim: input_claims.bytecode_read_raf,
                sumcheck_point: bytecode_point.to_vec(),
                r_address: bytecode_opening_point.r_address,
                r_cycle: bytecode_opening_point.r_cycle,
                full_opening_point: bytecode_opening_point.opening_point,
                bytecode_ra_opening_points,
                expected_output_claim: expected_outputs.bytecode_read_raf,
            },
            booleanity: VerifiedBooleanitySumcheck {
                input_claim: input_claims.booleanity,
                sumcheck_point: booleanity_point.to_vec(),
                r_address: booleanity_opening_point.r_address,
                r_cycle: booleanity_opening_point.r_cycle,
                opening_point: booleanity_opening_point.opening_point,
                reference_address: booleanity_reference_address,
                reference_cycle: booleanity_reference_cycle,
                expected_output_claim: expected_outputs.booleanity,
            },
            ram_hamming_booleanity: VerifiedStage6Sumcheck {
                input_claim: input_claims.ram_hamming_booleanity,
                sumcheck_point: ram_hamming_point.to_vec(),
                opening_point: ram_hamming_opening_point,
                expected_output_claim: expected_outputs.ram_hamming_booleanity,
            },
            ram_ra_virtualization: VerifiedRamRaVirtualizationSumcheck {
                input_claim: input_claims.ram_ra_virtualization,
                sumcheck_point: ram_ra_point.to_vec(),
                opening_point: ram_ra_opening_point,
                ram_ra_opening_points,
                expected_output_claim: expected_outputs.ram_ra_virtualization,
            },
            instruction_ra_virtualization: VerifiedInstructionRaVirtualizationSumcheck {
                input_claim: input_claims.instruction_ra_virtualization,
                sumcheck_point: instruction_ra_point.to_vec(),
                opening_point: instruction_ra_opening_point,
                instruction_ra_opening_points,
                expected_output_claim: expected_outputs.instruction_ra_virtualization,
            },
            inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: input_claims.inc_claim_reduction,
                sumcheck_point: inc_point.to_vec(),
                opening_point: inc_opening_point,
                expected_output_claim: expected_outputs.inc_claim_reduction,
            },
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: input_claims.field_registers_inc_claim_reduction,
                sumcheck_point: field_inc_point,
                opening_point: field_inc_opening_point,
                expected_output_claim: expected_outputs.field_registers_inc_claim_reduction,
            },
            trusted_advice_cycle_phase: trusted_advice,
            untrusted_advice_cycle_phase: untrusted_advice,
        },
    }))
}

fn validate_compressed_stage_claim<F: Field>(
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

fn advice_cycle_phase_input<F: Field>(
    claim: &JoltRelationClaims<F>,
    stage4: &Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let [advice_input] = advice::cycle_phase_input_openings(kind);
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

fn verify_advice_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    opening_claim: &AdviceCyclePhaseOutputClaim<F>,
    stage4: &Stage4ClearOutput<F>,
) -> Result<VerifiedAdviceCyclePhaseSumcheck<F>, VerifierError> {
    let booleanity_rounds = layout.log_k_chunk() + layout.log_t();
    let booleanity_offset = batch
        .max_num_vars
        .checked_sub(booleanity_rounds)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: "Stage 6 advice cycle phase offset underflow".to_string(),
        })?;
    let offset = booleanity_offset + layout.log_k_chunk();
    let advice_point = batch
        .try_instance_point_at(offset, claim.sumcheck.rounds)
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

fn advice_cycle_phase_public<F: Field, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
) -> Result<AdviceCyclePhasePublicOutput<F>, VerifierError> {
    let booleanity_rounds = layout.log_k_chunk() + layout.log_t();
    let booleanity_offset = batch
        .max_num_vars
        .checked_sub(booleanity_rounds)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: "Stage 6 advice cycle phase offset underflow".to_string(),
        })?;
    let offset = booleanity_offset + layout.log_k_chunk();
    let advice_point = batch
        .try_instance_point_at(offset, claim.sumcheck.rounds)
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

fn append_stage6_opening_claims<F, T>(transcript: &mut T, claims: &Stage6Claims<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.bytecode_read_raf.bytecode_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.bytecode_ra {
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
    #[cfg(feature = "field-inline")]
    transcript.append_labeled(
        b"opening_claim",
        &claims
            .field_inline
            .field_registers_inc_claim_reduction
            .field_rd_inc,
    );
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
}
