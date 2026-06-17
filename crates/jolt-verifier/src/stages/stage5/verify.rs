#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::registers as field_registers, FieldRegistersTraceDimensions,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram, registers,
    },
    InstructionReadRafChallenge, JoltChallengeId, JoltPublicId, JoltRelationId, JoltSumcheckDomain,
    RamRaClaimReductionChallenge, RegistersValEvaluationChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    try_eq_mle, IdentityPolynomial, LtPolynomial, MultilinearEvaluation, OperandPolynomial,
    OperandSide,
};
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::Transcript;
use num_traits::Zero;

use super::{
    inputs::{Deps, Stage5Claims},
    outputs::{
        InstructionReadRafPublicOutput, Stage5ClearOutput, Stage5Output, Stage5PublicOutput,
        Stage5SumcheckPublicOutput, Stage5ZkOutput, VerifiedInstructionReadRafSumcheck,
        VerifiedStage5Batch, VerifiedStage5Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::zk::committed,
    verifier::CheckedInputs, VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage5BatchInputClaims<F: Field> {
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    field_registers_val_evaluation: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage5BatchExpectedOutputClaims<F: Field> {
    instruction_read_raf: F,
    ram_ra_claim_reduction: F,
    registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    field_registers_val_evaluation: F,
}

#[cfg(feature = "field-inline")]
const fn field_inline_stage5_output_claim_count() -> usize {
    2
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_stage5_output_claim_count() -> usize {
    0
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage5Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage4" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage4" });
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
        preprocessing.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;

    let instruction_claims =
        instruction::read_raf::<PCS::Field>(formula_dimensions.instruction_read_raf);
    let ram_claims = ram::ra_claim_reduction::<PCS::Field>(trace_dimensions);
    let registers_claims = registers::val_evaluation::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_registers_claims =
        field_registers::val_evaluation::<PCS::Field>(FieldRegistersTraceDimensions::new(log_t));

    for claim in [&instruction_claims, &ram_claims, &registers_claims] {
        if claim.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: claim.id,
                degree: claim.sumcheck.degree,
            });
        }
        if !matches!(claim.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
            return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
                stage: claim.id,
            });
        }
    }
    #[cfg(feature = "field-inline")]
    {
        if field_registers_claims.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: JoltRelationId::RegistersValEvaluation,
                degree: field_registers_claims.sumcheck.degree,
            });
        }
    }

    let instruction_gamma = transcript.challenge_scalar();
    let instruction_gamma_squared = instruction_gamma * instruction_gamma;
    let ram_gamma = transcript.challenge_scalar();

    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let committed_output_claims = instruction_output_openings.lookup_table_flags.len()
        + instruction_output_openings.instruction_ra.len()
        + 1
        + 1
        + 2
        + field_inline_stage5_output_claim_count();

    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage5PublicOutput {
            challenges,
            batching_coefficients,
            instruction_gamma,
            ram_gamma,
        };

    if checked.zk {
        let Deps::Zk { stage2, stage4 } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage4" });
        };
        let statements = [
            SumcheckStatement::new(
                instruction_claims.sumcheck.rounds,
                instruction_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(ram_claims.sumcheck.rounds, ram_claims.sumcheck.degree),
            SumcheckStatement::new(
                registers_claims.sumcheck.rounds,
                registers_claims.sumcheck.degree,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let mut statements = statements.to_vec();
        #[cfg(feature = "field-inline")]
        statements.push(SumcheckStatement::new(
            field_registers_claims.sumcheck.rounds,
            field_registers_claims.sumcheck.degree,
        ));
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage5_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage5_sumcheck_proof,
                proof_label: "stage5_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::InstructionReadRaf,
            })?;

        let instruction_point = consistency
            .try_instance_point(instruction_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: error.to_string(),
            })?;
        let instruction_opening_point = formula_dimensions
            .instruction_read_raf
            .opening_point(&instruction_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: error.to_string(),
            })?;
        let instruction_ra_count = instruction_output_openings.instruction_ra.len();
        let instruction_ra_chunk_size = instruction_opening_point
            .r_address
            .len()
            .checked_div(instruction_ra_count)
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: "instruction read-RAF proof has no virtual RA claims".to_string(),
            })?;
        if instruction_ra_chunk_size * instruction_ra_count
            != instruction_opening_point.r_address.len()
        {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: format!(
                    "instruction address point length {} is not divisible by virtual RA count {}",
                    instruction_opening_point.r_address.len(),
                    instruction_ra_count
                ),
            });
        }
        let instruction_ra_opening_points = instruction_opening_point
            .r_address
            .chunks(instruction_ra_chunk_size)
            .map(|r_address_chunk| {
                [
                    r_address_chunk,
                    instruction_opening_point.r_cycle.as_slice(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();

        let [ram_ra_raf, ram_ra_read_write, ram_ra_val_check] =
            ram::ra_claim_reduction_input_openings();
        let ram_point = consistency
            .try_instance_point(ram_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            })?;
        let ram_cycle = trace_dimensions
            .cycle_opening_point(&ram_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            })?;
        let ram_raf_opening_point = &stage2
            .ram_ra_claim_reduction_inputs
            .ram_raf_evaluation_opening_point;
        let ram_read_write_opening_point = &stage2
            .ram_ra_claim_reduction_inputs
            .ram_read_write_opening_point;
        let ram_val_check_opening_point = &stage4.ram_val_check_opening_point;
        for (label, opening_point) in [
            ("RAM RAF evaluation", ram_raf_opening_point),
            ("RAM read-write", ram_read_write_opening_point),
            ("RAM value check", ram_val_check_opening_point),
        ] {
            if opening_point.len() != log_k + log_t {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamRaClaimReduction,
                    reason: format!(
                        "{label} opening point length mismatch: expected {}, got {}",
                        log_k + log_t,
                        opening_point.len()
                    ),
                });
            }
        }
        let (ram_raf_address, _ram_raf_cycle) = ram_raf_opening_point.split_at(log_k);
        let (ram_read_write_address, _ram_read_write_cycle) =
            ram_read_write_opening_point.split_at(log_k);
        let (ram_val_check_address, _ram_val_check_cycle) =
            ram_val_check_opening_point.split_at(log_k);
        if ram_raf_address != ram_read_write_address {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::RamRaClaimReduction,
                left: ram_ra_raf,
                right: ram_ra_read_write,
            });
        }
        if ram_val_check_address != ram_read_write_address {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::RamRaClaimReduction,
                left: ram_ra_val_check,
                right: ram_ra_read_write,
            });
        }
        let ram_opening_point = [ram_read_write_address, ram_cycle.as_slice()].concat();

        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        let registers_cycle = trace_dimensions
            .cycle_opening_point(&registers_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        if stage4.registers_read_write_opening_point.len() != REGISTER_ADDRESS_BITS + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: format!(
                    "register read-write opening point length mismatch: expected {}, got {}",
                    REGISTER_ADDRESS_BITS + log_t,
                    stage4.registers_read_write_opening_point.len()
                ),
            });
        }
        let (register_address, _registers_read_write_cycle) = stage4
            .registers_read_write_opening_point
            .split_at(REGISTER_ADDRESS_BITS);
        let registers_opening_point = [register_address, registers_cycle.as_slice()].concat();

        #[cfg(feature = "field-inline")]
        let field_registers_val_evaluation = {
            let field_registers_point = consistency
                .try_instance_point(field_registers_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RegistersValEvaluation,
                    reason: error.to_string(),
                })?;
            let field_registers_cycle = trace_dimensions
                .cycle_opening_point(&field_registers_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RegistersValEvaluation,
                    reason: error.to_string(),
                })?;
            let field_log_k = proof.protocol.field_inline.field_register_log_k;
            if stage4.field_registers_read_write_opening_point.len() != field_log_k + log_t {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RegistersValEvaluation,
                    reason: format!(
                        "field-register read-write opening point length mismatch: expected {}, got {}",
                        field_log_k + log_t,
                        stage4.field_registers_read_write_opening_point.len()
                    ),
                });
            }
            let (field_register_address, _field_registers_read_write_cycle) = stage4
                .field_registers_read_write_opening_point
                .split_at(field_log_k);
            let opening_point = [field_register_address, field_registers_cycle.as_slice()].concat();
            Stage5SumcheckPublicOutput {
                sumcheck_point: field_registers_point,
                opening_point,
            }
        };

        return Ok(Stage5Output::Zk(Stage5ZkOutput {
            public: public(
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
            instruction_read_raf: InstructionReadRafPublicOutput {
                sumcheck_point: instruction_point,
                r_address: instruction_opening_point.r_address.clone(),
                r_cycle: instruction_opening_point.r_cycle.clone(),
                full_opening_point: instruction_opening_point.opening_point,
                lookup_table_flag_opening_point: instruction_opening_point.r_cycle.clone(),
                instruction_ra_opening_points,
                instruction_raf_flag_opening_point: instruction_opening_point.r_cycle,
            },
            ram_ra_claim_reduction: Stage5SumcheckPublicOutput {
                sumcheck_point: ram_point,
                opening_point: ram_opening_point,
            },
            registers_val_evaluation: Stage5SumcheckPublicOutput {
                sumcheck_point: registers_point,
                opening_point: registers_opening_point,
            },
            #[cfg(feature = "field-inline")]
            field_inline: super::outputs::FieldInlineStage5ZkOutput {
                field_registers_val_evaluation,
            },
        }));
    }

    let Deps::Clear { stage2, stage4 } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage4" });
    };
    let claims = &proof.clear_claims()?.stage5;
    if claims.instruction_read_raf.lookup_table_flags.len()
        != instruction_output_openings.lookup_table_flags.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "lookup table flag claim count mismatch: expected {}, got {}",
                instruction_output_openings.lookup_table_flags.len(),
                claims.instruction_read_raf.lookup_table_flags.len()
            ),
        });
    }
    if claims.instruction_read_raf.instruction_ra.len()
        != instruction_output_openings.instruction_ra.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "instruction RA claim count mismatch: expected {}, got {}",
                instruction_output_openings.instruction_ra.len(),
                claims.instruction_read_raf.instruction_ra.len()
            ),
        });
    }

    let [(lookup_output_reduced, lookup_output_product)] =
        instruction::read_raf_consistency_openings();
    if stage2.batch.product_remainder.opening_point
        != stage2.batch.instruction_claim_reduction.opening_point
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }
    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output;
    let reduced_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);
    if reduced_lookup_output != product_lookup_output {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }

    let [lookup_output, left_lookup_operand, right_lookup_operand] =
        instruction::read_raf_input_openings();
    let [ram_ra_raf, ram_ra_read_write, ram_ra_val_check] =
        ram::ra_claim_reduction_input_openings();
    let [registers_val] = registers::val_evaluation_input_openings();
    let input_claims = Stage5BatchInputClaims {
        instruction_read_raf: instruction_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == lookup_output => Ok(reduced_lookup_output),
                id if id == left_lookup_operand => Ok(stage2
                    .output_claims
                    .instruction_claim_reduction
                    .left_lookup_operand),
                id if id == right_lookup_operand => Ok(stage2
                    .output_claims
                    .instruction_claim_reduction
                    .right_lookup_operand),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::Gamma) => {
                    Ok(instruction_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        ram_ra_claim_reduction: ram_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_ra_raf => Ok(stage2.output_claims.ram_raf_evaluation),
                id if id == ram_ra_read_write => Ok(stage2.output_claims.ram_read_write.ra),
                id if id == ram_ra_val_check => Ok(stage4.output_claims.ram_val_check.ram_ra),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => {
                    Ok(ram_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        registers_val_evaluation: registers_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == registers_val => {
                    Ok(stage4.output_claims.registers_read_write.registers_val)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation: stage4
            .output_claims
            .field_inline
            .field_registers_read_write
            .field_registers_val,
    };

    let sumcheck_claims = [
        SumcheckClaim::new(
            instruction_claims.sumcheck.rounds,
            instruction_claims.sumcheck.degree,
            input_claims.instruction_read_raf,
        ),
        SumcheckClaim::new(
            ram_claims.sumcheck.rounds,
            ram_claims.sumcheck.degree,
            input_claims.ram_ra_claim_reduction,
        ),
        SumcheckClaim::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
            input_claims.registers_val_evaluation,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let mut sumcheck_claims = sumcheck_claims.to_vec();
    #[cfg(feature = "field-inline")]
    sumcheck_claims.push(SumcheckClaim::new(
        field_registers_claims.sumcheck.rounds,
        field_registers_claims.sumcheck.degree,
        input_claims.field_registers_val_evaluation,
    ));
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage5_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;

    let instruction_point = batch
        .try_instance_point(instruction_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
    let instruction_opening_point = formula_dimensions
        .instruction_read_raf
        .opening_point(instruction_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
    let eq_reduction = try_eq_mle(
        &stage2.batch.instruction_claim_reduction.opening_point,
        &instruction_opening_point.r_cycle,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;
    let left_operand_eval = OperandPolynomial::new(2 * RISCV_XLEN, OperandSide::Left)
        .evaluate(&instruction_opening_point.r_address);
    let right_operand_eval = OperandPolynomial::new(2 * RISCV_XLEN, OperandSide::Right)
        .evaluate(&instruction_opening_point.r_address);
    let identity_eval =
        IdentityPolynomial::new(2 * RISCV_XLEN).evaluate(&instruction_opening_point.r_address);
    let mut table_values = [PCS::Field::zero(); LookupTableKind::<RISCV_XLEN>::COUNT];
    for table in LookupTableKind::<RISCV_XLEN>::iter() {
        table_values[table.index()] =
            table.evaluate_mle::<PCS::Field, PCS::Field>(&instruction_opening_point.r_address);
    }
    let instruction_output = instruction_claims.output.expression().try_evaluate(
        |id| {
            if let Some(index) = instruction_output_openings
                .lookup_table_flags
                .iter()
                .position(|opening| *opening == *id)
            {
                return Ok(claims.instruction_read_raf.lookup_table_flags[index]);
            }
            if let Some(index) = instruction_output_openings
                .instruction_ra
                .iter()
                .position(|opening| *opening == *id)
            {
                return Ok(claims.instruction_read_raf.instruction_ra[index]);
            }
            if *id == instruction_output_openings.instruction_raf_flag {
                return Ok(claims.instruction_read_raf.instruction_raf_flag);
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::EqTableValue(
                index,
            )) => table_values
                .get(*index)
                .copied()
                .map(|table_value| eq_reduction * table_value)
                .ok_or(VerifierError::MissingStageClaimChallenge { id: *id }),
            JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::EqRafConstant) => {
                Ok(eq_reduction
                    * (instruction_gamma * left_operand_eval
                        + instruction_gamma_squared * right_operand_eval))
            }
            JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::EqRafFlag) => {
                Ok(eq_reduction
                    * (instruction_gamma_squared * identity_eval
                        - instruction_gamma * left_operand_eval
                        - instruction_gamma_squared * right_operand_eval))
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let instruction_ra_chunk_size = instruction_opening_point
        .r_address
        .len()
        .checked_div(claims.instruction_read_raf.instruction_ra.len())
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: "instruction read-RAF proof has no virtual RA claims".to_string(),
        })?;
    if instruction_ra_chunk_size * claims.instruction_read_raf.instruction_ra.len()
        != instruction_opening_point.r_address.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "instruction address point length {} is not divisible by virtual RA count {}",
                instruction_opening_point.r_address.len(),
                claims.instruction_read_raf.instruction_ra.len()
            ),
        });
    }
    let instruction_ra_opening_points = instruction_opening_point
        .r_address
        .chunks(instruction_ra_chunk_size)
        .map(|r_address_chunk| {
            [
                r_address_chunk,
                instruction_opening_point.r_cycle.as_slice(),
            ]
            .concat()
        })
        .collect::<Vec<_>>();

    let ram_point = batch
        .try_instance_point(ram_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_cycle = trace_dimensions
        .cycle_opening_point(ram_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_raf_opening_point = &stage2.batch.ram_raf_evaluation.opening_point;
    let ram_read_write_opening_point = &stage2.batch.ram_read_write.opening_point;
    let ram_val_check_opening_point = &stage4.batch.ram_val_check.opening_point;
    for (label, opening_point) in [
        ("RAM RAF evaluation", ram_raf_opening_point),
        ("RAM read-write", ram_read_write_opening_point),
        ("RAM value check", ram_val_check_opening_point),
    ] {
        if opening_point.len() != log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: format!(
                    "{label} opening point length mismatch: expected {}, got {}",
                    log_k + log_t,
                    opening_point.len()
                ),
            });
        }
    }
    let (ram_raf_address, ram_raf_cycle) = ram_raf_opening_point.split_at(log_k);
    let (ram_read_write_address, ram_read_write_cycle) =
        ram_read_write_opening_point.split_at(log_k);
    let (ram_val_check_address, ram_val_check_cycle) = ram_val_check_opening_point.split_at(log_k);
    if ram_raf_address != ram_read_write_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamRaClaimReduction,
            left: ram_ra_raf,
            right: ram_ra_read_write,
        });
    }
    if ram_val_check_address != ram_read_write_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamRaClaimReduction,
            left: ram_ra_val_check,
            right: ram_ra_read_write,
        });
    }
    let ram_public_values = ram::RamRaClaimReductionPublicValues {
        eq_cycle_raf: try_eq_mle(ram_raf_cycle, &ram_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            }
        })?,
        eq_cycle_read_write: try_eq_mle(ram_read_write_cycle, &ram_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            }
        })?,
        eq_cycle_val_check: try_eq_mle(ram_val_check_cycle, &ram_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            }
        })?,
    };
    let [ram_ra_reduced] = ram::ra_claim_reduction_output_openings();
    let ram_output = ram_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_ra_reduced => Ok(claims.ram_ra_claim_reduction.ram_ra),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => {
                Ok(ram_gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::RamRaClaimReduction(public_id) => Ok(ram_public_values.value(*public_id)),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;
    let ram_opening_point = [ram_read_write_address, ram_cycle.as_slice()].concat();

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;
    let registers_cycle = trace_dimensions
        .cycle_opening_point(registers_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;
    let registers_read_write_opening_point = &stage4.batch.registers_read_write.opening_point;
    if registers_read_write_opening_point.len() != REGISTER_ADDRESS_BITS + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: format!(
                "register read-write opening point length mismatch: expected {}, got {}",
                REGISTER_ADDRESS_BITS + log_t,
                registers_read_write_opening_point.len()
            ),
        });
    }
    let (register_address, registers_read_write_cycle) =
        registers_read_write_opening_point.split_at(REGISTER_ADDRESS_BITS);
    let lt_cycle = LtPolynomial::evaluate(&registers_cycle, registers_read_write_cycle);
    let [rd_inc, rd_wa] = registers::val_evaluation_output_openings();
    let registers_output = registers_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == rd_inc => Ok(claims.registers_val_evaluation.rd_inc),
            id if id == rd_wa => Ok(claims.registers_val_evaluation.rd_wa),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RegistersValEvaluation(RegistersValEvaluationChallenge::LtCycle) => {
                Ok(lt_cycle)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;
    let registers_opening_point = [register_address, registers_cycle.as_slice()].concat();

    #[cfg(feature = "field-inline")]
    let (field_registers_point, field_registers_opening_point, field_registers_output) = {
        let field_registers_point = batch
            .try_instance_point(field_registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        let field_registers_cycle = trace_dimensions
            .cycle_opening_point(field_registers_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        let field_log_k = proof.protocol.field_inline.field_register_log_k;
        let field_registers_read_write_opening_point =
            &stage4.batch.field_registers_read_write.opening_point;
        if field_registers_read_write_opening_point.len() != field_log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: format!(
                    "field-register read-write opening point length mismatch: expected {}, got {}",
                    field_log_k + log_t,
                    field_registers_read_write_opening_point.len()
                ),
            });
        }
        let (field_register_address, field_registers_read_write_cycle) =
            field_registers_read_write_opening_point.split_at(field_log_k);
        let lt_cycle =
            LtPolynomial::evaluate(&field_registers_cycle, field_registers_read_write_cycle);
        let field_claims = &claims.field_inline.field_registers_val_evaluation;
        let output = lt_cycle * field_claims.field_rd_inc * field_claims.field_rd_wa;
        (
            field_registers_point.to_vec(),
            [field_register_address, field_registers_cycle.as_slice()].concat(),
            output,
        )
    };

    let expected_outputs = Stage5BatchExpectedOutputClaims {
        instruction_read_raf: instruction_output,
        ram_ra_claim_reduction: ram_output,
        registers_val_evaluation: registers_output,
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation: field_registers_output,
    };
    let coefficients = batch.batching_coefficients.as_slice();
    #[cfg(not(feature = "field-inline"))]
    let expected_final_claim = {
        let [instruction_coefficient, ram_coefficient, registers_coefficient] = coefficients else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: "Stage 5 batch verifier returned the wrong number of coefficients"
                    .to_string(),
            });
        };
        *instruction_coefficient * expected_outputs.instruction_read_raf
            + *ram_coefficient * expected_outputs.ram_ra_claim_reduction
            + *registers_coefficient * expected_outputs.registers_val_evaluation
    };
    #[cfg(feature = "field-inline")]
    let expected_final_claim = {
        let [instruction_coefficient, ram_coefficient, registers_coefficient, field_registers_coefficient] =
            coefficients
        else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: "Stage 5 batch verifier returned the wrong number of coefficients"
                    .to_string(),
            });
        };
        *instruction_coefficient * expected_outputs.instruction_read_raf
            + *ram_coefficient * expected_outputs.ram_ra_claim_reduction
            + *registers_coefficient * expected_outputs.registers_val_evaluation
            + *field_registers_coefficient * expected_outputs.field_registers_val_evaluation
    };
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::InstructionReadRaf,
        });
    }

    append_stage5_opening_claims(transcript, claims);

    Ok(Stage5Output::Clear(Stage5ClearOutput {
        public: public(
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        batch: VerifiedStage5Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.clone(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: input_claims.instruction_read_raf,
                sumcheck_point: instruction_point.to_vec(),
                r_address: instruction_opening_point.r_address.clone(),
                r_cycle: instruction_opening_point.r_cycle.clone(),
                full_opening_point: instruction_opening_point.opening_point,
                lookup_table_flag_opening_point: instruction_opening_point.r_cycle.clone(),
                instruction_ra_opening_points,
                instruction_raf_flag_opening_point: instruction_opening_point.r_cycle,
                expected_output_claim: expected_outputs.instruction_read_raf,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: input_claims.ram_ra_claim_reduction,
                sumcheck_point: ram_point.to_vec(),
                opening_point: ram_opening_point,
                expected_output_claim: expected_outputs.ram_ra_claim_reduction,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: input_claims.registers_val_evaluation,
                sumcheck_point: registers_point.to_vec(),
                opening_point: registers_opening_point,
                expected_output_claim: expected_outputs.registers_val_evaluation,
            },
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: input_claims.field_registers_val_evaluation,
                sumcheck_point: field_registers_point,
                opening_point: field_registers_opening_point,
                expected_output_claim: expected_outputs.field_registers_val_evaluation,
            },
        },
    }))
}

fn append_stage5_opening_claims<F, T>(transcript: &mut T, claims: &Stage5Claims<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.instruction_read_raf.lookup_table_flags {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.instruction_read_raf.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_read_raf.instruction_raf_flag,
    );
    transcript.append_labeled(b"opening_claim", &claims.ram_ra_claim_reduction.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_val_evaluation.rd_inc);
    transcript.append_labeled(b"opening_claim", &claims.registers_val_evaluation.rd_wa);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_val_evaluation;
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_inc);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_wa);
    }
}
