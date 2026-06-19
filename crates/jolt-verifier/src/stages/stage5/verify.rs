use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{JoltFormulaDimensions, TraceDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram, registers,
    },
    JoltRelationId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::Transcript;

use super::{
    inputs::{Deps, Stage5OutputClaims},
    instruction_read_raf::{
        InstructionReadRaf, InstructionReadRafInputClaims, InstructionReadRafOutputClaims,
    },
    outputs::{
        InstructionReadRafPublicOutput, Stage5ClearOutput, Stage5Output, Stage5PublicOutput,
        Stage5SumcheckPublicOutput, Stage5ZkOutput, VerifiedInstructionReadRafSumcheck,
        VerifiedStage5Batch, VerifiedStage5Sumcheck,
    },
    ram_ra_claim_reduction::{
        RamRaClaimReduction, RamRaClaimReductionInputClaims, RamRaClaimReductionOutputClaims,
    },
    registers_val_evaluation::{
        RegistersValEvaluation, RegistersValEvaluationInputClaims,
        RegistersValEvaluationOutputClaims,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::{OpeningClaim, SumcheckInstance},
        stage2::Stage2ClearOutput,
        stage4::Stage4ClearOutput,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

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
    let instruction_gamma = transcript.challenge_scalar();
    let ram_gamma = transcript.challenge_scalar();

    let instruction_relation =
        InstructionReadRaf::new(formula_dimensions.instruction_read_raf, instruction_gamma);
    let ram_relation = RamRaClaimReduction::new(trace_dimensions, log_k, ram_gamma);
    let registers_relation = RegistersValEvaluation::new(trace_dimensions);

    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let committed_output_claims = instruction_output_openings.lookup_table_flags.len()
        + instruction_output_openings.instruction_ra.len()
        + 1
        + ram::ra_claim_reduction_output_openings().len()
        + 2;

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

        let ram_point = consistency
            .try_instance_point(ram_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            })?;
        let ram_inputs = RamRaClaimReductionInputClaims {
            raf: stage2
                .ram_ra_claim_reduction_inputs
                .ram_raf_evaluation_opening_point
                .clone(),
            read_write: stage2
                .ram_ra_claim_reduction_inputs
                .ram_read_write_opening_point
                .clone(),
            val_check: stage4.ram_val_check_opening_point.clone(),
        };
        let ram_opening_point = ram_relation
            .derive_opening_points(&ram_point, &ram_inputs)?
            .ram_ra;

        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        let registers_inputs = RegistersValEvaluationInputClaims {
            registers_val: stage4.registers_read_write_opening_point.clone(),
        };
        let registers_opening_point = registers_relation
            .derive_opening_points(&registers_point, &registers_inputs)?
            .rd_inc;

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
    if stage2.output_claims.product_remainder_point()
        != stage2.output_claims.instruction_claim_reduction_point()
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }
    let product_lookup_output = stage2.output_claims.product_remainder.lookup_output.value;
    let reduced_lookup_output = stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .as_ref()
        .map_or(product_lookup_output, |claim| claim.value);
    if reduced_lookup_output != product_lookup_output {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }

    let instruction_inputs = InstructionReadRafInputClaims::from_upstream(stage2);
    let ram_inputs = RamRaClaimReductionInputClaims::from_upstream(stage2, stage4);
    let registers_inputs = RegistersValEvaluationInputClaims::from_upstream(stage4);
    let input_claims = Stage5InputClaims {
        instruction_read_raf: instruction_relation.input_claim(&instruction_inputs)?,
        ram_ra_claim_reduction: ram_relation.input_claim(&ram_inputs)?,
        registers_val_evaluation: registers_relation.input_claim(&registers_inputs)?,
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
    let instruction_outputs = InstructionReadRafOutputClaims {
        lookup_table_flags: claims
            .instruction_read_raf
            .lookup_table_flags
            .iter()
            .map(|&value| OpeningClaim {
                point: instruction_opening_point.r_cycle.clone(),
                value,
            })
            .collect(),
        instruction_ra: claims
            .instruction_read_raf
            .instruction_ra
            .iter()
            .zip(&instruction_ra_opening_points)
            .map(|(&value, point)| OpeningClaim {
                point: point.clone(),
                value,
            })
            .collect(),
        instruction_raf_flag: OpeningClaim {
            point: instruction_opening_point.r_cycle.clone(),
            value: claims.instruction_read_raf.instruction_raf_flag,
        },
    };
    let instruction_output =
        instruction_relation.expected_output(&instruction_inputs, &instruction_outputs)?;

    let ram_point = batch
        .try_instance_point(ram_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_output_points = ram_relation.derive_opening_points(ram_point, &ram_inputs)?;
    let ram_outputs = RamRaClaimReductionOutputClaims {
        ram_ra: OpeningClaim {
            point: ram_output_points.ram_ra,
            value: claims.ram_ra_claim_reduction.ram_ra,
        },
    };
    let ram_output = ram_relation.expected_output(&ram_inputs, &ram_outputs)?;
    let ram_opening_point = ram_outputs.ram_ra.point;

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;
    let registers_output_points =
        registers_relation.derive_opening_points(registers_point, &registers_inputs)?;
    let registers_outputs = RegistersValEvaluationOutputClaims {
        rd_inc: OpeningClaim {
            point: registers_output_points.rd_inc,
            value: claims.registers_val_evaluation.rd_inc,
        },
        rd_wa: OpeningClaim {
            point: registers_output_points.rd_wa,
            value: claims.registers_val_evaluation.rd_wa,
        },
    };
    let registers_output =
        registers_relation.expected_output(&registers_inputs, &registers_outputs)?;
    let registers_opening_point = registers_outputs.rd_inc.point;

    let expected_outputs = Stage5ExpectedOutputClaims {
        instruction_read_raf: instruction_output,
        ram_ra_claim_reduction: ram_output,
        registers_val_evaluation: registers_output,
    };
    let expected_final_claim =
        stage5_expected_final_claim(batch.batching_coefficients.as_slice(), &expected_outputs)?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::InstructionReadRaf,
        });
    }

    claims.append_to_transcript(transcript);

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
        },
    }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InputClaims<F: Field> {
    pub instruction_read_raf: F,
    pub ram_ra_claim_reduction: F,
    pub registers_val_evaluation: F,
}

pub struct Stage5InputClaimRequest<'a, F: Field> {
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub trace_dimensions: TraceDimensions,
    pub instruction_read_raf_dimensions: instruction::InstructionReadRafDimensions,
    pub ram_log_k: usize,
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

pub struct Stage5InstructionReadRafDependencyRequest<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub stage2: &'a Stage2ClearOutput<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ExpectedOutputClaims<F: Field> {
    pub instruction_read_raf: F,
    pub ram_ra_claim_reduction: F,
    pub registers_val_evaluation: F,
}

pub struct Stage5ExpectedOutputRequest<'a, F: Field> {
    pub instruction_read_raf_dimensions: instruction::InstructionReadRafDimensions,
    pub ram_log_k: usize,
    pub instruction_gamma: F,
    pub ram_gamma: F,
    pub instruction_fixed_cycle_point: &'a [F],
    pub instruction_r_address: &'a [F],
    pub instruction_r_cycle: &'a [F],
    pub ram_raf_fixed_cycle_point: &'a [F],
    pub ram_read_write_fixed_cycle_point: &'a [F],
    pub ram_val_check_fixed_cycle_point: &'a [F],
    pub ram_ra_claim_reduction_opening_point: &'a [F],
    pub registers_fixed_cycle_point: &'a [F],
    pub registers_val_evaluation_opening_point: &'a [F],
    pub claims: &'a Stage5OutputClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InstructionOpeningPoints<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub lookup_table_flag_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub instruction_raf_flag_opening_point: Vec<F>,
}

pub struct Stage5ValueOpeningPointRequest<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub ram_log_k: usize,
    pub ram_ra_claim_reduction_sumcheck_point: &'a [F],
    pub registers_val_evaluation_sumcheck_point: &'a [F],
    pub ram_raf_opening_point: &'a [F],
    pub ram_read_write_opening_point: &'a [F],
    pub ram_val_check_opening_point: &'a [F],
    pub registers_read_write_opening_point: &'a [F],
}

pub struct Stage5ValueOpeningPoints<F: Field> {
    pub ram_ra_claim_reduction_sumcheck_point: Vec<F>,
    pub ram_ra_claim_reduction_opening_point: Vec<F>,
    pub ram_raf_fixed_cycle_point: Vec<F>,
    pub ram_read_write_fixed_cycle_point: Vec<F>,
    pub ram_val_check_fixed_cycle_point: Vec<F>,
    pub registers_val_evaluation_sumcheck_point: Vec<F>,
    pub registers_val_evaluation_opening_point: Vec<F>,
    pub registers_fixed_cycle_point: Vec<F>,
}

pub struct Stage5DependencyOpeningPointRequest<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub ram_log_k: usize,
    pub ram_raf_opening_point: &'a [F],
    pub ram_read_write_opening_point: &'a [F],
    pub ram_val_check_opening_point: &'a [F],
    pub registers_read_write_opening_point: &'a [F],
}

pub struct Stage5DependencyOpeningPoints<'a, F: Field> {
    pub ram_address_point: &'a [F],
    pub ram_raf_fixed_cycle_point: &'a [F],
    pub ram_read_write_fixed_cycle_point: &'a [F],
    pub ram_val_check_fixed_cycle_point: &'a [F],
    pub register_address_point: &'a [F],
    pub register_fixed_cycle_point: &'a [F],
}

pub fn stage5_input_claims<F: Field>(
    request: Stage5InputClaimRequest<'_, F>,
) -> Result<Stage5InputClaims<F>, VerifierError> {
    let instruction_relation = InstructionReadRaf::new(
        request.instruction_read_raf_dimensions,
        request.instruction_gamma,
    );
    let ram_relation = RamRaClaimReduction::new(
        request.trace_dimensions,
        request.ram_log_k,
        request.ram_gamma,
    );
    let registers_relation = RegistersValEvaluation::new(request.trace_dimensions);

    Ok(Stage5InputClaims {
        instruction_read_raf: instruction_relation.input_claim(
            &InstructionReadRafInputClaims::from_upstream(request.stage2),
        )?,
        ram_ra_claim_reduction: ram_relation.input_claim(
            &RamRaClaimReductionInputClaims::from_upstream(request.stage2, request.stage4),
        )?,
        registers_val_evaluation: registers_relation.input_claim(
            &RegistersValEvaluationInputClaims::from_upstream(request.stage4),
        )?,
    })
}

pub fn stage5_expected_output_claims<F: Field>(
    request: Stage5ExpectedOutputRequest<'_, F>,
) -> Result<Stage5ExpectedOutputClaims<F>, VerifierError> {
    let log_t = request.instruction_read_raf_dimensions.log_t();
    let instruction_read_raf = expected_instruction_read_raf_output(&request)?;
    let ram_ra_claim_reduction = expected_ram_ra_claim_reduction_output(&request, log_t)?;
    let registers_val_evaluation = expected_registers_val_evaluation_output(&request, log_t)?;

    Ok(Stage5ExpectedOutputClaims {
        instruction_read_raf,
        ram_ra_claim_reduction,
        registers_val_evaluation,
    })
}

pub fn stage5_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage5ExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    let [instruction_coefficient, ram_coefficient, registers_coefficient] = coefficients else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: "Stage 5 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(
        *instruction_coefficient * expected_outputs.instruction_read_raf
            + *ram_coefficient * expected_outputs.ram_ra_claim_reduction
            + *registers_coefficient * expected_outputs.registers_val_evaluation,
    )
}

pub fn stage5_instruction_read_raf_dependencies<F: Field>(
    request: Stage5InstructionReadRafDependencyRequest<'_, F>,
) -> Result<(), VerifierError> {
    let expected_trace_vars = request.trace_dimensions.log_t();
    let product_point = request.stage2.output_claims.product_remainder_point();
    let reduced_point = request
        .stage2
        .output_claims
        .instruction_claim_reduction_point();
    for (label, point) in [
        ("product remainder", product_point),
        ("instruction claim reduction", reduced_point),
    ] {
        if point.len() != expected_trace_vars {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: format!(
                    "Stage 5 {label} opening point has {} variables, expected {expected_trace_vars}",
                    point.len()
                ),
            });
        }
    }

    let [(lookup_output_reduced, lookup_output_product)] =
        instruction::read_raf_consistency_openings();
    if product_point != reduced_point {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }

    let product_lookup_output = request
        .stage2
        .output_claims
        .product_remainder
        .lookup_output
        .value;
    let reduced_lookup_output = request
        .stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .as_ref()
        .map_or(product_lookup_output, |claim| claim.value);
    if reduced_lookup_output != product_lookup_output {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }

    Ok(())
}

pub fn stage5_instruction_opening_points<F: Field>(
    dimensions: instruction::InstructionReadRafDimensions,
    sumcheck_point: &[F],
) -> Result<Stage5InstructionOpeningPoints<F>, VerifierError> {
    let instruction_layout = dimensions.address_layout().map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        }
    })?;
    let opening_point = dimensions.opening_point(sumcheck_point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        }
    })?;
    let instruction_ra_opening_points = opening_point
        .r_address
        .chunks(instruction_layout.virtual_ra_chunk_bits())
        .map(|r_address_chunk| [r_address_chunk, opening_point.r_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    Ok(Stage5InstructionOpeningPoints {
        sumcheck_point: sumcheck_point.to_vec(),
        r_address: opening_point.r_address,
        lookup_table_flag_opening_point: opening_point.r_cycle.clone(),
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point: opening_point.r_cycle.clone(),
        r_cycle: opening_point.r_cycle,
        full_opening_point: opening_point.opening_point,
    })
}

pub fn stage5_dependency_opening_points<F: Field>(
    request: Stage5DependencyOpeningPointRequest<'_, F>,
) -> Result<Stage5DependencyOpeningPoints<'_, F>, VerifierError> {
    let log_t = request.trace_dimensions.log_t();
    let ram_expected_len = request.ram_log_k + log_t;
    for (label, opening_point) in [
        ("RAM RAF evaluation", request.ram_raf_opening_point),
        ("RAM read-write", request.ram_read_write_opening_point),
        ("RAM value check", request.ram_val_check_opening_point),
    ] {
        if opening_point.len() != ram_expected_len {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: format!(
                    "{label} opening point length mismatch: expected {ram_expected_len}, got {}",
                    opening_point.len()
                ),
            });
        }
    }

    let [ram_ra_raf, ram_ra_read_write, ram_ra_val_check] =
        ram::ra_claim_reduction_input_openings();
    let (ram_raf_address, ram_raf_fixed_cycle_point) =
        request.ram_raf_opening_point.split_at(request.ram_log_k);
    let (ram_read_write_address, ram_read_write_fixed_cycle_point) = request
        .ram_read_write_opening_point
        .split_at(request.ram_log_k);
    let (ram_val_check_address, ram_val_check_fixed_cycle_point) = request
        .ram_val_check_opening_point
        .split_at(request.ram_log_k);
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

    let register_expected_len = REGISTER_ADDRESS_BITS + log_t;
    if request.registers_read_write_opening_point.len() != register_expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: format!(
                "register read-write opening point length mismatch: expected {register_expected_len}, got {}",
                request.registers_read_write_opening_point.len()
            ),
        });
    }
    let (register_address_point, register_fixed_cycle_point) = request
        .registers_read_write_opening_point
        .split_at(REGISTER_ADDRESS_BITS);

    Ok(Stage5DependencyOpeningPoints {
        ram_address_point: ram_read_write_address,
        ram_raf_fixed_cycle_point,
        ram_read_write_fixed_cycle_point,
        ram_val_check_fixed_cycle_point,
        register_address_point,
        register_fixed_cycle_point,
    })
}

pub fn stage5_value_opening_points<F: Field>(
    request: Stage5ValueOpeningPointRequest<'_, F>,
) -> Result<Stage5ValueOpeningPoints<F>, VerifierError> {
    let dependency_points =
        stage5_dependency_opening_points(Stage5DependencyOpeningPointRequest {
            trace_dimensions: request.trace_dimensions,
            ram_log_k: request.ram_log_k,
            ram_raf_opening_point: request.ram_raf_opening_point,
            ram_read_write_opening_point: request.ram_read_write_opening_point,
            ram_val_check_opening_point: request.ram_val_check_opening_point,
            registers_read_write_opening_point: request.registers_read_write_opening_point,
        })?;
    let ram_cycle = request
        .trace_dimensions
        .cycle_opening_point(request.ram_ra_claim_reduction_sumcheck_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_ra_claim_reduction_opening_point =
        [dependency_points.ram_address_point, ram_cycle.as_slice()].concat();

    let registers_cycle = request
        .trace_dimensions
        .cycle_opening_point(request.registers_val_evaluation_sumcheck_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;
    let registers_val_evaluation_opening_point = [
        dependency_points.register_address_point,
        registers_cycle.as_slice(),
    ]
    .concat();

    Ok(Stage5ValueOpeningPoints {
        ram_ra_claim_reduction_sumcheck_point: request
            .ram_ra_claim_reduction_sumcheck_point
            .to_vec(),
        ram_ra_claim_reduction_opening_point,
        ram_raf_fixed_cycle_point: dependency_points.ram_raf_fixed_cycle_point.to_vec(),
        ram_read_write_fixed_cycle_point: dependency_points
            .ram_read_write_fixed_cycle_point
            .to_vec(),
        ram_val_check_fixed_cycle_point: dependency_points.ram_val_check_fixed_cycle_point.to_vec(),
        registers_val_evaluation_sumcheck_point: request
            .registers_val_evaluation_sumcheck_point
            .to_vec(),
        registers_val_evaluation_opening_point,
        registers_fixed_cycle_point: dependency_points.register_fixed_cycle_point.to_vec(),
    })
}

fn expected_instruction_read_raf_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
) -> Result<F, VerifierError> {
    let instruction_address_bits = request
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    if request.instruction_r_address.len() != instruction_address_bits {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction output address point has {} challenges, expected {instruction_address_bits}",
                request.instruction_r_address.len()
            ),
        });
    }
    let log_t = request.instruction_read_raf_dimensions.log_t();
    if request.instruction_r_cycle.len() != log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction output cycle point has {} challenges, expected {log_t}",
                request.instruction_r_cycle.len()
            ),
        });
    }
    let instruction_claims = &request.claims.instruction_read_raf;
    if instruction_claims.lookup_table_flags.len() != LookupTableKind::<RISCV_XLEN>::COUNT {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction table flag claim count is {}, expected {}",
                instruction_claims.lookup_table_flags.len(),
                LookupTableKind::<RISCV_XLEN>::COUNT
            ),
        });
    }
    if instruction_claims.instruction_ra.len()
        != request
            .instruction_read_raf_dimensions
            .num_virtual_ra_polys()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction virtual RA claim count is {}, expected {}",
                instruction_claims.instruction_ra.len(),
                request
                    .instruction_read_raf_dimensions
                    .num_virtual_ra_polys()
            ),
        });
    }

    let chunk_size = instruction_address_bits / instruction_claims.instruction_ra.len();
    let relation = InstructionReadRaf::new(
        request.instruction_read_raf_dimensions,
        request.instruction_gamma,
    );
    // Only `lookup_output`'s point is read (it carries the shared claim-reduction
    // opening point that `output_challenge` reduces against).
    let inputs = InstructionReadRafInputClaims {
        lookup_output: request.instruction_fixed_cycle_point.to_vec(),
        left_lookup_operand: request.instruction_fixed_cycle_point.to_vec(),
        right_lookup_operand: request.instruction_fixed_cycle_point.to_vec(),
    };
    let outputs = InstructionReadRafOutputClaims {
        lookup_table_flags: instruction_claims
            .lookup_table_flags
            .iter()
            .map(|&value| OpeningClaim {
                point: request.instruction_r_cycle.to_vec(),
                value,
            })
            .collect(),
        instruction_ra: instruction_claims
            .instruction_ra
            .iter()
            .zip(request.instruction_r_address.chunks(chunk_size))
            .map(|(&value, chunk)| OpeningClaim {
                point: [chunk, request.instruction_r_cycle].concat(),
                value,
            })
            .collect(),
        instruction_raf_flag: OpeningClaim {
            point: request.instruction_r_cycle.to_vec(),
            value: instruction_claims.instruction_raf_flag,
        },
    };
    relation.expected_output(&inputs, &outputs)
}

fn expected_ram_ra_claim_reduction_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
    log_t: usize,
) -> Result<F, VerifierError> {
    let opening_point = request.ram_ra_claim_reduction_opening_point;
    let expected_len = request.ram_log_k + log_t;
    if opening_point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: format!(
                "Stage 5 RAM RA claim-reduction opening point has {} variables, expected {expected_len}",
                opening_point.len()
            ),
        });
    }
    let (ram_address_point, _r_cycle) = opening_point.split_at(request.ram_log_k);
    let input_point = |fixed_cycle: &[F]| [ram_address_point, fixed_cycle].concat();
    let inputs = RamRaClaimReductionInputClaims {
        raf: input_point(request.ram_raf_fixed_cycle_point),
        read_write: input_point(request.ram_read_write_fixed_cycle_point),
        val_check: input_point(request.ram_val_check_fixed_cycle_point),
    };
    let outputs = RamRaClaimReductionOutputClaims {
        ram_ra: OpeningClaim {
            point: opening_point.to_vec(),
            value: request.claims.ram_ra_claim_reduction.ram_ra,
        },
    };
    let relation = RamRaClaimReduction::new(
        TraceDimensions::new(log_t),
        request.ram_log_k,
        request.ram_gamma,
    );
    relation.expected_output(&inputs, &outputs)
}

fn expected_registers_val_evaluation_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
    log_t: usize,
) -> Result<F, VerifierError> {
    let expected_len = REGISTER_ADDRESS_BITS + log_t;
    if request.registers_val_evaluation_opening_point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: format!(
                "Stage 5 register value-evaluation opening point has {} variables, expected {expected_len}",
                request.registers_val_evaluation_opening_point.len()
            ),
        });
    }
    let opening_point = request.registers_val_evaluation_opening_point;
    let (register_address, _r_cycle) = opening_point.split_at(REGISTER_ADDRESS_BITS);
    let relation = RegistersValEvaluation::new(TraceDimensions::new(log_t));
    let inputs = RegistersValEvaluationInputClaims {
        registers_val: [register_address, request.registers_fixed_cycle_point].concat(),
    };
    let registers_claims = &request.claims.registers_val_evaluation;
    let outputs = RegistersValEvaluationOutputClaims {
        rd_inc: OpeningClaim {
            point: opening_point.to_vec(),
            value: registers_claims.rd_inc,
        },
        rd_wa: OpeningClaim {
            point: opening_point.to_vec(),
            value: registers_claims.rd_wa,
        },
    };
    relation.expected_output(&inputs, &outputs)
}
