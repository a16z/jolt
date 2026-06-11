#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::registers as field_registers;
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::{program_image, registers as registers_claim_reduction},
        dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram,
        ram::{RamValCheckInit, RamValCheckInitContribution as FormulaInitContribution},
        registers,
    },
    JoltAdviceKind, JoltChallengeId, JoltRelationId, JoltSumcheckDomain, RamValCheckChallenge,
    RegistersReadWriteChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{block_selector_mle_msb, sparse_segments_mle_msb, try_eq_mle, LtPolynomial};
use jolt_program::preprocess::PublicInitialRam;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::{LabelWithCount, Transcript};
use num_traits::One;

use super::{
    inputs::{Deps, Stage4Claims},
    outputs::{
        RamValCheckInitialEvaluation, Stage4ClearOutput, Stage4Output, Stage4PublicOutput,
        Stage4ZkOutput, VerifiedRamValCheckAdviceContribution,
        VerifiedRamValCheckProgramImageContribution, VerifiedStage4Batch, VerifiedStage4Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::zk::committed,
    verifier::CheckedInputs, VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage4BatchInputClaims<F: Field> {
    registers_read_write: F,
    #[cfg(feature = "field-inline")]
    field_registers_read_write: F,
    ram_val_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage4BatchExpectedOutputClaims<F: Field> {
    registers_read_write: F,
    #[cfg(feature = "field-inline")]
    field_registers_read_write: F,
    ram_val_check: F,
}

const STAGE4_BATCH_BASE_OUTPUT_CLAIMS: usize = 7;
#[cfg(feature = "field-inline")]
const STAGE4_BATCH_FIELD_INLINE_OUTPUT_CLAIMS: usize = 5;
#[cfg(not(feature = "field-inline"))]
const STAGE4_BATCH_FIELD_INLINE_OUTPUT_CLAIMS: usize = 0;

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage4Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let register_dimensions = proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    #[cfg(feature = "field-inline")]
    let field_register_dimensions = proof.protocol.field_inline.read_write_dimensions(log_t);

    let registers_claims = registers::read_write_checking::<PCS::Field>(register_dimensions);
    #[cfg(feature = "field-inline")]
    let field_registers_claims =
        field_registers::read_write_checking::<PCS::Field>(field_register_dimensions);
    if registers_claims.sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: registers_claims.id,
            degree: registers_claims.sumcheck.degree,
        });
    }
    if !matches!(
        registers_claims.sumcheck.domain,
        JoltSumcheckDomain::BooleanHypercube
    ) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
            stage: registers_claims.id,
        });
    }
    #[cfg(feature = "field-inline")]
    {
        if field_registers_claims.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: JoltRelationId::RegistersReadWriteChecking,
                degree: field_registers_claims.sumcheck.degree,
            });
        }
    }

    let registers_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_registers_gamma = transcript.challenge_scalar();

    let (ram_read_write_opening_point, ram_output_check_opening_point) = match deps {
        Deps::Clear { stage2, .. } => (
            &stage2.batch.ram_read_write.opening_point,
            &stage2.batch.ram_output_check.opening_point,
        ),
        Deps::Zk { stage2, .. } => (
            &stage2.ram_val_check_inputs.ram_read_write_opening_point,
            &stage2.ram_val_check_inputs.ram_output_check_opening_point,
        ),
    };
    if ram_read_write_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM read-write opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_read_write_opening_point.len()
            ),
        });
    }
    let (r_address, r_cycle) = ram_read_write_opening_point.split_at(log_k);
    if ram_output_check_opening_point != r_address {
        let [ram_val, ram_val_final] = ram::val_check_input_openings();
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamValCheck,
            left: ram_val,
            right: ram_val_final,
        });
    }

    let ram_val_check_public_eval =
        public_initial_ram_evaluation(checked, preprocessing, r_address)?;

    append_ram_val_check_gamma_domain_separator(transcript);
    let ram_val_check_gamma = transcript.challenge_scalar();

    let ram_val_check_sumcheck = ram::val_check_sumcheck(trace_dimensions);
    if ram_val_check_sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: JoltRelationId::RamValCheck,
            degree: ram_val_check_sumcheck.degree,
        });
    }
    if !matches!(
        ram_val_check_sumcheck.domain,
        JoltSumcheckDomain::BooleanHypercube
    ) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
            stage: JoltRelationId::RamValCheck,
        });
    }

    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage4PublicOutput {
            challenges,
            batching_coefficients,
            registers_gamma,
            #[cfg(feature = "field-inline")]
            field_registers_gamma,
            ram_val_check_gamma,
        };

    if checked.zk {
        let Deps::Zk { .. } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage3" });
        };
        let mut statements = vec![SumcheckStatement::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
        )];
        #[cfg(feature = "field-inline")]
        statements.push(SumcheckStatement::new(
            field_registers_claims.sumcheck.rounds,
            field_registers_claims.sumcheck.degree,
        ));
        statements.push(SumcheckStatement::new(
            ram_val_check_sumcheck.rounds,
            ram_val_check_sumcheck.degree,
        ));
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage4_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage4_sumcheck_proof,
                proof_label: "stage4_sumcheck_proof",
                output_claim_count: stage4_committed_output_claims(checked, proof),
                stage: JoltRelationId::RegistersReadWriteChecking,
            })?;

        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        let registers_opening_point = register_dimensions
            .read_write_opening_point(&registers_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        #[cfg(feature = "field-inline")]
        let field_registers_read_write_opening_point = {
            let field_registers_point = consistency
                .try_instance_point(field_registers_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RegistersReadWriteChecking,
                    reason: error.to_string(),
                })?;
            field_register_dimensions
                .read_write_opening_point(&field_registers_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RegistersReadWriteChecking,
                    reason: error.to_string(),
                })?
                .opening_point
        };
        let ram_val_point = consistency
            .try_instance_point(ram_val_check_sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamValCheck,
                reason: error.to_string(),
            })?;
        let r_cycle_prime = ram_val_point.iter().rev().copied().collect::<Vec<_>>();
        if r_cycle_prime.len() != r_cycle.len() {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: format!(
                    "RAM value cycle point length mismatch: expected {}, got {}",
                    r_cycle.len(),
                    r_cycle_prime.len()
                ),
            });
        }
        let ram_val_opening_point = [r_address, r_cycle_prime.as_slice()].concat();

        return Ok(Stage4Output::Zk(Stage4ZkOutput {
            public: public(
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
            ram_val_check_public_eval,
            registers_read_write_opening_point: registers_opening_point.opening_point,
            #[cfg(feature = "field-inline")]
            field_registers_read_write_opening_point,
            ram_val_check_opening_point: ram_val_opening_point,
        }));
    }

    let Deps::Clear { stage2, stage3, .. } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage3" });
    };
    let claims = &proof.clear_claims()?.stage4;
    let ram_val_check_init = ram_val_check_initial_evaluation(
        checked,
        proof,
        claims,
        r_address,
        ram_val_check_public_eval,
    )?;

    let ram_val_check_claims = ram::val_check::<PCS::Field>(
        trace_dimensions,
        RamValCheckInit::decomposed(
            ram_val_check_init.public_eval,
            ram_val_check_init
                .program_image_contribution
                .iter()
                .map(|_| FormulaInitContribution::program_image(-PCS::Field::one()))
                .chain(
                    ram_val_check_init
                        .advice_contributions
                        .iter()
                        .map(|contribution| {
                            let neg_selector = -contribution.selector;
                            match contribution.kind {
                                JoltAdviceKind::Trusted => {
                                    FormulaInitContribution::trusted(neg_selector)
                                }
                                JoltAdviceKind::Untrusted => {
                                    FormulaInitContribution::untrusted(neg_selector)
                                }
                            }
                        }),
                ),
        ),
    );

    let [_right_operand_is_rs2, rs2_value_instruction, _right_operand_is_imm, _imm, _left_operand_is_rs1, rs1_value_instruction, _left_operand_is_pc, _unexpanded_pc] =
        instruction::input_virtualization_output_openings();
    let [_rd_write_value_reduced, rs1_value_reduced, rs2_value_reduced] =
        registers_claim_reduction::claim_reduction_output_openings();
    if stage3.output_claims.registers_claim_reduction.rs1_value
        != stage3.output_claims.instruction_input.rs1_value
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
            left: rs1_value_reduced,
            right: rs1_value_instruction,
        });
    }
    if stage3.output_claims.registers_claim_reduction.rs2_value
        != stage3.output_claims.instruction_input.rs2_value
    {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
            left: rs2_value_reduced,
            right: rs2_value_instruction,
        });
    }

    let [rd_write_value, rs1_value, rs2_value] = registers::read_write_checking_input_openings();
    let [ram_val, ram_val_final] = ram::val_check_input_openings();
    let input_claims = Stage4BatchInputClaims {
        registers_read_write: registers_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == rd_write_value => Ok(stage3
                    .output_claims
                    .registers_claim_reduction
                    .rd_write_value),
                id if id == rs1_value => {
                    Ok(stage3.output_claims.registers_claim_reduction.rs1_value)
                }
                id if id == rs2_value => {
                    Ok(stage3.output_claims.registers_claim_reduction.rs2_value)
                }
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => {
                    Ok(registers_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        #[cfg(feature = "field-inline")]
        field_registers_read_write: {
            let field_values = &stage2.output_claims.field_inline.product;
            field_values.field_rd_value
                + field_registers_gamma * field_values.field_rs1_value
                + field_registers_gamma * field_registers_gamma * field_values.field_rs2_value
        },
        ram_val_check: ram_val_check_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_val => Ok(stage2.output_claims.ram_read_write.val),
                id if id == ram_val_final => Ok(stage2.output_claims.ram_output_check),
                id if id == program_image::ram_val_check_contribution_opening() => claims
                    .program_image_contribution
                    .ok_or(VerifierError::MissingOpeningClaim { id }),
                id if id == ram::val_check_advice_opening(JoltAdviceKind::Untrusted) => claims
                    .advice
                    .untrusted
                    .ok_or(VerifierError::MissingOpeningClaim { id }),
                id if id == ram::val_check_advice_opening(JoltAdviceKind::Trusted) => claims
                    .advice
                    .trusted
                    .ok_or(VerifierError::MissingOpeningClaim { id }),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => {
                    Ok(ram_val_check_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
    };

    let mut sumcheck_claims = vec![SumcheckClaim::new(
        registers_claims.sumcheck.rounds,
        registers_claims.sumcheck.degree,
        input_claims.registers_read_write,
    )];
    #[cfg(feature = "field-inline")]
    sumcheck_claims.push(SumcheckClaim::new(
        field_registers_claims.sumcheck.rounds,
        field_registers_claims.sumcheck.degree,
        input_claims.field_registers_read_write,
    ));
    sumcheck_claims.push(SumcheckClaim::new(
        ram_val_check_claims.sumcheck.rounds,
        ram_val_check_claims.sumcheck.degree,
        input_claims.ram_val_check,
    ));
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage4_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::RegistersReadWriteChecking,
        reason: error.to_string(),
    })?;

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
    let registers_opening_point = register_dimensions
        .read_write_opening_point(registers_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
    let eq_cycle = try_eq_mle(
        &stage3.batch.registers_claim_reduction.opening_point,
        &registers_opening_point.r_cycle,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RegistersReadWriteChecking,
        reason: error.to_string(),
    })?;
    let [registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc] =
        registers::read_write_checking_output_openings();
    let registers_output = registers_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == registers_val => Ok(claims.registers_read_write.registers_val),
            id if id == rs1_ra => Ok(claims.registers_read_write.rs1_ra),
            id if id == rs2_ra => Ok(claims.registers_read_write.rs2_ra),
            id if id == rd_wa => Ok(claims.registers_read_write.rd_wa),
            id if id == rd_inc => Ok(claims.registers_read_write.rd_inc),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => {
                Ok(registers_gamma)
            }
            JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::EqCycle) => {
                Ok(eq_cycle)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    #[cfg(feature = "field-inline")]
    let (field_registers_point, field_registers_opening_point, field_registers_output) = {
        let field_registers_point = batch
            .try_instance_point(field_registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        let field_registers_opening_point = field_register_dimensions
            .read_write_opening_point(field_registers_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        let eq_cycle = try_eq_mle(
            &stage2.batch.field_registers_claim_reduction.opening_point,
            &field_registers_opening_point.r_cycle,
        )
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
        let field_claims = &claims.field_inline.field_registers_read_write;
        let output = eq_cycle
            * (field_claims.field_rd_wa
                * (field_claims.field_rd_inc + field_claims.field_registers_val)
                + field_registers_gamma
                    * field_claims.field_rs1_ra
                    * field_claims.field_registers_val
                + field_registers_gamma
                    * field_registers_gamma
                    * field_claims.field_rs2_ra
                    * field_claims.field_registers_val);
        (
            field_registers_point.to_vec(),
            field_registers_opening_point.opening_point,
            output,
        )
    };

    let ram_val_point = batch
        .try_instance_point(ram_val_check_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamValCheck,
            reason: error.to_string(),
        })?;
    let r_cycle_prime = ram_val_point.iter().rev().copied().collect::<Vec<_>>();
    if r_cycle_prime.len() != r_cycle.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM value cycle point length mismatch: expected {}, got {}",
                r_cycle.len(),
                r_cycle_prime.len()
            ),
        });
    }
    let ram_val_opening_point = [r_address, r_cycle_prime.as_slice()].concat();
    let lt_cycle = LtPolynomial::evaluate(&r_cycle_prime, r_cycle);
    let [ram_ra, ram_inc] = ram::val_check_output_openings();
    let ram_val_output = ram_val_check_claims.output.expression().try_evaluate(
        |id| match *id {
            id if id == ram_ra => Ok(claims.ram_val_check.ram_ra),
            id if id == ram_inc => Ok(claims.ram_val_check.ram_inc),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| match id {
            JoltChallengeId::RamValCheck(RamValCheckChallenge::LtCyclePlusGamma) => {
                Ok(lt_cycle + ram_val_check_gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let expected_outputs = Stage4BatchExpectedOutputClaims {
        registers_read_write: registers_output,
        #[cfg(feature = "field-inline")]
        field_registers_read_write: field_registers_output,
        ram_val_check: ram_val_output,
    };
    let coefficients = batch.batching_coefficients.as_slice();
    #[cfg(not(feature = "field-inline"))]
    let expected_final_claim = {
        let [registers_coefficient, ram_val_coefficient] = coefficients else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: "Stage 4 batch verifier returned the wrong number of coefficients"
                    .to_string(),
            });
        };
        *registers_coefficient * expected_outputs.registers_read_write
            + *ram_val_coefficient * expected_outputs.ram_val_check
    };
    #[cfg(feature = "field-inline")]
    let expected_final_claim = {
        let [registers_coefficient, field_registers_coefficient, ram_val_coefficient] =
            coefficients
        else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: "Stage 4 batch verifier returned the wrong number of coefficients"
                    .to_string(),
            });
        };
        *registers_coefficient * expected_outputs.registers_read_write
            + *field_registers_coefficient * expected_outputs.field_registers_read_write
            + *ram_val_coefficient * expected_outputs.ram_val_check
    };
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
        });
    }

    append_stage4_opening_claims(
        transcript,
        proof.untrusted_advice_commitment.is_some(),
        checked.trusted_advice_commitment_present,
        checked.precommitted.program_image.is_some(),
        claims,
    )?;

    Ok(Stage4Output::Clear(Stage4ClearOutput {
        public: public(
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        ram_val_check_init,
        batch: VerifiedStage4Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.clone(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            registers_read_write: VerifiedStage4Sumcheck {
                input_claim: input_claims.registers_read_write,
                sumcheck_point: registers_point.to_vec(),
                opening_point: registers_opening_point.opening_point,
                expected_output_claim: expected_outputs.registers_read_write,
            },
            #[cfg(feature = "field-inline")]
            field_registers_read_write: VerifiedStage4Sumcheck {
                input_claim: input_claims.field_registers_read_write,
                sumcheck_point: field_registers_point,
                opening_point: field_registers_opening_point,
                expected_output_claim: expected_outputs.field_registers_read_write,
            },
            ram_val_check: VerifiedStage4Sumcheck {
                input_claim: input_claims.ram_val_check,
                sumcheck_point: ram_val_point.to_vec(),
                opening_point: ram_val_opening_point,
                expected_output_claim: expected_outputs.ram_val_check,
            },
        },
    }))
}

fn ram_val_check_initial_evaluation<PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    claims: &Stage4Claims<PCS::Field>,
    r_address: &[PCS::Field],
    public_eval: PCS::Field,
) -> Result<RamValCheckInitialEvaluation<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let mut full_eval = public_eval;
    let program_image_contribution = collect_program_image_contribution(
        checked.precommitted.program_image.is_some(),
        claims.program_image_contribution,
        r_address,
        &mut full_eval,
    )?;
    let mut advice_contributions = Vec::new();
    let untrusted_present = proof.untrusted_advice_commitment.is_some();
    collect_advice_contribution(
        JoltAdviceKind::Untrusted,
        untrusted_present,
        claims.advice.untrusted,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;
    collect_advice_contribution(
        JoltAdviceKind::Trusted,
        checked.trusted_advice_commitment_present,
        claims.advice.trusted,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;

    Ok(RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution,
        advice_contributions,
        full_eval,
    })
}

fn collect_program_image_contribution<F: Field>(
    committed_program: bool,
    opening_claim: Option<F>,
    r_address: &[F],
    full_eval: &mut F,
) -> Result<Option<VerifiedRamValCheckProgramImageContribution<F>>, VerifierError> {
    let opening = program_image::ram_val_check_contribution_opening();
    if !committed_program {
        if opening_claim.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
        }
        return Ok(None);
    }

    let opening_claim = opening_claim.ok_or(VerifierError::MissingOpeningClaim { id: opening })?;
    *full_eval += opening_claim;
    Ok(Some(VerifiedRamValCheckProgramImageContribution {
        opening_claim,
        opening_point: r_address.to_vec(),
    }))
}

fn public_initial_ram_evaluation<PCS, VC>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    r_address: &[PCS::Field],
) -> Result<PCS::Field, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    // In committed program mode the image words are bound via the staged
    // `ProgramImageInitContributionRw` opening, so only inputs are public here.
    let public_initial_ram = match preprocessing.program.as_full() {
        Some(full) => PublicInitialRam::new(&full.ram, &checked.public_io),
        None => PublicInitialRam::inputs_only(&checked.public_io),
    }
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamValCheck,
        reason: error.to_string(),
    })?;
    for segment in &public_initial_ram.segments {
        let end = segment.start_index + segment.words.len();
        if end > checked.ram_K {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: format!(
                    "public initial RAM segment [{}, {}) exceeds RAM domain {}",
                    segment.start_index, end, checked.ram_K
                ),
            });
        }
    }

    Ok(sparse_segments_mle_msb(
        public_initial_ram
            .segments
            .iter()
            .map(|segment| (segment.start_index, segment.words.as_slice())),
        r_address,
    ))
}

fn stage4_committed_output_claims<PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
) -> usize
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    STAGE4_BATCH_BASE_OUTPUT_CLAIMS
        + STAGE4_BATCH_FIELD_INLINE_OUTPUT_CLAIMS
        + usize::from(proof.untrusted_advice_commitment.is_some())
        + usize::from(checked.trusted_advice_commitment_present)
        + usize::from(checked.precommitted.program_image.is_some())
}

fn collect_advice_contribution<F: Field>(
    kind: JoltAdviceKind,
    present: bool,
    opening_claim: Option<F>,
    checked: &CheckedInputs,
    r_address: &[F],
    full_eval: &mut F,
    contributions: &mut Vec<VerifiedRamValCheckAdviceContribution<F>>,
) -> Result<(), VerifierError> {
    let opening = ram::val_check_advice_opening(kind);
    if !present {
        if opening_claim.is_some() {
            return Err(VerifierError::UnexpectedOpeningClaim { id: opening });
        }
        return Ok(());
    }

    let opening_claim = opening_claim.ok_or(VerifierError::MissingOpeningClaim { id: opening })?;
    let layout = &checked.public_io.memory_layout;
    let (start_address, max_size) = match kind {
        JoltAdviceKind::Trusted => (layout.trusted_advice_start, layout.max_trusted_advice_size),
        JoltAdviceKind::Untrusted => (
            layout.untrusted_advice_start,
            layout.max_untrusted_advice_size,
        ),
    };
    if max_size == 0 {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!("{kind:?} advice commitment is present but configured size is zero"),
        });
    }

    let start_index = layout
        .remapped_word_address(start_address)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: error.to_string(),
        })? as usize;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    let selector =
        block_selector_mle_msb(start_index, advice_num_vars, r_address).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: error.to_string(),
            }
        })?;
    let opening_point = r_address[r_address.len() - advice_num_vars..].to_vec();
    *full_eval += selector * opening_claim;
    contributions.push(VerifiedRamValCheckAdviceContribution {
        kind,
        selector,
        opening_claim,
        opening_point,
    });
    Ok(())
}

fn append_stage4_opening_claims<F, T>(
    transcript: &mut T,
    untrusted_advice_commitment_present: bool,
    trusted_advice_commitment_present: bool,
    committed_program: bool,
    claims: &Stage4Claims<F>,
) -> Result<(), VerifierError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if untrusted_advice_commitment_present {
        let id = ram::val_check_advice_opening(JoltAdviceKind::Untrusted);
        let opening_claim = claims
            .advice
            .untrusted
            .ok_or(VerifierError::MissingOpeningClaim { id })?;
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    if trusted_advice_commitment_present {
        let id = ram::val_check_advice_opening(JoltAdviceKind::Trusted);
        let opening_claim = claims
            .advice
            .trusted
            .ok_or(VerifierError::MissingOpeningClaim { id })?;
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    if committed_program {
        let id = program_image::ram_val_check_contribution_opening();
        let opening_claim = claims
            .program_image_contribution
            .ok_or(VerifierError::MissingOpeningClaim { id })?;
        transcript.append_labeled(b"opening_claim", &opening_claim);
    }
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.registers_val);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rs1_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rs2_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rd_wa);
    transcript.append_labeled(b"opening_claim", &claims.registers_read_write.rd_inc);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_read_write;
        transcript.append_labeled(b"opening_claim", &field_claims.field_registers_val);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rs1_ra);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rs2_ra);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_wa);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_inc);
    }
    transcript.append_labeled(b"opening_claim", &claims.ram_val_check.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.ram_val_check.ram_inc);
    Ok(())
}

fn append_ram_val_check_gamma_domain_separator<T: Transcript>(transcript: &mut T) {
    transcript.append(&LabelWithCount(b"ram_val_check_gamma", 0));
    transcript.append_bytes(&[]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "field-inline")]
    use crate::stages::stage4::inputs::{
        FieldInlineStage4Claims, FieldRegistersReadWriteOutputOpeningClaims,
    };
    use crate::stages::stage4::inputs::{
        RamValCheckAdviceOpeningClaims, RamValCheckOutputOpeningClaims,
        RegistersReadWriteOutputOpeningClaims,
    };
    use jolt_field::{CanonicalBytes, FixedByteSize, Fr, FromPrimitiveInt};

    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
        state: [u8; 32],
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(0)
        }

        fn state(&self) -> [u8; 32] {
            self.state
        }
    }

    #[test]
    fn opening_claim_appends_follow_core_order_without_advice() {
        let claims = test_claims();
        let mut transcript = RecordingTranscript::new(b"stage4-openings");

        let result = append_stage4_opening_claims(&mut transcript, false, false, false, &claims);
        assert!(result.is_ok(), "stage 4 openings should append: {result:?}");

        let expected = [
            claims.registers_read_write.registers_val,
            claims.registers_read_write.rs1_ra,
            claims.registers_read_write.rs2_ra,
            claims.registers_read_write.rd_wa,
            claims.registers_read_write.rd_inc,
        ];
        let mut expected = expected.to_vec();
        #[cfg(feature = "field-inline")]
        {
            expected.extend([
                claims
                    .field_inline
                    .field_registers_read_write
                    .field_registers_val,
                claims.field_inline.field_registers_read_write.field_rs1_ra,
                claims.field_inline.field_registers_read_write.field_rs2_ra,
                claims.field_inline.field_registers_read_write.field_rd_wa,
                claims.field_inline.field_registers_read_write.field_rd_inc,
            ]);
        }
        expected.extend([claims.ram_val_check.ram_ra, claims.ram_val_check.ram_inc]);

        assert_opening_claim_payloads(&transcript, &expected);
    }

    #[test]
    fn ram_val_check_gamma_domain_separator_matches_core_empty_bytes_append() {
        let mut transcript = RecordingTranscript::new(b"stage4-gamma");

        append_ram_val_check_gamma_domain_separator(&mut transcript);

        assert_eq!(transcript.chunks.len(), 2);
        let mut packed = vec![0; 32];
        packed[..b"ram_val_check_gamma".len()].copy_from_slice(b"ram_val_check_gamma");
        assert_eq!(transcript.chunks[0], packed);
        assert!(transcript.chunks[1].is_empty());
    }

    fn test_claims() -> Stage4Claims<Fr> {
        Stage4Claims {
            advice: RamValCheckAdviceOpeningClaims {
                untrusted: Some(Fr::from_u64(1)),
                trusted: Some(Fr::from_u64(2)),
            },
            program_image_contribution: None,
            registers_read_write: RegistersReadWriteOutputOpeningClaims {
                registers_val: Fr::from_u64(3),
                rs1_ra: Fr::from_u64(4),
                rs2_ra: Fr::from_u64(5),
                rd_wa: Fr::from_u64(6),
                rd_inc: Fr::from_u64(7),
            },
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineStage4Claims {
                field_registers_read_write: FieldRegistersReadWriteOutputOpeningClaims {
                    field_registers_val: Fr::from_u64(10),
                    field_rs1_ra: Fr::from_u64(11),
                    field_rs2_ra: Fr::from_u64(12),
                    field_rd_wa: Fr::from_u64(13),
                    field_rd_inc: Fr::from_u64(14),
                },
            },
            ram_val_check: RamValCheckOutputOpeningClaims {
                ram_ra: Fr::from_u64(8),
                ram_inc: Fr::from_u64(9),
            },
        }
    }

    fn assert_opening_claim_payloads(transcript: &RecordingTranscript, expected: &[Fr]) {
        assert_eq!(transcript.chunks.len(), expected.len() * 2);
        let label = opening_claim_label();
        for (index, expected_payload) in expected.iter().copied().enumerate() {
            assert_eq!(transcript.chunks[2 * index], label);
            assert_eq!(
                transcript.chunks[2 * index + 1],
                scalar_bytes(expected_payload)
            );
        }
    }

    fn opening_claim_label() -> Vec<u8> {
        let mut label = vec![0; 32];
        label[..b"opening_claim".len()].copy_from_slice(b"opening_claim");
        label
    }

    fn scalar_bytes(value: Fr) -> Vec<u8> {
        let mut bytes = vec![0; Fr::NUM_BYTES];
        value.to_bytes_le(&mut bytes);
        bytes.reverse();
        bytes
    }
}
