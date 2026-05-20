use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction,
        dimensions::{TraceDimensions, PRODUCT_UNISKIP_DOMAIN_SIZE},
        ram::{
            self, RamOutputCheckPublicValues, RamRafEvaluationDimensions,
            RamRafEvaluationPublicValues,
        },
        spartan::{
            product_outer_opening, product_remainder, product_remainder_output_openings,
            product_should_branch_outer_opening, product_should_jump_outer_opening,
            product_uniskip, product_uniskip_opening, SpartanProductDimensions,
            SpartanProductPublicValues,
        },
    },
    InstructionClaimReductionChallenge, JoltChallengeId, JoltPublicId, JoltStageId,
    JoltSumcheckDomain, RamReadWriteChallenge, SpartanProductVirtualizationPublic,
};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    lagrange::{centered_lagrange_evals_array, centered_lagrange_kernel},
    range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle, IdentityPolynomial,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
    UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::{
    inputs::{Deps, Stage2BatchOutputOpeningClaims},
    outputs::{
        Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2RamValCheckInputs,
        Stage2ZkOutput, VerifiedProductUniSkip, VerifiedStage2Batch, VerifiedStage2Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::committed,
    verifier::CheckedInputs, VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage2BatchSumcheckInputClaims<F: Field> {
    ram_read_write: F,
    product_remainder: F,
    instruction_claim_reduction: F,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage2BatchExpectedOutputClaims<F: Field> {
    ram_read_write: F,
    product_remainder: F,
    instruction_claim_reduction: F,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

struct Stage2ZkProductUniSkip<F: Field, C> {
    tau_low: Vec<F>,
    tau_high: F,
    product_uniskip_challenge: F,
    consistency: jolt_sumcheck::CommittedSumcheckConsistency<F, C>,
}

enum Stage2ProductUniSkip<F: Field, C> {
    Clear(VerifiedProductUniSkip<F>),
    Zk(Stage2ZkProductUniSkip<F, C>),
}

struct Stage2ZkBatch<F: Field, C> {
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    ram_read_write_gamma: F,
    instruction_gamma: F,
    output_address_challenges: Vec<F>,
    consistency: jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    ram_val_check_inputs: Stage2RamValCheckInputs<F>,
}

enum Stage2Batch<F: Field, C> {
    Clear {
        verified: VerifiedStage2Batch<F>,
        output_claims: Stage2BatchOutputOpeningClaims<F>,
    },
    Zk(Stage2ZkBatch<F, C>),
}

const PRODUCT_UNISKIP_OUTPUT_CLAIMS: usize = 1;
const STAGE2_BATCH_OUTPUT_CLAIMS: usize = 15;

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage2Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage1" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage1" });
        }
        _ => {}
    }

    let product_uniskip =
        verify_product_uniskip::<PCS, VC, T, ZkProof>(checked, proof, transcript, deps)?;
    let batch = verify_regular_batch::<PCS, VC, T, ZkProof>(
        checked,
        proof,
        transcript,
        &product_uniskip,
        deps,
    )?;

    match (product_uniskip, batch) {
        (
            Stage2ProductUniSkip::Clear(product_uniskip),
            Stage2Batch::Clear {
                verified: batch,
                output_claims,
            },
        ) => {
            let [product_uniskip_challenge] = product_uniskip.sumcheck_point.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::SpartanProductVirtualization,
                    reason: "product uni-skip proof did not reduce to one challenge".to_string(),
                });
            };
            let public = Stage2PublicOutput {
                challenges: batch.sumcheck_point.as_slice().to_vec(),
                batching_coefficients: batch.batching_coefficients.clone(),
                product_uniskip_challenge: *product_uniskip_challenge,
                product_tau_low: product_uniskip.tau_low.clone(),
                product_tau_high: product_uniskip.tau_high,
                ram_read_write_gamma: batch.ram_read_write_gamma,
                instruction_gamma: batch.instruction_gamma,
                output_address_challenges: batch.output_address_challenges.clone(),
            };

            Ok(Stage2Output::Clear(Stage2ClearOutput {
                public,
                output_claims,
                product_uniskip,
                batch,
            }))
        }
        (Stage2ProductUniSkip::Zk(product_uniskip), Stage2Batch::Zk(batch)) => {
            let public = Stage2PublicOutput {
                challenges: batch.challenges,
                batching_coefficients: batch.batching_coefficients,
                product_uniskip_challenge: product_uniskip.product_uniskip_challenge,
                product_tau_low: product_uniskip.tau_low,
                product_tau_high: product_uniskip.tau_high,
                ram_read_write_gamma: batch.ram_read_write_gamma,
                instruction_gamma: batch.instruction_gamma,
                output_address_challenges: batch.output_address_challenges,
            };

            Ok(Stage2Output::Zk(Stage2ZkOutput {
                public,
                product_uniskip_consistency: product_uniskip.consistency,
                batch_consistency: batch.consistency,
                ram_val_check_inputs: batch.ram_val_check_inputs,
            }))
        }
        (Stage2ProductUniSkip::Clear(_), Stage2Batch::Zk(_)) => {
            Err(VerifierError::ExpectedClearProof {
                field: "stage2_sumcheck_proof",
            })
        }
        (Stage2ProductUniSkip::Zk(_), Stage2Batch::Clear { .. }) => {
            Err(VerifierError::ExpectedCommittedProof {
                field: "stage2_sumcheck_proof",
            })
        }
    }
}

fn verify_product_uniskip<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage2ProductUniSkip<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let stage = JoltStageId::SpartanProductVirtualization;
    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = SpartanProductDimensions::from(log_t);
    let stage1_public = match deps {
        Deps::Clear { stage1 } => &stage1.public,
        Deps::Zk { stage1 } => &stage1.public,
    };
    let mut tau_low = stage1_public
        .remainder_challenges
        .get(1..)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: "Stage 1 remainder challenge vector is empty".to_string(),
        })?
        .to_vec();
    if tau_low.len() != log_t {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: format!(
                "Stage 1 remainder challenge tail length mismatch: expected {log_t}, got {}",
                tau_low.len()
            ),
        });
    }
    tau_low.reverse();

    let tau_high = transcript.challenge();
    let uniskip_spec = dimensions.uniskip_sumcheck();
    if uniskip_spec.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage,
            degree: uniskip_spec.degree,
        });
    }
    let JoltSumcheckDomain::CenteredInteger { domain_size } = uniskip_spec.domain else {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: "Stage 2 product uni-skip sumcheck must use the centered-integer domain"
                .to_string(),
        });
    };
    match deps {
        Deps::Clear { stage1 } => {
            let claims = &proof.clear_claims()?.stage2;
            let weights =
                centered_lagrange_evals_array::<PCS::Field, PRODUCT_UNISKIP_DOMAIN_SIZE>(tau_high)
                    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                        stage,
                        reason: error.to_string(),
                    })?;

            let uniskip_claim = claims.product_uniskip_output_claim;
            let product_claims = product_uniskip::<PCS::Field>(dimensions);
            let product_outer = product_outer_opening();
            let product_should_branch = product_should_branch_outer_opening();
            let product_should_jump = product_should_jump_outer_opening();
            let uniskip_input_claim = product_claims.input.expression.try_evaluate(
                |id| match *id {
                    id if id == product_outer => Ok(stage1.outer.product),
                    id if id == product_should_branch => Ok(stage1.outer.should_branch),
                    id if id == product_should_jump => Ok(stage1.outer.should_jump),
                    id => Err(VerifierError::MissingOpeningClaim { id }),
                },
                |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                |id| match id {
                    JoltPublicId::SpartanProductVirtualization(
                        SpartanProductVirtualizationPublic::LagrangeWeight(index),
                    ) => weights
                        .get(*index)
                        .copied()
                        .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
                    _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
                },
            )?;

            let uniskip_reduction = proof
                .stages
                .stage2_uni_skip_first_round_proof
                .verify(
                    &SumcheckClaim::new(
                        uniskip_spec.rounds,
                        uniskip_spec.degree,
                        uniskip_input_claim,
                    ),
                    CenteredIntegerDomain::new(domain_size),
                    UNISKIP_ROUND_TRANSCRIPT_LABEL,
                    transcript,
                )
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: error.to_string(),
                })?;
            if uniskip_reduction.value != uniskip_claim {
                return Err(VerifierError::StageClaimOutputMismatch { stage });
            }

            transcript.append_labeled(b"opening_claim", &uniskip_claim);

            Ok(Stage2ProductUniSkip::Clear(VerifiedProductUniSkip {
                tau_low,
                tau_high,
                input_claim: uniskip_input_claim,
                sumcheck_point: uniskip_reduction.point,
                sumcheck_final_claim: uniskip_reduction.value,
                expected_output_claim: uniskip_claim,
            }))
        }
        Deps::Zk { .. } => {
            let consistency = proof
                .stages
                .stage2_uni_skip_first_round_proof
                .verify_committed_consistency(
                    SumcheckStatement::new(uniskip_spec.rounds, uniskip_spec.degree),
                    transcript,
                )
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: error.to_string(),
                })?;
            committed::require_output_claim_commitments(
                checked,
                &proof.stages.stage2_uni_skip_first_round_proof,
                "stage2_uni_skip_first_round_proof",
                PRODUCT_UNISKIP_OUTPUT_CLAIMS,
                stage,
            )?;
            let [round] = consistency.rounds.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: "product uni-skip committed consistency did not produce one challenge"
                        .to_string(),
                });
            };

            Ok(Stage2ProductUniSkip::Zk(Stage2ZkProductUniSkip {
                tau_low,
                tau_high,
                product_uniskip_challenge: round.challenge,
                consistency,
            }))
        }
    }
}

fn verify_regular_batch<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    product_uniskip: &Stage2ProductUniSkip<PCS::Field, VC::Output>,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage2Batch<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::from(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let ram_read_write_claims = ram::read_write_checking::<PCS::Field>(read_write_dimensions);
    let product_remainder_claims = product_remainder::<PCS::Field>(product_dimensions);
    let instruction_claim_reduction_claims =
        instruction_claim_reduction::claim_reduction::<PCS::Field>(trace_dimensions);
    let ram_raf_evaluation_claims = ram::raf_evaluation::<PCS::Field>(raf_dimensions);
    let ram_output_check_claims = ram::output_check::<PCS::Field>(read_write_dimensions);
    // Stage 2 batches these five sumchecks after the product uni-skip:
    // RAM read-write, Spartan product, instruction reduction, RAM RAF, RAM output.
    let ram_read_write_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let output_address_challenges = (0..log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    for claims in [
        &ram_read_write_claims,
        &product_remainder_claims,
        &instruction_claim_reduction_claims,
        &ram_raf_evaluation_claims,
        &ram_output_check_claims,
    ] {
        if claims.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: claims.id,
                degree: claims.sumcheck.degree,
            });
        }
        if !matches!(claims.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
            return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
                stage: claims.id,
            });
        }
    }

    match (deps, product_uniskip) {
        (Deps::Clear { stage1 }, Stage2ProductUniSkip::Clear(product_uniskip)) => {
            let claims = &proof.clear_claims()?.stage2;
            let [product_uniskip_challenge] = product_uniskip.sumcheck_point.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::SpartanProductVirtualization,
                    reason: "product uni-skip proof did not reduce to one challenge".to_string(),
                });
            };
            let product_uniskip_challenge = *product_uniskip_challenge;

            let [ram_read_value, ram_write_value] = ram::read_write_checking_input_openings();
            let product_uniskip_opening_id = product_uniskip_opening();
            let [instruction_lookup_output_spartan, instruction_left_lookup_operand_spartan, instruction_right_lookup_operand_spartan, instruction_left_instruction_input_spartan, instruction_right_instruction_input_spartan] =
                instruction_claim_reduction::claim_reduction_input_openings();
            let [ram_address_spartan] = ram::raf_evaluation_input_openings();

            let input_claims = Stage2BatchSumcheckInputClaims {
                ram_read_write: ram_read_write_claims.input.expression.try_evaluate(
                    |id| match *id {
                        id if id == ram_read_value => Ok(stage1.outer.ram_read_value),
                        id if id == ram_write_value => Ok(stage1.outer.ram_write_value),
                        id => Err(VerifierError::MissingOpeningClaim { id }),
                    },
                    |id| match id {
                        JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => {
                            Ok(ram_read_write_gamma)
                        }
                        _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    },
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?,
                product_remainder: product_remainder_claims.input.expression.try_evaluate(
                    |id| match *id {
                        id if id == product_uniskip_opening_id => {
                            Ok(claims.product_uniskip_output_claim)
                        }
                        id => Err(VerifierError::MissingOpeningClaim { id }),
                    },
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?,
                instruction_claim_reduction: instruction_claim_reduction_claims
                    .input
                    .expression
                    .try_evaluate(
                        |id| match *id {
                            id if id == instruction_lookup_output_spartan => {
                                Ok(stage1.outer.lookup_output)
                            }
                            id if id == instruction_left_lookup_operand_spartan => {
                                Ok(stage1.outer.left_lookup_operand)
                            }
                            id if id == instruction_right_lookup_operand_spartan => {
                                Ok(stage1.outer.right_lookup_operand)
                            }
                            id if id == instruction_left_instruction_input_spartan => {
                                Ok(stage1.outer.left_instruction_input)
                            }
                            id if id == instruction_right_instruction_input_spartan => {
                                Ok(stage1.outer.right_instruction_input)
                            }
                            id => Err(VerifierError::MissingOpeningClaim { id }),
                        },
                        |id| match id {
                            JoltChallengeId::InstructionClaimReduction(
                                InstructionClaimReductionChallenge::Gamma,
                            ) => Ok(instruction_gamma),
                            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                        },
                        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                    )?,
                ram_raf_evaluation: ram_raf_evaluation_claims.input.expression.try_evaluate(
                    |id| match *id {
                        id if id == ram_address_spartan => Ok(stage1.outer.ram_address),
                        id => Err(VerifierError::MissingOpeningClaim { id }),
                    },
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?,
                ram_output_check: ram_output_check_claims.input.expression.try_evaluate(
                    |id| Err(VerifierError::MissingOpeningClaim { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?,
            };

            // The claim order here must match the output-claim reconstruction below and
            // the transcript appends at the end of the stage.
            let sumcheck_claims = [
                SumcheckClaim::new(
                    ram_read_write_claims.sumcheck.rounds,
                    ram_read_write_claims.sumcheck.degree,
                    input_claims.ram_read_write,
                ),
                SumcheckClaim::new(
                    product_remainder_claims.sumcheck.rounds,
                    product_remainder_claims.sumcheck.degree,
                    input_claims.product_remainder,
                ),
                SumcheckClaim::new(
                    instruction_claim_reduction_claims.sumcheck.rounds,
                    instruction_claim_reduction_claims.sumcheck.degree,
                    input_claims.instruction_claim_reduction,
                ),
                SumcheckClaim::new(
                    ram_raf_evaluation_claims.sumcheck.rounds,
                    ram_raf_evaluation_claims.sumcheck.degree,
                    input_claims.ram_raf_evaluation,
                ),
                SumcheckClaim::new(
                    ram_output_check_claims.sumcheck.rounds,
                    ram_output_check_claims.sumcheck.degree,
                    input_claims.ram_output_check,
                ),
            ];
            let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
                &sumcheck_claims,
                &proof.stages.stage2_sumcheck_proof,
                transcript,
            )
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltStageId::RamReadWriteChecking,
                reason: error.to_string(),
            })?;

            let ram_read_write_point = batch
                .try_instance_point(ram_read_write_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let ram_read_write_opening_point = read_write_dimensions
                .read_write_opening_point(ram_read_write_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let eq_cycle = try_eq_mle(
                &product_uniskip.tau_low,
                &ram_read_write_opening_point.r_cycle,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::RamReadWriteChecking,
                reason: error.to_string(),
            })?;
            let [ram_val, ram_ra, ram_inc] = ram::read_write_checking_output_openings();
            let ram_read_write_output = ram_read_write_claims.output.expression.try_evaluate(
                |id| match *id {
                    id if id == ram_val => Ok(claims.batch_outputs.ram_read_write.val),
                    id if id == ram_ra => Ok(claims.batch_outputs.ram_read_write.ra),
                    id if id == ram_inc => Ok(claims.batch_outputs.ram_read_write.inc),
                    _ => Ok(PCS::Field::from_u64(0)),
                },
                |id| match id {
                    JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => {
                        Ok(ram_read_write_gamma)
                    }
                    JoltChallengeId::RamReadWrite(RamReadWriteChallenge::EqCycle) => Ok(eq_cycle),
                    _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                },
                |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
            )?;

            let product_point = batch
                .try_instance_point(product_remainder_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?;
            let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
            let product_lagrange_weights = centered_lagrange_evals_array::<
                PCS::Field,
                PRODUCT_UNISKIP_DOMAIN_SIZE,
            >(product_uniskip_challenge)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::SpartanProductVirtualization,
                reason: error.to_string(),
            })?;
            let product_tau_high_bound = centered_lagrange_kernel(
                PRODUCT_UNISKIP_DOMAIN_SIZE,
                product_uniskip.tau_high,
                product_uniskip_challenge,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::SpartanProductVirtualization,
                reason: error.to_string(),
            })?;
            let product_tau_low_eq = try_eq_mle(&product_uniskip.tau_low, &product_opening_point)
                .map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::SpartanProductVirtualization,
                    reason: error.to_string(),
                }
            })?;
            let product_publics = SpartanProductPublicValues {
                lagrange_weights: product_lagrange_weights,
                tau_kernel: product_tau_high_bound * product_tau_low_eq,
            };
            let [product_left_instruction_input, product_right_instruction_input, product_jump_flag, product_write_lookup_output_to_rd, product_lookup_output, product_branch_flag, product_next_is_noop, product_virtual_instruction] =
                product_remainder_output_openings();
            let product_remainder_output =
                product_remainder_claims.output.expression.try_evaluate(
                    |id| match *id {
                        id if id == product_left_instruction_input => Ok(claims
                            .batch_outputs
                            .product_remainder
                            .left_instruction_input),
                        id if id == product_right_instruction_input => Ok(claims
                            .batch_outputs
                            .product_remainder
                            .right_instruction_input),
                        id if id == product_jump_flag => {
                            Ok(claims.batch_outputs.product_remainder.jump_flag)
                        }
                        id if id == product_write_lookup_output_to_rd => Ok(claims
                            .batch_outputs
                            .product_remainder
                            .write_lookup_output_to_rd),
                        id if id == product_lookup_output => {
                            Ok(claims.batch_outputs.product_remainder.lookup_output)
                        }
                        id if id == product_branch_flag => {
                            Ok(claims.batch_outputs.product_remainder.branch_flag)
                        }
                        id if id == product_next_is_noop => {
                            Ok(claims.batch_outputs.product_remainder.next_is_noop)
                        }
                        id if id == product_virtual_instruction => {
                            Ok(claims.batch_outputs.product_remainder.virtual_instruction)
                        }
                        _ => Ok(PCS::Field::from_u64(0)),
                    },
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| match id {
                        JoltPublicId::SpartanProductVirtualization(public_id) => product_publics
                            .value(*public_id)
                            .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
                        _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
                    },
                )?;

            let instruction_point = batch
                .try_instance_point(instruction_claim_reduction_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::InstructionClaimReduction,
                    reason: error.to_string(),
                })?;
            let instruction_opening_point =
                instruction_point.iter().rev().copied().collect::<Vec<_>>();
            let eq_spartan = try_eq_mle(&instruction_opening_point, &product_uniskip.tau_low)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::InstructionClaimReduction,
                    reason: error.to_string(),
                })?;
            let product_and_instruction_points_match =
                product_opening_point == instruction_opening_point;
            let [instruction_lookup_output, instruction_left_lookup_operand, instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] =
                instruction_claim_reduction::claim_reduction_output_openings();
            let instruction_claim_reduction_output = instruction_claim_reduction_claims
                .output
                .expression
                .try_evaluate(
                    |id| match *id {
                        id if id == instruction_lookup_output => Ok(claims
                            .batch_outputs
                            .instruction_claim_reduction
                            .lookup_output
                            .unwrap_or_else(|| {
                                if product_and_instruction_points_match {
                                    claims.batch_outputs.product_remainder.lookup_output
                                } else {
                                    PCS::Field::from_u64(0)
                                }
                            })),
                        id if id == instruction_left_lookup_operand => Ok(claims
                            .batch_outputs
                            .instruction_claim_reduction
                            .left_lookup_operand),
                        id if id == instruction_right_lookup_operand => Ok(claims
                            .batch_outputs
                            .instruction_claim_reduction
                            .right_lookup_operand),
                        id if id == instruction_left_instruction_input => Ok(claims
                            .batch_outputs
                            .instruction_claim_reduction
                            .left_instruction_input
                            .unwrap_or_else(|| {
                                if product_and_instruction_points_match {
                                    claims
                                        .batch_outputs
                                        .product_remainder
                                        .left_instruction_input
                                } else {
                                    PCS::Field::from_u64(0)
                                }
                            })),
                        id if id == instruction_right_instruction_input => Ok(claims
                            .batch_outputs
                            .instruction_claim_reduction
                            .right_instruction_input
                            .unwrap_or_else(|| {
                                if product_and_instruction_points_match {
                                    claims
                                        .batch_outputs
                                        .product_remainder
                                        .right_instruction_input
                                } else {
                                    PCS::Field::from_u64(0)
                                }
                            })),
                        _ => Ok(PCS::Field::from_u64(0)),
                    },
                    |id| match id {
                        JoltChallengeId::InstructionClaimReduction(
                            InstructionClaimReductionChallenge::Gamma,
                        ) => Ok(instruction_gamma),
                        JoltChallengeId::InstructionClaimReduction(
                            InstructionClaimReductionChallenge::EqSpartan,
                        ) => Ok(eq_spartan),
                        _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    },
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?;

            let active_stage2_rounds = log_t + log_k;
            let phase1_offset = batch
                .try_round_offset(active_stage2_rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamRafEvaluation,
                    reason: error.to_string(),
                })?
                + read_write_dimensions.phase1_num_rounds();
            let ram_raf_evaluation_point = batch
                .try_instance_point_at(phase1_offset, ram_raf_evaluation_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_raf_address_point = read_write_dimensions
                .address_opening_point(ram_raf_evaluation_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_raf_opening_point = [
                ram_raf_address_point.as_slice(),
                product_uniskip.tau_low.as_slice(),
            ]
            .concat();
            if ram_raf_address_point.len() != log_k {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamRafEvaluation,
                    reason: format!(
                        "RAM RAF address point length mismatch: expected {log_k}, got {}",
                        ram_raf_address_point.len()
                    ),
                });
            }
            let ram_raf_unmap_address = IdentityPolynomial::new(log_k)
                .evaluate(&ram_raf_address_point)
                * PCS::Field::from_u64(8)
                + PCS::Field::from_u64(checked.public_io.memory_layout.get_lowest_address());
            let ram_raf_public_values = RamRafEvaluationPublicValues {
                unmap_address: ram_raf_unmap_address,
            };
            let [ram_ra_raf_evaluation] = ram::raf_evaluation_output_openings();
            let ram_raf_evaluation_output =
                ram_raf_evaluation_claims.output.expression.try_evaluate(
                    |id| match *id {
                        id if id == ram_ra_raf_evaluation => {
                            Ok(claims.batch_outputs.ram_raf_evaluation)
                        }
                        _ => Ok(PCS::Field::from_u64(0)),
                    },
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| match id {
                        JoltPublicId::RamRafEvaluation(public_id) => {
                            Ok(ram_raf_public_values.value(*public_id))
                        }
                        _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
                    },
                )?;

            let ram_output_check_point = batch
                .try_instance_point_at(phase1_offset, ram_output_check_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let ram_output_address_point = read_write_dimensions
                .address_opening_point(ram_output_check_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamOutputCheck,
                    reason: error.to_string(),
                }
            })?;
            let output_eq = try_eq_mle(&output_address_challenges, &ram_output_address_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let output_mask = range_mask_mle_msb(
                public_memory.io_mask_start,
                public_memory.io_mask_end,
                &ram_output_address_point,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltStageId::RamOutputCheck,
                reason: error.to_string(),
            })?;
            let io_num_vars = public_memory.io_num_vars();
            if ram_output_address_point.len() < io_num_vars {
                return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltStageId::RamOutputCheck,
            reason: format!(
                "RAM output address point is too short for public IO: address has {} variables, IO needs {io_num_vars}",
                ram_output_address_point.len()
            ),
        });
            }
            let (r_hi, r_lo) =
                ram_output_address_point.split_at(ram_output_address_point.len() - io_num_vars);
            let hi_scale = r_hi.iter().fold(PCS::Field::from_u64(1), |acc, challenge| {
                acc * (PCS::Field::from_u64(1) - *challenge)
            });
            let val_io = hi_scale
                * sparse_segments_mle_msb(
                    public_memory
                        .segments
                        .iter()
                        .map(|segment| (segment.start_index, segment.words.as_slice())),
                    r_lo,
                );
            let eq_io_mask = output_eq * output_mask;
            let ram_output_public_values = RamOutputCheckPublicValues {
                eq_io_mask,
                neg_eq_io_mask_val_io: -eq_io_mask * val_io,
            };
            let [ram_val_final] = ram::output_check_output_openings();
            let ram_output_check_output = ram_output_check_claims.output.expression.try_evaluate(
                |id| match *id {
                    id if id == ram_val_final => Ok(claims.batch_outputs.ram_output_check),
                    _ => Ok(PCS::Field::from_u64(0)),
                },
                |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                |id| match id {
                    JoltPublicId::RamOutputCheck(public_id) => {
                        Ok(ram_output_public_values.value(*public_id))
                    }
                    _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
                },
            )?;
            let expected_outputs = Stage2BatchExpectedOutputClaims {
                ram_read_write: ram_read_write_output,
                product_remainder: product_remainder_output,
                instruction_claim_reduction: instruction_claim_reduction_output,
                ram_raf_evaluation: ram_raf_evaluation_output,
                ram_output_check: ram_output_check_output,
            };

            // Reconstruct the final batched evaluation claim in the same order used
            // when the five input claims were batched.
            let [ram_read_write_coefficient, product_coefficient, instruction_coefficient, ram_raf_coefficient, ram_output_coefficient] =
                batch.batching_coefficients.as_slice()
            else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamReadWriteChecking,
                    reason: "Stage 2 batch verifier returned the wrong number of coefficients"
                        .to_string(),
                });
            };
            let expected_final_claim = *ram_read_write_coefficient
                * expected_outputs.ram_read_write
                + *product_coefficient * expected_outputs.product_remainder
                + *instruction_coefficient * expected_outputs.instruction_claim_reduction
                + *ram_raf_coefficient * expected_outputs.ram_raf_evaluation
                + *ram_output_coefficient * expected_outputs.ram_output_check;
            if batch.reduction.value != expected_final_claim {
                return Err(VerifierError::StageClaimOutputMismatch {
                    stage: JoltStageId::RamReadWriteChecking,
                });
            }

            transcript.append_labeled(b"opening_claim", &claims.batch_outputs.ram_read_write.val);
            transcript.append_labeled(b"opening_claim", &claims.batch_outputs.ram_read_write.ra);
            transcript.append_labeled(b"opening_claim", &claims.batch_outputs.ram_read_write.inc);
            transcript.append_labeled(
                b"opening_claim",
                &claims
                    .batch_outputs
                    .product_remainder
                    .left_instruction_input,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims
                    .batch_outputs
                    .product_remainder
                    .right_instruction_input,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims.batch_outputs.product_remainder.jump_flag,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims
                    .batch_outputs
                    .product_remainder
                    .write_lookup_output_to_rd,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims.batch_outputs.product_remainder.lookup_output,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims.batch_outputs.product_remainder.branch_flag,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims.batch_outputs.product_remainder.next_is_noop,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims.batch_outputs.product_remainder.virtual_instruction,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims
                    .batch_outputs
                    .instruction_claim_reduction
                    .left_lookup_operand,
            );
            transcript.append_labeled(
                b"opening_claim",
                &claims
                    .batch_outputs
                    .instruction_claim_reduction
                    .right_lookup_operand,
            );
            transcript.append_labeled(b"opening_claim", &claims.batch_outputs.ram_raf_evaluation);
            transcript.append_labeled(b"opening_claim", &claims.batch_outputs.ram_output_check);

            Ok(Stage2Batch::Clear {
                verified: VerifiedStage2Batch {
                    batching_coefficients: batch.batching_coefficients.clone(),
                    sumcheck_point: batch.reduction.point.clone(),
                    sumcheck_final_claim: batch.reduction.value,
                    expected_final_claim,
                    ram_read_write_gamma,
                    instruction_gamma,
                    output_address_challenges,
                    ram_read_write: VerifiedStage2Sumcheck {
                        input_claim: input_claims.ram_read_write,
                        sumcheck_point: ram_read_write_point.to_vec(),
                        opening_point: ram_read_write_opening_point.opening_point,
                        expected_output_claim: expected_outputs.ram_read_write,
                    },
                    product_remainder: VerifiedStage2Sumcheck {
                        input_claim: input_claims.product_remainder,
                        sumcheck_point: product_point.to_vec(),
                        opening_point: product_opening_point,
                        expected_output_claim: expected_outputs.product_remainder,
                    },
                    instruction_claim_reduction: VerifiedStage2Sumcheck {
                        input_claim: input_claims.instruction_claim_reduction,
                        sumcheck_point: instruction_point.to_vec(),
                        opening_point: instruction_opening_point,
                        expected_output_claim: expected_outputs.instruction_claim_reduction,
                    },
                    ram_raf_evaluation: VerifiedStage2Sumcheck {
                        input_claim: input_claims.ram_raf_evaluation,
                        sumcheck_point: ram_raf_evaluation_point.to_vec(),
                        opening_point: ram_raf_opening_point,
                        expected_output_claim: expected_outputs.ram_raf_evaluation,
                    },
                    ram_output_check: VerifiedStage2Sumcheck {
                        input_claim: input_claims.ram_output_check,
                        sumcheck_point: ram_output_check_point.to_vec(),
                        opening_point: ram_output_address_point,
                        expected_output_claim: expected_outputs.ram_output_check,
                    },
                },
                output_claims: claims.batch_outputs.clone(),
            })
        }
        (Deps::Zk { .. }, Stage2ProductUniSkip::Zk(_)) => {
            let statements = [
                SumcheckStatement::new(
                    ram_read_write_claims.sumcheck.rounds,
                    ram_read_write_claims.sumcheck.degree,
                ),
                SumcheckStatement::new(
                    product_remainder_claims.sumcheck.rounds,
                    product_remainder_claims.sumcheck.degree,
                ),
                SumcheckStatement::new(
                    instruction_claim_reduction_claims.sumcheck.rounds,
                    instruction_claim_reduction_claims.sumcheck.degree,
                ),
                SumcheckStatement::new(
                    ram_raf_evaluation_claims.sumcheck.rounds,
                    ram_raf_evaluation_claims.sumcheck.degree,
                ),
                SumcheckStatement::new(
                    ram_output_check_claims.sumcheck.rounds,
                    ram_output_check_claims.sumcheck.degree,
                ),
            ];
            let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
                &statements,
                &proof.stages.stage2_sumcheck_proof,
                transcript,
            )
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltStageId::RamReadWriteChecking,
                reason: error.to_string(),
            })?;
            committed::require_output_claim_commitments(
                checked,
                &proof.stages.stage2_sumcheck_proof,
                "stage2_sumcheck_proof",
                STAGE2_BATCH_OUTPUT_CLAIMS,
                JoltStageId::RamReadWriteChecking,
            )?;
            let ram_read_write_point = consistency
                .try_instance_point(ram_read_write_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let ram_read_write_opening_point = read_write_dimensions
                .read_write_opening_point(&ram_read_write_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let active_stage2_rounds = log_t + log_k;
            let phase1_offset =
                consistency
                    .try_round_offset(active_stage2_rounds)
                    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                        stage: JoltStageId::RamOutputCheck,
                        reason: error.to_string(),
                    })?
                    + read_write_dimensions.phase1_num_rounds();
            let ram_output_check_point = consistency
                .try_instance_point_at(phase1_offset, ram_output_check_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltStageId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let ram_output_check_opening_point = read_write_dimensions
                .address_opening_point(&ram_output_check_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltStageId::RamOutputCheck,
                    reason: error.to_string(),
                })?;

            Ok(Stage2Batch::Zk(Stage2ZkBatch {
                challenges: consistency.challenges(),
                batching_coefficients: consistency.batching_coefficients.clone(),
                ram_read_write_gamma,
                instruction_gamma,
                output_address_challenges,
                ram_val_check_inputs: Stage2RamValCheckInputs {
                    ram_read_write_opening_point: ram_read_write_opening_point.opening_point,
                    ram_output_check_opening_point,
                },
                consistency,
            }))
        }
        (Deps::Clear { .. }, Stage2ProductUniSkip::Zk(_)) => {
            Err(VerifierError::ExpectedClearProof {
                field: "stage2_uni_skip_first_round_proof",
            })
        }
        (Deps::Zk { .. }, Stage2ProductUniSkip::Clear(_)) => {
            Err(VerifierError::ExpectedCommittedProof {
                field: "stage2_uni_skip_first_round_proof",
            })
        }
    }
}
