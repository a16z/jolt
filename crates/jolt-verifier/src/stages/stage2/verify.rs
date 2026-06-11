#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::{
    claim_reductions::registers as field_registers_claim_reduction,
    dimensions::FieldRegistersTraceDimensions,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction,
        dimensions::TraceDimensions,
        ram::{
            self, RamOutputCheckPublicValues, RamRafEvaluationDimensions,
            RamRafEvaluationPublicValues,
        },
        spartan::{product_remainder, product_uniskip_opening, SpartanProductDimensions},
    },
    InstructionClaimReductionChallenge, JoltChallengeId, JoltPublicId, JoltRelationId,
    JoltSumcheckDomain, RamReadWriteChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle, IdentityPolynomial,
    MultilinearEvaluation,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
};
use jolt_transcript::FsTranscript;

use super::{
    inputs::{Deps, Stage2BatchOutputOpeningClaims},
    outputs::{
        Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2RamRaClaimReductionInputs,
        Stage2RamValCheckInputs, Stage2ZkOutput, VerifiedProductUniSkip, VerifiedStage2Batch,
        VerifiedStage2Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{stage1::Stage1ClearOutput, zk::committed},
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage2BatchSumcheckInputClaims<F: Field> {
    ram_read_write: F,
    product_remainder: F,
    instruction_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_claim_reduction: F,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage2BatchExpectedOutputClaims<F: Field> {
    ram_read_write: F,
    product_remainder: F,
    instruction_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_claim_reduction: F,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

struct Stage2ZkProductUniSkip<F: Field, C> {
    tau_low: Vec<F>,
    tau_high: F,
    product_uniskip_challenge: F,
    consistency: jolt_sumcheck::CommittedSumcheckConsistency<F, C>,
    output_claims: committed::CommittedOutputClaimOutput<C>,
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
    #[cfg(feature = "field-inline")]
    field_registers_claim_reduction_gamma: F,
    output_address_challenges: Vec<F>,
    consistency: jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    output_claims: committed::CommittedOutputClaimOutput<C>,
    ram_val_check_inputs: Stage2RamValCheckInputs<F>,
    ram_ra_claim_reduction_inputs: Stage2RamRaClaimReductionInputs<F>,
    #[cfg(feature = "field-inline")]
    field_inline: super::outputs::FieldInlineStage2ZkOutput<F>,
}

enum Stage2Batch<F: Field, C> {
    Clear {
        verified: VerifiedStage2Batch<F>,
        output_claims: Stage2BatchOutputOpeningClaims<F>,
    },
    Zk(Stage2ZkBatch<F, C>),
}

const PRODUCT_UNISKIP_OUTPUT_CLAIMS: usize = 1;
#[cfg(not(feature = "field-inline"))]
const STAGE2_BATCH_OUTPUT_CLAIMS: usize = 15;
#[cfg(feature = "field-inline")]
const STAGE2_BATCH_OUTPUT_CLAIMS: usize = 18;

fn selected_product_uniskip_sumcheck() -> jolt_claims::protocols::jolt::JoltSumcheckSpec {
    jolt_claims::protocols::jolt::JoltSumcheckSpec::centered_integer(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        1,
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
    )
}

fn selected_product_uniskip_input_claim<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    weights: &[F],
) -> Option<F> {
    let [product, should_branch, should_jump, rest @ ..] = weights else {
        return None;
    };
    let claim = *product * stage1.outer.product
        + *should_branch * stage1.outer.should_branch
        + *should_jump * stage1.outer.should_jump;

    #[cfg(feature = "field-inline")]
    {
        let [field_product_weight, field_inv_product_weight] = rest else {
            return None;
        };
        Some(
            claim
                + *field_product_weight * stage1.field_inline.field_product
                + *field_inv_product_weight * stage1.field_inline.field_inv_product,
        )
    }

    #[cfg(not(feature = "field-inline"))]
    {
        if !rest.is_empty() {
            return None;
        }
        Some(claim)
    }
}

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
    T: FsTranscript<PCS::Field>,
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
                    stage: JoltRelationId::SpartanProductVirtualization,
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
                #[cfg(feature = "field-inline")]
                field_registers_claim_reduction_gamma: batch.field_registers_claim_reduction_gamma,
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
                #[cfg(feature = "field-inline")]
                field_registers_claim_reduction_gamma: batch.field_registers_claim_reduction_gamma,
                output_address_challenges: batch.output_address_challenges,
            };

            Ok(Stage2Output::Zk(Stage2ZkOutput {
                public,
                product_uniskip_consistency: product_uniskip.consistency,
                product_uniskip_output_claims: product_uniskip.output_claims,
                batch_consistency: batch.consistency,
                batch_output_claims: batch.output_claims,
                ram_val_check_inputs: batch.ram_val_check_inputs,
                ram_ra_claim_reduction_inputs: batch.ram_ra_claim_reduction_inputs,
                #[cfg(feature = "field-inline")]
                field_inline: batch.field_inline,
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
    T: FsTranscript<PCS::Field>,
{
    let stage = JoltRelationId::SpartanProductVirtualization;
    let log_t = checked.trace_length.ilog2() as usize;
    let _dimensions = SpartanProductDimensions::new(log_t);
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
    let uniskip_spec = selected_product_uniskip_sumcheck();
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
            let weights = centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage,
                    reason: error.to_string(),
                })?;

            let uniskip_claim = claims.product_uniskip_output_claim;
            let uniskip_input_claim = selected_product_uniskip_input_claim(stage1, &weights)
                .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                    stage,
                    reason: format!(
                        "Stage 2 product uni-skip expected {} weights, got {}",
                        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                        weights.len()
                    ),
                })?;

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
                    transcript,
                )
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: error.to_string(),
                })?;
            if uniskip_reduction.value != uniskip_claim {
                return Err(VerifierError::StageClaimOutputMismatch { stage });
            }

            transcript.absorb_field(&uniskip_claim);

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
            let output_claims = committed::verify_output_claim_commitments(
                committed::CommittedOutputClaimInputs {
                    checked,
                    proof: &proof.stages.stage2_uni_skip_first_round_proof,
                    proof_label: "stage2_uni_skip_first_round_proof",
                    output_claim_count: PRODUCT_UNISKIP_OUTPUT_CLAIMS,
                    stage,
                },
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
                output_claims,
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
    T: FsTranscript<PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = proof.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let ram_read_write_claims = ram::read_write_checking::<PCS::Field>(read_write_dimensions);
    let product_remainder_claims = product_remainder::<PCS::Field>(product_dimensions);
    let instruction_claim_reduction_claims =
        instruction_claim_reduction::claim_reduction::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_registers_claim_reduction_claims = field_registers_claim_reduction::claim_reduction::<
        PCS::Field,
    >(FieldRegistersTraceDimensions::new(log_t));
    let ram_raf_evaluation_claims = ram::raf_evaluation::<PCS::Field>(raf_dimensions);
    let ram_output_check_claims = ram::output_check::<PCS::Field>(read_write_dimensions);
    // Stage 2 batches the regular post-product-uniskip sumchecks. Field-inline
    // inserts its claim reduction at the same product point when enabled.
    let ram_read_write_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_registers_claim_reduction_gamma = transcript.challenge_scalar();
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
    #[cfg(feature = "field-inline")]
    {
        if field_registers_claim_reduction_claims.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: JoltRelationId::SpartanProductVirtualization,
                degree: field_registers_claim_reduction_claims.sumcheck.degree,
            });
        }
    }

    match (deps, product_uniskip) {
        (Deps::Clear { stage1 }, Stage2ProductUniSkip::Clear(product_uniskip)) => {
            let claims = &proof.clear_claims()?.stage2;
            let [product_uniskip_challenge] = product_uniskip.sumcheck_point.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
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
                ram_read_write: ram_read_write_claims.input.expression().try_evaluate(
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
                product_remainder: product_remainder_claims.input.expression().try_evaluate(
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
                    .expression()
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
                #[cfg(feature = "field-inline")]
                field_registers_claim_reduction: stage1.field_inline.field_rd_value
                    + field_registers_claim_reduction_gamma * stage1.field_inline.field_rs1_value
                    + field_registers_claim_reduction_gamma
                        * field_registers_claim_reduction_gamma
                        * stage1.field_inline.field_rs2_value,
                ram_raf_evaluation: ram_raf_evaluation_claims.input.expression().try_evaluate(
                    |id| match *id {
                        id if id == ram_address_spartan => Ok(stage1.outer.ram_address),
                        id => Err(VerifierError::MissingOpeningClaim { id }),
                    },
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?,
                ram_output_check: ram_output_check_claims.input.expression().try_evaluate(
                    |id| Err(VerifierError::MissingOpeningClaim { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                    |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
                )?,
            };

            // The claim order here must match the output-claim reconstruction below and
            // the transcript appends at the end of the stage.
            let mut sumcheck_claims = vec![
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
            ];
            #[cfg(feature = "field-inline")]
            sumcheck_claims.push(SumcheckClaim::new(
                field_registers_claim_reduction_claims.sumcheck.rounds,
                field_registers_claim_reduction_claims.sumcheck.degree,
                input_claims.field_registers_claim_reduction,
            ));
            sumcheck_claims.extend([
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
            ]);
            let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
                &sumcheck_claims,
                &proof.stages.stage2_sumcheck_proof,
                transcript,
            )
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamReadWriteChecking,
                reason: error.to_string(),
            })?;

            let ram_read_write_point = batch
                .try_instance_point(ram_read_write_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let ram_read_write_opening_point = read_write_dimensions
                .read_write_opening_point(ram_read_write_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let eq_cycle = try_eq_mle(
                &product_uniskip.tau_low,
                &ram_read_write_opening_point.r_cycle,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamReadWriteChecking,
                reason: error.to_string(),
            })?;
            let [ram_val, ram_ra, ram_inc] = ram::read_write_checking_output_openings();
            let ram_read_write_output = ram_read_write_claims.output.expression().try_evaluate(
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
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?;
            let product_opening_point = product_point.iter().rev().copied().collect::<Vec<_>>();
            let product_lagrange_weights = centered_lagrange_evals(
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                product_uniskip_challenge,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::SpartanProductVirtualization,
                reason: error.to_string(),
            })?;
            let product_tau_high_bound = centered_lagrange_kernel(
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                product_uniskip.tau_high,
                product_uniskip_challenge,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::SpartanProductVirtualization,
                reason: error.to_string(),
            })?;
            let product_tau_low_eq = try_eq_mle(&product_uniskip.tau_low, &product_opening_point)
                .map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                }
            })?;
            let product_tau_kernel = product_tau_high_bound * product_tau_low_eq;
            let [instruction_product_weight, should_branch_weight, should_jump_weight, rest @ ..] =
                product_lagrange_weights.as_slice()
            else {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: format!(
                        "Stage 2 product remainder expected {} weights, got {}",
                        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                        product_lagrange_weights.len()
                    ),
                });
            };
            let product_left_base = *instruction_product_weight
                * claims
                    .batch_outputs
                    .product_remainder
                    .left_instruction_input
                + *should_branch_weight * claims.batch_outputs.product_remainder.lookup_output
                + *should_jump_weight * claims.batch_outputs.product_remainder.jump_flag;
            let product_right_base = *instruction_product_weight
                * claims
                    .batch_outputs
                    .product_remainder
                    .right_instruction_input
                + *should_branch_weight * claims.batch_outputs.product_remainder.branch_flag
                + *should_jump_weight
                    * (PCS::Field::from_u64(1)
                        - claims.batch_outputs.product_remainder.next_is_noop);
            #[cfg(feature = "field-inline")]
            let (product_left, product_right) = {
                let [field_product_weight, field_inv_product_weight] = rest else {
                    return Err(VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::SpartanProductVirtualization,
                        reason: format!(
                            "Stage 2 field product remainder expected two field weights, got {}",
                            rest.len()
                        ),
                    });
                };
                (
                    product_left_base
                        + (*field_product_weight + *field_inv_product_weight)
                            * claims.batch_outputs.field_inline.product.field_rs1_value,
                    product_right_base
                        + *field_product_weight
                            * claims.batch_outputs.field_inline.product.field_rs2_value
                        + *field_inv_product_weight
                            * claims.batch_outputs.field_inline.product.field_rd_value,
                )
            };
            #[cfg(not(feature = "field-inline"))]
            let (product_left, product_right) = {
                if !rest.is_empty() {
                    return Err(VerifierError::StageClaimPublicInputFailed {
                        stage: JoltRelationId::SpartanProductVirtualization,
                        reason: format!(
                            "Stage 2 product remainder expected no field weights, got {}",
                            rest.len()
                        ),
                    });
                }
                (product_left_base, product_right_base)
            };
            let product_remainder_output = product_tau_kernel * product_left * product_right;

            #[cfg(feature = "field-inline")]
            let (
                field_registers_claim_reduction_point,
                field_registers_claim_reduction_opening_point,
                field_registers_claim_reduction_output,
            ) = {
                let point = batch
                    .try_instance_point(field_registers_claim_reduction_claims.sumcheck.rounds)
                    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                        stage: JoltRelationId::SpartanProductVirtualization,
                        reason: error.to_string(),
                    })?;
                let opening_point = point.iter().rev().copied().collect::<Vec<_>>();
                let eq_spartan =
                    try_eq_mle(&opening_point, &product_uniskip.tau_low).map_err(|error| {
                        VerifierError::StageClaimPublicInputFailed {
                            stage: JoltRelationId::SpartanProductVirtualization,
                            reason: error.to_string(),
                        }
                    })?;
                let output = eq_spartan
                    * (claims.batch_outputs.field_inline.product.field_rd_value
                        + field_registers_claim_reduction_gamma
                            * claims.batch_outputs.field_inline.product.field_rs1_value
                        + field_registers_claim_reduction_gamma
                            * field_registers_claim_reduction_gamma
                            * claims.batch_outputs.field_inline.product.field_rs2_value);
                (point.to_vec(), opening_point, output)
            };

            let instruction_point = batch
                .try_instance_point(instruction_claim_reduction_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::InstructionClaimReduction,
                    reason: error.to_string(),
                })?;
            let instruction_opening_point =
                instruction_point.iter().rev().copied().collect::<Vec<_>>();
            let eq_spartan = try_eq_mle(&instruction_opening_point, &product_uniskip.tau_low)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::InstructionClaimReduction,
                    reason: error.to_string(),
                })?;
            let product_and_instruction_points_match =
                product_opening_point == instruction_opening_point;
            let [instruction_lookup_output, instruction_left_lookup_operand, instruction_right_lookup_operand, instruction_left_instruction_input, instruction_right_instruction_input] =
                instruction_claim_reduction::claim_reduction_output_openings();
            let instruction_claim_reduction_output = instruction_claim_reduction_claims
                .output
                .expression()
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
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?
                + read_write_dimensions.phase1_num_rounds();
            let ram_raf_evaluation_point = batch
                .try_instance_point_at(phase1_offset, ram_raf_evaluation_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_raf_address_point = read_write_dimensions
                .address_opening_point(ram_raf_evaluation_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_raf_opening_point = [
                ram_raf_address_point.as_slice(),
                product_uniskip.tau_low.as_slice(),
            ]
            .concat();
            if ram_raf_address_point.len() != log_k {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamRafEvaluation,
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
                ram_raf_evaluation_claims.output.expression().try_evaluate(
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
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let ram_output_address_point = read_write_dimensions
                .address_opening_point(ram_output_check_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                }
            })?;
            let output_eq = try_eq_mle(&output_address_challenges, &ram_output_address_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let output_mask = range_mask_mle_msb(
                public_memory.io_mask_start,
                public_memory.io_mask_end,
                &ram_output_address_point,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamOutputCheck,
                reason: error.to_string(),
            })?;
            let io_num_vars = public_memory.io_num_vars();
            if ram_output_address_point.len() < io_num_vars {
                return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
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
            let ram_output_check_output =
                ram_output_check_claims.output.expression().try_evaluate(
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
                #[cfg(feature = "field-inline")]
                field_registers_claim_reduction: field_registers_claim_reduction_output,
                ram_raf_evaluation: ram_raf_evaluation_output,
                ram_output_check: ram_output_check_output,
            };

            // Reconstruct the final batched evaluation claim in the same order used
            // when the five input claims were batched.
            let coefficients = batch.batching_coefficients.as_slice();
            #[cfg(not(feature = "field-inline"))]
            let expected_final_claim = {
                let [ram_read_write_coefficient, product_coefficient, instruction_coefficient, ram_raf_coefficient, ram_output_coefficient] =
                    coefficients
                else {
                    return Err(VerifierError::StageClaimSumcheckFailed {
                        stage: JoltRelationId::RamReadWriteChecking,
                        reason: "Stage 2 batch verifier returned the wrong number of coefficients"
                            .to_string(),
                    });
                };
                *ram_read_write_coefficient * expected_outputs.ram_read_write
                    + *product_coefficient * expected_outputs.product_remainder
                    + *instruction_coefficient * expected_outputs.instruction_claim_reduction
                    + *ram_raf_coefficient * expected_outputs.ram_raf_evaluation
                    + *ram_output_coefficient * expected_outputs.ram_output_check
            };
            #[cfg(feature = "field-inline")]
            let expected_final_claim = {
                let [ram_read_write_coefficient, product_coefficient, instruction_coefficient, field_registers_coefficient, ram_raf_coefficient, ram_output_coefficient] =
                    coefficients
                else {
                    return Err(VerifierError::StageClaimSumcheckFailed {
                        stage: JoltRelationId::RamReadWriteChecking,
                        reason: "Stage 2 batch verifier returned the wrong number of coefficients"
                            .to_string(),
                    });
                };
                *ram_read_write_coefficient * expected_outputs.ram_read_write
                    + *product_coefficient * expected_outputs.product_remainder
                    + *instruction_coefficient * expected_outputs.instruction_claim_reduction
                    + *field_registers_coefficient
                        * expected_outputs.field_registers_claim_reduction
                    + *ram_raf_coefficient * expected_outputs.ram_raf_evaluation
                    + *ram_output_coefficient * expected_outputs.ram_output_check
            };
            if batch.reduction.value != expected_final_claim {
                return Err(VerifierError::StageClaimOutputMismatch {
                    stage: JoltRelationId::RamReadWriteChecking,
                });
            }

            transcript.absorb_field(&claims.batch_outputs.ram_read_write.val);
            transcript.absorb_field(&claims.batch_outputs.ram_read_write.ra);
            transcript.absorb_field(&claims.batch_outputs.ram_read_write.inc);
            transcript.absorb_field(
                &claims
                    .batch_outputs
                    .product_remainder
                    .left_instruction_input,
            );
            transcript.absorb_field(
                &claims
                    .batch_outputs
                    .product_remainder
                    .right_instruction_input,
            );
            transcript.absorb_field(&claims.batch_outputs.product_remainder.jump_flag);
            transcript.absorb_field(
                &claims
                    .batch_outputs
                    .product_remainder
                    .write_lookup_output_to_rd,
            );
            transcript.absorb_field(&claims.batch_outputs.product_remainder.lookup_output);
            transcript.absorb_field(&claims.batch_outputs.product_remainder.branch_flag);
            transcript.absorb_field(&claims.batch_outputs.product_remainder.next_is_noop);
            transcript.absorb_field(&claims.batch_outputs.product_remainder.virtual_instruction);
            #[cfg(feature = "field-inline")]
            {
                transcript.absorb_field(&claims.batch_outputs.field_inline.product.field_rs1_value);
                transcript.absorb_field(&claims.batch_outputs.field_inline.product.field_rs2_value);
                transcript.absorb_field(&claims.batch_outputs.field_inline.product.field_rd_value);
            }
            transcript.absorb_field(
                &claims
                    .batch_outputs
                    .instruction_claim_reduction
                    .left_lookup_operand,
            );
            transcript.absorb_field(
                &claims
                    .batch_outputs
                    .instruction_claim_reduction
                    .right_lookup_operand,
            );
            transcript.absorb_field(&claims.batch_outputs.ram_raf_evaluation);
            transcript.absorb_field(&claims.batch_outputs.ram_output_check);

            Ok(Stage2Batch::Clear {
                verified: VerifiedStage2Batch {
                    batching_coefficients: batch.batching_coefficients.clone(),
                    sumcheck_point: batch.reduction.point.clone(),
                    sumcheck_final_claim: batch.reduction.value,
                    expected_final_claim,
                    ram_read_write_gamma,
                    instruction_gamma,
                    #[cfg(feature = "field-inline")]
                    field_registers_claim_reduction_gamma,
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
                    #[cfg(feature = "field-inline")]
                    field_registers_claim_reduction: VerifiedStage2Sumcheck {
                        input_claim: input_claims.field_registers_claim_reduction,
                        sumcheck_point: field_registers_claim_reduction_point,
                        opening_point: field_registers_claim_reduction_opening_point,
                        expected_output_claim: expected_outputs.field_registers_claim_reduction,
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
        (Deps::Zk { .. }, Stage2ProductUniSkip::Zk(product_uniskip)) => {
            let mut statements = vec![
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
            ];
            #[cfg(feature = "field-inline")]
            statements.push(SumcheckStatement::new(
                field_registers_claim_reduction_claims.sumcheck.rounds,
                field_registers_claim_reduction_claims.sumcheck.degree,
            ));
            statements.extend([
                SumcheckStatement::new(
                    ram_raf_evaluation_claims.sumcheck.rounds,
                    ram_raf_evaluation_claims.sumcheck.degree,
                ),
                SumcheckStatement::new(
                    ram_output_check_claims.sumcheck.rounds,
                    ram_output_check_claims.sumcheck.degree,
                ),
            ]);
            let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
                &statements,
                &proof.stages.stage2_sumcheck_proof,
                transcript,
            )
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamReadWriteChecking,
                reason: error.to_string(),
            })?;
            let output_claims = committed::verify_output_claim_commitments(
                committed::CommittedOutputClaimInputs {
                    checked,
                    proof: &proof.stages.stage2_sumcheck_proof,
                    proof_label: "stage2_sumcheck_proof",
                    output_claim_count: STAGE2_BATCH_OUTPUT_CLAIMS,
                    stage: JoltRelationId::RamReadWriteChecking,
                },
            )?;
            let ram_read_write_point = consistency
                .try_instance_point(ram_read_write_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let ram_read_write_opening_point = read_write_dimensions
                .read_write_opening_point(&ram_read_write_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            #[cfg(feature = "field-inline")]
            let field_registers_claim_reduction_opening_point = consistency
                .try_instance_point(field_registers_claim_reduction_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let active_stage2_rounds = log_t + log_k;
            let phase1_offset =
                consistency
                    .try_round_offset(active_stage2_rounds)
                    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                        stage: JoltRelationId::RamOutputCheck,
                        reason: error.to_string(),
                    })?
                    + read_write_dimensions.phase1_num_rounds();
            let ram_raf_evaluation_point = consistency
                .try_instance_point_at(phase1_offset, ram_raf_evaluation_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_raf_address_point = read_write_dimensions
                .address_opening_point(&ram_raf_evaluation_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            if ram_raf_address_point.len() != log_k {
                return Err(VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: format!(
                        "RAM RAF address point length mismatch: expected {log_k}, got {}",
                        ram_raf_address_point.len()
                    ),
                });
            }
            let ram_raf_opening_point = [
                ram_raf_address_point.as_slice(),
                product_uniskip.tau_low.as_slice(),
            ]
            .concat();
            let ram_output_check_point = consistency
                .try_instance_point_at(phase1_offset, ram_output_check_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;
            let ram_output_check_opening_point = read_write_dimensions
                .address_opening_point(&ram_output_check_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;

            Ok(Stage2Batch::Zk(Stage2ZkBatch {
                challenges: consistency.challenges(),
                batching_coefficients: consistency.batching_coefficients.clone(),
                ram_read_write_gamma,
                instruction_gamma,
                #[cfg(feature = "field-inline")]
                field_registers_claim_reduction_gamma,
                output_address_challenges,
                ram_val_check_inputs: Stage2RamValCheckInputs {
                    ram_read_write_opening_point: ram_read_write_opening_point
                        .opening_point
                        .clone(),
                    ram_output_check_opening_point,
                },
                ram_ra_claim_reduction_inputs: Stage2RamRaClaimReductionInputs {
                    ram_raf_evaluation_opening_point: ram_raf_opening_point,
                    ram_read_write_opening_point: ram_read_write_opening_point.opening_point,
                },
                #[cfg(feature = "field-inline")]
                field_inline: super::outputs::FieldInlineStage2ZkOutput {
                    field_registers_claim_reduction_opening_point,
                },
                consistency,
                output_claims,
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
