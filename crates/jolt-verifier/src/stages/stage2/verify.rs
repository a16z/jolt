use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction,
        dimensions::TraceDimensions,
        ram::{self, RamRafEvaluationDimensions},
        spartan::{product_remainder, SpartanProductDimensions},
    },
    JoltReadWriteConfig, JoltRelationId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{lagrange::centered_lagrange_evals, Point};
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
    UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::{
    batch::{Stage2BatchOpeningPointRefs, Stage2BatchRelations, Stage2BatchRelationsRequest},
    inputs::{
        product_uniskip_input_claim, Deps, Stage2BatchOutputClaims, Stage2ProductUniSkipInputValues,
    },
    outputs::{
        Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2RamRaClaimReductionInputs,
        Stage2RamValCheckInputs, Stage2ZkOutput, VerifiedProductUniSkip, VerifiedStage2Batch,
        VerifiedStage2Sumcheck,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{relations::SumcheckInstance, zk::committed},
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2BatchInputClaims<F: Field> {
    pub ram_read_write: F,
    pub product_remainder: F,
    pub instruction_claim_reduction: F,
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2BatchExpectedOutputClaims<F: Field> {
    pub ram_read_write: F,
    pub product_remainder: F,
    pub instruction_claim_reduction: F,
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
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
    output_address_challenges: Vec<F>,
    consistency: jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    output_claims: committed::CommittedOutputClaimOutput<C>,
    ram_val_check_inputs: Stage2RamValCheckInputs<F>,
    ram_ra_claim_reduction_inputs: Stage2RamRaClaimReductionInputs<F>,
}

enum Stage2Batch<F: Field, C> {
    Clear {
        verified: VerifiedStage2Batch<F>,
        output_claims: Stage2BatchOutputClaims<F>,
    },
    Zk(Stage2ZkBatch<F, C>),
}

const PRODUCT_UNISKIP_OUTPUT_CLAIMS: usize = 1;
const STAGE2_BATCH_OUTPUT_CLAIMS: usize = 15;

fn selected_product_uniskip_sumcheck() -> jolt_claims::protocols::jolt::JoltSumcheckSpec {
    jolt_claims::protocols::jolt::JoltSumcheckSpec::centered_integer(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        1,
        SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2BatchOpeningPoints<F: Field> {
    pub ram_read_write_sumcheck: Vec<F>,
    pub ram_read_write_opening: Vec<F>,
    pub ram_read_write_cycle: Vec<F>,
    pub product_sumcheck: Vec<F>,
    pub product_opening: Vec<F>,
    pub instruction_sumcheck: Vec<F>,
    pub instruction_opening: Vec<F>,
    pub ram_raf_sumcheck: Vec<F>,
    pub ram_raf_opening: Vec<F>,
    pub ram_output_check_sumcheck: Vec<F>,
    pub ram_output_check_opening: Vec<F>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2BatchPointRequest<'a, F: Field> {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
    pub challenges: &'a [F],
    pub product_tau_low: &'a [F],
}

/// Derives every stage 2 batch instance's sumcheck point and opening point from
/// the batched sumcheck `challenges`.
pub fn stage2_batch_opening_points<F: Field>(
    request: Stage2BatchPointRequest<'_, F>,
) -> Result<Stage2BatchOpeningPoints<F>, VerifierError> {
    let read_write_dimensions = request
        .rw_config
        .ram_dimensions(request.log_t, request.log_k);
    let read_write_rounds = request.log_t + request.log_k;
    if request.challenges.len() != read_write_rounds {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamReadWriteChecking,
            reason: format!(
                "Stage 2 regular batch returned {} challenges, expected {read_write_rounds}",
                request.challenges.len()
            ),
        });
    }

    let ram_read_write_sumcheck = request.challenges.to_vec();
    let ram_read_write = read_write_dimensions
        .read_write_opening_point(&ram_read_write_sumcheck)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamReadWriteChecking,
            reason: error.to_string(),
        })?;

    let product_offset = read_write_rounds - request.log_t;
    let product_sumcheck = request.challenges[product_offset..].to_vec();
    let mut product_opening = product_sumcheck.clone();
    product_opening.reverse();
    let instruction_sumcheck = product_sumcheck.clone();
    let instruction_opening = product_opening.clone();

    let ram_terminal_rounds =
        request.log_t + request.log_k - read_write_dimensions.phase1_num_rounds();
    let terminal_offset = read_write_dimensions.phase1_num_rounds();
    let terminal_end = terminal_offset + ram_terminal_rounds;
    let terminal_point = request
        .challenges
        .get(terminal_offset..terminal_end)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamReadWriteChecking,
            reason: format!(
                "Stage 2 terminal point range {terminal_offset}..{terminal_end} exceeds {} challenges",
                request.challenges.len()
            ),
        })?
        .to_vec();
    let ram_raf_address_point = read_write_dimensions
        .address_opening_point(&terminal_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRafEvaluation,
            reason: error.to_string(),
        })?;
    let ram_raf_opening = [ram_raf_address_point.as_slice(), request.product_tau_low].concat();
    let ram_output_check_opening = ram_raf_address_point;

    Ok(Stage2BatchOpeningPoints {
        ram_read_write_sumcheck,
        ram_read_write_opening: ram_read_write.opening_point,
        ram_read_write_cycle: ram_read_write.r_cycle,
        product_sumcheck: product_sumcheck.clone(),
        product_opening: product_opening.clone(),
        instruction_sumcheck,
        instruction_opening,
        ram_raf_sumcheck: terminal_point.clone(),
        ram_raf_opening,
        ram_output_check_sumcheck: terminal_point,
        ram_output_check_opening,
    })
}

pub fn stage2_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage2BatchExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    let [ram_read_write_coefficient, product_coefficient, instruction_coefficient, ram_raf_coefficient, ram_output_coefficient] =
        coefficients
    else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamReadWriteChecking,
            reason: "Stage 2 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(
        *ram_read_write_coefficient * expected_outputs.ram_read_write
            + *product_coefficient * expected_outputs.product_remainder
            + *instruction_coefficient * expected_outputs.instruction_claim_reduction
            + *ram_raf_coefficient * expected_outputs.ram_raf_evaluation
            + *ram_output_coefficient * expected_outputs.ram_output_check,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniSkipClearRequest<F: Field> {
    pub tau_low: Vec<F>,
    pub tau_high: F,
    pub input_claim: F,
    pub challenge: F,
    pub output_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RegularBatchClearRequest<F: Field> {
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub output_claim: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    pub output_address_challenges: Vec<F>,
    pub input_claims: Stage2BatchInputClaims<F>,
    pub opening_points: Stage2BatchOpeningPoints<F>,
    pub expected_outputs: Stage2BatchExpectedOutputClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ClearOutputRequest<F: Field> {
    pub output_claims: Stage2BatchOutputClaims<F>,
    pub product_uniskip: Stage2ProductUniSkipClearRequest<F>,
    pub batch: Stage2RegularBatchClearRequest<F>,
}

/// Assembles the clear-mode [`Stage2ClearOutput`] from pre-derived input claims,
/// opening points, and expected outputs, after checking the batched final claim
/// against the prover-supplied `output_claim`.
pub fn stage2_clear_output<F: Field>(
    request: Stage2ClearOutputRequest<F>,
) -> Result<Stage2ClearOutput<F>, VerifierError> {
    let expected_final_claim = stage2_expected_final_claim(
        &request.batch.batching_coefficients,
        &request.batch.expected_outputs,
    )?;
    if request.batch.output_claim != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::RamReadWriteChecking,
        });
    }

    let public = Stage2PublicOutput {
        challenges: request.batch.challenges.clone(),
        batching_coefficients: request.batch.batching_coefficients.clone(),
        product_uniskip_challenge: request.product_uniskip.challenge,
        product_tau_low: request.product_uniskip.tau_low.clone(),
        product_tau_high: request.product_uniskip.tau_high,
        ram_read_write_gamma: request.batch.ram_read_write_gamma,
        instruction_gamma: request.batch.instruction_gamma,
        output_address_challenges: request.batch.output_address_challenges.clone(),
    };

    Ok(Stage2ClearOutput {
        public,
        output_claims: request.output_claims,
        product_uniskip: VerifiedProductUniSkip {
            tau_low: request.product_uniskip.tau_low,
            tau_high: request.product_uniskip.tau_high,
            input_claim: request.product_uniskip.input_claim,
            sumcheck_point: Point::high_to_low(vec![request.product_uniskip.challenge]),
            sumcheck_final_claim: request.product_uniskip.output_claim,
            expected_output_claim: request.product_uniskip.output_claim,
        },
        batch: VerifiedStage2Batch {
            batching_coefficients: request.batch.batching_coefficients,
            sumcheck_point: Point::high_to_low(request.batch.challenges),
            sumcheck_final_claim: request.batch.output_claim,
            expected_final_claim,
            ram_read_write_gamma: request.batch.ram_read_write_gamma,
            instruction_gamma: request.batch.instruction_gamma,
            output_address_challenges: request.batch.output_address_challenges,
            ram_read_write: VerifiedStage2Sumcheck {
                input_claim: request.batch.input_claims.ram_read_write,
                sumcheck_point: request.batch.opening_points.ram_read_write_sumcheck,
                opening_point: request.batch.opening_points.ram_read_write_opening,
                expected_output_claim: request.batch.expected_outputs.ram_read_write,
            },
            product_remainder: VerifiedStage2Sumcheck {
                input_claim: request.batch.input_claims.product_remainder,
                sumcheck_point: request.batch.opening_points.product_sumcheck,
                opening_point: request.batch.opening_points.product_opening,
                expected_output_claim: request.batch.expected_outputs.product_remainder,
            },
            instruction_claim_reduction: VerifiedStage2Sumcheck {
                input_claim: request.batch.input_claims.instruction_claim_reduction,
                sumcheck_point: request.batch.opening_points.instruction_sumcheck,
                opening_point: request.batch.opening_points.instruction_opening,
                expected_output_claim: request.batch.expected_outputs.instruction_claim_reduction,
            },
            ram_raf_evaluation: VerifiedStage2Sumcheck {
                input_claim: request.batch.input_claims.ram_raf_evaluation,
                sumcheck_point: request.batch.opening_points.ram_raf_sumcheck,
                opening_point: request.batch.opening_points.ram_raf_opening,
                expected_output_claim: request.batch.expected_outputs.ram_raf_evaluation,
            },
            ram_output_check: VerifiedStage2Sumcheck {
                input_claim: request.batch.input_claims.ram_output_check,
                sumcheck_point: request.batch.opening_points.ram_output_check_sumcheck,
                opening_point: request.batch.opening_points.ram_output_check_opening,
                expected_output_claim: request.batch.expected_outputs.ram_output_check,
            },
        },
    })
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
                product_uniskip_output_claims: product_uniskip.output_claims,
                batch_consistency: batch.consistency,
                batch_output_claims: batch.output_claims,
                ram_val_check_inputs: batch.ram_val_check_inputs,
                ram_ra_claim_reduction_inputs: batch.ram_ra_claim_reduction_inputs,
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
            let uniskip_input_claim = product_uniskip_input_claim(
                Stage2ProductUniSkipInputValues::from_stage1(stage1),
                &weights,
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
    T: Transcript<Challenge = PCS::Field>,
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
    let ram_raf_evaluation_claims = ram::raf_evaluation::<PCS::Field>(raf_dimensions);
    let ram_output_check_claims = ram::output_check::<PCS::Field>(read_write_dimensions);
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
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: "product uni-skip proof did not reduce to one challenge".to_string(),
                });
            };
            let product_uniskip_challenge = *product_uniskip_challenge;

            // Build the five batch relations once; each owns its input/output
            // claim algebra (single-sourced with its jolt-claims formula and the
            // BlindFold constraint). The product uni-skip stays hand-coded above.
            let relations = Stage2BatchRelations::new(Stage2BatchRelationsRequest {
                log_t,
                log_k,
                rw_config: proof.rw_config,
                checked,
                stage1,
                product_uniskip_output_claim: claims.product_uniskip_output_claim,
                product_tau_low: product_uniskip.tau_low.clone(),
                product_tau_high: product_uniskip.tau_high,
                product_uniskip_challenge,
                ram_read_write_gamma,
                instruction_gamma,
                output_address_challenges: output_address_challenges.clone(),
            })?;
            let input_claims = relations.input_claims()?;

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
            let product_point = batch
                .try_instance_point(product_remainder_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?;
            let instruction_point = batch
                .try_instance_point(instruction_claim_reduction_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::InstructionClaimReduction,
                    reason: error.to_string(),
                })?;
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
            let ram_output_check_point = batch
                .try_instance_point_at(phase1_offset, ram_output_check_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;

            // Each relation maps its sumcheck point to its produced opening points.
            let ram_read_write_points = relations
                .ram_read_write
                .derive_opening_points(ram_read_write_point, &relations.ram_read_write_inputs)?;
            let product_remainder_points = relations
                .product_remainder
                .derive_opening_points(product_point, &relations.product_remainder_inputs)?;
            let instruction_reduction_points = relations
                .instruction_reduction
                .derive_opening_points(instruction_point, &relations.instruction_reduction_inputs)?;
            let ram_raf_points = relations
                .ram_raf
                .derive_opening_points(ram_raf_evaluation_point, &relations.ram_raf_inputs)?;
            let ram_output_points = relations
                .ram_output
                .derive_opening_points(ram_output_check_point, &relations.ram_output_inputs)?;

            // Reconstruct every expected output from the produced opening points and
            // committed values (the relation bundle fills the aliased openings).
            let expected_outputs = relations.expected_outputs(
                Stage2BatchOpeningPointRefs {
                    ram_read_write: &ram_read_write_points.val,
                    product_remainder: &product_remainder_points.left_instruction_input,
                    instruction_reduction: &instruction_reduction_points.left_lookup_operand,
                    ram_raf: &ram_raf_points.ram_ra,
                    ram_output: &ram_output_points.val_final,
                },
                &claims.batch_outputs,
            )?;

            let expected_final_claim =
                stage2_expected_final_claim(&batch.batching_coefficients, &expected_outputs)?;
            if batch.reduction.value != expected_final_claim {
                return Err(VerifierError::StageClaimOutputMismatch {
                    stage: JoltRelationId::RamReadWriteChecking,
                });
            }

            claims.batch_outputs.append_to_transcript(transcript);

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
                        opening_point: ram_read_write_points.val.clone(),
                        expected_output_claim: expected_outputs.ram_read_write,
                    },
                    product_remainder: VerifiedStage2Sumcheck {
                        input_claim: input_claims.product_remainder,
                        sumcheck_point: product_point.to_vec(),
                        opening_point: product_remainder_points.left_instruction_input.clone(),
                        expected_output_claim: expected_outputs.product_remainder,
                    },
                    instruction_claim_reduction: VerifiedStage2Sumcheck {
                        input_claim: input_claims.instruction_claim_reduction,
                        sumcheck_point: instruction_point.to_vec(),
                        opening_point: instruction_reduction_points.left_lookup_operand.clone(),
                        expected_output_claim: expected_outputs.instruction_claim_reduction,
                    },
                    ram_raf_evaluation: VerifiedStage2Sumcheck {
                        input_claim: input_claims.ram_raf_evaluation,
                        sumcheck_point: ram_raf_evaluation_point.to_vec(),
                        opening_point: ram_raf_points.ram_ra.clone(),
                        expected_output_claim: expected_outputs.ram_raf_evaluation,
                    },
                    ram_output_check: VerifiedStage2Sumcheck {
                        input_claim: input_claims.ram_output_check,
                        sumcheck_point: ram_output_check_point.to_vec(),
                        opening_point: ram_output_points.val_final.clone(),
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
