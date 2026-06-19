use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction,
        dimensions::TraceDimensions,
        ram::{self, RamRafEvaluationDimensions},
        spartan::{product_remainder, product_uniskip_opening, SpartanProductDimensions},
    },
    InstructionClaimReductionChallenge, JoltChallengeId, JoltReadWriteConfig, JoltRelationId,
    JoltSumcheckDomain, RamReadWriteChallenge,
};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    range_mask_mle_msb, sparse_segments_mle_msb, try_eq_mle, IdentityPolynomial,
    MultilinearEvaluation, Point,
};
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
    UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::{
    inputs::{
        product_uniskip_input_claim, Deps, Stage2BatchOutputClaims, Stage2ProductUniSkipInputValues,
    },
    outputs::{
        Stage2ClearOutput, Stage2Output, Stage2PublicOutput, Stage2RamRaClaimReductionInputs,
        Stage2RamValCheckInputs, Stage2ZkOutput, VerifiedProductUniSkip, VerifiedStage2Batch,
        VerifiedStage2Sumcheck,
    },
    InstructionClaimReduction, InstructionClaimReductionInputClaims,
    InstructionClaimReductionOutputClaims, ProductRemainder, ProductRemainderInputClaims,
    ProductRemainderOutputClaims, RamOutputCheck, RamOutputCheckInputClaims,
    RamOutputCheckOutputClaims, RamRafEvaluation, RamRafEvaluationInputClaims,
    RamRafEvaluationOutputClaims, RamReadWriteChecking, RamReadWriteInputClaims,
    RamReadWriteOutputClaims,
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::{OpeningClaim, SumcheckInstance},
        stage1::Stage1ClearOutput,
        zk::committed,
    },
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

fn stage2_public_input_failed(stage: JoltRelationId, error: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2BatchInputClaimRequest<'a, F: Field> {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub product_uniskip_output_claim: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
}

/// Evaluates each stage 2 batch instance's input claim from the stage 1 openings
/// and gamma challenges.
pub fn stage2_batch_input_claims<F: Field>(
    request: Stage2BatchInputClaimRequest<'_, F>,
) -> Result<Stage2BatchInputClaims<F>, VerifierError> {
    let trace_dimensions = TraceDimensions::new(request.log_t);
    let read_write_dimensions = request
        .rw_config
        .ram_dimensions(request.log_t, request.log_k);
    let product_dimensions = SpartanProductDimensions::new(request.log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let ram_read_write_claims = ram::read_write_checking::<F>(read_write_dimensions);
    let product_remainder_claims = product_remainder::<F>(product_dimensions);
    let instruction_claim_reduction_claims =
        instruction_claim_reduction::claim_reduction::<F>(trace_dimensions);
    let ram_raf_evaluation_claims = ram::raf_evaluation::<F>(raf_dimensions);
    let ram_output_check_claims = ram::output_check::<F>(read_write_dimensions);

    let [ram_read_value, ram_write_value] = ram::read_write_checking_input_openings();
    let product_uniskip_opening_id = product_uniskip_opening();
    let [instruction_lookup_output_spartan, instruction_left_lookup_operand_spartan, instruction_right_lookup_operand_spartan, instruction_left_instruction_input_spartan, instruction_right_instruction_input_spartan] =
        instruction_claim_reduction::claim_reduction_input_openings();
    let [ram_address_spartan] = ram::raf_evaluation_input_openings();

    Ok(Stage2BatchInputClaims {
        ram_read_write: ram_read_write_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_read_value => Ok(request.stage1.outer.ram_read_value),
                id if id == ram_write_value => Ok(request.stage1.outer.ram_write_value),
                id => Err(VerifierError::MissingOpeningClaim { id }),
            },
            |id| match id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => {
                    Ok(request.ram_read_write_gamma)
                }
                _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
            },
            |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
        )?,
        product_remainder: product_remainder_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == product_uniskip_opening_id => Ok(request.product_uniskip_output_claim),
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
                        Ok(request.stage1.outer.lookup_output)
                    }
                    id if id == instruction_left_lookup_operand_spartan => {
                        Ok(request.stage1.outer.left_lookup_operand)
                    }
                    id if id == instruction_right_lookup_operand_spartan => {
                        Ok(request.stage1.outer.right_lookup_operand)
                    }
                    id if id == instruction_left_instruction_input_spartan => {
                        Ok(request.stage1.outer.left_instruction_input)
                    }
                    id if id == instruction_right_instruction_input_spartan => {
                        Ok(request.stage1.outer.right_instruction_input)
                    }
                    id => Err(VerifierError::MissingOpeningClaim { id }),
                },
                |id| match id {
                    JoltChallengeId::InstructionClaimReduction(
                        InstructionClaimReductionChallenge::Gamma,
                    ) => Ok(request.instruction_gamma),
                    _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
                },
                |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
            )?,
        ram_raf_evaluation: ram_raf_evaluation_claims.input.expression().try_evaluate(
            |id| match *id {
                id if id == ram_address_spartan => Ok(request.stage1.outer.ram_address),
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
    })
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

#[derive(Clone, Copy, Debug)]
pub struct Stage2ProductUniSkipOutputClaimData<'a, F: Field> {
    pub tau_low: &'a [F],
    pub tau_high: F,
    pub challenge: F,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2ExpectedOutputRequest<'a, F: Field> {
    pub log_k: usize,
    pub checked: &'a CheckedInputs,
    pub product_uniskip: Stage2ProductUniSkipOutputClaimData<'a, F>,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    pub output_address_challenges: &'a [F],
    pub opening_points: &'a Stage2BatchOpeningPoints<F>,
    pub claims: &'a Stage2BatchOutputClaims<F>,
}

/// Reconstructs each stage 2 batch instance's expected output claim from the
/// committed opening claims, gamma challenges, and derived opening points.
pub fn stage2_expected_outputs<F: Field>(
    request: Stage2ExpectedOutputRequest<'_, F>,
) -> Result<Stage2BatchExpectedOutputClaims<F>, VerifierError> {
    let eq_cycle = try_eq_mle(
        request.product_uniskip.tau_low,
        &request.opening_points.ram_read_write_cycle,
    )
    .map_err(|error| stage2_public_input_failed(JoltRelationId::RamReadWriteChecking, error))?;
    let ram_read_write = eq_cycle
        * request.claims.ram_read_write.ra
        * (request.claims.ram_read_write.val
            + request.ram_read_write_gamma
                * (request.claims.ram_read_write.val + request.claims.ram_read_write.inc));

    let weights = centered_lagrange_evals(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        request.product_uniskip.challenge,
    )
    .map_err(|error| {
        stage2_public_input_failed(JoltRelationId::SpartanProductVirtualization, error)
    })?;
    let product_tau_high_bound = centered_lagrange_kernel(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        request.product_uniskip.tau_high,
        request.product_uniskip.challenge,
    )
    .map_err(|error| {
        stage2_public_input_failed(JoltRelationId::SpartanProductVirtualization, error)
    })?;
    let product_tau_low_eq = try_eq_mle(
        request.product_uniskip.tau_low,
        &request.opening_points.product_opening,
    )
    .map_err(|error| {
        stage2_public_input_failed(JoltRelationId::SpartanProductVirtualization, error)
    })?;
    let [instruction_product_weight, should_branch_weight, should_jump_weight, rest @ ..] =
        weights.as_slice()
    else {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::SpartanProductVirtualization,
            reason: format!(
                "Stage 2 product remainder expected {} weights, got {}",
                SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                weights.len()
            ),
        });
    };
    let product_left_base = *instruction_product_weight
        * request.claims.product_remainder.left_instruction_input
        + *should_branch_weight * request.claims.product_remainder.lookup_output
        + *should_jump_weight * request.claims.product_remainder.jump_flag;
    let product_right_base = *instruction_product_weight
        * request.claims.product_remainder.right_instruction_input
        + *should_branch_weight * request.claims.product_remainder.branch_flag
        + *should_jump_weight * (F::from_u64(1) - request.claims.product_remainder.next_is_noop);
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
    let product_remainder =
        product_tau_high_bound * product_tau_low_eq * product_left * product_right;

    let eq_spartan = try_eq_mle(
        &request.opening_points.instruction_opening,
        request.product_uniskip.tau_low,
    )
    .map_err(|error| {
        stage2_public_input_failed(JoltRelationId::InstructionClaimReduction, error)
    })?;
    let product_and_instruction_points_match = request.opening_points.product_opening.as_slice()
        == request.opening_points.instruction_opening.as_slice();
    let default_instruction_opening = |opening| {
        if product_and_instruction_points_match {
            opening
        } else {
            F::zero()
        }
    };
    let gamma = request.instruction_gamma;
    let gamma2 = gamma * gamma;
    let gamma3 = gamma2 * gamma;
    let gamma4 = gamma3 * gamma;
    let instruction_claim_reduction = eq_spartan
        * (request
            .claims
            .instruction_claim_reduction
            .lookup_output
            .unwrap_or_else(|| {
                default_instruction_opening(request.claims.product_remainder.lookup_output)
            })
            + gamma
                * request
                    .claims
                    .instruction_claim_reduction
                    .left_lookup_operand
            + gamma2
                * request
                    .claims
                    .instruction_claim_reduction
                    .right_lookup_operand
            + gamma3
                * request
                    .claims
                    .instruction_claim_reduction
                    .left_instruction_input
                    .unwrap_or_else(|| {
                        default_instruction_opening(
                            request.claims.product_remainder.left_instruction_input,
                        )
                    })
            + gamma4
                * request
                    .claims
                    .instruction_claim_reduction
                    .right_instruction_input
                    .unwrap_or_else(|| {
                        default_instruction_opening(
                            request.claims.product_remainder.right_instruction_input,
                        )
                    }));

    if request.opening_points.ram_raf_opening.len() < request.log_k {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRafEvaluation,
            reason: format!(
                "RAM RAF opening point is too short: expected at least {}, got {}",
                request.log_k,
                request.opening_points.ram_raf_opening.len()
            ),
        });
    }
    let ram_raf_address_point = &request.opening_points.ram_raf_opening[..request.log_k];
    let ram_raf_unmap_address =
        IdentityPolynomial::new(request.log_k).evaluate(ram_raf_address_point) * F::from_u64(8)
            + F::from_u64(request.checked.public_io.memory_layout.get_lowest_address());
    let ram_raf_evaluation = ram_raf_unmap_address * request.claims.ram_raf_evaluation;

    let public_memory = PublicIoMemory::new(&request.checked.public_io)
        .map_err(|error| stage2_public_input_failed(JoltRelationId::RamOutputCheck, error))?;
    let output_eq = try_eq_mle(
        request.output_address_challenges,
        &request.opening_points.ram_output_check_opening,
    )
    .map_err(|error| stage2_public_input_failed(JoltRelationId::RamOutputCheck, error))?;
    let output_mask = range_mask_mle_msb(
        public_memory.io_mask_start,
        public_memory.io_mask_end,
        &request.opening_points.ram_output_check_opening,
    )
    .map_err(|error| stage2_public_input_failed(JoltRelationId::RamOutputCheck, error))?;
    let io_num_vars = public_memory.io_num_vars();
    if request.opening_points.ram_output_check_opening.len() < io_num_vars {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
            reason: format!(
                "RAM output address point is too short for public IO: address has {} variables, IO needs {io_num_vars}",
                request.opening_points.ram_output_check_opening.len()
            ),
        });
    }
    let (r_hi, r_lo) = request
        .opening_points
        .ram_output_check_opening
        .split_at(request.opening_points.ram_output_check_opening.len() - io_num_vars);
    let hi_scale = r_hi.iter().fold(F::from_u64(1), |acc, challenge| {
        acc * (F::from_u64(1) - *challenge)
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
    let ram_output_check = eq_io_mask * request.claims.ram_output_check - eq_io_mask * val_io;

    Ok(Stage2BatchExpectedOutputClaims {
        ram_read_write,
        product_remainder,
        instruction_claim_reduction,
        ram_raf_evaluation,
        ram_output_check,
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

            // Build the five batch relations. Each owns its input/output claim
            // algebra (single-sourced with its jolt-claims formula and the
            // BlindFold constraint); the product uni-skip stays hand-coded above.
            let lowest_address = checked.public_io.memory_layout.get_lowest_address();
            let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
                stage2_public_input_failed(JoltRelationId::RamOutputCheck, error)
            })?;
            let ram_read_write = RamReadWriteChecking::new(
                read_write_dimensions,
                log_k,
                ram_read_write_gamma,
                product_uniskip.tau_low.clone(),
            );
            let product_remainder = ProductRemainder::new(
                product_dimensions,
                product_uniskip_challenge,
                product_uniskip.tau_high,
                product_uniskip.tau_low.clone(),
            );
            let instruction_reduction = InstructionClaimReduction::new(
                trace_dimensions,
                instruction_gamma,
                product_uniskip.tau_low.clone(),
            );
            let ram_raf = RamRafEvaluation::new(
                read_write_dimensions,
                raf_dimensions,
                log_k,
                lowest_address,
                product_uniskip.tau_low.clone(),
            );
            let ram_output =
                RamOutputCheck::new(read_write_dimensions, output_address_challenges.clone(), public_memory);

            let ram_read_write_inputs = RamReadWriteInputClaims::from_upstream(stage1);
            let product_remainder_inputs =
                ProductRemainderInputClaims::from_uniskip_output(claims.product_uniskip_output_claim);
            let instruction_reduction_inputs =
                InstructionClaimReductionInputClaims::from_upstream(stage1);
            let ram_raf_inputs = RamRafEvaluationInputClaims::from_upstream(stage1);
            let ram_output_inputs = RamOutputCheckInputClaims::from_upstream();

            let ram_read_write_input_claim = ram_read_write.input_claim(&ram_read_write_inputs)?;
            let product_remainder_input_claim =
                product_remainder.input_claim(&product_remainder_inputs)?;
            let instruction_reduction_input_claim =
                instruction_reduction.input_claim(&instruction_reduction_inputs)?;
            let ram_raf_input_claim = ram_raf.input_claim(&ram_raf_inputs)?;
            let ram_output_input_claim = ram_output.input_claim(&ram_output_inputs)?;

            // The claim order here must match the output-claim reconstruction below and
            // the transcript appends at the end of the stage.
            let mut sumcheck_claims = vec![
                SumcheckClaim::new(
                    ram_read_write_claims.sumcheck.rounds,
                    ram_read_write_claims.sumcheck.degree,
                    ram_read_write_input_claim,
                ),
                SumcheckClaim::new(
                    product_remainder_claims.sumcheck.rounds,
                    product_remainder_claims.sumcheck.degree,
                    product_remainder_input_claim,
                ),
                SumcheckClaim::new(
                    instruction_claim_reduction_claims.sumcheck.rounds,
                    instruction_claim_reduction_claims.sumcheck.degree,
                    instruction_reduction_input_claim,
                ),
            ];
            sumcheck_claims.extend([
                SumcheckClaim::new(
                    ram_raf_evaluation_claims.sumcheck.rounds,
                    ram_raf_evaluation_claims.sumcheck.degree,
                    ram_raf_input_claim,
                ),
                SumcheckClaim::new(
                    ram_output_check_claims.sumcheck.rounds,
                    ram_output_check_claims.sumcheck.degree,
                    ram_output_input_claim,
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
            let ram_read_write_points =
                ram_read_write.derive_opening_points(ram_read_write_point, &ram_read_write_inputs)?;
            let product_remainder_points =
                product_remainder.derive_opening_points(product_point, &product_remainder_inputs)?;
            let instruction_reduction_points = instruction_reduction
                .derive_opening_points(instruction_point, &instruction_reduction_inputs)?;
            let ram_raf_points =
                ram_raf.derive_opening_points(ram_raf_evaluation_point, &ram_raf_inputs)?;
            let ram_output_points =
                ram_output.derive_opening_points(ram_output_check_point, &ram_output_inputs)?;

            // Pair each produced opening point with its committed value to form the
            // located claims the output `Expr`s consume.
            let batch_outputs = &claims.batch_outputs;
            let located = |point: &[PCS::Field], value: PCS::Field| OpeningClaim {
                point: point.to_vec(),
                value,
            };
            let ram_read_write_located = RamReadWriteOutputClaims {
                val: located(&ram_read_write_points.val, batch_outputs.ram_read_write.val),
                ra: located(&ram_read_write_points.ra, batch_outputs.ram_read_write.ra),
                inc: located(&ram_read_write_points.inc, batch_outputs.ram_read_write.inc),
            };
            let product_remainder_located = ProductRemainderOutputClaims {
                left_instruction_input: located(
                    &product_remainder_points.left_instruction_input,
                    batch_outputs.product_remainder.left_instruction_input,
                ),
                right_instruction_input: located(
                    &product_remainder_points.right_instruction_input,
                    batch_outputs.product_remainder.right_instruction_input,
                ),
                jump_flag: located(
                    &product_remainder_points.jump_flag,
                    batch_outputs.product_remainder.jump_flag,
                ),
                write_lookup_output_to_rd: located(
                    &product_remainder_points.write_lookup_output_to_rd,
                    batch_outputs.product_remainder.write_lookup_output_to_rd,
                ),
                lookup_output: located(
                    &product_remainder_points.lookup_output,
                    batch_outputs.product_remainder.lookup_output,
                ),
                branch_flag: located(
                    &product_remainder_points.branch_flag,
                    batch_outputs.product_remainder.branch_flag,
                ),
                next_is_noop: located(
                    &product_remainder_points.next_is_noop,
                    batch_outputs.product_remainder.next_is_noop,
                ),
                virtual_instruction: located(
                    &product_remainder_points.virtual_instruction,
                    batch_outputs.product_remainder.virtual_instruction,
                ),
            };
            // The reduced instruction openings share the product remainder's point;
            // the three aliased openings, absent on the wire, reuse the
            // product-remainder values (or zero if the points disagree — a defensive
            // fallback that mirrors the legacy reconstruction).
            let reduction = &batch_outputs.instruction_claim_reduction;
            let product_values = &batch_outputs.product_remainder;
            let instruction_opening = instruction_reduction_points.left_lookup_operand.as_slice();
            let points_match =
                product_remainder_points.left_instruction_input.as_slice() == instruction_opening;
            let aliased = |value: Option<PCS::Field>, product: PCS::Field| {
                let resolved = value.unwrap_or(if points_match {
                    product
                } else {
                    PCS::Field::from_u64(0)
                });
                Some(OpeningClaim {
                    point: instruction_opening.to_vec(),
                    value: resolved,
                })
            };
            let instruction_reduction_located = InstructionClaimReductionOutputClaims {
                lookup_output: aliased(reduction.lookup_output, product_values.lookup_output),
                left_lookup_operand: located(instruction_opening, reduction.left_lookup_operand),
                right_lookup_operand: located(instruction_opening, reduction.right_lookup_operand),
                left_instruction_input: aliased(
                    reduction.left_instruction_input,
                    product_values.left_instruction_input,
                ),
                right_instruction_input: aliased(
                    reduction.right_instruction_input,
                    product_values.right_instruction_input,
                ),
            };
            let ram_raf_located = RamRafEvaluationOutputClaims {
                ram_ra: located(&ram_raf_points.ram_ra, batch_outputs.ram_raf_evaluation),
            };
            let ram_output_located = RamOutputCheckOutputClaims {
                val_final: located(&ram_output_points.val_final, batch_outputs.ram_output_check),
            };

            let expected_outputs = Stage2BatchExpectedOutputClaims {
                ram_read_write: ram_read_write
                    .expected_output(&ram_read_write_inputs, &ram_read_write_located)?,
                product_remainder: product_remainder
                    .expected_output(&product_remainder_inputs, &product_remainder_located)?,
                instruction_claim_reduction: instruction_reduction
                    .expected_output(&instruction_reduction_inputs, &instruction_reduction_located)?,
                ram_raf_evaluation: ram_raf.expected_output(&ram_raf_inputs, &ram_raf_located)?,
                ram_output_check: ram_output.expected_output(&ram_output_inputs, &ram_output_located)?,
            };

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
                        input_claim: ram_read_write_input_claim,
                        sumcheck_point: ram_read_write_point.to_vec(),
                        opening_point: ram_read_write_located.val.point.clone(),
                        expected_output_claim: expected_outputs.ram_read_write,
                    },
                    product_remainder: VerifiedStage2Sumcheck {
                        input_claim: product_remainder_input_claim,
                        sumcheck_point: product_point.to_vec(),
                        opening_point: product_remainder_located
                            .left_instruction_input
                            .point
                            .clone(),
                        expected_output_claim: expected_outputs.product_remainder,
                    },
                    instruction_claim_reduction: VerifiedStage2Sumcheck {
                        input_claim: instruction_reduction_input_claim,
                        sumcheck_point: instruction_point.to_vec(),
                        opening_point: instruction_reduction_located
                            .left_lookup_operand
                            .point
                            .clone(),
                        expected_output_claim: expected_outputs.instruction_claim_reduction,
                    },
                    ram_raf_evaluation: VerifiedStage2Sumcheck {
                        input_claim: ram_raf_input_claim,
                        sumcheck_point: ram_raf_evaluation_point.to_vec(),
                        opening_point: ram_raf_located.ram_ra.point.clone(),
                        expected_output_claim: expected_outputs.ram_raf_evaluation,
                    },
                    ram_output_check: VerifiedStage2Sumcheck {
                        input_claim: ram_output_input_claim,
                        sumcheck_point: ram_output_check_point.to_vec(),
                        opening_point: ram_output_located.val_final.point.clone(),
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
