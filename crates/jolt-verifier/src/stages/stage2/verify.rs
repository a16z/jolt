use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction,
        dimensions::TraceDimensions,
        ram::{self, RamRafEvaluationDimensions},
        spartan::{product_remainder, SpartanProductDimensions},
    },
    JoltRelationId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::lagrange::centered_lagrange_evals;
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
};
use jolt_transcript::FsTranscript;

use super::{
    instruction_claim_reduction::{
        InstructionClaimReduction, InstructionClaimReductionInputClaims,
        InstructionClaimReductionOutputClaims,
    },
    outputs::{
        product_uniskip_input_claim, Stage2BatchOutputClaims, Stage2ClearOutput, Stage2Output,
        Stage2ProductUniSkipInputValues, Stage2PublicOutput, Stage2ZkOutput,
        VerifiedProductUniSkip,
    },
    product_remainder::{ProductRemainder, ProductRemainderInputClaims},
    ram_output_check::{RamOutputCheck, RamOutputCheckInputClaims},
    ram_raf_evaluation::{RamRafEvaluation, RamRafEvaluationInputClaims},
    ram_read_write_checking::{RamReadWriteChecking, RamReadWriteInputClaims},
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, OpeningClaim, SumcheckInstance,
        },
        stage1::Stage1Output,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

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
    ram_read_write_gamma: F,
    instruction_gamma: F,
    output_address_challenges: Vec<F>,
    consistency: jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    output_claims: committed::CommittedOutputClaimOutput<C>,
    output_points: Stage2BatchOutputClaims<Vec<F>>,
}

// The clear variant carries the opening claims (point + value); the ZK variant
// carries committed consistency plus the point-only `output_points`.
enum Stage2Batch<F: Field, C> {
    Clear {
        public: Stage2PublicOutput<F>,
        output_claims: Stage2BatchOutputClaims<OpeningClaim<F>>,
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

/// Pair every produced batch opening point with its committed value into the
/// `OpeningClaim` (point + value) form the output `Expr`s and later stages consume.
/// The three aliased instruction-claim-reduction openings, absent on the wire,
/// reuse the product-remainder openings at the shared point (or zero when the
/// points disagree — a defensive fallback that mirrors the legacy reconstruction).
/// Shared by the verifier and the prover so the opening-claim form is built once.
pub fn stage2_batch_output_claims_with_points<F: Field>(
    claims: &Stage2BatchOutputClaims<F>,
    points: &Stage2BatchOutputClaims<Vec<F>>,
) -> Stage2BatchOutputClaims<OpeningClaim<F>> {
    // The reduced instruction openings share one point; the three aliased openings,
    // absent on the wire, reuse the product-remainder values (or zero when the
    // product/instruction points disagree). This cross-relation fill cannot go
    // through the field-wise `zip_openings`, so it stays explicit.
    let reduction = &claims.instruction_claim_reduction;
    let product = &claims.product_remainder;
    let reduction_point = points
        .instruction_claim_reduction
        .left_lookup_operand
        .as_slice();
    let points_match =
        points.product_remainder.left_instruction_input.as_slice() == reduction_point;
    let opening = |value: F| OpeningClaim {
        point: reduction_point.to_vec(),
        value,
    };
    let aliased = |value: Option<F>, product_value: F| {
        Some(opening(value.unwrap_or(if points_match {
            product_value
        } else {
            F::from_u64(0)
        })))
    };
    let instruction_claim_reduction = InstructionClaimReductionOutputClaims {
        lookup_output: aliased(reduction.lookup_output, product.lookup_output),
        left_lookup_operand: opening(reduction.left_lookup_operand),
        right_lookup_operand: opening(reduction.right_lookup_operand),
        left_instruction_input: aliased(
            reduction.left_instruction_input,
            product.left_instruction_input,
        ),
        right_instruction_input: aliased(
            reduction.right_instruction_input,
            product.right_instruction_input,
        ),
    };
    Stage2BatchOutputClaims {
        ram_read_write: zip_openings(&claims.ram_read_write, &points.ram_read_write),
        product_remainder: zip_openings(&claims.product_remainder, &points.product_remainder),
        instruction_claim_reduction,
        ram_raf_evaluation: zip_openings(&claims.ram_raf_evaluation, &points.ram_raf_evaluation),
        ram_output_check: zip_openings(&claims.ram_output_check, &points.ram_output_check),
    }
}

/// Combine the five stage 2 batch expected output claims with the batch's
/// coefficients, in canonical order. Shared by the verifier and the prover.
pub fn stage2_expected_final_claim<F: Field>(
    coefficients: &[F],
    ram_read_write: F,
    product_remainder: F,
    instruction_claim_reduction: F,
    ram_raf_evaluation: F,
    ram_output_check: F,
) -> Result<F, VerifierError> {
    let [ram_read_write_coefficient, product_coefficient, instruction_coefficient, ram_raf_coefficient, ram_output_coefficient] =
        coefficients
    else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamReadWriteChecking,
            reason: "Stage 2 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(*ram_read_write_coefficient * ram_read_write
        + *product_coefficient * product_remainder
        + *instruction_coefficient * instruction_claim_reduction
        + *ram_raf_coefficient * ram_raf_evaluation
        + *ram_output_coefficient * ram_output_check)
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
) -> Result<Stage2Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsTranscript<PCS::Field>,
{
    match (checked.zk, stage1) {
        (true, Stage1Output::Clear(_)) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage1" });
        }
        (false, Stage1Output::Zk(_)) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage1" });
        }
        _ => {}
    }

    let product_uniskip =
        verify_product_uniskip::<PCS, VC, T, ZkProof>(checked, proof, transcript, stage1)?;
    let batch = verify_regular_batch::<PCS, VC, T, ZkProof>(
        checked,
        proof,
        transcript,
        &product_uniskip,
        stage1,
    )?;

    match (product_uniskip, batch) {
        (
            Stage2ProductUniSkip::Clear(product_uniskip),
            Stage2Batch::Clear {
                public,
                output_claims,
            },
        ) => Ok(Stage2Output::Clear(Stage2ClearOutput {
            public,
            output_claims,
            product_uniskip,
        })),
        (Stage2ProductUniSkip::Zk(product_uniskip), Stage2Batch::Zk(batch)) => {
            let public = Stage2PublicOutput {
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
                output_points: batch.output_points,
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
    stage1: &Stage1Output<PCS::Field, VC::Output>,
) -> Result<Stage2ProductUniSkip<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsTranscript<PCS::Field>,
{
    let stage = JoltRelationId::SpartanProductVirtualization;
    let log_t = checked.trace_length.ilog2() as usize;
    let _dimensions = SpartanProductDimensions::new(log_t);
    let stage1_public = stage1.public();
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
    match stage1 {
        Stage1Output::Clear(stage1) => {
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
                sumcheck_point: uniskip_reduction.point,
            }))
        }
        Stage1Output::Zk(_) => {
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
    stage1: &Stage1Output<PCS::Field, VC::Output>,
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
        check_relation_boolean_hypercube(claims)?;
    }

    match (stage1, product_uniskip) {
        (Stage1Output::Clear(stage1), Stage2ProductUniSkip::Clear(product_uniskip)) => {
            let claims = &proof.clear_claims()?.stage2;
            let [product_uniskip_challenge] = product_uniskip.sumcheck_point.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: "product uni-skip proof did not reduce to one challenge".to_string(),
                });
            };
            let product_uniskip_challenge = *product_uniskip_challenge;

            // Build the five batch relations inline; each owns its input/output
            // claim algebra (single-sourced with its jolt-claims formula and the
            // BlindFold constraint). The product uni-skip stays hand-coded above.
            let lowest_address = checked.public_io.memory_layout.get_lowest_address();
            let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                }
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
            let ram_output = RamOutputCheck::new(
                read_write_dimensions,
                output_address_challenges.clone(),
                public_memory,
            );

            let ram_read_write_inputs = RamReadWriteInputClaims::from_upstream(stage1);
            let product_remainder_inputs = ProductRemainderInputClaims::from_uniskip_output(
                claims.product_uniskip_output_claim,
            );
            let instruction_reduction_inputs =
                InstructionClaimReductionInputClaims::from_upstream(stage1);
            let ram_raf_inputs = RamRafEvaluationInputClaims::from_upstream(stage1);
            let ram_output_inputs = RamOutputCheckInputClaims::from_upstream();

            // The claim order here must match the output-claim reconstruction below
            // and the transcript appends at the end of the stage.
            let sumcheck_claims = vec![
                SumcheckClaim::new(
                    ram_read_write_claims.sumcheck.rounds,
                    ram_read_write_claims.sumcheck.degree,
                    ram_read_write.input_claim(&ram_read_write_inputs)?,
                ),
                SumcheckClaim::new(
                    product_remainder_claims.sumcheck.rounds,
                    product_remainder_claims.sumcheck.degree,
                    product_remainder.input_claim(&product_remainder_inputs)?,
                ),
                SumcheckClaim::new(
                    instruction_claim_reduction_claims.sumcheck.rounds,
                    instruction_claim_reduction_claims.sumcheck.degree,
                    instruction_reduction.input_claim(&instruction_reduction_inputs)?,
                ),
                SumcheckClaim::new(
                    ram_raf_evaluation_claims.sumcheck.rounds,
                    ram_raf_evaluation_claims.sumcheck.degree,
                    ram_raf.input_claim(&ram_raf_inputs)?,
                ),
                SumcheckClaim::new(
                    ram_output_check_claims.sumcheck.rounds,
                    ram_output_check_claims.sumcheck.degree,
                    ram_output.input_claim(&ram_output_inputs)?,
                ),
            ];
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

            // Each relation maps its sumcheck point to its produced opening points;
            // pair them with the committed values into the opening claims.
            let points = Stage2BatchOutputClaims {
                ram_read_write: ram_read_write
                    .derive_opening_points(ram_read_write_point, &ram_read_write_inputs)?,
                product_remainder: product_remainder
                    .derive_opening_points(product_point, &product_remainder_inputs)?,
                instruction_claim_reduction: instruction_reduction
                    .derive_opening_points(instruction_point, &instruction_reduction_inputs)?,
                ram_raf_evaluation: ram_raf
                    .derive_opening_points(ram_raf_evaluation_point, &ram_raf_inputs)?,
                ram_output_check: ram_output
                    .derive_opening_points(ram_output_check_point, &ram_output_inputs)?,
            };
            let output_claims =
                stage2_batch_output_claims_with_points(&claims.batch_outputs, &points);
            output_claims.validate()?;

            let expected_final_claim = stage2_expected_final_claim(
                &batch.batching_coefficients,
                ram_read_write
                    .expected_output(&ram_read_write_inputs, &output_claims.ram_read_write)?,
                product_remainder
                    .expected_output(&product_remainder_inputs, &output_claims.product_remainder)?,
                instruction_reduction.expected_output(
                    &instruction_reduction_inputs,
                    &output_claims.instruction_claim_reduction,
                )?,
                ram_raf.expected_output(&ram_raf_inputs, &output_claims.ram_raf_evaluation)?,
                ram_output.expected_output(&ram_output_inputs, &output_claims.ram_output_check)?,
            )?;
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

            let public = Stage2PublicOutput {
                product_uniskip_challenge,
                product_tau_low: product_uniskip.tau_low.clone(),
                product_tau_high: product_uniskip.tau_high,
                ram_read_write_gamma,
                instruction_gamma,
                output_address_challenges,
            };
            Ok(Stage2Batch::Clear {
                public,
                output_claims,
            })
        }
        (Stage1Output::Zk(_), Stage2ProductUniSkip::Zk(product_uniskip)) => {
            let statements = vec![
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
            // Map each relation's committed sumcheck point to its produced opening
            // points, building the point-only counterpart of the clear arm's
            // `output_claims`. The relations match the clear arm; their
            // `derive_opening_points` ignore inputs, so the ZK arm passes empty
            // point-cell inputs.
            let ram_read_write_point = consistency
                .try_instance_point(ram_read_write_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let product_point = consistency
                .try_instance_point(product_remainder_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?;
            let instruction_point = consistency
                .try_instance_point(instruction_claim_reduction_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::InstructionClaimReduction,
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
            let ram_output_check_point = consistency
                .try_instance_point_at(phase1_offset, ram_output_check_claims.sumcheck.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;

            let lowest_address = checked.public_io.memory_layout.get_lowest_address();
            let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                }
            })?;
            let ram_read_write = RamReadWriteChecking::new(
                read_write_dimensions,
                log_k,
                ram_read_write_gamma,
                product_uniskip.tau_low.clone(),
            );
            let product_remainder = ProductRemainder::new(
                product_dimensions,
                product_uniskip.product_uniskip_challenge,
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
            let ram_output = RamOutputCheck::new(
                read_write_dimensions,
                output_address_challenges.clone(),
                public_memory,
            );

            let empty = Vec::<PCS::Field>::new;
            let output_points = Stage2BatchOutputClaims {
                ram_read_write: ram_read_write.derive_opening_points(
                    &ram_read_write_point,
                    &RamReadWriteInputClaims {
                        ram_read_value: empty(),
                        ram_write_value: empty(),
                    },
                )?,
                product_remainder: product_remainder.derive_opening_points(
                    &product_point,
                    &ProductRemainderInputClaims {
                        product_uniskip: empty(),
                    },
                )?,
                instruction_claim_reduction: instruction_reduction.derive_opening_points(
                    &instruction_point,
                    &InstructionClaimReductionInputClaims {
                        lookup_output: empty(),
                        left_lookup_operand: empty(),
                        right_lookup_operand: empty(),
                        left_instruction_input: empty(),
                        right_instruction_input: empty(),
                    },
                )?,
                ram_raf_evaluation: ram_raf.derive_opening_points(
                    &ram_raf_evaluation_point,
                    &RamRafEvaluationInputClaims {
                        ram_address: empty(),
                    },
                )?,
                ram_output_check: ram_output.derive_opening_points(
                    &ram_output_check_point,
                    &RamOutputCheckInputClaims::<Vec<PCS::Field>>::default(),
                )?,
            };

            Ok(Stage2Batch::Zk(Stage2ZkBatch {
                ram_read_write_gamma,
                instruction_gamma,
                output_address_challenges,
                consistency,
                output_claims,
                output_points,
            }))
        }
        (Stage1Output::Clear(_), Stage2ProductUniSkip::Zk(_)) => {
            Err(VerifierError::ExpectedClearProof {
                field: "stage2_uni_skip_first_round_proof",
            })
        }
        (Stage1Output::Zk(_), Stage2ProductUniSkip::Clear(_)) => {
            Err(VerifierError::ExpectedCommittedProof {
                field: "stage2_uni_skip_first_round_proof",
            })
        }
    }
}
