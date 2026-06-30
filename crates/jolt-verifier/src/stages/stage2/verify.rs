use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::TraceDimensions, ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    relations, JoltRelationId, JoltSumcheckDomain,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
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
    instruction_claim_reduction::{
        instruction_claim_reduction_inputs_from_upstream, InstructionClaimReduction,
        InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
        InstructionClaimReductionOutputClaims,
    },
    outputs::{
        Stage2BatchChallenges, Stage2BatchInputClaims, Stage2BatchOutputClaims,
        Stage2BatchSumchecks, Stage2ClearOutput, Stage2Output, Stage2ZkOutput,
        VerifiedProductUniSkip,
    },
    product_remainder::{
        product_remainder_inputs_from_uniskip_output, ProductRemainder, ProductRemainderInputClaims,
    },
    product_uniskip::{product_uniskip_inputs_from_stage1, ProductUniskip},
    ram_output_check::{
        ram_output_check_inputs_from_upstream, RamOutputCheck, RamOutputCheckInputClaims,
    },
    ram_raf_evaluation::{
        ram_raf_evaluation_inputs_from_upstream, RamRafEvaluation, RamRafEvaluationInputClaims,
    },
    ram_read_write_checking::{
        ram_read_write_inputs_from_upstream, RamReadWriteChallenges, RamReadWriteChecking,
        RamReadWriteInputClaims,
    },
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, ConcreteSumcheck, OpeningClaim,
        },
        stage1::{Stage1ClearOutput, Stage1Output},
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
    batch_challenges: Stage2BatchChallenges<F>,
    output_address_challenges: Vec<F>,
    consistency: jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    output_claims: committed::CommittedOutputClaimOutput<C>,
    output_points: Stage2BatchOutputClaims<F, Vec<F>>,
}

// The clear variant carries the opening claims (point + value); the ZK variant
// carries committed consistency plus the point-only `output_points`. Boxing the
// common clear variant to shrink the rarer ZK one would add indirection to every
// clear-path access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the located opening claims read on the hot path; boxing it would penalize the common case"
)]
enum Stage2Batch<F: Field, C> {
    Clear {
        output_claims: Stage2BatchOutputClaims<F, OpeningClaim<F>>,
    },
    Zk(Stage2ZkBatch<F, C>),
}

const PRODUCT_UNISKIP_OUTPUT_CLAIMS: usize = 1;
const STAGE2_BATCH_OUTPUT_CLAIMS: usize = 15;

fn selected_product_uniskip_rounds() -> usize {
    1
}

fn selected_product_uniskip_degree() -> usize {
    SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE
}

fn selected_product_uniskip_domain() -> jolt_claims::protocols::jolt::JoltSumcheckDomain {
    jolt_claims::protocols::jolt::JoltSumcheckDomain::centered_integer(
        SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
    )
}

/// Pair every produced batch opening point with its committed value into the
/// `OpeningClaim` (point + value) form the output `Expr`s and later stages consume.
/// The three aliased instruction-claim-reduction openings, absent on the wire,
/// reuse the product-remainder openings at the shared point (or zero when the
/// points disagree — a defensive fallback that mirrors the legacy reconstruction).
/// Shared by the verifier and the prover so the opening-claim form is built once.
pub fn stage2_batch_output_claims_with_points<F: Field>(
    claims: &Stage2BatchOutputClaims<F, F>,
    points: &Stage2BatchOutputClaims<F, Vec<F>>,
) -> Stage2BatchOutputClaims<F, OpeningClaim<F>> {
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

/// Assemble the stage-2 batch consumed openings from the upstream clear outputs
/// into the generated `Stage2BatchInputClaims` aggregate. This is the single place
/// the batch's Outputs→Inputs dataflow is expressed: each per-relation
/// `*_from_upstream` helper wires which upstream opening feeds which downstream
/// input. The product-remainder input is the product uni-skip's output claim (a
/// separate stage-2 sub-sumcheck), not an upstream stage's opening.
fn stage2_batch_inputs_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    product_uniskip_output_claim: F,
) -> Stage2BatchInputClaims<F, OpeningClaim<F>> {
    Stage2BatchInputClaims {
        ram_read_write: ram_read_write_inputs_from_upstream(stage1),
        product_remainder: product_remainder_inputs_from_uniskip_output(
            product_uniskip_output_claim,
        ),
        instruction_claim_reduction: instruction_claim_reduction_inputs_from_upstream(stage1),
        ram_raf_evaluation: ram_raf_evaluation_inputs_from_upstream(stage1),
        ram_output_check: ram_output_check_inputs_from_upstream(),
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
    T: Transcript<Challenge = PCS::Field>,
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
        (Stage2ProductUniSkip::Clear(product_uniskip), Stage2Batch::Clear { output_claims }) => {
            Ok(Stage2Output::Clear(Stage2ClearOutput {
                output_claims,
                product_uniskip,
            }))
        }
        (Stage2ProductUniSkip::Zk(product_uniskip), Stage2Batch::Zk(batch)) => {
            Ok(Stage2Output::Zk(Stage2ZkOutput {
                challenges: batch.batch_challenges,
                product_uniskip_challenge: product_uniskip.product_uniskip_challenge,
                product_tau_high: product_uniskip.tau_high,
                output_address_challenges: batch.output_address_challenges,
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
    T: Transcript<Challenge = PCS::Field>,
{
    let stage = JoltRelationId::SpartanProductVirtualization;
    let log_t = checked.trace_length.ilog2() as usize;
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let stage1_remainder = stage1.remainder_point();
    let mut tau_low = stage1_remainder
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
    let uniskip_rounds = selected_product_uniskip_rounds();
    let uniskip_degree = selected_product_uniskip_degree();
    let uniskip_domain = selected_product_uniskip_domain();
    if uniskip_degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage,
            degree: uniskip_degree,
        });
    }
    let JoltSumcheckDomain::CenteredInteger { domain_size } = uniskip_domain else {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: "Stage 2 product uni-skip sumcheck must use the centered-integer domain"
                .to_string(),
        });
    };
    match stage1 {
        Stage1Output::Clear(stage1) => {
            let claims = &proof.clear_claims()?.stage2;
            let uniskip = ProductUniskip::new(product_dimensions, tau_high);
            let uniskip_inputs = product_uniskip_inputs_from_stage1(stage1);

            let uniskip_claim = claims.product_uniskip_output_claim;
            let uniskip_input_claim =
                uniskip.input_claim(&uniskip_inputs, &NoChallenges::default())?;

            let uniskip_reduction = proof
                .stages
                .stage2_uni_skip_first_round_proof
                .verify(
                    &SumcheckClaim::new(uniskip_rounds, uniskip_degree, uniskip_input_claim),
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
                sumcheck_point: uniskip_reduction.point,
            }))
        }
        Stage1Output::Zk(_) => {
            let consistency = proof
                .stages
                .stage2_uni_skip_first_round_proof
                .verify_committed_consistency(
                    SumcheckStatement::new(uniskip_rounds, uniskip_degree),
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

    let ram_read_write_rel = relations::ram::ReadWriteChecking::new(read_write_dimensions);
    let product_remainder_rel = relations::spartan::ProductRemainder::new(product_dimensions);
    let instruction_claim_reduction_rel =
        relations::claim_reductions::instruction::ClaimReduction::new(trace_dimensions);
    let ram_raf_evaluation_rel = relations::ram::RafEvaluation::new(raf_dimensions);
    let ram_output_check_rel = relations::ram::OutputCheck::new(read_write_dimensions);

    struct RelSpec {
        rounds: usize,
        degree: usize,
        domain: JoltSumcheckDomain,
    }
    let ram_read_write_claims = RelSpec {
        rounds: ram_read_write_rel.rounds(),
        degree: ram_read_write_rel.degree(),
        domain: ram_read_write_rel.domain(),
    };
    let product_remainder_claims = RelSpec {
        rounds: product_remainder_rel.rounds(),
        degree: product_remainder_rel.degree(),
        domain: product_remainder_rel.domain(),
    };
    let instruction_claim_reduction_claims = RelSpec {
        rounds: instruction_claim_reduction_rel.rounds(),
        degree: instruction_claim_reduction_rel.degree(),
        domain: instruction_claim_reduction_rel.domain(),
    };
    let ram_raf_evaluation_claims = RelSpec {
        rounds: ram_raf_evaluation_rel.rounds(),
        degree: ram_raf_evaluation_rel.degree(),
        domain: ram_raf_evaluation_rel.domain(),
    };
    let ram_output_check_claims = RelSpec {
        rounds: ram_output_check_rel.rounds(),
        degree: ram_output_check_rel.degree(),
        domain: ram_output_check_rel.domain(),
    };
    // The RAM read-write and instruction-reduction batching gammas (each a single
    // `challenge_scalar`, matching their default `draw_challenges`), drawn in inline
    // order into explicit locals — fixing the Fiat-Shamir draw order independent of
    // struct-field evaluation — then assembled into the generated batch-challenge
    // aggregate. The other three batch relations draw no challenges (`NoChallenges`).
    // The aggregate is carried on the ZK arm's `Stage2ZkOutput.challenges`. The
    // non-challenge `output_address_challenges` point draw stays in place, after both
    // gammas.
    let ram_read_write_challenges = RamReadWriteChallenges {
        gamma: transcript.challenge_scalar(),
    };
    let instruction_challenges = InstructionClaimReductionChallenges {
        gamma: transcript.challenge_scalar(),
    };
    let batch_challenges = Stage2BatchChallenges {
        ram_read_write: ram_read_write_challenges,
        product_remainder: NoChallenges::default(),
        instruction_claim_reduction: instruction_challenges,
        ram_raf_evaluation: NoChallenges::default(),
        ram_output_check: NoChallenges::default(),
    };
    let output_address_challenges = (0..log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    for (relation, domain, degree) in [
        (
            relations::ram::ReadWriteChecking::id(),
            ram_read_write_claims.domain,
            ram_read_write_claims.degree,
        ),
        (
            relations::spartan::ProductRemainder::id(),
            product_remainder_claims.domain,
            product_remainder_claims.degree,
        ),
        (
            relations::claim_reductions::instruction::ClaimReduction::id(),
            instruction_claim_reduction_claims.domain,
            instruction_claim_reduction_claims.degree,
        ),
        (
            relations::ram::RafEvaluation::id(),
            ram_raf_evaluation_claims.domain,
            ram_raf_evaluation_claims.degree,
        ),
        (
            relations::ram::OutputCheck::id(),
            ram_output_check_claims.domain,
            ram_output_check_claims.degree,
        ),
    ] {
        check_relation_boolean_hypercube(relation, domain, degree)?;
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
            let sumchecks = Stage2BatchSumchecks {
                ram_read_write: RamReadWriteChecking::new(
                    read_write_dimensions,
                    log_k,
                    product_uniskip.tau_low.clone(),
                ),
                product_remainder: ProductRemainder::new(
                    product_dimensions,
                    product_uniskip_challenge,
                    product_uniskip.tau_high,
                    product_uniskip.tau_low.clone(),
                ),
                instruction_claim_reduction: InstructionClaimReduction::new(
                    trace_dimensions,
                    product_uniskip.tau_low.clone(),
                ),
                ram_raf_evaluation: RamRafEvaluation::new(
                    read_write_dimensions,
                    raf_dimensions,
                    log_k,
                    lowest_address,
                    product_uniskip.tau_low.clone(),
                ),
                ram_output_check: RamOutputCheck::new(
                    read_write_dimensions,
                    output_address_challenges.clone(),
                    public_memory,
                ),
            };

            let inputs =
                stage2_batch_inputs_from_upstream(stage1, claims.product_uniskip_output_claim);

            // The claim order here must match the output-claim reconstruction below
            // and the transcript appends at the end of the stage.
            let sumcheck_claims = vec![
                SumcheckClaim::new(
                    ram_read_write_claims.rounds,
                    ram_read_write_claims.degree,
                    sumchecks
                        .ram_read_write
                        .input_claim(&inputs.ram_read_write, &batch_challenges.ram_read_write)?,
                ),
                SumcheckClaim::new(
                    product_remainder_claims.rounds,
                    product_remainder_claims.degree,
                    sumchecks.product_remainder.input_claim(
                        &inputs.product_remainder,
                        &batch_challenges.product_remainder,
                    )?,
                ),
                SumcheckClaim::new(
                    instruction_claim_reduction_claims.rounds,
                    instruction_claim_reduction_claims.degree,
                    sumchecks.instruction_claim_reduction.input_claim(
                        &inputs.instruction_claim_reduction,
                        &batch_challenges.instruction_claim_reduction,
                    )?,
                ),
                SumcheckClaim::new(
                    ram_raf_evaluation_claims.rounds,
                    ram_raf_evaluation_claims.degree,
                    sumchecks.ram_raf_evaluation.input_claim(
                        &inputs.ram_raf_evaluation,
                        &batch_challenges.ram_raf_evaluation,
                    )?,
                ),
                SumcheckClaim::new(
                    ram_output_check_claims.rounds,
                    ram_output_check_claims.degree,
                    sumchecks.ram_output_check.input_claim(
                        &inputs.ram_output_check,
                        &batch_challenges.ram_output_check,
                    )?,
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
                .try_instance_point(ram_read_write_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let product_point = batch
                .try_instance_point(product_remainder_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?;
            let instruction_point = batch
                .try_instance_point(instruction_claim_reduction_claims.rounds)
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
                .try_instance_point_at(phase1_offset, ram_raf_evaluation_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_output_check_point = batch
                .try_instance_point_at(phase1_offset, ram_output_check_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamOutputCheck,
                    reason: error.to_string(),
                })?;

            // Each relation maps its sumcheck point to its produced opening points;
            // pair them with the committed values into the opening claims.
            let points = Stage2BatchOutputClaims {
                ram_read_write: sumchecks
                    .ram_read_write
                    .derive_opening_points(ram_read_write_point, &inputs.ram_read_write)?,
                product_remainder: sumchecks
                    .product_remainder
                    .derive_opening_points(product_point, &inputs.product_remainder)?,
                instruction_claim_reduction: sumchecks
                    .instruction_claim_reduction
                    .derive_opening_points(
                        instruction_point,
                        &inputs.instruction_claim_reduction,
                    )?,
                ram_raf_evaluation: sumchecks
                    .ram_raf_evaluation
                    .derive_opening_points(ram_raf_evaluation_point, &inputs.ram_raf_evaluation)?,
                ram_output_check: sumchecks
                    .ram_output_check
                    .derive_opening_points(ram_output_check_point, &inputs.ram_output_check)?,
            };
            let output_claims =
                stage2_batch_output_claims_with_points(&claims.batch_outputs, &points);
            output_claims.validate()?;

            let expected_final_claim = stage2_expected_final_claim(
                &batch.batching_coefficients,
                sumchecks.ram_read_write.expected_output(
                    &inputs.ram_read_write,
                    &output_claims.ram_read_write,
                    &batch_challenges.ram_read_write,
                )?,
                sumchecks.product_remainder.expected_output(
                    &inputs.product_remainder,
                    &output_claims.product_remainder,
                    &batch_challenges.product_remainder,
                )?,
                sumchecks.instruction_claim_reduction.expected_output(
                    &inputs.instruction_claim_reduction,
                    &output_claims.instruction_claim_reduction,
                    &batch_challenges.instruction_claim_reduction,
                )?,
                sumchecks.ram_raf_evaluation.expected_output(
                    &inputs.ram_raf_evaluation,
                    &output_claims.ram_raf_evaluation,
                    &batch_challenges.ram_raf_evaluation,
                )?,
                sumchecks.ram_output_check.expected_output(
                    &inputs.ram_output_check,
                    &output_claims.ram_output_check,
                    &batch_challenges.ram_output_check,
                )?,
            )?;
            if batch.reduction.value != expected_final_claim {
                return Err(VerifierError::StageClaimOutputMismatch {
                    stage: JoltRelationId::RamReadWriteChecking,
                });
            }

            claims.batch_outputs.append_to_transcript(transcript);

            Ok(Stage2Batch::Clear { output_claims })
        }
        (Stage1Output::Zk(_), Stage2ProductUniSkip::Zk(product_uniskip)) => {
            let statements = vec![
                SumcheckStatement::new(ram_read_write_claims.rounds, ram_read_write_claims.degree),
                SumcheckStatement::new(
                    product_remainder_claims.rounds,
                    product_remainder_claims.degree,
                ),
                SumcheckStatement::new(
                    instruction_claim_reduction_claims.rounds,
                    instruction_claim_reduction_claims.degree,
                ),
                SumcheckStatement::new(
                    ram_raf_evaluation_claims.rounds,
                    ram_raf_evaluation_claims.degree,
                ),
                SumcheckStatement::new(
                    ram_output_check_claims.rounds,
                    ram_output_check_claims.degree,
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
                .try_instance_point(ram_read_write_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamReadWriteChecking,
                    reason: error.to_string(),
                })?;
            let product_point = consistency
                .try_instance_point(product_remainder_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::SpartanProductVirtualization,
                    reason: error.to_string(),
                })?;
            let instruction_point = consistency
                .try_instance_point(instruction_claim_reduction_claims.rounds)
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
                .try_instance_point_at(phase1_offset, ram_raf_evaluation_claims.rounds)
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage: JoltRelationId::RamRafEvaluation,
                    reason: error.to_string(),
                })?;
            let ram_output_check_point = consistency
                .try_instance_point_at(phase1_offset, ram_output_check_claims.rounds)
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
            let sumchecks = Stage2BatchSumchecks {
                ram_read_write: RamReadWriteChecking::new(
                    read_write_dimensions,
                    log_k,
                    product_uniskip.tau_low.clone(),
                ),
                product_remainder: ProductRemainder::new(
                    product_dimensions,
                    product_uniskip.product_uniskip_challenge,
                    product_uniskip.tau_high,
                    product_uniskip.tau_low.clone(),
                ),
                instruction_claim_reduction: InstructionClaimReduction::new(
                    trace_dimensions,
                    product_uniskip.tau_low.clone(),
                ),
                ram_raf_evaluation: RamRafEvaluation::new(
                    read_write_dimensions,
                    raf_dimensions,
                    log_k,
                    lowest_address,
                    product_uniskip.tau_low.clone(),
                ),
                ram_output_check: RamOutputCheck::new(
                    read_write_dimensions,
                    output_address_challenges.clone(),
                    public_memory,
                ),
            };

            // The relations' `derive_opening_points` ignore their inputs, so the ZK
            // arm passes empty point-cell inputs assembled into the same generated
            // aggregate the clear arm builds.
            let empty = Vec::<PCS::Field>::new;
            let inputs = Stage2BatchInputClaims::<PCS::Field, Vec<PCS::Field>> {
                ram_read_write: RamReadWriteInputClaims {
                    ram_read_value: empty(),
                    ram_write_value: empty(),
                },
                product_remainder: ProductRemainderInputClaims {
                    product_uniskip: empty(),
                },
                instruction_claim_reduction: InstructionClaimReductionInputClaims {
                    lookup_output: empty(),
                    left_lookup_operand: empty(),
                    right_lookup_operand: empty(),
                    left_instruction_input: empty(),
                    right_instruction_input: empty(),
                },
                ram_raf_evaluation: RamRafEvaluationInputClaims {
                    ram_address: empty(),
                },
                ram_output_check: RamOutputCheckInputClaims::<Vec<PCS::Field>>::default(),
            };
            let output_points = Stage2BatchOutputClaims {
                ram_read_write: sumchecks
                    .ram_read_write
                    .derive_opening_points(&ram_read_write_point, &inputs.ram_read_write)?,
                product_remainder: sumchecks
                    .product_remainder
                    .derive_opening_points(&product_point, &inputs.product_remainder)?,
                instruction_claim_reduction: sumchecks
                    .instruction_claim_reduction
                    .derive_opening_points(
                        &instruction_point,
                        &inputs.instruction_claim_reduction,
                    )?,
                ram_raf_evaluation: sumchecks
                    .ram_raf_evaluation
                    .derive_opening_points(&ram_raf_evaluation_point, &inputs.ram_raf_evaluation)?,
                ram_output_check: sumchecks
                    .ram_output_check
                    .derive_opening_points(&ram_output_check_point, &inputs.ram_output_check)?,
            };

            Ok(Stage2Batch::Zk(Stage2ZkBatch {
                batch_challenges,
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
