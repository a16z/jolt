use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::TraceDimensions, ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    JoltRelationId,
};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    CenteredIntegerDomain, CommittedSumcheckConsistency, SumcheckClaim, SumcheckStatement,
    UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::{
    instruction_claim_reduction::{
        instruction_claim_reduction_input_values_from_upstream, InstructionClaimReduction,
        InstructionClaimReductionInputClaims,
    },
    outputs::{
        Stage2BatchInputClaims, Stage2BatchInputPoints, Stage2BatchSumchecks, Stage2ClearOutput,
        Stage2Output, Stage2ZkOutput,
    },
    product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainder,
        ProductRemainderInputClaims,
    },
    product_uniskip::{product_uniskip_input_values_from_stage1, ProductUniskip},
    ram_output_check::{RamOutputCheck, RamOutputCheckInputClaims},
    ram_raf_evaluation::{
        ram_raf_evaluation_input_values_from_upstream, RamRafEvaluation,
        RamRafEvaluationInputClaims,
    },
    ram_read_write_checking::{
        ram_read_write_input_values_from_upstream, RamReadWriteChecking, RamReadWriteInputClaims,
    },
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::ConcreteSumcheck,
        stage1::{Stage1ClearOutput, Stage1Output},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// The number of opening claims the stage-2 batch commits/absorbs: 15, not the 16
/// the members' output expressions reference, because the three aliased
/// instruction-claim-reduction openings (`lookup_output`,
/// `left`/`right_instruction_input`) are absorbed once via their canonical
/// product-remainder source (see
/// [`Stage2BatchOutputClaims::opening_values`](super::outputs::Stage2BatchOutputClaims)).
const STAGE2_BATCH_OUTPUT_CLAIMS: usize = 15;

/// The product uni-skip step's outputs: the tau bindings and the uni-skip
/// reduction challenge are extracted mode-agnostically (clear: the single-entry
/// reduction point; ZK: the committed round challenge) so the batch relations —
/// `ProductRemainder::new` in particular — can be built before the mode branch.
struct ProductUniskipStep<F: Field, C> {
    tau_low: Vec<F>,
    tau_high: F,
    challenge: F,
    verified: ProductUniskipVerified<F, C>,
}

enum ProductUniskipVerified<F: Field, C> {
    Clear,
    Zk {
        consistency: CommittedSumcheckConsistency<F, C>,
        output_claims: committed::CommittedOutputClaimOutput<C>,
    },
}

/// Assemble the stage-2 batch consumed opening *values* from the upstream clear
/// outputs into the generated `Stage2BatchInputClaims` aggregate. Each per-relation
/// `*_from_upstream` helper wires which upstream opening feeds which downstream
/// input. The product-remainder input is the product uni-skip's output claim (a
/// separate stage-2 sub-sumcheck), not an upstream stage's opening.
fn stage2_batch_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    product_uniskip_output_claim: F,
) -> Stage2BatchInputClaims<F> {
    Stage2BatchInputClaims {
        ram_read_write: ram_read_write_input_values_from_upstream(stage1),
        product_remainder: product_remainder_input_values_from_uniskip_output(
            product_uniskip_output_claim,
        ),
        instruction_claim_reduction: instruction_claim_reduction_input_values_from_upstream(stage1),
        ram_raf_evaluation: ram_raf_evaluation_input_values_from_upstream(stage1),
        ram_output_check: RamOutputCheckInputClaims::default(),
    }
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

    let uniskip =
        verify_product_uniskip::<PCS, VC, T, ZkProof>(checked, proof, transcript, stage1)?;

    // Build the five batch relations once, pre-branch; each owns its input/output
    // claim algebra (single-sourced with its jolt-claims formula and the BlindFold
    // constraint). The product uni-skip stays hand-coded above.
    let lowest_address = checked.public_io.memory_layout.get_lowest_address();
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;
    let mut sumchecks = Stage2BatchSumchecks {
        ram_read_write: RamReadWriteChecking::new(
            read_write_dimensions,
            log_k,
            uniskip.tau_low.clone(),
        ),
        product_remainder: ProductRemainder::new(
            product_dimensions,
            uniskip.challenge,
            uniskip.tau_high,
            uniskip.tau_low.clone(),
        ),
        instruction_claim_reduction: InstructionClaimReduction::new(
            trace_dimensions,
            uniskip.tau_low.clone(),
        ),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write_dimensions,
            raf_dimensions,
            log_k,
            lowest_address,
            uniskip.tau_low.clone(),
        ),
        // Two-phase construction: the output-check address reference point is
        // drawn AFTER the batch gammas, so the instance starts with a placeholder
        // and is completed right after the draws below (see
        // `set_output_address_challenges`).
        ram_output_check: RamOutputCheck::new(read_write_dimensions, Vec::new(), public_memory),
    };

    // Draw each relation's batching gamma in declaration order: the RAM read-write
    // gamma, then the instruction claim-reduction gamma (each a single
    // `challenge_scalar`; the other three batch relations draw nothing —
    // `NoChallenges`). The drawn challenges feed the input/output claims and
    // populate the stage aggregate carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    // The RAM output-check address reference point, drawn after both gammas. MUST
    // stay `challenge()` (not `challenge_scalar()`): both decode the same 16-byte
    // squeeze, but differently, so switching would silently change the address
    // point values without changing the transcript bytes.
    let output_address_challenges = (0..log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();
    sumchecks
        .ram_output_check
        .set_output_address_challenges(output_address_challenges.clone());

    // Every member's input points are empty (each derives its output points from its
    // own sumcheck point).
    let input_points = Stage2BatchInputPoints {
        ram_read_write: RamReadWriteInputClaims {
            ram_read_value: Vec::new(),
            ram_write_value: Vec::new(),
        },
        product_remainder: ProductRemainderInputClaims {
            product_uniskip: Vec::new(),
        },
        instruction_claim_reduction: InstructionClaimReductionInputClaims {
            lookup_output: Vec::new(),
            left_lookup_operand: Vec::new(),
            right_lookup_operand: Vec::new(),
            left_instruction_input: Vec::new(),
            right_instruction_input: Vec::new(),
        },
        ram_raf_evaluation: RamRafEvaluationInputClaims {
            ram_address: Vec::new(),
        },
        ram_output_check: RamOutputCheckInputClaims::default(),
    };

    if checked.zk {
        let ProductUniskipVerified::Zk {
            consistency: product_uniskip_consistency,
            output_claims: product_uniskip_output_claims,
        } = uniskip.verified
        else {
            return Err(VerifierError::ExpectedCommittedProof {
                field: "stage2_uni_skip_first_round_proof",
            });
        };
        let consistency = sumchecks.verify_zk(&proof.stages.stage2_sumcheck_proof, transcript)?;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage2_sumcheck_proof,
            "stage2_sumcheck_proof",
            STAGE2_BATCH_OUTPUT_CLAIMS,
            JoltRelationId::RamReadWriteChecking,
        )?;
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        return Ok(Stage2Output::Zk(Stage2ZkOutput {
            challenges,
            product_uniskip_challenge: uniskip.challenge,
            product_tau_low: uniskip.tau_low,
            product_tau_high: uniskip.tau_high,
            output_address_challenges,
            product_uniskip_consistency,
            product_uniskip_output_claims,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
        }));
    }

    let ProductUniskipVerified::Clear = uniskip.verified else {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage2_uni_skip_first_round_proof",
        });
    };
    let stage1 = stage1.clear()?;
    let claims = &proof.clear_claims()?.stage2;

    let input_values =
        stage2_batch_input_values_from_upstream(stage1, claims.product_uniskip_output_claim);

    let batch = sumchecks.verify_clear(
        &input_values,
        &challenges,
        &proof.stages.stage2_sumcheck_proof,
        transcript,
    )?;

    let output_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;

    let output_values = claims.batch_outputs.clone();
    output_values.validate(&output_points)?;

    // The reduction's output `Expr` reads the aliased openings, absent on the wire,
    // so the fold is fed the alias-filled effective aggregates (raw `None` openings
    // would error as `MissingOpeningClaim`).
    let (effective_values, effective_points) = output_values.effective_aggregates(&output_points);
    let expected_final_claim = sumchecks.expected_final_claim(
        &batch.coefficients,
        &input_points,
        &effective_values,
        &effective_points,
        &challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 2 });
    }

    claims.batch_outputs.append_to_transcript(transcript);

    Ok(Stage2Output::Clear(Stage2ClearOutput {
        output_values,
        output_points,
        product_tau_low: uniskip.tau_low,
    }))
}

fn verify_product_uniskip<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
) -> Result<ProductUniskipStep<PCS::Field, VC::Output>, VerifierError>
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
    let uniskip_rounds = 1;
    let uniskip_degree = SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE;
    match stage1 {
        Stage1Output::Clear(stage1) => {
            let claims = &proof.clear_claims()?.stage2;
            let uniskip = ProductUniskip::new(product_dimensions, tau_high);
            let uniskip_input_values = product_uniskip_input_values_from_stage1(stage1);

            let uniskip_claim = claims.product_uniskip_output_claim;
            let uniskip_input_claim =
                uniskip.input_claim(&uniskip_input_values, &NoChallenges::default())?;

            let uniskip_reduction = proof
                .stages
                .stage2_uni_skip_first_round_proof
                .verify(
                    &SumcheckClaim::new(uniskip_rounds, uniskip_degree, uniskip_input_claim),
                    CenteredIntegerDomain::new(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE),
                    UNISKIP_ROUND_TRANSCRIPT_LABEL,
                    transcript,
                )
                .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: error.to_string(),
                })?;
            if uniskip_reduction.value != uniskip_claim {
                return Err(VerifierError::StageClaimOutputMismatch { stage: 2 });
            }

            transcript.append_labeled(b"opening_claim", &uniskip_claim);

            let [challenge] = uniskip_reduction.point.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: "product uni-skip proof did not reduce to one challenge".to_string(),
                });
            };
            Ok(ProductUniskipStep {
                tau_low,
                tau_high,
                challenge: *challenge,
                verified: ProductUniskipVerified::Clear,
            })
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
                checked,
                &proof.stages.stage2_uni_skip_first_round_proof,
                "stage2_uni_skip_first_round_proof",
                1,
                stage,
            )?;
            let [round] = consistency.rounds.as_slice() else {
                return Err(VerifierError::StageClaimSumcheckFailed {
                    stage,
                    reason: "product uni-skip committed consistency did not produce one challenge"
                        .to_string(),
                });
            };

            Ok(ProductUniskipStep {
                tau_low,
                tau_high,
                challenge: round.challenge,
                verified: ProductUniskipVerified::Zk {
                    consistency,
                    output_claims,
                },
            })
        }
    }
}
