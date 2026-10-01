#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::TraceDimensions, ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    JoltRelationId,
};
use jolt_claims::NoChallenges;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_program::preprocess::PublicIoMemory;
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::sites::STAGE2;

#[cfg(feature = "field-inline")]
use super::field_registers_claim_reduction::{
    field_registers_claim_reduction_input_values_from_upstream, FieldRegistersClaimReduction,
};
use super::{
    instruction_claim_reduction::{
        instruction_claim_reduction_input_values_from_upstream, InstructionClaimReduction,
    },
    outputs::{
        Stage2BatchInputClaims, Stage2BatchSumchecks, Stage2ClearOutput, Stage2Output,
        Stage2ZkOutput,
    },
    product_remainder::{product_remainder_input_values_from_uniskip_output, ProductRemainder},
    product_uniskip::{product_uniskip_input_values_from_stage1, ProductUniskip},
    ram_output_check::{RamOutputCheck, RamOutputCheckInputClaims},
    ram_raf_evaluation::{ram_raf_evaluation_input_values_from_upstream, RamRafEvaluation},
    ram_read_write_checking::{ram_read_write_input_values_from_upstream, RamReadWriteChecking},
};
use crate::{
    stages::{
        relations::ConcreteSumcheck,
        stage1::{Stage1ClearOutput, Stage1Output},
        uniskip,
        zk::{committed, outputs::CommittedOutputClaimOutput},
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// The product uni-skip step's outputs: the tau bindings and the uni-skip
/// reduction challenge are extracted mode-agnostically (clear: the single-entry
/// reduction point; ZK: the committed round challenge) so the batch relations —
/// `ProductRemainder::new` in particular — can be built before the mode branch.
struct ProductUniskipStep<F: JoltField, C> {
    tau_low: Vec<F>,
    tau_high: F,
    challenge: F,
    verified: ProductUniskipVerified<F, C>,
}

enum ProductUniskipVerified<F: JoltField, C> {
    Clear { output_claim: F },
    Zk(uniskip::UniskipZk<F, C>),
}

/// Assemble the stage-2 batch consumed opening *values* from the upstream clear outputs into
/// the generated `Stage2BatchInputClaims` aggregate. Each per-relation `*_from_upstream`
/// helper wires which upstream opening feeds which downstream input. The product-remainder
/// input is the product uni-skip's output claim (a separate stage-2 sub-sumcheck), not an
/// upstream stage's opening.
pub fn stage2_batch_input_values_from_upstream<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
    product_uniskip_output_claim: F,
) -> Stage2BatchInputClaims<F> {
    Stage2BatchInputClaims {
        ram_read_write: ram_read_write_input_values_from_upstream(stage1),
        product_remainder: product_remainder_input_values_from_uniskip_output(
            product_uniskip_output_claim,
        ),
        instruction_claim_reduction: instruction_claim_reduction_input_values_from_upstream(stage1),
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction: field_registers_claim_reduction_input_values_from_upstream(
            stage1,
        ),
        ram_raf_evaluation: ram_raf_evaluation_input_values_from_upstream(stage1),
        ram_output_check: RamOutputCheckInputClaims::default(),
    }
}

pub fn verify<F, C, H>(
    checked: &CheckedInputs,
    transcript: &mut VerifierTranscript<'_, H>,
    stage1: &Stage1Output<F, C>,
) -> Result<Stage2Output<F, C>, VerifierError>
where
    F: JoltField,
    C: CanonicalDecode,
    H: Sponge,
{
    transcript.site(STAGE2);
    match (checked.zk, stage1) {
        (true, Stage1Output::Clear(_)) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage1" });
        }
        (false, Stage1Output::Zk(_)) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage1" });
        }
        _ => {}
    }

    let log_t = crate::num::ilog2(checked.trace_length);
    let log_k = crate::num::ilog2(checked.ram_K);
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = checked.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    let uniskip = verify_product_uniskip(checked, transcript, stage1)?;

    // Build the batch relations once, pre-branch; each owns its input/output
    // claim algebra (single-sourced with its jolt-claims formula and the BlindFold
    // constraint). The product uni-skip stays hand-coded above.
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
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction: FieldRegistersClaimReduction::new(
            FieldRegistersTraceDimensions::new(log_t),
            uniskip.tau_low.clone(),
        ),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write_dimensions,
            raf_dimensions,
            log_k,
            lowest_address,
            uniskip.tau_low.clone(),
        ),
        ram_output_check: RamOutputCheck::new(read_write_dimensions, public_memory),
    };

    // Draw each relation's challenges in declaration order: the RAM read-write gamma, the
    // instruction claim-reduction gamma, under `field-inline` the field-inline claim-reduction
    // gamma (each a single `challenge_scalar`), then the RAM output-check address reference
    // point (the last member's `draw_challenges` override — one raw `challenge()` per RAM
    // address variable, landing after the gammas as the inline draw did). The drawn challenges
    // feed the input/output claims and populate the stage aggregate carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    // Every member's input points are empty (each derives its output points from its
    // own sumcheck point).
    let input_points = sumchecks.empty_input_points();

    if checked.zk {
        let ProductUniskipVerified::Zk(product_uniskip) = uniskip.verified else {
            return Err(VerifierError::ExpectedCommittedProof {
                field: "stage2_uni_skip_first_round_proof",
            });
        };
        // Both modes omit aliases from the canonical member output rows.
        let shape = committed::output_claim_shape(checked, sumchecks.output_claim_count())?;
        let (consistency, commitments) = sumchecks.verify_zk(shape.row_count(), transcript)?;
        let batch_output_claims = CommittedOutputClaimOutput { shape, commitments };
        let output_points =
            sumchecks.derive_opening_points(&consistency.challenges(), &input_points)?;

        return Ok(Stage2Output::Zk(Stage2ZkOutput {
            challenges,
            product_uniskip_challenge: uniskip.challenge,
            product_tau_low: uniskip.tau_low,
            product_tau_high: uniskip.tau_high,
            product_uniskip_consistency: product_uniskip.consistency,
            product_uniskip_output_claims: product_uniskip.output_claims,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
        }));
    }

    let ProductUniskipVerified::Clear {
        output_claim: product_uniskip_output_claim,
    } = uniskip.verified
    else {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage2_uni_skip_first_round_proof",
        });
    };
    let stage1 = stage1.clear()?;

    let input_values =
        stage2_batch_input_values_from_upstream(stage1, product_uniskip_output_claim);

    let (output_points, output_values) =
        sumchecks.verify_clear(&input_values, &input_points, &challenges, transcript, 2)?;

    Ok(Stage2Output::Clear(Stage2ClearOutput {
        output_values,
        output_points,
        product_tau_low: uniskip.tau_low,
    }))
}

/// The product uni-skip's low binding tau_low: the tail (`[1..]`) of stage
/// 1's raw remainder point, reversed. Shared by `verify_product_uniskip` and
/// the prove-side stage-2 recipe, so the derivation cannot drift.
pub fn product_tau_low<F: JoltField>(
    stage1_remainder: &[F],
    log_t: usize,
) -> Result<Vec<F>, VerifierError> {
    let mut tau_low = stage1_remainder
        .get(1..)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
            reason: "Stage 1 remainder challenge vector is empty".to_string(),
        })?
        .to_vec();
    if tau_low.len() != log_t {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
            reason: format!(
                "Stage 1 remainder challenge tail length mismatch: expected {log_t}, got {}",
                tau_low.len()
            ),
        });
    }
    tau_low.reverse();
    Ok(tau_low)
}

fn verify_product_uniskip<F, C, H>(
    checked: &CheckedInputs,
    transcript: &mut VerifierTranscript<'_, H>,
    stage1: &Stage1Output<F, C>,
) -> Result<ProductUniskipStep<F, C>, VerifierError>
where
    F: JoltField,
    C: CanonicalDecode,
    H: Sponge,
{
    let log_t = crate::num::ilog2(checked.trace_length);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let tau_low = product_tau_low(&stage1.remainder_point(), log_t)?;

    let tau_high = uniskip::draw_spartan_product_tau_high(transcript);
    let uniskip_params = uniskip::UniskipParams::spartan_product();
    match stage1 {
        Stage1Output::Clear(stage1) => {
            let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
            let uniskip_input_values = product_uniskip_input_values_from_stage1(stage1);
            let uniskip_input_claim =
                uniskip_relation.input_claim(&uniskip_input_values, &NoChallenges::default())?;

            let verified = uniskip::verify_clear(&uniskip_params, uniskip_input_claim, transcript)?;
            Ok(ProductUniskipStep {
                tau_low,
                tau_high,
                challenge: verified.challenge,
                verified: ProductUniskipVerified::Clear {
                    output_claim: verified.output_claim,
                },
            })
        }
        Stage1Output::Zk(_) => {
            let verified = uniskip::verify_zk(checked, &uniskip_params, transcript)?;
            Ok(ProductUniskipStep {
                tau_low,
                tau_high,
                challenge: verified.challenge,
                verified: ProductUniskipVerified::Zk(verified),
            })
        }
    }
}
