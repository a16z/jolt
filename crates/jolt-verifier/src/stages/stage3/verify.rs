//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

use jolt_claims::protocols::jolt::{geometry::dimensions::TraceDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::{
    instruction_input::{instruction_input_input_values_from_upstream, InstructionInput},
    outputs::{
        Stage3ClearOutput, Stage3InputClaims, Stage3Output, Stage3Sumchecks, Stage3ZkOutput,
    },
    registers_claim_reduction::{
        registers_claim_reduction_input_values_from_upstream, RegistersClaimReduction,
    },
    spartan_shift::{spartan_shift_input_values_from_upstream, SpartanShift},
};
use crate::{
    proof::JoltProof,
    stages::{
        stage1::{Stage1BatchOutputClaims, Stage1Output},
        stage2::{Stage2BatchOutputClaims, Stage2Output},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// The number of opening claims the stage-3 batch commits/absorbs: 13, not the 16
/// the members' output expressions reference, because three cross-relation aliases
/// (`instruction_input.unexpanded_pc`, the register-reduction `rs1`/`rs2`) are
/// absorbed once via their canonical source (see
/// [`Stage3OutputClaims::opening_values`](super::outputs::Stage3OutputClaims)).
const STAGE3_BATCH_OUTPUT_CLAIMS: usize = 13;

/// Assemble the stage-3 consumed opening *values* from the upstream outputs into
/// the generated `Stage3InputClaims` aggregate. This is the single place the
/// stage's Outputs→Inputs dataflow is expressed: each per-relation `*_from_upstream`
/// helper wires which upstream opening feeds which downstream input.
fn stage3_input_values_from_upstream<F: Field>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
) -> Stage3InputClaims<F> {
    Stage3InputClaims {
        shift: spartan_shift_input_values_from_upstream(stage1, stage2),
        instruction_input: instruction_input_input_values_from_upstream(stage2),
        registers_claim_reduction: registers_claim_reduction_input_values_from_upstream(stage1),
    }
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
) -> Result<Stage3Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = TraceDimensions::new(log_t);

    // The shift/register relations evaluate their `EqPlusOne`/`EqSpartan` publics
    // against upstream stage-2 data, read mode-agnostically so the one construction
    // serves both paths.
    let tau_low = stage2.product_tau_low().to_vec();
    let product_remainder_point = stage2
        .batch_output_points()
        .product_remainder_point()
        .to_vec();

    let sumchecks = Stage3Sumchecks {
        shift: SpartanShift::new(dimensions, tau_low.clone(), product_remainder_point.clone()),
        instruction_input: InstructionInput::new(dimensions, product_remainder_point),
        registers_claim_reduction: RegistersClaimReduction::new(dimensions, tau_low),
    };

    // Draw each relation's batching gamma in declaration order (shift, instruction
    // input, register reduction); each is a single `challenge_scalar`. The drawn
    // challenges feed the input/output claims and populate the stage aggregate
    // carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage3_sumcheck_proof, transcript)?;
        let batch_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage3_sumcheck_proof,
            "stage3_sumcheck_proof",
            STAGE3_BATCH_OUTPUT_CLAIMS,
            JoltRelationId::SpartanShift,
        )?;
        let output_points = sumchecks
            .derive_opening_points(&consistency.challenges(), &sumchecks.empty_input_points())?;

        return Ok(Stage3Output::Zk(Stage3ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
        }));
    }

    let stage1 = stage1.clear()?;
    let stage2 = stage2.clear()?;
    let claims = &proof.clear_claims()?.stage3;

    let input_values =
        stage3_input_values_from_upstream(&stage1.output_values, &stage2.output_values);
    let input_points = sumchecks.empty_input_points();

    let batch = sumchecks.verify_clear(
        &input_values,
        &challenges,
        &proof.stages.stage3_sumcheck_proof,
        transcript,
    )?;

    let output_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;

    let expected_final_claim = sumchecks.expected_final_claim(
        &batch.coefficients,
        &input_points,
        claims,
        &output_points,
        &challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 3 });
    }

    // After the per-relation output checks (which catch any single-claim offset),
    // enforce the cross-relation opening aliases the downstream stages relied on.
    claims.validate()?;

    claims.append_to_transcript(transcript);

    Ok(Stage3Output::Clear(Stage3ClearOutput {
        output_values: claims.clone(),
        output_points,
    }))
}
