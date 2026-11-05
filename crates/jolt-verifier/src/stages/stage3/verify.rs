use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_transcript::FieldTranscript;
use jolt_zk::masking::MaskedSession;

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
        pow::{check_pow, PowSite},
        stage1::{Stage1BatchOutputClaims, Stage1Output},
        stage2::{Stage2BatchOutputClaims, Stage2Output},
        zk::masked_channel,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// Assemble the stage-3 consumed opening *values* from the upstream outputs into
/// the generated `Stage3InputClaims` aggregate. This is the single place the
/// stage's Outputs→Inputs dataflow is expressed: each per-relation `*_from_upstream`
/// helper wires which upstream opening feeds which downstream input.
pub fn stage3_input_values_from_upstream<F: JoltField>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
) -> Stage3InputClaims<F> {
    Stage3InputClaims {
        shift: spartan_shift_input_values_from_upstream(stage1, stage2),
        instruction_input: instruction_input_input_values_from_upstream(stage2),
        registers_claim_reduction: registers_claim_reduction_input_values_from_upstream(stage1),
    }
}

#[jolt_verifier_derive::fs_scope(Stage3)]
pub fn verify<PCS, VC, T, ZkProof, O, P>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof, O>,
    transcript: &mut T,
    masked: Option<&mut MaskedSession<'_, PCS::Field, P>>,
    stage1: &Stage1Output<PCS::Field>,
    stage2: &Stage2Output<PCS::Field>,
) -> Result<Stage3Output<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FieldTranscript<PCS::Field>,
{
    let log_t = crate::num::ilog2(checked.trace_length);
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
    // input, register reduction); each is a single `challenge()`. The drawn
    // challenges feed the input/output claims and populate the stage aggregate
    // carried downstream.
    check_pow(PowSite::Stage3, &proof.stages.pow_nonces, transcript)?;
    let challenges = sumchecks.draw_challenges(transcript)?;

    if !checked.zk.is_masked() {
        let claims = &proof.clear_claims()?.stage3;
        let stage1 = stage1.clear()?;
        let stage2 = stage2.clear()?;
        sumchecks.validate_output_claims(claims)?;

        let input_values =
            stage3_input_values_from_upstream(&stage1.output_values, &stage2.output_values);
        let input_points = sumchecks.empty_input_points();

        let output_points = sumchecks.verify_clear(
            &input_values,
            &input_points,
            &challenges,
            claims,
            &proof.stages.stage3_sumcheck_proof,
            transcript,
            3,
        )?;

        sumchecks.append_output_claims(transcript, claims);

        return Ok(Stage3Output::Clear(Stage3ClearOutput {
            output_values: claims.clone(),
            output_points,
        }));
    }

    {
        let batch = sumchecks.verify_masked(
            &sumchecks.empty_input_points(),
            &mut masked_channel(masked, transcript)?,
        )?;

        Ok(Stage3Output::Zk(Stage3ZkOutput {
            challenges,
            batch_rounds: batch.rounds,
            batch_output_claims: batch.output_claims,
            output_points: batch.output_points,
        }))
    }
}
