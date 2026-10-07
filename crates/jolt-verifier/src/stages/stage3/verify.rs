use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_transcript::{Channel, Sponge, VerifierTranscript};

use crate::sites::STAGE3;

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
    stages::{
        stage1::{Stage1BatchOutputClaims, Stage1Output},
        stage2::{Stage2BatchOutputClaims, Stage2Output},
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

pub fn verify<F, C, H>(
    checked: &CheckedInputs,
    transcript: &mut VerifierTranscript<'_, H>,
    stage1: &Stage1Output<F, C>,
    stage2: &Stage2Output<F, C>,
) -> Result<Stage3Output<F, C>, VerifierError>
where
    F: JoltField,
    C: CanonicalDecode,
    H: Sponge,
{
    transcript.site(STAGE3);
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
    // input, register reduction); each is a single `challenge_scalar`. The drawn
    // challenges feed the input/output claims and populate the stage aggregate
    // carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    if !checked.zk {
        let stage1 = stage1.clear()?;
        let stage2 = stage2.clear()?;

        let input_values =
            stage3_input_values_from_upstream(&stage1.output_values, &stage2.output_values);
        let input_points = sumchecks.empty_input_points();

        let (output_points, output_values) =
            sumchecks.verify_clear(&input_values, &input_points, &challenges, transcript, 3)?;

        return Ok(Stage3Output::Clear(Stage3ClearOutput {
            output_values,
            output_points,
        }));
    }

    {
        let batch = sumchecks.verify_zk(
            checked.committed_row_len()?,
            &sumchecks.empty_input_points(),
            transcript,
        )?;

        Ok(Stage3Output::Zk(Stage3ZkOutput {
            challenges,
            batch_consistency: batch.consistency,
            batch_output_claims: batch.output_claims,
            output_points: batch.output_points,
        }))
    }
}
