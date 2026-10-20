use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_field::JoltField;
use jolt_transcript::{Channel, Sponge, VerifierTranscript};
use jolt_zk::masking::CommittedPadsView;

use crate::sites::STAGE1;

use super::outer_remainder::{outer_remainder_input_values_from_uniskip_output, OuterRemainder};
use super::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1Challenges, Stage1ClearOutput,
    Stage1Output, Stage1ZkOutput,
};
use crate::{
    stages::{uniskip, zk::masked_channel},
    verifier::CheckedInputs,
    VerifierError,
};

pub fn verify<F, H, P>(
    checked: &CheckedInputs,
    transcript: &mut VerifierTranscript<'_, H>,
    pads: Option<&mut CommittedPadsView<P>>,
) -> Result<Stage1Output<F>, VerifierError>
where
    F: JoltField,
    H: Sponge,
{
    transcript.site(STAGE1);
    let uniskip_params = uniskip::UniskipParams::spartan_outer();
    let log_t = crate::num::ilog2(checked.trace_length);
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let tau = uniskip::draw_spartan_outer_tau(transcript, log_t);

    if !checked.zk {
        // The uni-skip first round consumes no openings: its symbolic `input_expression`
        // is `zero` (jolt-claims `spartan/outer_uniskip.rs`), so the input claim is the
        // constant zero. BlindFold still single-sources this claim from that same symbolic
        // Expr, and muldiv (host) catches any drift between the two.
        let uniskip = uniskip::verify_clear(&uniskip_params, F::from_u64(0), transcript)?;

        // Built after the uni-skip step so the relation carries `tau` and the
        // uni-skip reduction challenge; the coefficient table completes itself from
        // the bound point captured by `derive_opening_points`. Construction and the
        // (no-op) member draw are transcript-neutral, so their position relative to
        // the uni-skip is immaterial.
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(dimensions, tau, uniskip.challenge),
        };
        let batch_challenges = sumchecks.draw_challenges(transcript)?;
        let input_points = sumchecks.empty_input_points();

        // The remainder consumes the uni-skip's reduced opening as its input claim
        // (the relation's `input_claim` is the bare consumed opening).
        let input_values = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(uniskip.output_claim),
        };

        let (output_points, output_values) = sumchecks.verify_clear(
            &input_values,
            &input_points,
            &batch_challenges,
            transcript,
            1,
        )?;

        return Ok(Stage1Output::Clear(Stage1ClearOutput {
            output_values,
            output_points,
        }));
    }

    {
        let mut channel = masked_channel(pads, transcript)?;
        let uniskip = uniskip::verify_masked(&uniskip_params, &mut channel)?;
        let uniskip_challenge = uniskip.challenge;

        // Built after the uni-skip step so the relation carries `tau` and the
        // uni-skip reduction challenge (two of its three coefficient-table
        // inputs); transcript-neutral, since the remainder draws no member
        // challenges.
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(dimensions, tau.clone(), uniskip_challenge),
        };
        let input_points = sumchecks.empty_input_points();

        let batch = sumchecks.verify_masked(&input_points, &mut channel)?;

        Ok(Stage1Output::Zk(Stage1ZkOutput {
            challenges: Stage1Challenges {
                tau,
                uniskip_challenge,
            },
            uniskip_rounds: uniskip.rounds,
            uniskip_output_claims: uniskip.output_claims,
            remainder_rounds: batch.rounds,
            remainder_output_claims: batch.output_claims,
            output_points: batch.output_points,
        }))
    }
}
