use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_transcript::{Sponge, VerifierTranscript};

use super::outer_remainder::{outer_remainder_input_values_from_uniskip_output, OuterRemainder};
use super::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1Challenges, Stage1ClearOutput,
    Stage1Output, Stage1ZkOutput,
};
use crate::{
    stages::{uniskip, zk::committed, zk::outputs::CommittedOutputClaimOutput},
    verifier::CheckedInputs,
    VerifierError,
};

#[jolt_verifier_derive::fs_scope(Stage1)]
pub fn verify<F, C, H>(
    checked: &CheckedInputs,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<Stage1Output<F, C>, VerifierError>
where
    F: JoltField,
    C: CanonicalDecode,
    H: Sponge,
{
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
        let uniskip = uniskip::verify_zk(checked, &uniskip_params, transcript)?;
        let uniskip_challenge = uniskip.challenge;

        // Built after the uni-skip step so the relation carries `tau` and the
        // uni-skip reduction challenge (two of its three coefficient-table
        // inputs); transcript-neutral, since the remainder draws no member
        // challenges.
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(dimensions, tau.clone(), uniskip_challenge),
        };
        let input_points = sumchecks.empty_input_points();

        // Clear and committed proofs share the composed member's canonical order.
        let shape = committed::output_claim_shape(checked, sumchecks.output_claim_count())?;
        let (remainder_consistency, commitments) =
            sumchecks.verify_zk(shape.row_count(), transcript)?;
        let output_points =
            sumchecks.derive_opening_points(&remainder_consistency.challenges(), &input_points)?;

        Ok(Stage1Output::Zk(Stage1ZkOutput {
            challenges: Stage1Challenges {
                tau,
                uniskip_challenge,
            },
            uniskip_consistency: uniskip.consistency,
            uniskip_output_claims: uniskip.output_claims,
            remainder_consistency,
            remainder_output_claims: CommittedOutputClaimOutput { shape, commitments },
            output_points,
        }))
    }
}
