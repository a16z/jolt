use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::Ring;
use jolt_openings::CommitmentScheme;
use jolt_transcript::FieldTranscript;
use jolt_zk::masking::MaskedSession;

use super::outer_remainder::{outer_remainder_input_values_from_uniskip_output, OuterRemainder};
use super::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1Challenges, Stage1ClearOutput,
    Stage1Output, Stage1ZkOutput,
};
use crate::{
    proof::JoltProof,
    stages::{
        pow::{check_pow, PowSite},
        uniskip,
        zk::masked_channel,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[jolt_verifier_derive::fs_scope(Stage1)]
pub fn verify<PCS, VC, T, ZkProof, O, P>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof, O>,
    transcript: &mut T,
    masked: Option<&mut MaskedSession<'_, PCS::Field, P>>,
) -> Result<Stage1Output<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FieldTranscript<PCS::Field>,
{
    let uniskip_params = uniskip::UniskipParams::spartan_outer();
    let log_t = crate::num::ilog2(checked.trace_length);
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let pow_nonces = &proof.stages.pow_nonces;
    check_pow(PowSite::Stage1Tau, pow_nonces, transcript)?;
    let tau = uniskip::draw_spartan_outer_tau(transcript, log_t)?;

    if !checked.zk.is_masked() {
        let claims = &proof.clear_claims()?.stage1;
        let uni_skip_first_round_proof = &proof.stages.stage1_uni_skip_first_round_proof;
        let sumcheck_proof = &proof.stages.stage1_sumcheck_proof;
        // The uni-skip first round consumes no openings: its symbolic `input_expression`
        // is `zero` (jolt-claims `spartan/outer_uniskip.rs`), so the input claim is the
        // constant zero. BlindFold still single-sources this claim from that same symbolic
        // Expr, and muldiv (host) catches any drift between the two.
        let uniskip_input_claim = PCS::Field::from_u64(0);
        let uniskip_output_claim = claims.uniskip_output_claim;
        let uniskip_challenge = uniskip::verify_clear(
            uni_skip_first_round_proof,
            &uniskip_params,
            uniskip_input_claim,
            uniskip_output_claim,
            transcript,
        )?;

        // Built after the uni-skip step so the relation carries `tau` and the
        // uni-skip reduction challenge; the coefficient table completes itself from
        // the bound point captured by `derive_opening_points`. Construction and the
        // (no-op) member draw are transcript-neutral, so their position relative to
        // the uni-skip is immaterial.
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(dimensions, tau, uniskip_challenge),
        };
        check_pow(PowSite::Stage1, pow_nonces, transcript)?;
        let batch_challenges = sumchecks.draw_challenges(transcript)?;
        let input_points = sumchecks.empty_input_points();

        sumchecks.validate_output_claims(&claims.outer)?;

        // The remainder consumes the uni-skip's reduced opening as its input claim
        // (the relation's `input_claim` is the bare consumed opening).
        let input_values = Stage1BatchInputClaims {
            outer_remainder: outer_remainder_input_values_from_uniskip_output(uniskip_output_claim),
        };

        let output_points = sumchecks.verify_clear(
            &input_values,
            &input_points,
            &batch_challenges,
            &claims.outer,
            sumcheck_proof,
            transcript,
            1,
        )?;

        // Append the 35 produced openings in canonical (declaration) order, matching
        // the prover's commitment order.
        sumchecks.append_output_claims(transcript, &claims.outer);

        return Ok(Stage1Output::Clear(Stage1ClearOutput {
            output_values: claims.outer.clone(),
            output_points,
        }));
    }

    {
        let mut channel = masked_channel(masked, transcript)?;
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

        check_pow(PowSite::Stage1, pow_nonces, channel.transcript())?;
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
