use jolt_claims::protocols::jolt::{geometry::spartan::SpartanOuterDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::FromPrimitiveInt;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::outer_remainder::{outer_remainder_input_values_from_uniskip_output, OuterRemainder};
use super::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1Challenges, Stage1ClearOutput,
    Stage1Output, Stage1ZkOutput,
};
use crate::{
    proof::JoltProof,
    stages::{uniskip, zk::committed},
    verifier::CheckedInputs,
    VerifierError,
};

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
) -> Result<Stage1Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let uniskip_params = uniskip::UniskipParams::spartan_outer();
    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let tau = uniskip::draw_spartan_outer_tau(transcript, log_t);

    if !checked.zk {
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
        let uniskip = uniskip::verify_zk(
            checked,
            &proof.stages.stage1_uni_skip_first_round_proof,
            &uniskip_params,
            transcript,
        )?;
        let uniskip_challenge = uniskip.challenge;

        // Built after the uni-skip step so the relation carries `tau` and the
        // uni-skip reduction challenge (two of its three coefficient-table
        // inputs); transcript-neutral, since the remainder draws no member
        // challenges.
        let sumchecks = Stage1BatchSumchecks {
            outer_remainder: OuterRemainder::new(dimensions, tau.clone(), uniskip_challenge),
        };
        let input_points = sumchecks.empty_input_points();

        let remainder_consistency =
            sumchecks.verify_zk(&proof.stages.stage1_sumcheck_proof, transcript)?;
        let remainder_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage1_sumcheck_proof,
            "stage1_sumcheck_proof",
            sumchecks.output_claim_count(),
            JoltRelationId::SpartanOuter,
        )?;
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
            remainder_output_claims,
            output_points,
        }))
    }
}
