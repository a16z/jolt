use jolt_claims::protocols::jolt::{geometry::spartan::SpartanOuterDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::FromPrimitiveInt;
use jolt_openings::CommitmentScheme;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    CenteredIntegerDomain, SumcheckClaim, SumcheckStatement, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::outer_remainder::{outer_remainder_input_values_from_uniskip_output, OuterRemainder};
use super::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1Challenges, Stage1ClearOutput,
    Stage1Output, Stage1ZkOutput,
};
use crate::{proof::JoltProof, stages::zk::committed, verifier::CheckedInputs, VerifierError};

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
    let stage = JoltRelationId::SpartanOuter;

    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let tau = transcript.challenge_vector(log_t + 2);

    if checked.zk {
        let uniskip_consistency = proof
            .stages
            .stage1_uni_skip_first_round_proof
            .verify_committed_consistency(
                SumcheckStatement::new(1, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE),
                transcript,
            )
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: error.to_string(),
            })?;
        let uniskip_output_claims = committed::verify_output_claim_commitments(
            checked,
            &proof.stages.stage1_uni_skip_first_round_proof,
            "stage1_uni_skip_first_round_proof",
            1,
            stage,
        )?;
        let [round] = uniskip_consistency.rounds.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "uni-skip committed consistency did not produce one challenge".to_string(),
            });
        };
        let uniskip_challenge = round.challenge;

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
            stage,
        )?;
        let output_points =
            sumchecks.derive_opening_points(&remainder_consistency.challenges(), &input_points)?;

        return Ok(Stage1Output::Zk(Stage1ZkOutput {
            challenges: Stage1Challenges {
                tau,
                uniskip_challenge,
            },
            uniskip_consistency,
            uniskip_output_claims,
            remainder_consistency,
            remainder_output_claims,
            output_points,
        }));
    }

    let claims = &proof.clear_claims()?.stage1;
    // The uni-skip first round consumes no openings: its symbolic `input_expression`
    // is `zero` (jolt-claims `spartan/outer_uniskip.rs`), so the input claim is the
    // constant zero. BlindFold still single-sources this claim from that same symbolic
    // Expr, and muldiv (host) catches any drift between the two.
    let uniskip_input_claim = PCS::Field::from_u64(0);
    let uniskip_reduction = proof
        .stages
        .stage1_uni_skip_first_round_proof
        .verify(
            &SumcheckClaim::new(
                1,
                SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
                uniskip_input_claim,
            ),
            CenteredIntegerDomain::new(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
            UNISKIP_ROUND_TRANSCRIPT_LABEL,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    if uniskip_reduction.value != claims.uniskip_output_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 1 });
    }
    let uniskip_output_claim = claims.uniskip_output_claim;

    // Match the prover transcript: the uni-skip output is absorbed as an
    // opening claim before deriving the remainder batching challenge (the
    // singleton batch's RLC coefficient squeeze inside `verify_clear`).
    transcript.append_labeled(b"opening_claim", &uniskip_output_claim);

    let [uniskip_challenge] = uniskip_reduction.point.as_slice() else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: "uni-skip proof did not reduce to one challenge".to_string(),
        });
    };
    let uniskip_challenge = *uniskip_challenge;

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

    let batch = sumchecks.verify_clear(
        &input_values,
        &batch_challenges,
        &proof.stages.stage1_sumcheck_proof,
        transcript,
    )?;

    let output_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;

    let expected_final_claim = sumchecks.expected_final_claim(
        &batch.coefficients,
        &input_points,
        &claims.outer,
        &output_points,
        &batch_challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 1 });
    }

    // Append the 35 produced openings in canonical (declaration) order, matching
    // the prover's commitment order.
    sumchecks.append_output_claims(transcript, &claims.outer);

    Ok(Stage1Output::Clear(Stage1ClearOutput {
        output_values: claims.outer.clone(),
        output_points,
    }))
}
