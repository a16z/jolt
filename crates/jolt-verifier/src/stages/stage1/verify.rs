use jolt_claims::protocols::jolt::{
    geometry::spartan::SpartanOuterDimensions, JoltRelationId, JoltSumcheckDomain,
};
use jolt_claims::NoChallenges;
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

use super::outer_remainder::{
    outer_remainder_input_points_from_uniskip_output,
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use super::outer_uniskip::{OuterUniskip, OuterUniskipInputClaims};
use super::outputs::{
    Stage1BatchInputClaims, Stage1BatchInputPoints, Stage1BatchSumchecks, Stage1Challenges,
    Stage1ClearOutput, Stage1Output, Stage1ZkOutput,
};
use crate::stages::relations::ConcreteSumcheck;
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

    let uniskip_domain = JoltSumcheckDomain::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE);
    let JoltSumcheckDomain::CenteredInteger { domain_size } = uniskip_domain else {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: "Stage 1 uni-skip sumcheck must use the centered-integer domain".to_string(),
        });
    };

    let sumchecks = Stage1BatchSumchecks {
        outer_remainder: OuterRemainder::new(dimensions.clone()),
    };
    // A transcript no-op (the remainder draws no challenges), but it produces the
    // aggregate value the generated clear drivers take.
    let batch_challenges = sumchecks.draw_challenges(transcript)?;
    let input_points = Stage1BatchInputPoints {
        outer_remainder: outer_remainder_input_points_from_uniskip_output::<PCS::Field>(),
    };

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
        let uniskip_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage1_uni_skip_first_round_proof,
                proof_label: "stage1_uni_skip_first_round_proof",
                output_claim_count: 1,
                stage,
            })?;
        let [round] = uniskip_consistency.rounds.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "uni-skip committed consistency did not produce one challenge".to_string(),
            });
        };
        let uniskip_challenge = round.challenge;

        let remainder_consistency =
            sumchecks.verify_zk(&proof.stages.stage1_sumcheck_proof, transcript)?;
        let remainder_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage1_sumcheck_proof,
                proof_label: "stage1_sumcheck_proof",
                output_claim_count: sumchecks.output_claim_count(),
                stage,
            })?;
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
    let uniskip_relation = OuterUniskip::<PCS::Field>::new(dimensions);
    // The uni-skip first round consumes no openings: its `input_expression` is
    // `zero`, so the relation's `input_claim` is the constant zero. Computing it
    // through the relation (rather than hard-coding `0`) keeps the claim algebra
    // single-sourced with the BlindFold input constraint.
    let uniskip_input_claim = uniskip_relation.input_claim(
        &OuterUniskipInputClaims::default(),
        &NoChallenges::default(),
    )?;
    debug_assert_eq!(uniskip_input_claim, PCS::Field::from_u64(0));
    let uniskip_reduction = proof
        .stages
        .stage1_uni_skip_first_round_proof
        .verify(
            &SumcheckClaim::new(
                1,
                SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
                uniskip_input_claim,
            ),
            CenteredIntegerDomain::new(domain_size),
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

    // The remainder's coefficient table depends on its own bound point, so it is
    // bound now that the batch has reduced; `expected_final_claim` evaluates the
    // output `Expr` against it.
    sumchecks.outer_remainder.bind_coefficients(
        &tau,
        uniskip_challenge,
        batch.reduction.point.as_slice(),
    )?;
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
    claims.outer.append_to_transcript(transcript);

    Ok(Stage1Output::Clear(Stage1ClearOutput {
        output_values: claims.outer.clone(),
        output_points,
        uniskip_output_claim,
    }))
}
