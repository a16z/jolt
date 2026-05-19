use jolt_claims::protocols::jolt::{
    formulas::spartan::SpartanOuterDimensions, JoltStageId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::FromPrimitiveInt;
use jolt_openings::CommitmentScheme;
use jolt_r1cs::constraints::rv64::Rv64SpartanOuterRemainder;
use jolt_sumcheck::{
    append_sumcheck_claim, BooleanHypercube, CenteredIntegerDomain, ClearProof, LabeledRoundPoly,
    SumcheckClaim, SumcheckProof, SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::outputs::{Stage1Output, VerifiedSpartanOuterSumcheck};
use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, verifier::CheckedInputs,
    VerifierError,
};

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
) -> Result<Stage1Output<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    if checked.zk {
        return Err(VerifierError::Unimplemented);
    }

    let stage = JoltStageId::SpartanOuter;
    let claims = &proof.transparent_claims()?.stage1;

    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let tau = transcript.challenge_vector(log_t + 2);

    let uniskip_spec = dimensions.uniskip_sumcheck();
    if uniskip_spec.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage,
            degree: uniskip_spec.degree,
        });
    }
    let JoltSumcheckDomain::CenteredInteger { domain_size } = uniskip_spec.domain else {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: "Stage 1 uni-skip sumcheck must use the centered-integer domain".to_string(),
        });
    };
    let uniskip_input_claim = PCS::Field::from_u64(0);
    let SumcheckProof::Clear(ClearProof::Full(uniskip_proof)) =
        &proof.stages.stage1_uni_skip_first_round_proof
    else {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage1_uni_skip_first_round_proof",
        });
    };
    let uniskip_round_polynomials = uniskip_proof
        .round_polynomials
        .iter()
        .map(LabeledRoundPoly::uniskip)
        .collect::<Vec<_>>();
    let uniskip_reduction = SumcheckVerifier::verify(
        &SumcheckClaim::new(
            uniskip_spec.rounds,
            uniskip_spec.degree,
            uniskip_input_claim,
        ),
        &uniskip_round_polynomials,
        CenteredIntegerDomain::new(domain_size),
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: error.to_string(),
    })?;
    if uniskip_reduction.value != claims.uniskip_output_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage });
    }
    let uniskip = VerifiedSpartanOuterSumcheck {
        input_claim: uniskip_input_claim,
        sumcheck_point: uniskip_reduction.point,
        sumcheck_final_claim: uniskip_reduction.value,
        expected_output_claim: claims.uniskip_output_claim,
    };

    // Core absorbs the uni-skip output as an opening claim before deriving the
    // batching challenge for the remainder sumcheck.
    transcript.append_labeled(b"opening_claim", &uniskip.expected_output_claim);

    let [uniskip_challenge] = uniskip.sumcheck_point.as_slice() else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: "uni-skip proof did not reduce to one challenge".to_string(),
        });
    };
    let uniskip_challenge = *uniskip_challenge;

    let remainder_input_claim = uniskip.expected_output_claim;
    append_sumcheck_claim(transcript, &remainder_input_claim);
    let remainder_batching_coefficient = transcript.challenge_scalar();
    let batched_remainder_input_claim = remainder_input_claim * remainder_batching_coefficient;

    let remainder_spec = dimensions.remainder_sumcheck();
    if remainder_spec.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage,
            degree: remainder_spec.degree,
        });
    }
    if !matches!(remainder_spec.domain, JoltSumcheckDomain::BooleanHypercube) {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: "Stage 1 remainder sumcheck must use the Boolean hypercube".to_string(),
        });
    }
    let SumcheckProof::Clear(ClearProof::Compressed(remainder_proof)) =
        &proof.stages.stage1_sumcheck_proof
    else {
        return Err(VerifierError::ExpectedClearProof {
            field: "stage1_sumcheck_proof",
        });
    };
    let remainder_reduction = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(
            remainder_spec.rounds,
            remainder_spec.degree,
            batched_remainder_input_claim,
        ),
        remainder_proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: error.to_string(),
    })?;
    let r1cs_input_claims = claims.outer.r1cs_input_claims(&dimensions)?;
    let remainder_challenges = remainder_reduction.point.as_slice();
    let remainder_shape =
        Rv64SpartanOuterRemainder::new(&dimensions, &tau, uniskip_challenge, remainder_challenges)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: error.to_string(),
            })?;
    let expected_remainder_output_claim = remainder_shape
        .expected_output_claim(&r1cs_input_claims)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?
        * remainder_batching_coefficient;
    if remainder_reduction.value != expected_remainder_output_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage });
    }
    let remainder = VerifiedSpartanOuterSumcheck {
        input_claim: batched_remainder_input_claim,
        sumcheck_point: remainder_reduction.point,
        sumcheck_final_claim: remainder_reduction.value,
        expected_output_claim: expected_remainder_output_claim,
    };
    for opening_claim in &r1cs_input_claims {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }

    Ok(Stage1Output {
        uniskip_challenge,
        remainder_batching_coefficient,
        remainder_challenges: remainder.sumcheck_point.as_slice().to_vec(),
        uniskip,
        remainder,
        outer: claims.outer.clone(),
    })
}
