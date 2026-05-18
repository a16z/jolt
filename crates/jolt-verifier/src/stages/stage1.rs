//! Stage 1 Spartan outer verifier.

use jolt_claims::protocols::jolt::{
    formulas::spartan::{outer_opening, outer_uniskip_opening, SpartanOuterDimensions},
    JoltStageId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_openings::{append_opening_claim, CommitmentScheme};
use jolt_r1cs::constraints::rv64::Rv64SpartanOuterRemainder;
use jolt_sumcheck::{
    append_sumcheck_claim, BooleanHypercube, CenteredIntegerDomain, ClearProof, LabeledRoundPoly,
    SumcheckClaim, SumcheckProof, SumcheckVerifier, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Output<F: Field> {
    pub uniskip_challenge: F,
    pub remainder_batching_coefficient: F,
    pub remainder_challenges: Vec<F>,
    pub uniskip: VerifiedSpartanOuterSumcheck<F>,
    pub remainder: VerifiedSpartanOuterSumcheck<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSpartanOuterSumcheck<F: Field> {
    pub input_claim: F,
    pub sumcheck_point: jolt_poly::Point<F>,
    pub sumcheck_final_claim: F,
    pub expected_output_claim: F,
}

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
    let opening_claim_payload = proof
        .opening_claims
        .as_deref()
        .ok_or(VerifierError::MissingOpeningClaims)?;

    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let tau = transcript.challenge_vector(log_t + 2);
    let opening_claim = |id| {
        opening_claim_payload
            .iter()
            .find_map(|&(claim_id, opening_claim)| (claim_id == id).then_some(opening_claim))
            .ok_or(VerifierError::MissingOpeningClaim { id })
    };
    let opening_claims = SpartanOuterOpeningClaims {
        uniskip_claim: opening_claim(outer_uniskip_opening())?,
        r1cs_input_claims: dimensions
            .variables()
            .iter()
            .copied()
            .map(|variable| opening_claim(outer_opening(variable)))
            .collect::<Result<_, _>>()?,
    };

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
    if uniskip_reduction.value != opening_claims.uniskip_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage });
    }
    let uniskip = VerifiedSpartanOuterSumcheck {
        input_claim: uniskip_input_claim,
        sumcheck_point: uniskip_reduction.point,
        sumcheck_final_claim: uniskip_reduction.value,
        expected_output_claim: opening_claims.uniskip_claim,
    };

    // Core absorbs the uni-skip output as an opening claim before deriving the
    // batching challenge for the remainder sumcheck.
    append_opening_claim(transcript, &uniskip.expected_output_claim);

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
    let remainder_challenges = remainder_reduction.point.as_slice();
    let remainder_shape =
        Rv64SpartanOuterRemainder::new(&dimensions, &tau, uniskip_challenge, remainder_challenges)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: error.to_string(),
            })?;
    let expected_remainder_output_claim = remainder_shape
        .expected_output_claim(&opening_claims.r1cs_input_claims)
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
    for opening_claim in &opening_claims.r1cs_input_claims {
        append_opening_claim(transcript, opening_claim);
    }

    Ok(Stage1Output {
        uniskip_challenge,
        remainder_batching_coefficient,
        remainder_challenges: remainder.sumcheck_point.as_slice().to_vec(),
        uniskip,
        remainder,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SpartanOuterOpeningClaims<F: Field> {
    uniskip_claim: F,
    r1cs_input_claims: Vec<F>,
}
