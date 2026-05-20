use jolt_claims::protocols::jolt::{
    formulas::spartan::SpartanOuterDimensions, JoltStageId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::FromPrimitiveInt;
use jolt_openings::CommitmentScheme;
use jolt_r1cs::constraints::rv64::Rv64SpartanOuterRemainder;
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
    UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::outputs::{
    Stage1ClearOutput, Stage1Output, Stage1PublicOutput, Stage1ZkOutput,
    VerifiedSpartanOuterSumcheck,
};
use crate::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::committed,
    verifier::CheckedInputs, VerifierError,
};

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    _preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
) -> Result<Stage1Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let stage = JoltStageId::SpartanOuter;

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
    let uniskip_statement = SumcheckStatement::new(uniskip_spec.rounds, uniskip_spec.degree);
    let (uniskip_challenge, clear_uniskip, zk_uniskip_consistency) = if checked.zk {
        let consistency = proof
            .stages
            .stage1_uni_skip_first_round_proof
            .verify_committed_consistency(uniskip_statement, transcript)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: error.to_string(),
            })?;
        committed::require_output_claim_commitments(
            checked,
            &proof.stages.stage1_uni_skip_first_round_proof,
            "stage1_uni_skip_first_round_proof",
            1,
            stage,
        )?;
        let [round] = consistency.rounds.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "uni-skip committed consistency did not produce one challenge".to_string(),
            });
        };
        (round.challenge, None, Some(consistency))
    } else {
        let claims = &proof.clear_claims()?.stage1;
        let uniskip_input_claim = PCS::Field::from_u64(0);
        let uniskip_reduction = proof
            .stages
            .stage1_uni_skip_first_round_proof
            .verify(
                &SumcheckClaim::new(
                    uniskip_spec.rounds,
                    uniskip_spec.degree,
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
            return Err(VerifierError::StageClaimOutputMismatch { stage });
        }
        let uniskip = VerifiedSpartanOuterSumcheck {
            input_claim: uniskip_input_claim,
            sumcheck_point: uniskip_reduction.point,
            sumcheck_final_claim: uniskip_reduction.value,
            expected_output_claim: claims.uniskip_output_claim,
        };

        // Core absorbs the uni-skip output as an opening claim before deriving
        // the batching challenge for the remainder sumcheck.
        transcript.append_labeled(b"opening_claim", &uniskip.expected_output_claim);

        let [uniskip_challenge] = uniskip.sumcheck_point.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "uni-skip proof did not reduce to one challenge".to_string(),
            });
        };
        (*uniskip_challenge, Some(uniskip), None)
    };

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
    let remainder_statement = SumcheckStatement::new(remainder_spec.rounds, remainder_spec.degree);
    let (
        remainder_batching_coefficient,
        remainder_challenges,
        clear_remainder,
        zk_remainder_consistency,
    ) = if checked.zk {
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &[remainder_statement],
            &proof.stages.stage1_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
        committed::require_output_claim_commitments(
            checked,
            &proof.stages.stage1_sumcheck_proof,
            "stage1_sumcheck_proof",
            dimensions.variables().len(),
            stage,
        )?;
        let [remainder_batching_coefficient] = consistency.batching_coefficients.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason:
                    "Stage 1 committed remainder returned the wrong number of batching coefficients"
                        .to_string(),
            });
        };
        let remainder_challenges = consistency.challenges();
        (
            *remainder_batching_coefficient,
            remainder_challenges,
            None,
            Some(consistency),
        )
    } else {
        let claims = &proof.clear_claims()?.stage1;
        let uniskip = clear_uniskip
            .as_ref()
            .ok_or(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "clear Stage 1 uni-skip output is missing".to_string(),
            })?;
        let remainder_input_claim = uniskip.expected_output_claim;
        let remainder_batch = BatchedSumcheckVerifier::verify_compressed_boolean(
            &[SumcheckClaim::new(
                remainder_spec.rounds,
                remainder_spec.degree,
                remainder_input_claim,
            )],
            &proof.stages.stage1_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
        let [remainder_batching_coefficient] = remainder_batch.batching_coefficients.as_slice()
        else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "Stage 1 remainder returned the wrong number of batching coefficients"
                    .to_string(),
            });
        };
        let batched_remainder_input_claim = remainder_input_claim * *remainder_batching_coefficient;
        let remainder_reduction = remainder_batch.reduction;
        let r1cs_input_claims = claims.outer.r1cs_input_claims(&dimensions)?;
        let remainder_challenges = remainder_reduction.point.as_slice();
        let remainder_formula = Rv64SpartanOuterRemainder::new(
            &dimensions,
            &tau,
            uniskip_challenge,
            remainder_challenges,
        )
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })?;
        let expected_remainder_output_claim = remainder_formula
            .expected_output_claim(&r1cs_input_claims)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: error.to_string(),
            })?
            * *remainder_batching_coefficient;
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

        let remainder_challenges = remainder.sumcheck_point.as_slice().to_vec();
        (
            *remainder_batching_coefficient,
            remainder_challenges,
            Some(remainder),
            None,
        )
    };

    let public = Stage1PublicOutput {
        uniskip_challenge,
        remainder_batching_coefficient,
        remainder_challenges,
    };

    if checked.zk {
        let uniskip_consistency =
            zk_uniskip_consistency.ok_or(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "ZK Stage 1 uni-skip consistency is missing".to_string(),
            })?;
        let remainder_consistency =
            zk_remainder_consistency.ok_or(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "ZK Stage 1 remainder consistency is missing".to_string(),
            })?;

        return Ok(Stage1Output::Zk(Stage1ZkOutput {
            public,
            uniskip_consistency,
            remainder_consistency,
        }));
    }

    let claims = &proof.clear_claims()?.stage1;
    let uniskip = clear_uniskip.ok_or(VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: "clear Stage 1 uni-skip output is missing".to_string(),
    })?;
    let remainder = clear_remainder.ok_or(VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: "clear Stage 1 remainder output is missing".to_string(),
    })?;

    Ok(Stage1Output::Clear(Stage1ClearOutput {
        public,
        uniskip,
        remainder,
        outer: claims.outer.clone(),
    }))
}
