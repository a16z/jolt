use jolt_claims::protocols::jolt::{
    geometry::spartan::SpartanOuterDimensions, JoltRelationId, JoltSumcheckDomain, JoltSumcheckSpec,
};
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::FromPrimitiveInt;
use jolt_openings::CommitmentScheme;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_REMAINDER_DEGREE, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
    SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    BatchedSumcheckVerifier, CenteredIntegerDomain, SumcheckClaim, SumcheckStatement,
    UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

use super::outer_remainder::{
    outer_remainder_inputs_from_uniskip_output, outer_remainder_outputs_from_spartan_outer_claims,
    OuterRemainder,
};
use super::outer_uniskip::OuterUniskip;
use super::outputs::{
    spartan_outer_opening_order, Stage1ClearOutput, Stage1Output, Stage1PublicOutput,
    Stage1ZkOutput, VerifiedSpartanOuterSumcheck,
};
use crate::stages::relations::{zip_openings, ConcreteSumcheck, OutputAppend};
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

    let uniskip_spec = JoltSumcheckSpec::centered_integer(
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        1,
        SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
    );
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
        let uniskip_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage1_uni_skip_first_round_proof,
                proof_label: "stage1_uni_skip_first_round_proof",
                output_claim_count: 1,
                stage,
            })?;
        let [round] = consistency.rounds.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "uni-skip committed consistency did not produce one challenge".to_string(),
            });
        };
        (
            round.challenge,
            None,
            Some((consistency, uniskip_output_claims)),
        )
    } else {
        let claims = &proof.clear_claims()?.stage1;
        let uniskip_relation = OuterUniskip::<PCS::Field>::new(dimensions.clone());
        // The uni-skip first round consumes no openings: its `input_expression` is
        // `zero`, so the relation's `input_claim` is the constant zero. Computing it
        // through the relation (rather than hard-coding `0`) keeps the claim algebra
        // single-sourced with the BlindFold input constraint.
        let uniskip_input_claim = uniskip_relation.input_claim(
            &super::outer_uniskip::OuterUniskipInputClaims::default(),
            &NoChallenges::default(),
        )?;
        debug_assert_eq!(uniskip_input_claim, PCS::Field::from_u64(0));
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

        // Match the prover transcript: the uni-skip output is absorbed as an
        // opening claim before deriving the remainder batching challenge.
        transcript.append_labeled(b"opening_claim", &uniskip.expected_output_claim);

        let [uniskip_challenge] = uniskip.sumcheck_point.as_slice() else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "uni-skip proof did not reduce to one challenge".to_string(),
            });
        };
        (*uniskip_challenge, Some(uniskip), None)
    };

    let remainder_spec = JoltSumcheckSpec::boolean(1 + log_t, SPARTAN_OUTER_REMAINDER_DEGREE);
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
        let remainder_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage1_sumcheck_proof,
                proof_label: "stage1_sumcheck_proof",
                output_claim_count: spartan_outer_opening_order(&dimensions).len(),
                stage,
            })?;
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
            Some((consistency, remainder_output_claims)),
        )
    } else {
        let claims = &proof.clear_claims()?.stage1;
        let uniskip = clear_uniskip
            .as_ref()
            .ok_or(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "clear Stage 1 uni-skip output is missing".to_string(),
            })?;
        // The remainder consumes the uni-skip's reduced opening as its input claim.
        // The input claim drawn into the singleton batch is the relation's
        // `input_claim` (the bare consumed opening), which equals
        // `uniskip.expected_output_claim`; it must be known before the sumcheck binds
        // (the `OuterRemainder` coefficients depend on the remainder challenges, which
        // are only available after binding, so the relation itself is built below).
        let remainder_inputs =
            outer_remainder_inputs_from_uniskip_output(uniskip.expected_output_claim);
        let remainder_input_claim = uniskip.expected_output_claim;
        let no_challenges = NoChallenges::default();
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
        // The singleton batch's RLC coefficient is drawn inside
        // `verify_compressed_boolean`, in the inter-sumcheck gap after the uni-skip
        // output was absorbed above. This squeeze is part of the prover transcript and
        // must not be removed.
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

        // Build the remainder relation now that the bound point is known: its
        // constructor expands the quadratic R1CS form into `SpartanOuterPublic`
        // coefficients once (the same source BlindFold uses), which
        // `expected_output` then evaluates against the produced openings.
        let remainder_relation = OuterRemainder::<PCS::Field>::new(
            dimensions.clone(),
            &tau,
            uniskip_challenge,
            remainder_reduction.point.as_slice(),
        )?;
        debug_assert_eq!(
            remainder_relation.input_claim(&remainder_inputs, &no_challenges)?,
            remainder_input_claim,
        );
        let remainder_points = remainder_relation
            .derive_opening_points(remainder_reduction.point.as_slice(), &remainder_inputs)?;
        // The produced opening values come from the serialized outer claims (the wire
        // form); pairing them with the shared remainder point yields the located
        // openings the output `Expr` evaluates against.
        let remainder_output_values =
            outer_remainder_outputs_from_spartan_outer_claims(&claims.outer);
        let remainder_output_claims = zip_openings(&remainder_output_values, &remainder_points);
        let expected_remainder_output_claim = remainder_relation.expected_output(
            &remainder_inputs,
            &remainder_output_claims,
            &no_challenges,
        )? * *remainder_batching_coefficient;
        if remainder_reduction.value != expected_remainder_output_claim {
            return Err(VerifierError::StageClaimOutputMismatch { stage });
        }
        let remainder = VerifiedSpartanOuterSumcheck {
            input_claim: batched_remainder_input_claim,
            sumcheck_point: remainder_reduction.point,
            sumcheck_final_claim: remainder_reduction.value,
            expected_output_claim: expected_remainder_output_claim,
        };
        // Append the 35 produced openings in canonical (declaration) order, matching
        // the prover's commitment order. `OuterRemainderOutputClaims`' field order is
        // `dimensions.variables()` (= `SPARTAN_OUTER_R1CS_INPUTS`), so this is the same
        // order as the previous explicit `r1cs_input_claims` loop.
        remainder_output_claims.append_openings(transcript);

        let remainder_challenges = remainder.sumcheck_point.as_slice().to_vec();
        (
            *remainder_batching_coefficient,
            remainder_challenges,
            Some(remainder),
            None,
        )
    };

    let public = Stage1PublicOutput {
        tau,
        uniskip_challenge,
        remainder_batching_coefficient,
        remainder_challenges,
    };

    if checked.zk {
        let (uniskip_consistency, uniskip_output_claims) =
            zk_uniskip_consistency.ok_or(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "ZK Stage 1 uni-skip consistency is missing".to_string(),
            })?;
        let (remainder_consistency, remainder_output_claims) =
            zk_remainder_consistency.ok_or(VerifierError::StageClaimSumcheckFailed {
                stage,
                reason: "ZK Stage 1 remainder consistency is missing".to_string(),
            })?;

        return Ok(Stage1Output::Zk(Stage1ZkOutput {
            public,
            uniskip_consistency,
            uniskip_output_claims,
            remainder_consistency,
            remainder_output_claims,
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
