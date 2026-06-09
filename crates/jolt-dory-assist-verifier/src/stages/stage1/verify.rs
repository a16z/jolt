use jolt_claims::protocols::dory_assist::{
    formulas::protocol::{protocol_claims, CANONICAL_RELATION_ORDER},
    DoryAssistChallengeId, DoryAssistExpr, DoryAssistOpeningId, DoryAssistPublicId,
    DoryAssistRelationClaims, DoryAssistSumcheckDomain,
};
use jolt_claims::ConsistencyClaim;
use jolt_dory::DoryScheme;
use jolt_field::Fq;
use jolt_openings::{CommitmentScheme, EvaluationClaim};
use jolt_poly::UnivariatePolynomial;
use jolt_sumcheck::{SumcheckClaim, SUMCHECK_ROUND_TRANSCRIPT_LABEL};
use jolt_transcript::{Label, LabelWithCount, Transcript, U64Word};

use super::{
    inputs::{canonical_stage1_relation_specs, Stage1Inputs, Stage1RelationProof},
    outputs::{DoryAssistChallengeValue, Stage1Output, Stage1RelationOutput},
};
use crate::{
    proof::{DoryAssistOpeningClaim, DoryAssistProofClaims},
    verifier::{squeeze_fq, squeeze_fq_challenge},
    DoryAssistStage, DoryAssistVerifierError,
};

pub fn verify<T>(
    inputs: Stage1Inputs<'_, '_>,
    transcript: &mut T,
) -> Result<Stage1Output, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    if inputs.proof.relations.is_empty() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage1.relations",
            reason: "relations must be nonempty".to_string(),
        });
    }

    let expected_relations = canonical_stage1_relation_specs(inputs.dimensions);
    let actual_relations = inputs
        .proof
        .relations
        .iter()
        .map(Stage1RelationProof::spec)
        .collect::<Vec<_>>();
    if actual_relations != expected_relations {
        return Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage1,
            reason: format!(
                "stage 1 relations must match canonical Dory-assist relation catalog: expected {expected_relations:?}, got {:?}",
                actual_relations
            ),
        });
    }

    transcript.append(&Label(b"dory_assist_stage1"));
    transcript.append(&Label(inputs.checked.mode_name().as_bytes()));
    let semantic_relations = canonical_stage1_relation_claims(inputs.dimensions);

    transcript.append(&Label(b"stage1_relations"));
    transcript.append(&U64Word(inputs.proof.relation_count() as u64));
    let mut relation_outputs = Vec::with_capacity(inputs.proof.relations.len());
    for (relation, semantic_relation) in inputs.proof.relations.iter().zip(&semantic_relations) {
        absorb_relation(relation, transcript);

        let relation_challenges = sample_relation_challenges(semantic_relation, transcript);
        let input_claim = evaluate_relation_expression(
            semantic_relation.input.expression(),
            inputs.claims,
            &relation_challenges,
        )?;
        let reduction = verify_relation_sumcheck(relation, input_claim, transcript)?;
        let expected_output_claim = evaluate_relation_expression(
            semantic_relation.output.expression(),
            inputs.claims,
            &relation_challenges,
        )?;
        if reduction.value != expected_output_claim {
            return Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                reason: format!(
                    "relation {:?} sumcheck final claim {:?} did not match semantic output claim {:?}",
                    relation.id, reduction.value, expected_output_claim
                ),
            });
        }
        verify_relation_consistency(inputs.claims, semantic_relation, &relation_challenges)?;

        let opening_claims = relation_opening_claims(inputs.claims, semantic_relation)?;
        append_opening_claims(&opening_claims, transcript);
        relation_outputs.push(Stage1RelationOutput {
            id: relation.id,
            relation_challenges,
            input_claim,
            sumcheck_point: reduction.point,
            sumcheck_final_claim: reduction.value,
            expected_output_claim,
            opening_claims,
        });
    }
    let challenge = squeeze_fq_challenge(transcript, b"dory_stage1_challenge");

    Ok(Stage1Output {
        relation_count: inputs.proof.relation_count(),
        relation_outputs,
        challenge,
    })
}

fn absorb_relation<T>(relation: &Stage1RelationProof, transcript: &mut T)
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    transcript.append(&Label(b"stage1_relation_id"));
    transcript.append(&U64Word(relation_transcript_tag(relation) as u64));
    transcript.append(&Label(b"stage1_sumcheck_domain"));
    transcript.append(&U64Word(0));
    transcript.append(&Label(b"stage1_sumcheck_rounds"));
    transcript.append(&U64Word(relation.sumcheck.rounds as u64));
    transcript.append(&Label(b"stage1_sumcheck_degree"));
    transcript.append(&U64Word(relation.sumcheck.degree as u64));
}

fn verify_relation_sumcheck<T>(
    relation: &Stage1RelationProof,
    input_claim: Fq,
    transcript: &mut T,
) -> Result<EvaluationClaim<Fq>, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    if !matches!(
        relation.sumcheck.domain,
        DoryAssistSumcheckDomain::BooleanHypercube
    ) {
        return Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage1,
            reason: format!(
                "relation {:?} must use the Boolean hypercube domain",
                relation.id
            ),
        });
    }
    if relation.sumcheck.degree == 0 {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage1.relation.sumcheck.degree",
            reason: "sumcheck degree must be nonzero".to_string(),
        });
    }

    let claim = SumcheckClaim {
        num_vars: relation.sumcheck.rounds,
        degree: relation.sumcheck.degree,
        claimed_sum: input_claim,
    };

    if relation.sumcheck_proof.round_polynomials.len() != claim.num_vars {
        return Err(stage_sumcheck_failed(
            relation,
            format!(
                "expected {} rounds, proof contains {}",
                claim.num_vars,
                relation.sumcheck_proof.round_polynomials.len()
            ),
        ));
    }

    let mut running_sum = claim.claimed_sum;
    let mut challenges = Vec::with_capacity(claim.num_vars);
    for (round, round_proof) in relation.sumcheck_proof.round_polynomials.iter().enumerate() {
        if round_proof.degree() > claim.degree {
            return Err(stage_sumcheck_failed(
                relation,
                format!(
                    "degree bound exceeded: degree {}, max {}",
                    round_proof.degree(),
                    claim.degree
                ),
            ));
        }

        let coeffs = round_proof.coeffs_except_linear_term();
        if coeffs.is_empty() {
            return Err(stage_sumcheck_failed(
                relation,
                format!(
                    "round {round}: compressed round polynomial requires >= 2 coefficients, got 0"
                ),
            ));
        }

        transcript.append(&LabelWithCount(
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            coeffs.len() as u64,
        ));
        for coeff in coeffs {
            transcript.append(coeff);
        }

        let challenge = squeeze_fq(transcript);
        running_sum = round_proof.evaluate_with_hint(running_sum, challenge);
        challenges.push(challenge);
    }

    Ok(EvaluationClaim::new(challenges, running_sum))
}

fn sample_relation_challenges<T>(
    relation: &DoryAssistRelationClaims<Fq>,
    transcript: &mut T,
) -> Vec<DoryAssistChallengeValue>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    relation
        .required_challenges()
        .into_iter()
        .map(|id| DoryAssistChallengeValue {
            id,
            value: squeeze_fq(transcript),
        })
        .collect()
}

fn verify_relation_consistency(
    claims: &DoryAssistProofClaims,
    relation: &DoryAssistRelationClaims<Fq>,
    relation_challenges: &[DoryAssistChallengeValue],
) -> Result<(), DoryAssistVerifierError> {
    for (index, consistency) in relation.consistency.iter().enumerate() {
        let ConsistencyClaim::EqualExpressions { left, right } = consistency;
        let left_value = evaluate_relation_expression(left, claims, relation_challenges)?;
        let right_value = evaluate_relation_expression(right, claims, relation_challenges)?;

        if left_value != right_value {
            return Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                reason: format!(
                    "relation {:?} consistency claim {index} evaluated unequal expressions: left {left_value:?}, right {right_value:?}",
                    relation.id
                ),
            });
        }
    }

    Ok(())
}

fn evaluate_relation_expression(
    expression: &DoryAssistExpr<Fq>,
    claims: &DoryAssistProofClaims,
    relation_challenges: &[DoryAssistChallengeValue],
) -> Result<Fq, DoryAssistVerifierError> {
    expression.try_evaluate(
        |id| resolve_stage1_opening_claim(claims, id),
        |id| resolve_relation_challenge(relation_challenges, id),
        |id| resolve_stage1_public_claim(claims, id),
    )
}

fn resolve_stage1_opening_claim(
    claims: &DoryAssistProofClaims,
    id: &DoryAssistOpeningId,
) -> Result<Fq, DoryAssistVerifierError> {
    claims
        .stage1
        .opening_claim(id)
        .ok_or(DoryAssistVerifierError::MissingOpeningClaim { id: *id })
}

fn resolve_relation_challenge(
    challenges: &[DoryAssistChallengeValue],
    id: &DoryAssistChallengeId,
) -> Result<Fq, DoryAssistVerifierError> {
    challenges
        .iter()
        .find(|challenge| challenge.id == *id)
        .map(|challenge| challenge.value)
        .ok_or(DoryAssistVerifierError::MissingStageClaimChallenge { id: *id })
}

fn resolve_stage1_public_claim(
    claims: &DoryAssistProofClaims,
    id: &DoryAssistPublicId,
) -> Result<Fq, DoryAssistVerifierError> {
    claims
        .stage1
        .public_claim(id)
        .ok_or(DoryAssistVerifierError::MissingStageClaimPublic { id: *id })
}

fn relation_opening_claims(
    claims: &DoryAssistProofClaims,
    relation: &DoryAssistRelationClaims<Fq>,
) -> Result<Vec<DoryAssistOpeningClaim>, DoryAssistVerifierError> {
    relation
        .required_openings()
        .into_iter()
        .map(|id| {
            Ok(DoryAssistOpeningClaim {
                id,
                value: resolve_stage1_opening_claim(claims, &id)?,
            })
        })
        .collect()
}

fn append_opening_claims<T>(opening_claims: &[DoryAssistOpeningClaim], transcript: &mut T)
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    for opening_claim in opening_claims {
        transcript.append_labeled(b"opening_claim", &opening_claim.value);
    }
}

fn stage_sumcheck_failed(
    relation: &Stage1RelationProof,
    reason: String,
) -> DoryAssistVerifierError {
    DoryAssistVerifierError::StageSumcheckFailed {
        stage: DoryAssistStage::Stage1,
        relation: relation.id,
        reason,
    }
}

#[expect(
    clippy::expect_used,
    reason = "stage 1 relation IDs are a subset of the canonical protocol catalog"
)]
fn canonical_stage1_relation_claims(
    dimensions: jolt_claims::protocols::dory_assist::DoryAssistDimensions,
) -> Vec<DoryAssistRelationClaims<Fq>> {
    let protocol = protocol_claims::<Fq>(dimensions);
    canonical_stage1_relation_specs(dimensions)
        .iter()
        .map(|spec| {
            protocol
                .relation(spec.id)
                .expect("stage 1 relation belongs to canonical Dory-assist protocol")
                .clone()
        })
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "verified stage 1 relations are drawn from the canonical Dory-assist catalog"
)]
fn relation_transcript_tag(relation: &Stage1RelationProof) -> usize {
    CANONICAL_RELATION_ORDER
        .iter()
        .position(|id| *id == relation.id)
        .expect("stage 1 relation has a canonical transcript tag")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::dory_assist::{
        formulas::gt, DoryAssistConsistencyClaim, DoryAssistRelationId, DoryAssistSumcheckSpec,
    };
    use jolt_claims::{opening, public};
    use jolt_field::FromPrimitiveInt;

    #[test]
    fn consistency_claim_accepts_equal_opening_values() {
        let relation = synthetic_consistency_relation();
        let claims = DoryAssistProofClaims::default();

        assert_eq!(verify_relation_consistency(&claims, &relation, &[]), Ok(()));
    }

    #[test]
    fn consistency_claim_rejects_unequal_opening_values() {
        let relation = synthetic_consistency_relation();
        let mut claims = DoryAssistProofClaims::default();
        claims.stage1.gt_exponentiation.shifted_accumulator = Fq::from_u64(1);

        assert!(matches!(
            verify_relation_consistency(&claims, &relation, &[]),
            Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage1,
                ..
            })
        ));
    }

    #[test]
    fn consistency_claim_can_use_public_terms() {
        let left =
            opening(gt::exp_accumulator_opening()) + public(DoryAssistPublicId::GtShiftEqKernel);
        let right = opening(gt::exp_shifted_accumulator_opening())
            + public(DoryAssistPublicId::GtShiftEqKernel);
        let relation = DoryAssistRelationClaims::new(
            DoryAssistRelationId::GtExponentiation,
            DoryAssistSumcheckSpec::boolean(1, 1),
            DoryAssistExpr::zero(),
            DoryAssistExpr::zero(),
        )
        .with_consistency([DoryAssistConsistencyClaim::equal_expressions(left, right)]);

        assert_eq!(
            verify_relation_consistency(&DoryAssistProofClaims::default(), &relation, &[]),
            Ok(())
        );
    }

    fn synthetic_consistency_relation() -> DoryAssistRelationClaims<Fq> {
        DoryAssistRelationClaims::new(
            DoryAssistRelationId::GtExponentiation,
            DoryAssistSumcheckSpec::boolean(1, 1),
            DoryAssistExpr::zero(),
            DoryAssistExpr::zero(),
        )
        .with_consistency([DoryAssistConsistencyClaim::same_evaluation(
            gt::exp_accumulator_opening(),
            gt::exp_shifted_accumulator_opening(),
        )])
    }
}
