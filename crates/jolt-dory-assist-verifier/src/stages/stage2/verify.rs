use jolt_claims::protocols::dory_assist::{
    formulas::dory_reduce::DoryReducePublicFoldConstraint, DoryAssistCopyConstraint,
    DoryAssistOpeningId, DoryAssistRelationId, DoryAssistValueRef, DoryAssistVirtualPolynomial,
};
use jolt_dory::DoryScheme;
use jolt_field::{Fq, FromPrimitiveInt};
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use jolt_transcript::{Label, Transcript, U64Word};

use super::{
    inputs::{canonical_stage2_copy_constraints, Stage2Inputs},
    outputs::{Stage2CopyConstraintOutput, Stage2DoryReducePublicFoldOutput, Stage2Output},
};
use crate::{
    proof::DoryAssistProofClaims,
    stages::stage1::{outputs::Stage1RelationOutput, Stage1Output},
    verifier::squeeze_fq_challenge,
    DoryAssistStage, DoryAssistVerifierError,
};

pub fn verify<T>(
    inputs: Stage2Inputs<'_, '_>,
    transcript: &mut T,
) -> Result<Stage2Output, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    if inputs.proof.copy_constraints.is_empty() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage2.copy_constraints",
            reason: "copy_constraints must be nonempty".to_string(),
        });
    }

    let expected_constraints = canonical_stage2_copy_constraints(inputs.dimensions);
    if inputs.proof.copy_constraints != expected_constraints {
        return Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage2,
            reason: format!(
                "stage 2 copy constraints must match canonical Dory-assist direct-copy stencil: expected {expected_constraints:?}, got {:?}",
                inputs.proof.copy_constraints
            ),
        });
    }

    transcript.append(&Label(b"dory_assist_stage2"));
    transcript.append(&Label(inputs.checked.mode_name().as_bytes()));
    transcript.append(&Label(b"stage1_relations"));
    transcript.append(&U64Word(inputs.stage1.relation_count as u64));
    transcript.append(&Label(b"stage2_copy_constraints"));
    transcript.append(&U64Word(inputs.proof.relation_count() as u64));

    let mut copy_constraints = Vec::with_capacity(inputs.proof.copy_constraints.len());
    for (index, constraint) in inputs.proof.copy_constraints.iter().enumerate() {
        transcript.append(&Label(b"stage2_copy_constraint"));
        transcript.append(&U64Word(index as u64));
        let source_value = resolve_copy_value(inputs.claims, inputs.stage1, constraint.source)?;
        let target_value = resolve_copy_value(inputs.claims, inputs.stage1, constraint.target)?;
        transcript.append(&Label(b"stage2_copy_source"));
        transcript.append(&source_value);
        transcript.append(&Label(b"stage2_copy_target"));
        transcript.append(&target_value);

        if source_value != target_value {
            return Err(copy_constraint_mismatch(
                constraint,
                source_value,
                target_value,
            ));
        }

        copy_constraints.push(Stage2CopyConstraintOutput {
            constraint: *constraint,
            source_value,
            target_value,
        });
    }

    let public_folds = verify_dory_reduce_public_folds(&inputs, transcript)?;
    let challenge = squeeze_fq_challenge(transcript, b"dory_stage2_challenge");

    Ok(Stage2Output {
        relation_count: inputs.proof.relation_count(),
        copy_constraints,
        dory_reduce_public_folds: public_folds,
        challenge,
    })
}

fn verify_dory_reduce_public_folds<T>(
    inputs: &Stage2Inputs<'_, '_>,
    transcript: &mut T,
) -> Result<Vec<Stage2DoryReducePublicFoldOutput>, DoryAssistVerifierError>
where
    T: Transcript<Challenge = <DoryScheme as CommitmentScheme>::Field>,
{
    let constraints =
        jolt_claims::protocols::dory_assist::formulas::dory_reduce::public_fold_constraints(
            inputs.dimensions.dory_reduce,
        );
    transcript.append(&Label(b"stage2_dory_reduce_public_folds"));
    transcript.append(&U64Word(constraints.len() as u64));

    let mut outputs = Vec::with_capacity(constraints.len());
    for (index, constraint) in constraints.into_iter().enumerate() {
        transcript.append(&Label(b"stage2_dory_reduce_public_fold"));
        transcript.append(&U64Word(index as u64));

        let expected_value = evaluate_public_fold(inputs.claims, inputs.stage1, &constraint)?;
        let target_value = resolve_copy_value(inputs.claims, inputs.stage1, constraint.target)?;
        transcript.append(&Label(b"s2_dory_reduce_fold_expected"));
        transcript.append(&expected_value);
        transcript.append(&Label(b"s2_dory_reduce_fold_target"));
        transcript.append(&target_value);

        if expected_value != target_value {
            return Err(DoryAssistVerifierError::StageOutputMismatch {
                stage: DoryAssistStage::Stage2,
                reason: format!(
                    "Dory-reduce public fold {constraint:?} resolved to unequal values: expected {expected_value:?}, target {target_value:?}"
                ),
            });
        }

        outputs.push(Stage2DoryReducePublicFoldOutput {
            constraint,
            expected_value,
            target_value,
        });
    }

    Ok(outputs)
}

fn evaluate_public_fold(
    claims: &DoryAssistProofClaims,
    stage1: &Stage1Output,
    constraint: &DoryReducePublicFoldConstraint,
) -> Result<Fq, DoryAssistVerifierError> {
    let opening = constraint.target.witness_opening().ok_or_else(|| {
        DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage2,
            reason: format!(
                "Dory-reduce public fold target has no witness opening: {:?}",
                constraint.target
            ),
        }
    })?;
    ensure_stage1_recorded_opening(stage1, opening)?;

    let relation = relation_output_for_opening(stage1, opening)?;
    let weights = EqPolynomial::new(relation.sumcheck_point.as_slice().to_vec()).evaluations();
    if constraint.sources.len() > weights.len() {
        return Err(DoryAssistVerifierError::InvalidProofShape {
            component: "stage2.dory_reduce_public_fold.sources",
            reason: format!(
                "public fold has {} sources but the relation point only supports {} rows",
                constraint.sources.len(),
                weights.len()
            ),
        });
    }

    constraint
        .sources
        .iter()
        .zip(weights)
        .try_fold(Fq::default(), |acc, (id, weight)| {
            let value = claims
                .stage1
                .public_claim(id)
                .ok_or(DoryAssistVerifierError::MissingStageClaimPublic { id: *id })?;
            Ok(acc + value * weight)
        })
}

fn resolve_copy_value(
    claims: &DoryAssistProofClaims,
    stage1: &Stage1Output,
    value_ref: DoryAssistValueRef,
) -> Result<Fq, DoryAssistVerifierError> {
    match value_ref {
        DoryAssistValueRef::Witness { .. } => {
            let opening = value_ref.witness_opening().ok_or_else(|| {
                DoryAssistVerifierError::StageClaimMismatch {
                    stage: DoryAssistStage::Stage2,
                    reason: format!("copy witness endpoint has no opening: {value_ref:?}"),
                }
            })?;
            ensure_stage1_recorded_opening(stage1, opening)?;
            resolve_witness_value(claims, value_ref, opening)
        }
        DoryAssistValueRef::Public { id, .. } => claims
            .stage1
            .public_claim(&id)
            .ok_or(DoryAssistVerifierError::MissingStageClaimPublic { id }),
        DoryAssistValueRef::Constant(value) => Ok(Fq::from_u64(value as u64)),
        DoryAssistValueRef::Challenge(id) => Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage2,
            reason: format!("copy constraints cannot directly reference challenge {id:?}"),
        }),
    }
}

fn resolve_witness_value(
    claims: &DoryAssistProofClaims,
    value_ref: DoryAssistValueRef,
    opening: DoryAssistOpeningId,
) -> Result<Fq, DoryAssistVerifierError> {
    match value_ref {
        DoryAssistValueRef::Witness {
            relation: DoryAssistRelationId::GtMultiplication,
            polynomial: DoryAssistVirtualPolynomial::Gt(polynomial),
            row,
            component,
            ..
        } => claims
            .stage1
            .gt_multiplication
            .row_claim(row, component, polynomial)
            .ok_or(DoryAssistVerifierError::MissingOpeningClaim { id: opening }),
        DoryAssistValueRef::Witness { .. } => claims
            .stage1
            .opening_claim(&opening)
            .ok_or(DoryAssistVerifierError::MissingOpeningClaim { id: opening }),
        DoryAssistValueRef::Public { .. }
        | DoryAssistValueRef::Challenge(_)
        | DoryAssistValueRef::Constant(_) => Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage2,
            reason: format!("copy witness resolver received non-witness endpoint {value_ref:?}"),
        }),
    }
}

fn ensure_stage1_recorded_opening(
    stage1: &Stage1Output,
    opening: DoryAssistOpeningId,
) -> Result<(), DoryAssistVerifierError> {
    let recorded = stage1
        .relation_outputs
        .iter()
        .flat_map(|relation| &relation.opening_claims)
        .any(|claim| claim.id == opening);

    if recorded {
        Ok(())
    } else {
        Err(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage2,
            reason: format!("copy endpoint opening {opening:?} was not verified by stage 1"),
        })
    }
}

fn relation_output_for_opening(
    stage1: &Stage1Output,
    opening: DoryAssistOpeningId,
) -> Result<&Stage1RelationOutput, DoryAssistVerifierError> {
    stage1
        .relation_outputs
        .iter()
        .find(|relation| {
            relation
                .opening_claims
                .iter()
                .any(|claim| claim.id == opening)
        })
        .ok_or(DoryAssistVerifierError::StageClaimMismatch {
            stage: DoryAssistStage::Stage2,
            reason: format!("copy endpoint opening {opening:?} was not verified by stage 1"),
        })
}

fn copy_constraint_mismatch(
    constraint: &DoryAssistCopyConstraint,
    source_value: Fq,
    target_value: Fq,
) -> DoryAssistVerifierError {
    DoryAssistVerifierError::StageOutputMismatch {
        stage: DoryAssistStage::Stage2,
        reason: format!(
            "copy constraint {constraint:?} resolved to unequal values: source {source_value:?}, target {target_value:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests fail loudly on malformed local fixtures"
    )]

    use super::*;
    use crate::proof::DoryAssistOpeningClaim;
    use jolt_claims::protocols::dory_assist::{DoryAssistPublicId, DoryReducePolynomial};
    use jolt_field::FromPrimitiveInt;
    use jolt_poly::{Point, HIGH_TO_LOW};

    #[test]
    fn public_fold_evaluates_round_indexed_sources_at_relation_point() {
        let target = DoryAssistValueRef::witness(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryAssistVirtualPolynomial::DoryReduce(DoryReducePolynomial::Beta),
            0,
            0,
        );
        let constraint = DoryReducePublicFoldConstraint::new(
            jolt_claims::protocols::dory_assist::DoryAssistValueType::Scalar,
            vec![
                DoryAssistPublicId::TranscriptScalar(0),
                DoryAssistPublicId::TranscriptScalar(1),
            ],
            target,
        );
        let mut claims = DoryAssistProofClaims::default();
        claims.stage1.public.input.transcript_scalars = vec![Fq::from_u64(3), Fq::from_u64(5)];
        let point = Fq::from_u64(7);
        let expected = Fq::from_u64(3) * (Fq::from_u64(1) - point) + Fq::from_u64(5) * point;
        let opening = target.witness_opening().expect("target opening");
        let stage1 = Stage1Output {
            relation_count: 1,
            relation_outputs: vec![Stage1RelationOutput {
                id: DoryAssistRelationId::DoryReduceGtTransition,
                relation_challenges: Vec::new(),
                input_claim: Fq::default(),
                sumcheck_point: Point::<HIGH_TO_LOW, Fq>::high_to_low(vec![point]),
                sumcheck_final_claim: Fq::default(),
                expected_output_claim: Fq::default(),
                opening_claims: vec![DoryAssistOpeningClaim {
                    id: opening,
                    value: expected,
                }],
            }],
            challenge: Fq::default(),
        };

        let actual =
            evaluate_public_fold(&claims, &stage1, &constraint).expect("public fold evaluates");

        assert_eq!(actual, expected);
    }

    #[test]
    fn public_fold_rejects_sources_larger_than_relation_domain() {
        let target = DoryAssistValueRef::witness(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryAssistVirtualPolynomial::DoryReduce(DoryReducePolynomial::Beta),
            0,
            0,
        );
        let constraint = DoryReducePublicFoldConstraint::new(
            jolt_claims::protocols::dory_assist::DoryAssistValueType::Scalar,
            vec![
                DoryAssistPublicId::TranscriptScalar(0),
                DoryAssistPublicId::TranscriptScalar(1),
            ],
            target,
        );
        let opening = target.witness_opening().expect("target opening");
        let stage1 = Stage1Output {
            relation_count: 1,
            relation_outputs: vec![Stage1RelationOutput {
                id: DoryAssistRelationId::DoryReduceGtTransition,
                relation_challenges: Vec::new(),
                input_claim: Fq::default(),
                sumcheck_point: Point::<HIGH_TO_LOW, Fq>::default(),
                sumcheck_final_claim: Fq::default(),
                expected_output_claim: Fq::default(),
                opening_claims: vec![DoryAssistOpeningClaim {
                    id: opening,
                    value: Fq::default(),
                }],
            }],
            challenge: Fq::default(),
        };

        let result = evaluate_public_fold(&DoryAssistProofClaims::default(), &stage1, &constraint);

        assert!(matches!(
            result,
            Err(DoryAssistVerifierError::InvalidProofShape {
                component: "stage2.dory_reduce_public_fold.sources",
                ..
            })
        ));
    }
}
