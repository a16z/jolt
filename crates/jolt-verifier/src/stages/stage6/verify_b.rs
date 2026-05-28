//! Stage 6b verifier helpers.

use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::advice, AdviceClaimReductionLayout, AdviceClaimReductionPublic,
    JoltAdviceKind, JoltPublicId, JoltRelationClaims, JoltRelationId,
};
use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::{
    stages::{
        stage4::Stage4ClearOutput,
        stage6::{
            inputs::{AdviceCyclePhaseOutputClaim, Stage6Claims},
            outputs::{AdviceCyclePhasePublicOutput, VerifiedAdviceCyclePhaseSumcheck},
        },
    },
    VerifierError,
};

pub(super) fn aliased_booleanity_bytecode_openings<F: Field>(
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> usize {
    bytecode_ra_opening_points
        .iter()
        .filter(|point| point.as_slice() == booleanity_opening_point)
        .count()
}

pub(super) fn advice_cycle_phase_input<F: Field>(
    claim: &JoltRelationClaims<F>,
    stage4: &Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let advice_input = advice::ram_val_check_advice_opening(kind);
    claim.input.expression().try_evaluate(
        |id| match *id {
            id if id == advice_input => stage4
                .ram_val_check_init
                .advice_contributions
                .iter()
                .find(|contribution| contribution.kind == kind)
                .map(|contribution| contribution.opening_claim)
                .ok_or(VerifierError::MissingOpeningClaim { id }),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )
}

pub(super) fn verify_advice_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    opening_claim: &AdviceCyclePhaseOutputClaim<F>,
    stage4: &Stage4ClearOutput<F>,
) -> Result<VerifiedAdviceCyclePhaseSumcheck<F>, VerifierError> {
    let advice_point = advice_cycle_phase_point(batch, claim)?;
    let opening_point = layout
        .cycle_phase_opening_point(advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let cycle_phase_variables = layout
        .cycle_phase_variable_challenges(advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let contribution = stage4
        .ram_val_check_init
        .advice_contributions
        .iter()
        .find(|contribution| contribution.kind == kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    let output_openings = advice::cycle_phase_output_openings(kind, layout.dimensions());
    let expected_output_claim = claim.output.expression().try_evaluate(
        |id| {
            if output_openings.contains(id) {
                Ok(opening_claim.opening_claim)
            } else {
                Err(VerifierError::MissingOpeningClaim { id: *id })
            }
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| match id {
            JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                public_kind,
            )) if *public_kind == kind => layout
                .cycle_phase_final_output_scale(&contribution.opening_point, advice_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::AdviceClaimReductionCyclePhase,
                    reason: error.to_string(),
                }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    Ok(VerifiedAdviceCyclePhaseSumcheck {
        kind,
        input_claim: contribution.opening_claim,
        sumcheck_point: advice_point.to_vec(),
        opening_point,
        cycle_phase_variables,
        expected_output_claim,
    })
}

pub(super) fn advice_cycle_phase_public<F: Field, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
) -> Result<AdviceCyclePhasePublicOutput<F>, VerifierError> {
    let advice_point = advice_cycle_phase_committed_point(batch, claim)?;
    let opening_point = layout
        .cycle_phase_opening_point(&advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let cycle_phase_variables = layout
        .cycle_phase_variable_challenges(&advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;

    Ok(AdviceCyclePhasePublicOutput {
        kind,
        sumcheck_point: advice_point,
        opening_point,
        cycle_phase_variables,
    })
}

fn advice_cycle_phase_point<'a, F: Field>(
    batch: &'a jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
) -> Result<&'a [F], VerifierError> {
    batch
        .try_instance_point(claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })
}

fn advice_cycle_phase_committed_point<F, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
) -> Result<Vec<F>, VerifierError>
where
    F: Field + Copy,
{
    batch
        .try_instance_point(claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })
}

pub(super) fn append_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6Claims<F>,
    bytecode_read_raf_points: &[Vec<F>],
    booleanity_point: &[F],
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.bytecode_read_raf.bytecode_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.ram_hamming_booleanity.ram_hamming_weight,
    );
    for opening_claim in &claims.ram_ra_virtualization.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims
        .instruction_ra_virtualization
        .committed_instruction_ra
    {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(b"opening_claim", &claims.inc_claim_reduction.ram_inc);
    transcript.append_labeled(b"opening_claim", &claims.inc_claim_reduction.rd_inc);
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "tests fail loudly on helper errors")]

    use super::*;
    use jolt_claims::protocols::jolt::formulas::dimensions::{
        CommitmentMatrixShape, TracePolynomialOrder,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_sumcheck::{
        BatchedCommittedSumcheckConsistency, BatchedEvaluationClaim, CommittedSumcheckConsistency,
        EvaluationClaim, VerifiedCommittedRound,
    };

    #[test]
    fn clear_advice_cycle_phase_uses_batched_suffix_for_cycle_major_layout() {
        let layout = advice_layout(TracePolynomialOrder::CycleMajor);
        assert!(layout.dimensions().has_address_phase());
        let claim = advice::cycle_phase::<Fr>(JoltAdviceKind::Trusted, layout.dimensions());
        let batch = clear_batch(claim.sumcheck.rounds + 3);

        let point = advice_cycle_phase_point(&batch, &claim).expect("suffix point");
        let expected = suffix(batch.reduction.point.as_slice(), claim.sumcheck.rounds);

        assert_eq!(point, expected.as_slice());
    }

    #[test]
    fn committed_advice_cycle_phase_uses_batched_suffix_for_address_major_layout() {
        let layout = advice_layout(TracePolynomialOrder::AddressMajor);
        assert!(layout.dimensions().has_address_phase());
        let claim = advice::cycle_phase::<Fr>(JoltAdviceKind::Untrusted, layout.dimensions());
        let batch = committed_batch(claim.sumcheck.rounds + 5);

        let point = advice_cycle_phase_committed_point(&batch, &claim).expect("suffix point");
        let challenges = batch
            .consistency
            .rounds
            .iter()
            .map(|round| round.challenge)
            .collect::<Vec<_>>();

        assert_eq!(point, suffix(&challenges, claim.sumcheck.rounds));
    }

    fn advice_layout(trace_order: TracePolynomialOrder) -> AdviceClaimReductionLayout {
        AdviceClaimReductionLayout::new(
            trace_order,
            8,
            4,
            CommitmentMatrixShape::balanced(12),
            CommitmentMatrixShape::balanced(8),
        )
    }

    fn clear_batch(max_num_vars: usize) -> BatchedEvaluationClaim<Fr> {
        BatchedEvaluationClaim {
            reduction: EvaluationClaim {
                point: challenges(max_num_vars).into(),
                value: Fr::from_u64(0),
            },
            batching_coefficients: Vec::new(),
            max_num_vars,
            max_degree: 2,
        }
    }

    fn committed_batch(max_num_vars: usize) -> BatchedCommittedSumcheckConsistency<Fr, ()> {
        BatchedCommittedSumcheckConsistency {
            consistency: CommittedSumcheckConsistency {
                rounds: challenges(max_num_vars)
                    .into_iter()
                    .map(|challenge| VerifiedCommittedRound {
                        commitment: (),
                        degree: 2,
                        challenge,
                    })
                    .collect(),
            },
            batching_coefficients: Vec::new(),
            max_num_vars,
            max_degree: 2,
        }
    }

    fn challenges(count: usize) -> Vec<Fr> {
        (1..=count as u64).map(Fr::from_u64).collect()
    }

    fn suffix(values: &[Fr], len: usize) -> Vec<Fr> {
        values[values.len() - len..].to_vec()
    }
}
