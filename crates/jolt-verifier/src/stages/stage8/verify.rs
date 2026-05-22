use super::{
    inputs::Deps,
    outputs::{Stage8ClearOutput, Stage8Output, Stage8ZkOutput},
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltCommitments, JoltProof},
    stages::{
        stage6::{Stage6ClearOutput, Stage6ZkOutput},
        stage7::{Stage7ClearOutput, Stage7ZkOutput},
    },
    verifier::CheckedInputs,
    VerifierError,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::advice, committed_openings::advice_commitment_embedding_scale,
        dimensions::JoltFormulaDimensions, ra::JoltRaPolynomialLayout,
    },
    AdviceClaimReductionLayout, JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId,
    JoltRelationId,
};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::{
    AdditivelyHomomorphic, CommitmentScheme, EvaluationClaim, VerifierOpeningClaim, ZkOpeningScheme,
};
use jolt_poly::{EqPolynomial, Point};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

struct AdviceFinalOpening<F: Field> {
    point: Vec<F>,
    opening_claim: F,
}

struct AdviceFinalOpeningPoint<F: Field> {
    point: Vec<F>,
}

pub fn verify<F, PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
    deps: Deps<'_, F, VC::Output>,
) -> Result<Stage8Output<F, PCS::Output, VC::Output>, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: Clone + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage8" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage8" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode.code_size,
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;
    let layout = formula_dimensions.ra_layout;

    let opening_point = match deps {
        Deps::Clear { stage7, .. } => stage7
            .batch
            .hamming_weight_claim_reduction
            .opening_point
            .clone(),
        Deps::Zk { stage7, .. } => stage7.hamming_weight_claim_reduction.opening_point.clone(),
    };
    let committed_chunk_bits = proof.one_hot_config.committed_chunk_bits();
    if opening_point.len() < committed_chunk_bits {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "final opening point has {} variables, expected at least {committed_chunk_bits}",
                opening_point.len()
            ),
        });
    }
    let r_address_stage7 = &opening_point[..committed_chunk_bits];
    let dense_embedding_scale = EqPolynomial::<PCS::Field>::zero_selector(r_address_stage7);
    require_commitment_layout(&proof.commitments, layout)?;

    let pcs_opening_point = Point::high_to_low(
        proof
            .trace_polynomial_order
            .commitment_opening_point(&opening_point, log_t)
            .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                reason: error.to_string(),
            })?,
    );
    let common_point = Point::high_to_low(opening_point.clone());

    if checked.zk {
        let Deps::Zk { stage6, stage7 } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage8" });
        };

        let trusted_advice = if checked.trusted_advice_commitment_present {
            Some(final_advice_opening_point(
                JoltAdviceKind::Trusted,
                checked,
                proof,
                stage6,
                stage7,
            )?)
        } else {
            None
        };
        let untrusted_advice = if proof.untrusted_advice_commitment.is_some() {
            Some(final_advice_opening_point(
                JoltAdviceKind::Untrusted,
                checked,
                proof,
                stage6,
                stage7,
            )?)
        } else {
            None
        };
        let final_opening_count = 2
            + layout.total()
            + usize::from(trusted_advice.is_some())
            + usize::from(untrusted_advice.is_some());
        let trusted_advice_commitment = match (trusted_advice.is_some(), trusted_advice_commitment)
        {
            (true, Some(commitment)) => Some(commitment),
            (true, None) => {
                return Err(VerifierError::MissingFinalOpeningCommitment {
                    polynomial: JoltCommittedPolynomial::TrustedAdvice,
                });
            }
            (false, _) => None,
        };
        let untrusted_advice_commitment = match (
            untrusted_advice.is_some(),
            proof.untrusted_advice_commitment.as_ref(),
        ) {
            (true, Some(commitment)) => Some(commitment),
            (true, None) => {
                return Err(VerifierError::MissingFinalOpeningCommitment {
                    polynomial: JoltCommittedPolynomial::UntrustedAdvice,
                });
            }
            (false, _) => None,
        };

        let mut opening_ids = Vec::with_capacity(final_opening_count);
        let mut commitments = Vec::with_capacity(final_opening_count);
        let mut scaling_factors = Vec::with_capacity(final_opening_count);
        {
            let mut push_opening =
                |id: JoltOpeningId, commitment: &PCS::Output, scale: PCS::Field| {
                    scaling_factors.push(scale);
                    opening_ids.push(id);
                    commitments.push(commitment.clone());
                };

            // Core's final PCS batch order intentionally differs from proof payload order.
            push_opening(
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamInc,
                    JoltRelationId::IncClaimReduction,
                ),
                &proof.commitments.ram_inc,
                dense_embedding_scale,
            );
            push_opening(
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RdInc,
                    JoltRelationId::IncClaimReduction,
                ),
                &proof.commitments.rd_inc,
                dense_embedding_scale,
            );
            for (index, commitment) in proof.commitments.ra.instruction.iter().enumerate() {
                push_opening(
                    JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(index),
                        JoltRelationId::HammingWeightClaimReduction,
                    ),
                    commitment,
                    PCS::Field::one(),
                );
            }
            for (index, commitment) in proof.commitments.ra.bytecode.iter().enumerate() {
                push_opening(
                    JoltOpeningId::committed(
                        JoltCommittedPolynomial::BytecodeRa(index),
                        JoltRelationId::HammingWeightClaimReduction,
                    ),
                    commitment,
                    PCS::Field::one(),
                );
            }
            for (index, commitment) in proof.commitments.ra.ram.iter().enumerate() {
                push_opening(
                    JoltOpeningId::committed(
                        JoltCommittedPolynomial::RamRa(index),
                        JoltRelationId::HammingWeightClaimReduction,
                    ),
                    commitment,
                    PCS::Field::one(),
                );
            }
            if let Some(commitment) = trusted_advice_commitment {
                let final_opening =
                    trusted_advice
                        .as_ref()
                        .ok_or(VerifierError::MissingOpeningClaim {
                            id: advice::final_advice_opening(JoltAdviceKind::Trusted),
                        })?;
                push_opening(
                    advice::final_advice_opening(JoltAdviceKind::Trusted),
                    commitment,
                    advice_commitment_embedding_scale(&opening_point, &final_opening.point),
                );
            }
            if let Some(commitment) = untrusted_advice_commitment {
                let final_opening =
                    untrusted_advice
                        .as_ref()
                        .ok_or(VerifierError::MissingOpeningClaim {
                            id: advice::final_advice_opening(JoltAdviceKind::Untrusted),
                        })?;
                push_opening(
                    advice::final_advice_opening(JoltAdviceKind::Untrusted),
                    commitment,
                    advice_commitment_embedding_scale(&opening_point, &final_opening.point),
                );
            }
        }

        let gamma_powers = transcript.challenge_scalar_powers(opening_ids.len());
        let joint_commitment = PCS::combine(&commitments, &gamma_powers);
        let constraint_coefficients = gamma_powers
            .iter()
            .zip(scaling_factors)
            .map(|(gamma, scale)| *gamma * scale)
            .collect::<Vec<_>>();

        let hiding_evaluation_commitment = PCS::verify_zk(
            &joint_commitment,
            &opening_point,
            &proof.joint_opening_proof,
            &preprocessing.pcs_setup,
            transcript,
        )
        .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        })?;
        PCS::bind_zk_opening_inputs(
            transcript,
            common_point.as_slice(),
            &hiding_evaluation_commitment,
        );

        return Ok(Stage8Output::Zk(Stage8ZkOutput {
            opening_ids,
            constraint_coefficients,
            opening_point: common_point,
            pcs_opening_point,
            joint_commitment,
            hiding_evaluation_commitment,
        }));
    }

    let Deps::Clear { stage6, stage7 } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage8" });
    };
    let trusted_advice = if checked.trusted_advice_commitment_present {
        Some(final_advice_opening(
            JoltAdviceKind::Trusted,
            checked,
            proof,
            stage6,
            stage7,
        )?)
    } else {
        None
    };
    let untrusted_advice = if proof.untrusted_advice_commitment.is_some() {
        Some(final_advice_opening(
            JoltAdviceKind::Untrusted,
            checked,
            proof,
            stage6,
            stage7,
        )?)
    } else {
        None
    };

    let final_opening_count = 2
        + layout.total()
        + usize::from(trusted_advice.is_some())
        + usize::from(untrusted_advice.is_some());
    let trusted_advice_commitment = match (trusted_advice.is_some(), trusted_advice_commitment) {
        (true, Some(commitment)) => Some(commitment),
        (true, None) => {
            return Err(VerifierError::MissingFinalOpeningCommitment {
                polynomial: JoltCommittedPolynomial::TrustedAdvice,
            });
        }
        (false, _) => None,
    };
    let untrusted_advice_commitment = match (
        untrusted_advice.is_some(),
        proof.untrusted_advice_commitment.as_ref(),
    ) {
        (true, Some(commitment)) => Some(commitment),
        (true, None) => {
            return Err(VerifierError::MissingFinalOpeningCommitment {
                polynomial: JoltCommittedPolynomial::UntrustedAdvice,
            });
        }
        (false, _) => None,
    };
    let mut opening_ids = Vec::with_capacity(final_opening_count);
    let mut opening_claims = Vec::with_capacity(final_opening_count);
    let mut constraint_coefficients = Vec::with_capacity(final_opening_count);
    let mut scaling_factors = Vec::with_capacity(final_opening_count);

    {
        let mut push_opening = |id: JoltOpeningId,
                                commitment: &PCS::Output,
                                opening_claim: PCS::Field,
                                scale: PCS::Field| {
            scaling_factors.push(scale);
            opening_ids.push(id);
            opening_claims.push(VerifierOpeningClaim {
                commitment: commitment.clone(),
                evaluation: EvaluationClaim::new(pcs_opening_point.clone(), opening_claim * scale),
            });
        };

        // Core's final PCS batch order intentionally differs from proof payload order.
        push_opening(
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RamInc,
                JoltRelationId::IncClaimReduction,
            ),
            &proof.commitments.ram_inc,
            stage6.output_claims.inc_claim_reduction.ram_inc,
            dense_embedding_scale,
        );
        push_opening(
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RdInc,
                JoltRelationId::IncClaimReduction,
            ),
            &proof.commitments.rd_inc,
            stage6.output_claims.inc_claim_reduction.rd_inc,
            dense_embedding_scale,
        );

        for (index, commitment) in proof.commitments.ra.instruction.iter().enumerate() {
            let id = JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            );
            let opening_claim = *stage7
                .output_claims
                .hamming_weight_claim_reduction
                .instruction_ra
                .get(index)
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            push_opening(id, commitment, opening_claim, PCS::Field::one());
        }

        for (index, commitment) in proof.commitments.ra.bytecode.iter().enumerate() {
            let id = JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            );
            let opening_claim = *stage7
                .output_claims
                .hamming_weight_claim_reduction
                .bytecode_ra
                .get(index)
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            push_opening(id, commitment, opening_claim, PCS::Field::one());
        }

        for (index, commitment) in proof.commitments.ra.ram.iter().enumerate() {
            let id = JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(index),
                JoltRelationId::HammingWeightClaimReduction,
            );
            let opening_claim = *stage7
                .output_claims
                .hamming_weight_claim_reduction
                .ram_ra
                .get(index)
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            push_opening(id, commitment, opening_claim, PCS::Field::one());
        }

        if let Some(commitment) = trusted_advice_commitment {
            let id = advice::final_advice_opening(JoltAdviceKind::Trusted);
            let final_opening = trusted_advice
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            push_opening(
                id,
                commitment,
                final_opening.opening_claim,
                advice_commitment_embedding_scale(&opening_point, &final_opening.point),
            );
        }

        if let Some(commitment) = untrusted_advice_commitment {
            let id = advice::final_advice_opening(JoltAdviceKind::Untrusted);
            let final_opening = untrusted_advice
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            push_opening(
                id,
                commitment,
                final_opening.opening_claim,
                advice_commitment_embedding_scale(&opening_point, &final_opening.point),
            );
        }
    }

    transcript.append(&LabelWithCount(b"rlc_claims", opening_claims.len() as u64));
    for claim in &opening_claims {
        claim.evaluation.value.append_to_transcript(transcript);
    }
    let gamma_powers = transcript.challenge_scalar_powers(opening_claims.len());

    let joint_claim = gamma_powers
        .iter()
        .zip(&opening_claims)
        .fold(PCS::Field::zero(), |claim, (gamma, opening)| {
            claim + *gamma * opening.evaluation.value
        });
    let commitments = opening_claims
        .iter()
        .map(|claim| claim.commitment.clone())
        .collect::<Vec<_>>();
    let joint_commitment = PCS::combine(&commitments, &gamma_powers);
    constraint_coefficients.extend(
        gamma_powers
            .iter()
            .zip(scaling_factors)
            .map(|(gamma, scale)| *gamma * scale),
    );

    PCS::verify(
        &joint_commitment,
        pcs_opening_point.as_slice(),
        joint_claim,
        &proof.joint_opening_proof,
        &preprocessing.pcs_setup,
        transcript,
    )
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })?;
    PCS::bind_opening_inputs(transcript, common_point.as_slice(), &joint_claim);

    Ok(Stage8Output::Clear(Stage8ClearOutput {
        opening_claims,
        opening_ids,
        constraint_coefficients,
        opening_point: common_point,
        pcs_opening_point,
        joint_claim,
        joint_commitment,
    }))
}

fn require_commitment_layout<C>(
    commitments: &JoltCommitments<C>,
    layout: JoltRaPolynomialLayout,
) -> Result<(), VerifierError> {
    let expected = 2 + layout.total();
    let got = 2
        + commitments.ra.instruction.len()
        + commitments.ra.bytecode.len()
        + commitments.ra.ram.len();
    if got != expected {
        return Err(VerifierError::InvalidCommitmentCount { expected, got });
    }
    if commitments.ra.instruction.len() != layout.instruction()
        || commitments.ra.bytecode.len() != layout.bytecode()
        || commitments.ra.ram.len() != layout.ram()
    {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "commitment layout mismatch: expected instruction={}, bytecode={}, ram={}; got instruction={}, bytecode={}, ram={}",
                layout.instruction(),
                layout.bytecode(),
                layout.ram(),
                commitments.ra.instruction.len(),
                commitments.ra.bytecode.len(),
                commitments.ra.ram.len()
            ),
        });
    }
    Ok(())
}

fn final_advice_opening<PCS, VC, ZkProof>(
    kind: JoltAdviceKind,
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    stage6: &Stage6ClearOutput<PCS::Field>,
    stage7: &Stage7ClearOutput<PCS::Field>,
) -> Result<AdviceFinalOpening<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let max_advice_size = match kind {
        JoltAdviceKind::Trusted => checked.public_io.memory_layout.max_trusted_advice_size,
        JoltAdviceKind::Untrusted => checked.public_io.memory_layout.max_untrusted_advice_size,
    } as usize;
    let layout = AdviceClaimReductionLayout::balanced(
        proof.trace_polynomial_order,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
        max_advice_size,
    );
    let id = advice::final_advice_opening(kind);

    match (kind, layout.dimensions().has_address_phase()) {
        (JoltAdviceKind::Trusted, true) => {
            let verified = stage7
                .batch
                .trusted_advice_address_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            let claim = stage7
                .output_claims
                .advice_address_phase
                .trusted
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpening {
                point: verified.opening_point.clone(),
                opening_claim: claim.opening_claim,
            })
        }
        (JoltAdviceKind::Untrusted, true) => {
            let verified = stage7
                .batch
                .untrusted_advice_address_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            let claim = stage7
                .output_claims
                .advice_address_phase
                .untrusted
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpening {
                point: verified.opening_point.clone(),
                opening_claim: claim.opening_claim,
            })
        }
        (JoltAdviceKind::Trusted, false) => {
            let verified = stage6
                .batch
                .trusted_advice_cycle_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            let claim = stage6
                .output_claims
                .advice_cycle_phase
                .trusted
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpening {
                point: verified.opening_point.clone(),
                opening_claim: claim.opening_claim,
            })
        }
        (JoltAdviceKind::Untrusted, false) => {
            let verified = stage6
                .batch
                .untrusted_advice_cycle_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            let claim = stage6
                .output_claims
                .advice_cycle_phase
                .untrusted
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpening {
                point: verified.opening_point.clone(),
                opening_claim: claim.opening_claim,
            })
        }
    }
}

fn final_advice_opening_point<PCS, VC, ZkProof>(
    kind: JoltAdviceKind,
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    stage6: &Stage6ZkOutput<PCS::Field, VC::Output>,
    stage7: &Stage7ZkOutput<PCS::Field, VC::Output>,
) -> Result<AdviceFinalOpeningPoint<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let max_advice_size = match kind {
        JoltAdviceKind::Trusted => checked.public_io.memory_layout.max_trusted_advice_size,
        JoltAdviceKind::Untrusted => checked.public_io.memory_layout.max_untrusted_advice_size,
    } as usize;
    let layout = AdviceClaimReductionLayout::balanced(
        proof.trace_polynomial_order,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
        max_advice_size,
    );
    let id = advice::final_advice_opening(kind);

    match (kind, layout.dimensions().has_address_phase()) {
        (JoltAdviceKind::Trusted, true) => {
            let verified = stage7
                .trusted_advice_address_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpeningPoint {
                point: verified.opening_point.clone(),
            })
        }
        (JoltAdviceKind::Untrusted, true) => {
            let verified = stage7
                .untrusted_advice_address_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpeningPoint {
                point: verified.opening_point.clone(),
            })
        }
        (JoltAdviceKind::Trusted, false) => {
            let verified = stage6
                .trusted_advice_cycle_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpeningPoint {
                point: verified.opening_point.clone(),
            })
        }
        (JoltAdviceKind::Untrusted, false) => {
            let verified = stage6
                .untrusted_advice_cycle_phase
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            Ok(AdviceFinalOpeningPoint {
                point: verified.opening_point.clone(),
            })
        }
    }
}
