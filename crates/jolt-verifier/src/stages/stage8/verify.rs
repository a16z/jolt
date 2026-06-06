use super::{
    inputs::Deps,
    outputs::{Stage8ClearOutput, Stage8OpeningId, Stage8Output, Stage8ZkOutput},
};
#[cfg(feature = "pcs-assist")]
use crate::pcs_assist::{PcsAssistClearInput, PcsAssistZkInput};
use crate::{
    config::JoltProtocolConfig,
    pcs_assist::PcsProofAssist,
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltCommitments, JoltProof},
    stages::{
        stage6::{Stage6ClearOutput, Stage6ZkOutput},
        stage7::{Stage7ClearOutput, Stage7ZkOutput},
    },
    verifier::CheckedInputs,
    VerifierError,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::claim_reductions::increments as field_increments;
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

#[cfg(feature = "field-inline")]
const fn field_inline_final_opening_count() -> usize {
    1
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_final_opening_count() -> usize {
    0
}

pub fn verify<F, PCS, VC, T, ZkProof, PcsAssist>(
    checked: &CheckedInputs,
    config: &JoltProtocolConfig<PcsAssist::Config>,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
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
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = F>,
{
    #[cfg(not(feature = "pcs-assist"))]
    let _ = config;

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
            + field_inline_final_opening_count()
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
                |id: Stage8OpeningId, commitment: &PCS::Output, scale: PCS::Field| {
                    scaling_factors.push(scale);
                    opening_ids.push(id);
                    commitments.push(commitment.clone());
                };

            // Core's final PCS batch order intentionally differs from proof payload order.
            push_opening(
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RamInc,
                    JoltRelationId::IncClaimReduction,
                )
                .into(),
                &proof.commitments.ram_inc,
                dense_embedding_scale,
            );
            push_opening(
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::RdInc,
                    JoltRelationId::IncClaimReduction,
                )
                .into(),
                &proof.commitments.rd_inc,
                dense_embedding_scale,
            );
            #[cfg(feature = "field-inline")]
            push_opening(
                field_increments::field_rd_inc_reduced_opening().into(),
                &proof.commitments.field_inline.field_registers.rd_inc,
                dense_embedding_scale,
            );
            for (index, commitment) in proof.commitments.ra.instruction.iter().enumerate() {
                push_opening(
                    JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(index),
                        JoltRelationId::HammingWeightClaimReduction,
                    )
                    .into(),
                    commitment,
                    PCS::Field::one(),
                );
            }
            for (index, commitment) in proof.commitments.ra.bytecode.iter().enumerate() {
                push_opening(
                    JoltOpeningId::committed(
                        JoltCommittedPolynomial::BytecodeRa(index),
                        JoltRelationId::HammingWeightClaimReduction,
                    )
                    .into(),
                    commitment,
                    PCS::Field::one(),
                );
            }
            for (index, commitment) in proof.commitments.ra.ram.iter().enumerate() {
                push_opening(
                    JoltOpeningId::committed(
                        JoltCommittedPolynomial::RamRa(index),
                        JoltRelationId::HammingWeightClaimReduction,
                    )
                    .into(),
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
                    advice::final_advice_opening(JoltAdviceKind::Trusted).into(),
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
                    advice::final_advice_opening(JoltAdviceKind::Untrusted).into(),
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

        #[cfg(not(feature = "pcs-assist"))]
        let hiding_evaluation_commitment = {
            PCS::verify_zk(
                &joint_commitment,
                &opening_point,
                &proof.joint_opening_proof,
                &preprocessing.pcs_setup,
                transcript,
            )
            .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            })?
        };

        #[cfg(feature = "pcs-assist")]
        let hiding_evaluation_commitment = verify_zk_pcs_assist::<PCS, T, PcsAssist>(
            config,
            proof.pcs_assist.as_ref(),
            PcsAssistZkInput {
                setup: &preprocessing.pcs_setup,
                pcs_proof: &proof.joint_opening_proof,
                commitment: &joint_commitment,
                point: opening_point.as_slice(),
            },
            transcript,
        )?;
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
        + field_inline_final_opening_count()
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
        let mut push_opening = |id: Stage8OpeningId,
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
            )
            .into(),
            &proof.commitments.ram_inc,
            stage6.output_claims.inc_claim_reduction.ram_inc,
            dense_embedding_scale,
        );
        push_opening(
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RdInc,
                JoltRelationId::IncClaimReduction,
            )
            .into(),
            &proof.commitments.rd_inc,
            stage6.output_claims.inc_claim_reduction.rd_inc,
            dense_embedding_scale,
        );
        #[cfg(feature = "field-inline")]
        {
            push_opening(
                field_increments::field_rd_inc_reduced_opening().into(),
                &proof.commitments.field_inline.field_registers.rd_inc,
                stage6
                    .output_claims
                    .field_inline
                    .field_registers_inc_claim_reduction
                    .field_rd_inc,
                dense_embedding_scale,
            );
        }

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
            push_opening(id.into(), commitment, opening_claim, PCS::Field::one());
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
            push_opening(id.into(), commitment, opening_claim, PCS::Field::one());
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
            push_opening(id.into(), commitment, opening_claim, PCS::Field::one());
        }

        if let Some(commitment) = trusted_advice_commitment {
            let id = advice::final_advice_opening(JoltAdviceKind::Trusted);
            let final_opening = trusted_advice
                .as_ref()
                .ok_or(VerifierError::MissingOpeningClaim { id })?;
            push_opening(
                id.into(),
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
                id.into(),
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

    #[cfg(not(feature = "pcs-assist"))]
    {
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
    }

    #[cfg(feature = "pcs-assist")]
    verify_clear_pcs_assist::<PCS, T, PcsAssist>(
        config,
        proof.pcs_assist.as_ref(),
        PcsAssistClearInput {
            setup: &preprocessing.pcs_setup,
            pcs_proof: &proof.joint_opening_proof,
            commitment: &joint_commitment,
            point: pcs_opening_point.as_slice(),
            eval: joint_claim,
        },
        transcript,
    )?;
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

#[cfg(feature = "pcs-assist")]
fn verify_clear_pcs_assist<PCS, T, PcsAssist>(
    config: &JoltProtocolConfig<PcsAssist::Config>,
    assist_proof: Option<&PcsAssist::Proof>,
    input: PcsAssistClearInput<'_, PCS>,
    transcript: &mut T,
) -> Result<(), VerifierError>
where
    PCS: CommitmentScheme,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    let assist_config = config
        .pcs_assist
        .as_ref()
        .ok_or(VerifierError::MissingPcsAssistConfig)?;
    let assist_proof = assist_proof.ok_or(VerifierError::MissingPcsAssistProof)?;
    PcsAssist::verify_clear(assist_config, input, assist_proof, transcript).map_err(|error| {
        VerifierError::PcsAssistVerificationFailed {
            reason: error.to_string(),
        }
    })
}

#[cfg(feature = "pcs-assist")]
fn verify_zk_pcs_assist<PCS, T, PcsAssist>(
    config: &JoltProtocolConfig<PcsAssist::Config>,
    assist_proof: Option<&PcsAssist::Proof>,
    input: PcsAssistZkInput<'_, PCS>,
    transcript: &mut T,
) -> Result<<PCS as ZkOpeningScheme>::HidingCommitment, VerifierError>
where
    PCS: ZkOpeningScheme,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    let assist_config = config
        .pcs_assist
        .as_ref()
        .ok_or(VerifierError::MissingPcsAssistConfig)?;
    let assist_proof = assist_proof.ok_or(VerifierError::MissingPcsAssistProof)?;
    PcsAssist::verify_zk(assist_config, input, assist_proof, transcript).map_err(|error| {
        VerifierError::PcsAssistVerificationFailed {
            reason: error.to_string(),
        }
    })
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

fn final_advice_opening<PCS, VC, ZkProof, PcsAssist>(
    kind: JoltAdviceKind,
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    stage6: &Stage6ClearOutput<PCS::Field>,
    stage7: &Stage7ClearOutput<PCS::Field>,
) -> Result<AdviceFinalOpening<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
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

fn final_advice_opening_point<PCS, VC, ZkProof, PcsAssist>(
    kind: JoltAdviceKind,
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    stage6: &Stage6ZkOutput<PCS::Field, VC::Output>,
    stage7: &Stage7ZkOutput<PCS::Field, VC::Output>,
) -> Result<AdviceFinalOpeningPoint<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
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

#[cfg(all(test, feature = "pcs-assist"))]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert successful verifier results"
)]
mod tests {
    use std::fmt;

    use super::*;
    use jolt_crypto::Commitment;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_openings::OpeningsError;
    use jolt_poly::{MultilinearPoly, Polynomial};
    use jolt_transcript::U64Word;
    use num_traits::Zero;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestPcs;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct TestCommitment(u64);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct TestPcsProof(u64);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct TestSetup(u64);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct TestHidingCommitment(u64);

    impl Commitment for TestPcs {
        type Output = TestCommitment;
    }

    impl CommitmentScheme for TestPcs {
        type Field = Fr;
        type Proof = TestPcsProof;
        type ProverSetup = ();
        type VerifierSetup = TestSetup;
        type Polynomial = Polynomial<Fr>;
        type OpeningHint = ();
        type SetupParams = ();

        fn setup(_params: Self::SetupParams) -> (Self::ProverSetup, Self::VerifierSetup) {
            ((), TestSetup::default())
        }

        fn verifier_setup(_prover_setup: &Self::ProverSetup) -> Self::VerifierSetup {
            TestSetup::default()
        }

        fn commit<P: MultilinearPoly<Self::Field> + ?Sized>(
            _poly: &P,
            _setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            (TestCommitment::default(), ())
        }

        fn open(
            _poly: &Self::Polynomial,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Option<Self::OpeningHint>,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Self::Proof {
            TestPcsProof::default()
        }

        fn verify(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _eval: Self::Field,
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<(), OpeningsError> {
            Ok(())
        }

        fn bind_opening_inputs(
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
            _point: &[Self::Field],
            _eval: &Self::Field,
        ) {
        }
    }

    impl ZkOpeningScheme for TestPcs {
        type HidingCommitment = TestHidingCommitment;
        type Blind = ();

        fn commit_zk<P: MultilinearPoly<Self::Field> + ?Sized>(
            poly: &P,
            setup: &Self::ProverSetup,
        ) -> (Self::Output, Self::OpeningHint) {
            Self::commit(poly, setup)
        }

        fn open_zk(
            _poly: &Self::Polynomial,
            _point: &[Self::Field],
            _eval: Self::Field,
            _setup: &Self::ProverSetup,
            _hint: Self::OpeningHint,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> (Self::Proof, Self::HidingCommitment, Self::Blind) {
            (TestPcsProof::default(), TestHidingCommitment::default(), ())
        }

        fn verify_zk(
            _commitment: &Self::Output,
            _point: &[Self::Field],
            _proof: &Self::Proof,
            _setup: &Self::VerifierSetup,
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
        ) -> Result<Self::HidingCommitment, OpeningsError> {
            Ok(TestHidingCommitment::default())
        }

        fn bind_zk_opening_inputs(
            _transcript: &mut impl Transcript<Challenge = Self::Field>,
            _point: &[Self::Field],
            _hiding_commitment: &Self::HidingCommitment,
        ) {
        }
    }

    impl AppendToTranscript for TestHidingCommitment {
        fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
            transcript.append(&U64Word(self.0));
        }

        fn transcript_payload_len(&self) -> Option<u64> {
            Some(32)
        }
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct TestAssistConfig {
        version: u64,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    enum TestAssistProof {
        Clear {
            config: TestAssistConfig,
            setup: TestSetup,
            pcs_proof: TestPcsProof,
            commitment: TestCommitment,
            point: Vec<Fr>,
            eval: Fr,
        },
        Zk {
            config: TestAssistConfig,
            setup: TestSetup,
            pcs_proof: TestPcsProof,
            commitment: TestCommitment,
            point: Vec<Fr>,
            hiding_commitment: TestHidingCommitment,
        },
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
    struct TestAssist;

    #[derive(Debug)]
    struct TestAssistError(String);

    impl fmt::Display for TestAssistError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for TestAssistError {}

    impl PcsProofAssist<TestPcs> for TestAssist {
        type Proof = TestAssistProof;
        type Config = TestAssistConfig;
        type Error = TestAssistError;

        fn selected_config() -> Self::Config {
            selected_assist_config()
        }

        fn verify_clear<T>(
            config: &Self::Config,
            input: PcsAssistClearInput<'_, TestPcs>,
            proof: &Self::Proof,
            transcript: &mut T,
        ) -> Result<(), Self::Error>
        where
            T: Transcript<Challenge = Fr>,
        {
            let TestAssistProof::Clear {
                config: expected_config,
                setup,
                pcs_proof,
                commitment,
                point,
                eval,
            } = proof
            else {
                return Err(TestAssistError("expected clear assist proof".to_string()));
            };

            verify_common_inputs(
                CommonAssistInput {
                    config,
                    setup: input.setup,
                    pcs_proof: input.pcs_proof,
                    commitment: input.commitment,
                    point: input.point,
                },
                CommonAssistInput {
                    config: expected_config,
                    setup,
                    pcs_proof,
                    commitment,
                    point,
                },
            )?;
            if input.eval != *eval {
                return Err(TestAssistError("clear eval mismatch".to_string()));
            }
            transcript.append(&U64Word(11));
            Ok(())
        }

        fn verify_zk<T>(
            config: &Self::Config,
            input: PcsAssistZkInput<'_, TestPcs>,
            proof: &Self::Proof,
            transcript: &mut T,
        ) -> Result<TestHidingCommitment, Self::Error>
        where
            T: Transcript<Challenge = Fr>,
        {
            let TestAssistProof::Zk {
                config: expected_config,
                setup,
                pcs_proof,
                commitment,
                point,
                hiding_commitment,
            } = proof
            else {
                return Err(TestAssistError("expected zk assist proof".to_string()));
            };

            verify_common_inputs(
                CommonAssistInput {
                    config,
                    setup: input.setup,
                    pcs_proof: input.pcs_proof,
                    commitment: input.commitment,
                    point: input.point,
                },
                CommonAssistInput {
                    config: expected_config,
                    setup,
                    pcs_proof,
                    commitment,
                    point,
                },
            )?;
            transcript.append(&U64Word(22));
            Ok(*hiding_commitment)
        }
    }

    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
        state: [u8; 32],
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::zero()
        }

        fn state(&self) -> &[u8; 32] {
            &self.state
        }
    }

    #[test]
    fn pcs_assist_clear_dispatch_passes_final_opening_inputs() {
        let config = protocol_config();
        let setup = TestSetup(3);
        let pcs_proof = TestPcsProof(5);
        let commitment = TestCommitment(8);
        let point = vec![fr(1), fr(2), fr(3)];
        let eval = fr(13);
        let assist_proof = TestAssistProof::Clear {
            config: selected_assist_config(),
            setup,
            pcs_proof,
            commitment,
            point: point.clone(),
            eval,
        };
        let mut transcript = RecordingTranscript::new(b"pcs-assist-clear");

        let result = verify_clear_pcs_assist::<TestPcs, _, TestAssist>(
            &config,
            Some(&assist_proof),
            PcsAssistClearInput {
                setup: &setup,
                pcs_proof: &pcs_proof,
                commitment: &commitment,
                point: point.as_slice(),
                eval,
            },
            &mut transcript,
        );

        assert!(result.is_ok());
        assert!(transcript_contains_word(&transcript, 11));
    }

    #[test]
    fn pcs_assist_zk_dispatch_passes_final_opening_inputs() {
        let config = protocol_config();
        let setup = TestSetup(4);
        let pcs_proof = TestPcsProof(6);
        let commitment = TestCommitment(9);
        let point = vec![fr(21), fr(34)];
        let hiding_commitment = TestHidingCommitment(55);
        let assist_proof = TestAssistProof::Zk {
            config: selected_assist_config(),
            setup,
            pcs_proof,
            commitment,
            point: point.clone(),
            hiding_commitment,
        };
        let mut transcript = RecordingTranscript::new(b"pcs-assist-zk");

        let result = verify_zk_pcs_assist::<TestPcs, _, TestAssist>(
            &config,
            Some(&assist_proof),
            PcsAssistZkInput {
                setup: &setup,
                pcs_proof: &pcs_proof,
                commitment: &commitment,
                point: point.as_slice(),
            },
            &mut transcript,
        )
        .unwrap();

        assert_eq!(result, hiding_commitment);
        assert!(transcript_contains_word(&transcript, 22));
    }

    #[test]
    fn pcs_assist_dispatch_rejects_missing_config_or_proof() {
        let setup = TestSetup(3);
        let pcs_proof = TestPcsProof(5);
        let commitment = TestCommitment(8);
        let point = vec![fr(1)];
        let eval = fr(2);
        let assist_proof = TestAssistProof::Clear {
            config: selected_assist_config(),
            setup,
            pcs_proof,
            commitment,
            point: point.clone(),
            eval,
        };
        let mut missing_config = protocol_config();
        missing_config.pcs_assist = None;
        let mut transcript = RecordingTranscript::new(b"pcs-assist-missing-config");

        assert!(matches!(
            verify_clear_pcs_assist::<TestPcs, _, TestAssist>(
                &missing_config,
                Some(&assist_proof),
                PcsAssistClearInput {
                    setup: &setup,
                    pcs_proof: &pcs_proof,
                    commitment: &commitment,
                    point: point.as_slice(),
                    eval,
                },
                &mut transcript,
            ),
            Err(VerifierError::MissingPcsAssistConfig)
        ));

        let mut transcript = RecordingTranscript::new(b"pcs-assist-missing-proof");
        assert!(matches!(
            verify_clear_pcs_assist::<TestPcs, _, TestAssist>(
                &protocol_config(),
                None,
                PcsAssistClearInput {
                    setup: &setup,
                    pcs_proof: &pcs_proof,
                    commitment: &commitment,
                    point: point.as_slice(),
                    eval,
                },
                &mut transcript,
            ),
            Err(VerifierError::MissingPcsAssistProof)
        ));
    }

    #[test]
    fn pcs_assist_dispatch_surfaces_assist_rejection() {
        let config = protocol_config();
        let setup = TestSetup(3);
        let pcs_proof = TestPcsProof(5);
        let commitment = TestCommitment(8);
        let point = vec![fr(1), fr(2)];
        let assist_proof = TestAssistProof::Clear {
            config: selected_assist_config(),
            setup,
            pcs_proof,
            commitment,
            point: point.clone(),
            eval: fr(13),
        };
        let mut transcript = RecordingTranscript::new(b"pcs-assist-reject");

        let result = verify_clear_pcs_assist::<TestPcs, _, TestAssist>(
            &config,
            Some(&assist_proof),
            PcsAssistClearInput {
                setup: &setup,
                pcs_proof: &pcs_proof,
                commitment: &commitment,
                point: point.as_slice(),
                eval: fr(99),
            },
            &mut transcript,
        );

        assert!(matches!(
            result,
            Err(VerifierError::PcsAssistVerificationFailed { .. })
        ));
    }

    struct CommonAssistInput<'a> {
        config: &'a TestAssistConfig,
        setup: &'a TestSetup,
        pcs_proof: &'a TestPcsProof,
        commitment: &'a TestCommitment,
        point: &'a [Fr],
    }

    fn verify_common_inputs(
        actual: CommonAssistInput<'_>,
        expected: CommonAssistInput<'_>,
    ) -> Result<(), TestAssistError> {
        if actual.config != expected.config {
            return Err(TestAssistError("assist config mismatch".to_string()));
        }
        if actual.setup != expected.setup {
            return Err(TestAssistError("setup mismatch".to_string()));
        }
        if actual.pcs_proof != expected.pcs_proof {
            return Err(TestAssistError("PCS proof mismatch".to_string()));
        }
        if actual.commitment != expected.commitment {
            return Err(TestAssistError("commitment mismatch".to_string()));
        }
        if actual.point != expected.point {
            return Err(TestAssistError("point mismatch".to_string()));
        }
        Ok(())
    }

    fn selected_assist_config() -> TestAssistConfig {
        TestAssistConfig { version: 7 }
    }

    fn protocol_config() -> JoltProtocolConfig<TestAssistConfig> {
        JoltProtocolConfig::selected_for_zk::<TestPcs, TestAssist>(false)
    }

    fn fr(value: u64) -> Fr {
        <Fr as FromPrimitiveInt>::from_u64(value)
    }

    fn transcript_contains_word(transcript: &RecordingTranscript, word: u64) -> bool {
        let suffix = word.to_be_bytes();
        transcript
            .chunks
            .iter()
            .any(|chunk| chunk.ends_with(&suffix))
    }
}
