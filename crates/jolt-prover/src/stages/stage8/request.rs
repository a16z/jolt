#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::claim_reductions::increments as field_increments;
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::advice, committed_openings::advice_commitment_embedding_scale},
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
};
use jolt_field::Field;
use jolt_poly::{EqPolynomial, Point};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};
use jolt_verifier::stages::{
    stage6::Stage6ClearOutput, stage7::outputs::Stage7ClearOutput, stage8::outputs::Stage8OpeningId,
};

use super::input::Stage8ProverConfig;
use super::output::Stage8OpeningStructure;
use crate::ProverError;

/// Derive the deterministic Stage 8 final-opening structure (clear path).
///
/// Mirrors `jolt-verifier/src/stages/stage8/verify.rs`: take the common opening
/// point from Stage 7 hamming-weight output, compute the dense increment
/// embedding scale and the PCS opening point, build the final opening IDs and
/// scaled opening-claim values in the verifier's batch order (RamInc, RdInc,
/// instruction RA, bytecode RA, RAM RA), append `rlc_claims` and the scaled
/// values to the transcript, squeeze the RLC powers, and compute the joint claim
/// and constraint coefficients.
///
/// This is the Stage 8 backend request: it describes WHAT the joint
/// `joint_opening_proof` opens (the opening ids, points, and claim) and HOW the
/// committed polynomials/hints combine (the RLC powers), without yet
/// materializing any polynomial. ZK is not yet wired.
pub fn derive_stage8_opening_structure<F, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage8OpeningStructure<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    Ok(derive_stage8_structure_and_gamma(config, stage6, stage7, transcript)?.0)
}

/// As [`derive_stage8_opening_structure`], but also returns the raw RLC powers
/// `gamma_powers` (the coefficients that combine the committed polynomials and
/// retained hints for the joint opening — distinct from the BlindFold
/// `constraint_coefficients`, which fold in the scaling factors).
pub fn derive_stage8_structure_and_gamma<F, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
) -> Result<(Stage8OpeningStructure<F>, Vec<F>), ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    derive_stage8_structure_and_gamma_inner(config, stage6, stage7, transcript, true)
}

pub fn derive_stage8_zk_structure_and_gamma<F, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
) -> Result<(Stage8OpeningStructure<F>, Vec<F>), ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    derive_stage8_structure_and_gamma_inner(config, stage6, stage7, transcript, false)
}

fn derive_stage8_structure_and_gamma_inner<F, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    transcript: &mut T,
    append_scaled_claims: bool,
) -> Result<(Stage8OpeningStructure<F>, Vec<F>), ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let layout = config.layout;
    let opening_point_vec = stage7
        .batch
        .hamming_weight_claim_reduction
        .opening_point
        .clone();
    if opening_point_vec.len() < config.committed_chunk_bits {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 opening point has {} variables, expected at least {}",
                opening_point_vec.len(),
                config.committed_chunk_bits
            ),
        });
    }
    let dense_embedding_scale =
        EqPolynomial::<F>::zero_selector(&opening_point_vec[..config.committed_chunk_bits]);
    let pcs_opening_point = Point::high_to_low(
        config
            .trace_polynomial_order
            .commitment_opening_point(&opening_point_vec, config.log_t)
            .map_err(|error| ProverError::InvalidStageRequest {
                reason: error.to_string(),
            })?,
    );
    let mut opening_ids: Vec<Stage8OpeningId> = Vec::new();
    let mut scaled_opening_values: Vec<F> = Vec::new();
    let mut scaling_factors: Vec<F> = Vec::new();

    // Core's final PCS batch order intentionally differs from proof payload order.
    opening_ids.push(
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        )
        .into(),
    );
    scaled_opening_values
        .push(stage6.output_claims.inc_claim_reduction.ram_inc * dense_embedding_scale);
    scaling_factors.push(dense_embedding_scale);

    opening_ids.push(
        JoltOpeningId::committed(
            JoltCommittedPolynomial::RdInc,
            JoltRelationId::IncClaimReduction,
        )
        .into(),
    );
    scaled_opening_values
        .push(stage6.output_claims.inc_claim_reduction.rd_inc * dense_embedding_scale);
    scaling_factors.push(dense_embedding_scale);

    #[cfg(feature = "field-inline")]
    {
        opening_ids.push(field_increments::field_rd_inc_reduced_opening().into());
        scaled_opening_values.push(
            stage6
                .output_claims
                .field_inline
                .field_registers_inc_claim_reduction
                .field_rd_inc
                * dense_embedding_scale,
        );
        scaling_factors.push(dense_embedding_scale);
    }

    let hamming = &stage7.output_claims.hamming_weight_claim_reduction;
    push_ra_openings(
        &mut opening_ids,
        &mut scaled_opening_values,
        &mut scaling_factors,
        layout.instruction(),
        &hamming.instruction_ra,
        JoltCommittedPolynomial::InstructionRa,
    )?;
    push_ra_openings(
        &mut opening_ids,
        &mut scaled_opening_values,
        &mut scaling_factors,
        layout.bytecode(),
        &hamming.bytecode_ra,
        JoltCommittedPolynomial::BytecodeRa,
    )?;
    push_ra_openings(
        &mut opening_ids,
        &mut scaled_opening_values,
        &mut scaling_factors,
        layout.ram(),
        &hamming.ram_ra,
        JoltCommittedPolynomial::RamRa,
    )?;
    push_advice_opening(
        &mut opening_ids,
        &mut scaled_opening_values,
        &mut scaling_factors,
        JoltAdviceKind::Trusted,
        config.trusted_advice_layout.as_ref(),
        &opening_point_vec,
        stage6,
        stage7,
    )?;
    push_advice_opening(
        &mut opening_ids,
        &mut scaled_opening_values,
        &mut scaling_factors,
        JoltAdviceKind::Untrusted,
        config.untrusted_advice_layout.as_ref(),
        &opening_point_vec,
        stage6,
        stage7,
    )?;
    let common_point = Point::high_to_low(opening_point_vec);

    if append_scaled_claims {
        transcript.append(&LabelWithCount(
            b"rlc_claims",
            scaled_opening_values.len() as u64,
        ));
        for value in &scaled_opening_values {
            value.append_to_transcript(transcript);
        }
    }
    let gamma_powers = transcript.challenge_scalar_powers(scaled_opening_values.len());

    let joint_claim = gamma_powers
        .iter()
        .zip(&scaled_opening_values)
        .fold(F::zero(), |claim, (gamma, value)| claim + *gamma * *value);
    let constraint_coefficients = gamma_powers
        .iter()
        .zip(&scaling_factors)
        .map(|(gamma, scale)| *gamma * *scale)
        .collect::<Vec<_>>();

    Ok((
        Stage8OpeningStructure {
            opening_ids,
            scaled_opening_values,
            constraint_coefficients,
            opening_point: common_point,
            pcs_opening_point,
            joint_claim,
        },
        gamma_powers,
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "Advice final-opening assembly appends to three verifier-ordered vectors and needs both Stage 6/7 outputs."
)]
fn push_advice_opening<F: Field>(
    opening_ids: &mut Vec<Stage8OpeningId>,
    scaled_opening_values: &mut Vec<F>,
    scaling_factors: &mut Vec<F>,
    kind: JoltAdviceKind,
    layout: Option<&jolt_claims::protocols::jolt::AdviceClaimReductionLayout>,
    common_opening_point: &[F],
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<(), ProverError> {
    let Some(layout) = layout else {
        return Ok(());
    };
    let id = advice::final_advice_opening(kind);
    let (opening_point, opening_claim) = match (kind, layout.dimensions().has_address_phase()) {
        (JoltAdviceKind::Trusted, true) => {
            let verified = stage7
                .batch
                .trusted_advice_address_phase
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 7 verified address phase"))?;
            let claim = stage7
                .output_claims
                .advice_address_phase
                .trusted
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 7 address output claim"))?;
            (verified.opening_point.as_slice(), claim.opening_claim)
        }
        (JoltAdviceKind::Untrusted, true) => {
            let verified = stage7
                .batch
                .untrusted_advice_address_phase
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 7 verified address phase"))?;
            let claim = stage7
                .output_claims
                .advice_address_phase
                .untrusted
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 7 address output claim"))?;
            (verified.opening_point.as_slice(), claim.opening_claim)
        }
        (JoltAdviceKind::Trusted, false) => {
            let verified = stage6
                .batch
                .trusted_advice_cycle_phase
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 6 verified cycle phase"))?;
            let claim = stage6
                .output_claims
                .advice_cycle_phase
                .trusted
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 6 cycle output claim"))?;
            (verified.opening_point.as_slice(), claim.opening_claim)
        }
        (JoltAdviceKind::Untrusted, false) => {
            let verified = stage6
                .batch
                .untrusted_advice_cycle_phase
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 6 verified cycle phase"))?;
            let claim = stage6
                .output_claims
                .advice_cycle_phase
                .untrusted
                .as_ref()
                .ok_or_else(|| missing_advice(kind, "Stage 6 cycle output claim"))?;
            (verified.opening_point.as_slice(), claim.opening_claim)
        }
    };
    let scale = advice_commitment_embedding_scale(common_opening_point, opening_point);
    opening_ids.push(id.into());
    scaled_opening_values.push(opening_claim * scale);
    scaling_factors.push(scale);
    Ok(())
}

fn missing_advice(kind: JoltAdviceKind, context: &'static str) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("Stage 8 missing {kind:?} advice final opening data: {context}"),
    }
}

fn push_ra_openings<F: Field>(
    opening_ids: &mut Vec<Stage8OpeningId>,
    scaled_opening_values: &mut Vec<F>,
    scaling_factors: &mut Vec<F>,
    count: usize,
    claims: &[F],
    make_polynomial: impl Fn(usize) -> JoltCommittedPolynomial,
) -> Result<(), ProverError> {
    for index in 0..count {
        let claim = *claims
            .get(index)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: format!("Stage 8 missing hamming-weight RA opening claim {index}"),
            })?;
        opening_ids.push(
            JoltOpeningId::committed(
                make_polynomial(index),
                JoltRelationId::HammingWeightClaimReduction,
            )
            .into(),
        );
        scaled_opening_values.push(claim);
        scaling_factors.push(F::one());
    }
    Ok(())
}
