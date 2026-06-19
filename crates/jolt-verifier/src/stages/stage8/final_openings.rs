//! Stage 8 final batched-opening order, claims, and RLC structure.
//!
//! These helpers single-source the canonical final-opening order and the
//! Fiat-Shamir absorption that the stage 8 prover and verifier must agree on.
//! [`verify`](super::verify::verify) builds its batch entries from the same
//! [`stage8_final_opening_order`] this module exposes, so the prover and
//! verifier cannot drift on opening order or transcript absorption.

use jolt_claims::protocols::jolt::{
    formulas::{
        committed_openings::{
            self, commitment_embedding_scale, final_opening_point, FinalOpeningPointInputs,
        },
        dimensions::TracePolynomialOrder,
        ra::JoltRaPolynomialLayout,
    },
    AdviceClaimReductionLayout, JoltAdviceKind, JoltCommittedPolynomial,
};
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use super::outputs::Stage8OpeningId;
use crate::{
    stages::{
        relations::OpeningClaim, stage6::Stage6ClearOutput, stage7::outputs::Stage7ClearOutput,
    },
    VerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8FinalOpening {
    Jolt(JoltCommittedPolynomial),
}

impl Stage8FinalOpening {
    pub fn id(self) -> Stage8OpeningId {
        match self {
            Self::Jolt(polynomial) => committed_openings::final_opening_id(polynomial).into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8FinalOpeningClaim<F: Field> {
    pub opening: Stage8FinalOpening,
    pub id: Stage8OpeningId,
    pub opening_claim: F,
    pub scale: F,
}

impl<F: Field> Stage8FinalOpeningClaim<F> {
    pub fn scaled_value(&self) -> F {
        self.opening_claim * self.scale
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8FinalOpeningStructure<F: Field> {
    pub opening_ids: Vec<Stage8OpeningId>,
    pub scaled_opening_values: Vec<F>,
    pub constraint_coefficients: Vec<F>,
    /// Unified big-endian opening point, low-to-high stored as `HIGH_TO_LOW`.
    /// Identical to [`Self::pcs_opening_point`]; both carry the same point so the
    /// prover's `bind_opening_inputs` matches the verifier exactly.
    pub opening_point: Point<HIGH_TO_LOW, F>,
    pub pcs_opening_point: Point<HIGH_TO_LOW, F>,
    pub joint_claim: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8FinalOpeningBatch<F: Field> {
    pub claims: Vec<Stage8FinalOpeningClaim<F>>,
    pub structure: Stage8FinalOpeningStructure<F>,
    pub gamma_powers: Vec<F>,
}

pub struct Stage8FinalOpeningClaimsInput<'a, F: Field> {
    pub layout: JoltRaPolynomialLayout,
    /// Unified big-endian opening point from [`final_opening_point`].
    pub opening_point: &'a [F],
    /// Stage 7 hamming-weight claim-reduction opening point (the own point of
    /// the one-hot `Ra` polynomials).
    pub hamming_weight_opening_point: &'a [F],
    /// Stage 6 increment claim-reduction opening point (the own point of the
    /// dense `Inc` polynomials).
    pub inc_claim_reduction_opening_point: &'a [F],
    /// `Some(..)` signals trusted advice is part of the batch.
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    /// `Some(..)` signals untrusted advice is part of the batch.
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub stage6: &'a Stage6ClearOutput<F>,
    pub stage7: &'a Stage7ClearOutput<F>,
}

pub struct Stage8FinalOpeningBatchInput<'a, F: Field> {
    pub log_t: usize,
    pub committed_chunk_bits: usize,
    pub layout: JoltRaPolynomialLayout,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub stage6: &'a Stage6ClearOutput<F>,
    pub stage7: &'a Stage7ClearOutput<F>,
}

/// Canonical stage 8 final-opening order.
///
/// Mirrors [`final_opening_polynomial_order`](committed_openings::final_opening_polynomial_order)
/// for the non-committed-program case so the prover and verifier RLC batches
/// share one source of truth.
pub fn stage8_final_opening_order(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
) -> Vec<Stage8FinalOpening> {
    let mut openings = Vec::with_capacity(stage8_final_opening_count(
        layout,
        include_trusted_advice,
        include_untrusted_advice,
    ));
    openings.push(Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamInc));
    openings.push(Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RdInc));
    openings.extend(
        (0..layout.instruction())
            .map(JoltCommittedPolynomial::InstructionRa)
            .map(Stage8FinalOpening::Jolt),
    );
    openings.extend(
        (0..layout.bytecode())
            .map(JoltCommittedPolynomial::BytecodeRa)
            .map(Stage8FinalOpening::Jolt),
    );
    openings.extend(
        (0..layout.ram())
            .map(JoltCommittedPolynomial::RamRa)
            .map(Stage8FinalOpening::Jolt),
    );
    if include_trusted_advice {
        openings.push(Stage8FinalOpening::Jolt(
            JoltCommittedPolynomial::TrustedAdvice,
        ));
    }
    if include_untrusted_advice {
        openings.push(Stage8FinalOpening::Jolt(
            JoltCommittedPolynomial::UntrustedAdvice,
        ));
    }
    openings
}

pub fn stage8_final_opening_count(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
) -> usize {
    2 + layout.total() + usize::from(include_trusted_advice) + usize::from(include_untrusted_advice)
}

/// Resolves the per-polynomial opening claim, embedding scale, and opening id
/// for every member of [`stage8_final_opening_order`], pulling claims from the
/// stage 6/7 clear outputs and advice points from the precommitted finals.
pub fn stage8_final_opening_claims<F: Field>(
    input: Stage8FinalOpeningClaimsInput<'_, F>,
) -> Result<Vec<Stage8FinalOpeningClaim<F>>, VerifierError> {
    let final_openings = stage8_final_opening_order(
        input.layout,
        input.trusted_advice_layout.is_some(),
        input.untrusted_advice_layout.is_some(),
    );
    let mut claims = Vec::with_capacity(final_openings.len());
    for opening in final_openings {
        let (opening_claim, scale) = match opening {
            Stage8FinalOpening::Jolt(polynomial) => jolt_final_opening_claim_and_scale(
                polynomial,
                input.opening_point,
                input.hamming_weight_opening_point,
                input.inc_claim_reduction_opening_point,
                input.stage6,
                input.stage7,
            )?,
        };
        claims.push(Stage8FinalOpeningClaim {
            opening,
            id: opening.id(),
            opening_claim,
            scale,
        });
    }
    Ok(claims)
}

pub fn stage8_clear_final_opening_batch<F, T>(
    input: Stage8FinalOpeningBatchInput<'_, F>,
    transcript: &mut T,
) -> Result<Stage8FinalOpeningBatch<F>, VerifierError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    stage8_final_opening_batch(input, transcript, true)
}

pub fn stage8_zk_final_opening_batch<F, T>(
    input: Stage8FinalOpeningBatchInput<'_, F>,
    transcript: &mut T,
) -> Result<Stage8FinalOpeningBatch<F>, VerifierError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    stage8_final_opening_batch(input, transcript, false)
}

fn stage8_final_opening_batch<F, T>(
    input: Stage8FinalOpeningBatchInput<'_, F>,
    transcript: &mut T,
    append_scaled_claims: bool,
) -> Result<Stage8FinalOpeningBatch<F>, VerifierError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let hamming_weight_opening_point = input.stage7.hamming_weight_opening_point.clone();
    let inc_claim_reduction_opening_point =
        input.stage6.batch.inc_claim_reduction.opening_point.clone();

    let anchor_points: Vec<&[F]> = input
        .stage7
        .precommitted_final_openings
        .iter()
        .map(|opening| opening.point.as_slice())
        .collect();
    let opening_point = final_opening_point(FinalOpeningPointInputs {
        log_t: input.log_t,
        log_k_chunk: input.committed_chunk_bits,
        trace_order: input.trace_polynomial_order,
        hamming_weight_opening_point: &hamming_weight_opening_point,
        inc_claim_reduction_opening_point: &inc_claim_reduction_opening_point,
        precommitted_anchor_points: &anchor_points,
    })
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })?;

    let claims = stage8_final_opening_claims(Stage8FinalOpeningClaimsInput {
        layout: input.layout,
        opening_point: &opening_point,
        hamming_weight_opening_point: &hamming_weight_opening_point,
        inc_claim_reduction_opening_point: &inc_claim_reduction_opening_point,
        trusted_advice_layout: input.trusted_advice_layout,
        untrusted_advice_layout: input.untrusted_advice_layout,
        stage6: input.stage6,
        stage7: input.stage7,
    })?;
    let opening_ids = claims.iter().map(|claim| claim.id).collect::<Vec<_>>();
    let scaled_opening_values = claims
        .iter()
        .map(Stage8FinalOpeningClaim::scaled_value)
        .collect::<Vec<_>>();
    let scaling_factors = claims.iter().map(|claim| claim.scale).collect::<Vec<_>>();

    let pcs_opening_point = Point::high_to_low(opening_point);
    let opening_point = pcs_opening_point.clone();

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
        .zip(scaling_factors)
        .map(|(gamma, scale)| *gamma * scale)
        .collect::<Vec<_>>();

    Ok(Stage8FinalOpeningBatch {
        claims,
        structure: Stage8FinalOpeningStructure {
            opening_ids,
            scaled_opening_values,
            constraint_coefficients,
            opening_point,
            pcs_opening_point,
            joint_claim,
        },
        gamma_powers,
    })
}

fn jolt_final_opening_claim_and_scale<F: Field>(
    polynomial: JoltCommittedPolynomial,
    opening_point: &[F],
    hamming_weight_opening_point: &[F],
    inc_claim_reduction_opening_point: &[F],
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<(F, F), VerifierError> {
    match polynomial {
        JoltCommittedPolynomial::RamInc => Ok((
            stage6.output_claims.inc_claim_reduction.ram_inc,
            commitment_embedding_scale(opening_point, inc_claim_reduction_opening_point),
        )),
        JoltCommittedPolynomial::RdInc => Ok((
            stage6.output_claims.inc_claim_reduction.rd_inc,
            commitment_embedding_scale(opening_point, inc_claim_reduction_opening_point),
        )),
        JoltCommittedPolynomial::InstructionRa(index) => hamming_weight_opening_claim(
            polynomial,
            index,
            &stage7
                .output_claims
                .hamming_weight_claim_reduction
                .instruction_ra,
            opening_point,
            hamming_weight_opening_point,
        ),
        JoltCommittedPolynomial::BytecodeRa(index) => hamming_weight_opening_claim(
            polynomial,
            index,
            &stage7
                .output_claims
                .hamming_weight_claim_reduction
                .bytecode_ra,
            opening_point,
            hamming_weight_opening_point,
        ),
        JoltCommittedPolynomial::RamRa(index) => hamming_weight_opening_claim(
            polynomial,
            index,
            &stage7.output_claims.hamming_weight_claim_reduction.ram_ra,
            opening_point,
            hamming_weight_opening_point,
        ),
        JoltCommittedPolynomial::TrustedAdvice => {
            advice_opening_claim_and_scale(JoltAdviceKind::Trusted, opening_point, stage7)
        }
        JoltCommittedPolynomial::UntrustedAdvice => {
            advice_opening_claim_and_scale(JoltAdviceKind::Untrusted, opening_point, stage7)
        }
        JoltCommittedPolynomial::BytecodeChunk(_) | JoltCommittedPolynomial::ProgramImageInit => {
            // Committed-program members are not part of the prover-driven order
            // produced by `stage8_final_opening_order`; `verify()` handles them
            // directly from the precommitted finals.
            Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "committed-program polynomial {polynomial:?} is not part of the stage 8 prover order"
                ),
            })
        }
    }
}

fn hamming_weight_opening_claim<F: Field>(
    polynomial: JoltCommittedPolynomial,
    index: usize,
    claims: &[OpeningClaim<F>],
    opening_point: &[F],
    hamming_weight_opening_point: &[F],
) -> Result<(F, F), VerifierError> {
    let claim = claims.get(index).map(|claim| claim.value).ok_or_else(|| {
        VerifierError::MissingOpeningClaim {
            id: committed_openings::final_opening_id(polynomial),
        }
    })?;
    Ok((
        claim,
        commitment_embedding_scale(opening_point, hamming_weight_opening_point),
    ))
}

fn advice_opening_claim_and_scale<F: Field>(
    kind: JoltAdviceKind,
    opening_point: &[F],
    stage7: &Stage7ClearOutput<F>,
) -> Result<(F, F), VerifierError> {
    let polynomial = match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
    };
    let opening = stage7
        .precommitted_final_openings
        .iter()
        .find(|opening| opening.polynomial == polynomial)
        .ok_or(VerifierError::MissingOpeningClaim {
            id: committed_openings::final_opening_id(polynomial),
        })?;
    let opening_claim =
        opening
            .opening_claim
            .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
                reason: format!("missing clear advice opening claim for {polynomial:?}"),
            })?;
    let scale = commitment_embedding_scale(opening_point, opening.point.as_slice());
    Ok((opening_claim, scale))
}
