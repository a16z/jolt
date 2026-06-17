#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::claim_reductions::increments as field_increments, FieldInlineCommittedPolynomial,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::advice,
        committed_openings::{self, advice_commitment_embedding_scale},
        dimensions::TracePolynomialOrder,
        ra::JoltRaPolynomialLayout,
    },
    AdviceClaimReductionLayout, JoltAdviceKind, JoltCommittedPolynomial,
};
use jolt_field::Field;
use jolt_poly::{EqPolynomial, Point, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use super::outputs::Stage8OpeningId;
use crate::{
    stages::{stage6::Stage6ClearOutput, stage7::Stage7ClearOutput},
    VerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8FinalOpening {
    Jolt(JoltCommittedPolynomial),
    #[cfg(feature = "field-inline")]
    FieldInline(FieldInlineCommittedPolynomial),
}

impl Stage8FinalOpening {
    pub fn id(self) -> Stage8OpeningId {
        match self {
            Self::Jolt(polynomial) => committed_openings::final_opening_id(polynomial).into(),
            #[cfg(feature = "field-inline")]
            Self::FieldInline(FieldInlineCommittedPolynomial::FieldRdInc) => {
                field_increments::field_rd_inc_reduced_opening().into()
            }
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
    pub common_opening_point: &'a [F],
    pub dense_embedding_scale: F,
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
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
    #[cfg(feature = "field-inline")]
    openings.push(Stage8FinalOpening::FieldInline(
        FieldInlineCommittedPolynomial::FieldRdInc,
    ));
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
                input.dense_embedding_scale,
                input.common_opening_point,
                input.trusted_advice_layout,
                input.untrusted_advice_layout,
                input.stage6,
                input.stage7,
            )?,
            #[cfg(feature = "field-inline")]
            Stage8FinalOpening::FieldInline(polynomial) => {
                field_inline_final_opening_claim_and_scale(
                    polynomial,
                    input.dense_embedding_scale,
                    input.stage6,
                )
            }
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

pub fn stage8_final_opening_count(
    layout: JoltRaPolynomialLayout,
    include_trusted_advice: bool,
    include_untrusted_advice: bool,
) -> usize {
    2 + field_inline_final_opening_count()
        + layout.total()
        + usize::from(include_trusted_advice)
        + usize::from(include_untrusted_advice)
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
    let common_opening_point = input
        .stage7
        .batch
        .hamming_weight_claim_reduction
        .opening_point
        .clone();
    if common_opening_point.len() < input.committed_chunk_bits {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "final opening point has {} variables, expected at least {}",
                common_opening_point.len(),
                input.committed_chunk_bits
            ),
        });
    }
    let dense_embedding_scale =
        EqPolynomial::<F>::zero_selector(&common_opening_point[..input.committed_chunk_bits]);
    let pcs_opening_point = Point::high_to_low(
        input
            .trace_polynomial_order
            .commitment_opening_point(&common_opening_point, input.log_t)
            .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                reason: error.to_string(),
            })?,
    );
    let claims = stage8_final_opening_claims(Stage8FinalOpeningClaimsInput {
        layout: input.layout,
        common_opening_point: &common_opening_point,
        dense_embedding_scale,
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
    let opening_point = Point::high_to_low(common_opening_point);

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

#[cfg(feature = "field-inline")]
const fn field_inline_final_opening_count() -> usize {
    1
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_final_opening_count() -> usize {
    0
}

fn jolt_final_opening_claim_and_scale<F: Field>(
    polynomial: JoltCommittedPolynomial,
    dense_embedding_scale: F,
    common_opening_point: &[F],
    trusted_advice_layout: Option<&AdviceClaimReductionLayout>,
    untrusted_advice_layout: Option<&AdviceClaimReductionLayout>,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<(F, F), VerifierError> {
    match polynomial {
        JoltCommittedPolynomial::RamInc => Ok((
            stage6.output_claims.inc_claim_reduction.ram_inc,
            dense_embedding_scale,
        )),
        JoltCommittedPolynomial::RdInc => Ok((
            stage6.output_claims.inc_claim_reduction.rd_inc,
            dense_embedding_scale,
        )),
        JoltCommittedPolynomial::InstructionRa(index) => hamming_weight_opening_claim(
            polynomial,
            index,
            &stage7
                .output_claims
                .hamming_weight_claim_reduction
                .instruction_ra,
        ),
        JoltCommittedPolynomial::BytecodeRa(index) => hamming_weight_opening_claim(
            polynomial,
            index,
            &stage7
                .output_claims
                .hamming_weight_claim_reduction
                .bytecode_ra,
        ),
        JoltCommittedPolynomial::RamRa(index) => hamming_weight_opening_claim(
            polynomial,
            index,
            &stage7.output_claims.hamming_weight_claim_reduction.ram_ra,
        ),
        JoltCommittedPolynomial::TrustedAdvice => advice_opening_claim_and_scale(
            JoltAdviceKind::Trusted,
            trusted_advice_layout,
            common_opening_point,
            stage6,
            stage7,
        ),
        JoltCommittedPolynomial::UntrustedAdvice => advice_opening_claim_and_scale(
            JoltAdviceKind::Untrusted,
            untrusted_advice_layout,
            common_opening_point,
            stage6,
            stage7,
        ),
    }
}

#[cfg(feature = "field-inline")]
fn field_inline_final_opening_claim_and_scale<F: Field>(
    polynomial: FieldInlineCommittedPolynomial,
    dense_embedding_scale: F,
    stage6: &Stage6ClearOutput<F>,
) -> (F, F) {
    match polynomial {
        FieldInlineCommittedPolynomial::FieldRdInc => (
            stage6
                .output_claims
                .field_inline
                .field_registers_inc_claim_reduction
                .field_rd_inc,
            dense_embedding_scale,
        ),
    }
}

fn hamming_weight_opening_claim<F: Field>(
    polynomial: JoltCommittedPolynomial,
    index: usize,
    claims: &[F],
) -> Result<(F, F), VerifierError> {
    let claim = claims
        .get(index)
        .copied()
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: committed_openings::final_opening_id(polynomial),
        })?;
    Ok((claim, F::one()))
}

fn advice_opening_claim_and_scale<F: Field>(
    kind: JoltAdviceKind,
    layout: Option<&AdviceClaimReductionLayout>,
    common_opening_point: &[F],
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<(F, F), VerifierError> {
    let final_opening = stage8_clear_final_advice_opening(kind, layout, stage6, stage7)?;
    let scale = advice_commitment_embedding_scale(common_opening_point, &final_opening.point);
    Ok((final_opening.opening_claim, scale))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8ClearAdviceFinalOpening<F: Field> {
    pub point: Vec<F>,
    pub opening_claim: F,
}

pub fn stage8_clear_final_advice_opening<F: Field>(
    kind: JoltAdviceKind,
    layout: Option<&AdviceClaimReductionLayout>,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
) -> Result<Stage8ClearAdviceFinalOpening<F>, VerifierError> {
    let layout = layout.ok_or_else(|| VerifierError::MissingOpeningClaim {
        id: advice::final_advice_opening(kind),
    })?;
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
            Ok(Stage8ClearAdviceFinalOpening {
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
            Ok(Stage8ClearAdviceFinalOpening {
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
            Ok(Stage8ClearAdviceFinalOpening {
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
            Ok(Stage8ClearAdviceFinalOpening {
                point: verified.opening_point.clone(),
                opening_claim: claim.opening_claim,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::*;
    use jolt_claims::protocols::jolt::JoltRelationId;

    fn layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 2).unwrap_or_else(|error| {
            panic!("test layout should be valid: {error}");
        })
    }

    #[test]
    fn order_matches_stage8_rlc_batch() {
        let order = stage8_final_opening_order(layout(), true, true);

        #[cfg(not(feature = "field-inline"))]
        assert_eq!(
            order,
            vec![
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamInc),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RdInc),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::InstructionRa(0)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::InstructionRa(1)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::BytecodeRa(0)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamRa(0)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamRa(1)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::TrustedAdvice),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::UntrustedAdvice),
            ]
        );

        #[cfg(feature = "field-inline")]
        assert_eq!(
            order,
            vec![
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamInc),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RdInc),
                Stage8FinalOpening::FieldInline(FieldInlineCommittedPolynomial::FieldRdInc),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::InstructionRa(0)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::InstructionRa(1)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::BytecodeRa(0)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamRa(0)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::RamRa(1)),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::TrustedAdvice),
                Stage8FinalOpening::Jolt(JoltCommittedPolynomial::UntrustedAdvice),
            ]
        );
    }

    #[test]
    fn ids_use_sumcheck_sources() {
        let ids = stage8_final_opening_order(layout(), true, false)
            .into_iter()
            .map(Stage8FinalOpening::id)
            .collect::<Vec<_>>();

        let mut expected = vec![
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::RamInc,
                JoltRelationId::IncClaimReduction,
            )),
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::RdInc,
                JoltRelationId::IncClaimReduction,
            )),
        ];
        #[cfg(feature = "field-inline")]
        expected.push(Stage8OpeningId::FieldInline(
            field_increments::field_rd_inc_reduced_opening(),
        ));
        expected.extend([
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(0),
                JoltRelationId::HammingWeightClaimReduction,
            )),
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::InstructionRa(1),
                JoltRelationId::HammingWeightClaimReduction,
            )),
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::BytecodeRa(0),
                JoltRelationId::HammingWeightClaimReduction,
            )),
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(0),
                JoltRelationId::HammingWeightClaimReduction,
            )),
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::committed(
                JoltCommittedPolynomial::RamRa(1),
                JoltRelationId::HammingWeightClaimReduction,
            )),
            Stage8OpeningId::Jolt(jolt_claims::protocols::jolt::JoltOpeningId::trusted_advice(
                JoltRelationId::AdviceClaimReduction,
            )),
        ]);

        assert_eq!(ids, expected);
    }
}
