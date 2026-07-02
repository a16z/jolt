//! Resolving the final openings of the precommitted polynomials for stage 8.
//!
//! Each precommitted claim reduction (advice, committed bytecode, program image)
//! is completed either by stage 7's address phase or by the stage 6b cycle phase
//! (whichever ran the last round). Stage 8 consumes the resolved openings as the
//! anchors and batch members of the final PCS opening, so the resolution happens
//! here, next to that consumer, before any stage-8 transcript operation.

use jolt_claims::protocols::jolt::geometry::claim_reductions::{
    advice,
    bytecode::{self as bytecode_reduction},
    program_image,
};
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind,
    JoltCommittedPolynomial, JoltRelationId, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_field::Field;

use crate::stages::stage6b::outputs::{Stage6bOutputClaims, Stage6bOutputPoints};
use crate::stages::stage7::outputs::{Stage7OutputClaims, Stage7OutputPoints};
use crate::stages::PrecommittedSchedule;
use crate::VerifierError;

/// Final opening of a precommitted polynomial, resolved from whichever stage
/// completed its claim reduction (stage 6b cycle phase or stage 7 address
/// phase). Stage 8 consumes these as anchors and batch members of the final
/// PCS opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedFinalOpening<F: Field> {
    pub polynomial: JoltCommittedPolynomial,
    pub point: Vec<F>,
    /// `None` in ZK mode, where opening claims stay committed.
    pub opening_claim: Option<F>,
}

/// Opening point and (clear-mode) claim payload recorded by the stage that
/// completed a precommitted claim reduction. `T` is a single claim for advice and
/// the program image, and the per-chunk claim slice for the committed bytecode.
struct PrecommittedFinalSource<'a, F, T = F> {
    point: &'a [F],
    opening_claim: Option<T>,
}

impl<'a, F, T> PrecommittedFinalSource<'a, F, T> {
    fn zk(point: &'a [F]) -> Self {
        Self {
            point,
            opening_claim: None,
        }
    }

    fn clear(point: &'a [F], opening_claim: T) -> Self {
        Self {
            point,
            opening_claim: Some(opening_claim),
        }
    }
}

/// Resolve the final openings of the precommitted polynomials from whichever phase
/// completed each reduction: stage 7's address phase (points off `stage7_points`,
/// clear values off the stage-7 output claims) or the stage 6b cycle phase (points
/// off `stage6_points`, clear values off the stage-6b output claims). In ZK every
/// opening claim stays committed (`None`) and only points are read; in clear mode a
/// source requires both its point and its value. The walk order — trusted advice,
/// untrusted advice, bytecode chunks, program image — fixes stage 8's anchor order.
pub fn precommitted_final_openings<F: Field>(
    schedule: &PrecommittedSchedule,
    stage7_points: &Stage7OutputPoints<F>,
    stage6_points: &Stage6bOutputPoints<F>,
    clear: Option<(&Stage7OutputClaims<F>, &Stage6bOutputClaims<F>)>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let is_clear = clear.is_some();
    let mut openings = Vec::new();
    for (kind, layout) in [
        (JoltAdviceKind::Trusted, schedule.trusted_advice.as_ref()),
        (
            JoltAdviceKind::Untrusted,
            schedule.untrusted_advice.as_ref(),
        ),
    ] {
        if let Some(layout) = layout {
            let address_value = clear.and_then(|(stage7, _)| advice_address_value(stage7, kind));
            let address_phase =
                resolve_source(is_clear, stage7_points.advice_point(kind), address_value);
            let cycle_value = clear.and_then(|(_, stage6)| stage6.advice_cycle_phase_claim(kind));
            let cycle_phase = resolve_source(
                is_clear,
                stage6_points.advice_cycle_phase_opening_point(kind),
                cycle_value,
            );
            openings.push(advice_final_opening(
                kind,
                layout,
                address_phase,
                cycle_phase,
            )?);
        }
    }
    if let Some(layout) = schedule.bytecode.as_ref() {
        let address_value = clear.and_then(|(stage7, _)| {
            stage7
                .bytecode_address_phase
                .as_ref()
                .map(|values| values.chunks.clone())
        });
        let address_phase = resolve_source(is_clear, stage7_points.bytecode_point(), address_value);
        // The stage-6b cycle phase completes the reduction only when it produced the
        // final chunk claims (no intermediate remained), so the clear source is Some
        // only under that guard; the point-only ZK source is unguarded.
        let cycle_value = clear.and_then(|(_, stage6)| {
            stage6
                .bytecode_reduction
                .as_ref()
                .filter(|reduction| {
                    reduction.intermediate.is_none() && !reduction.chunks.is_empty()
                })
                .map(|reduction| reduction.chunks.clone())
        });
        let cycle_phase = resolve_source(
            is_clear,
            stage6_points.bytecode_reduction_opening_point(),
            cycle_value,
        );
        openings.extend(bytecode_final_openings(layout, address_phase, cycle_phase)?);
    }
    if let Some(layout) = schedule.program_image.as_ref() {
        let address_value = clear.and_then(|(stage7, _)| {
            stage7
                .program_image_address_phase
                .as_ref()
                .map(|values| values.program_image)
        });
        let address_phase =
            resolve_source(is_clear, stage7_points.program_image_point(), address_value);
        let cycle_value = clear.and_then(|(_, stage6)| {
            stage6
                .program_image_reduction
                .as_ref()
                .map(|claim| claim.program_image)
        });
        let cycle_phase = resolve_source(
            is_clear,
            stage6_points.program_image_opening_point(),
            cycle_value,
        );
        openings.push(program_image_final_opening(
            layout,
            address_phase,
            cycle_phase,
        )?);
    }
    Ok(openings)
}

/// Build a completing source from a phase's opening point and (clear-only) value.
/// In clear mode both the point and the value must be present (the `zip` semantics
/// the twin clear/zk drivers had); in ZK only the point is read and the claim stays
/// committed (`None`).
fn resolve_source<F: Field, T>(
    is_clear: bool,
    point: Option<&[F]>,
    value: Option<T>,
) -> Option<PrecommittedFinalSource<'_, F, T>> {
    point.and_then(|point| {
        if is_clear {
            value.map(|value| PrecommittedFinalSource::clear(point, value))
        } else {
            Some(PrecommittedFinalSource::zk(point))
        }
    })
}

/// The stage-7 advice address-phase output *value* for `kind` (only that kind's
/// slot is filled on the wire).
fn advice_address_value<F: Field>(
    claims: &Stage7OutputClaims<F>,
    kind: JoltAdviceKind,
) -> Option<F> {
    match kind {
        JoltAdviceKind::Trusted => claims
            .trusted_advice
            .as_ref()
            .and_then(|claims| claims.trusted),
        JoltAdviceKind::Untrusted => claims
            .untrusted_advice
            .as_ref()
            .and_then(|claims| claims.untrusted),
    }
}

/// Resolves the final opening of an advice polynomial from whichever phase
/// completed its reduction: this stage's address phase, or the stage 6b cycle
/// phase when no active address rounds remain.
fn advice_final_opening<F: Field>(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F>>,
) -> Result<PrecommittedFinalOpening<F>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: advice::final_advice_opening(kind),
    })?;
    let polynomial = match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
    };
    Ok(PrecommittedFinalOpening {
        polynomial,
        point: source.point.to_vec(),
        opening_claim: source.opening_claim,
    })
}

/// Resolves the final per-chunk openings of the committed bytecode from whichever
/// phase completed the reduction: this stage's address phase, or the stage 6b
/// cycle phase when no active address rounds remain.
fn bytecode_final_openings<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F, Vec<F>>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F, Vec<F>>>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: bytecode_reduction::final_bytecode_chunk_opening(0),
    })?;
    if let Some(chunk_claims) = &source.opening_claim {
        if chunk_claims.len() != layout.chunk_count() {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeClaimReduction,
                reason: format!(
                    "final bytecode chunk claim count mismatch: expected {}, got {}",
                    layout.chunk_count(),
                    chunk_claims.len()
                ),
            });
        }
    }
    Ok((0..layout.chunk_count())
        .map(|chunk_idx| PrecommittedFinalOpening {
            polynomial: JoltCommittedPolynomial::BytecodeChunk(chunk_idx),
            point: source.point.to_vec(),
            opening_claim: source
                .opening_claim
                .as_ref()
                .map(|chunk_claims| chunk_claims[chunk_idx]),
        })
        .collect())
}

/// Resolves the final opening of the committed program image from whichever phase
/// completed the reduction: this stage's address phase, or the stage 6b cycle
/// phase when no active address rounds remain.
fn program_image_final_opening<F: Field>(
    layout: &ProgramImageClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F>>,
) -> Result<PrecommittedFinalOpening<F>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: program_image::final_program_image_opening(),
    })?;
    Ok(PrecommittedFinalOpening {
        polynomial: JoltCommittedPolynomial::ProgramImageInit,
        point: source.point.to_vec(),
        opening_claim: source.opening_claim,
    })
}
