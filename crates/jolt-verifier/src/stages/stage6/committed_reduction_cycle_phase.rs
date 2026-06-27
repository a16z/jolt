//! The stage 6b committed-program claim-reduction cycle phases.
//!
//! In committed-program mode the advice, program-image, and per-chunk bytecode
//! polynomials are reduced over the shared precommitted schedule. Each reduction's
//! cycle phase runs here (stage 6b); when the polynomial still has active
//! address-phase rounds it stages an intermediate opening consumed by stage 7,
//! otherwise it completes here, opening the committed polynomial under a
//! `FinalScale` / `ChunkOutputWeight` public.
//!
//! Because the produced opening point is the (reverse-ordered) cycle opening
//! point while the scale is evaluated at the Dory-permuted point, each relation
//! OVERRIDES [`ConcreteSumcheck::expected_output`] to recover the scale from the
//! produced opening point via the layout's `cycle_phase_*_at_opening_point`
//! helpers — see [`PrecommittedClaimReduction::cycle_phase_permuted_from_opening_point`].
//! The output expression is bypassed (not the `resolve_public` path) because the
//! produced opening id is dynamic in `has_address_phase`; the override computes
//! exactly the formula value, so the clear path and BlindFold stay in sync.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
    AdviceCyclePhaseInputClaims, AdviceCyclePhaseOutputClaims,
};
pub use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
    BytecodeReductionCyclePhaseChallenges, BytecodeReductionCyclePhaseInputClaims,
    BytecodeReductionCyclePhaseOutputClaims,
};
pub use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
    ProgramImageReductionCyclePhaseInputClaims, ProgramImageReductionCyclePhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::claim_reductions::bytecode::BytecodeOutputWeightInputs, AdviceClaimReductionLayout,
    BytecodeClaimReductionLayout, JoltAdviceKind, JoltRelationId, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;

use super::outputs::BytecodeReductionWeights;
use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage4::Stage4ClearOutput;
use crate::VerifierError;

/// Wire the consumed RAM value-check advice opening, keyed by kind. (Verifier-side
/// constructor for the moved [`AdviceCyclePhaseInputClaims`].)
pub fn advice_cycle_phase_inputs_from_upstream<F: Field>(
    stage4: &Stage4ClearOutput<F>,
) -> AdviceCyclePhaseInputClaims<OpeningClaim<F>> {
    let opening = |kind| {
        stage4
            .ram_val_check_init
            .advice_contributions
            .iter()
            .find(|contribution| contribution.kind == kind)
            .map(|contribution| OpeningClaim {
                point: contribution.opening.point.clone(),
                value: contribution.opening.value,
            })
    };
    AdviceCyclePhaseInputClaims {
        trusted: opening(JoltAdviceKind::Trusted),
        untrusted: opening(JoltAdviceKind::Untrusted),
    }
}

pub struct AdviceCyclePhase<F: Field> {
    symbolic: relations::claim_reductions::advice::CyclePhase,
    kind: JoltAdviceKind,
    layout: AdviceClaimReductionLayout,
    /// The RAM address point of the staged advice opening from RAM value-check;
    /// the `FinalScale` public compares the produced opening point against it.
    reference_opening_point: Vec<F>,
}

impl<F: Field> AdviceCyclePhase<F> {
    pub fn new(
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::advice::CyclePhase::new((
                kind,
                layout.dimensions(),
            )),
            kind,
            layout: layout.clone(),
            reference_opening_point,
        }
    }

    pub fn kind(&self) -> JoltAdviceKind {
        self.kind
    }

    fn output<'a>(
        &self,
        outputs: &'a AdviceCyclePhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<&'a OpeningClaim<F>, VerifierError> {
        match self.kind {
            JoltAdviceKind::Trusted => outputs.trusted.as_ref(),
            JoltAdviceKind::Untrusted => outputs.untrusted.as_ref(),
        }
        .ok_or_else(|| advice_public_failed("advice cycle phase produced no opening"))
    }
}

fn advice_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::AdviceClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for AdviceCyclePhase<F> {
    type Symbolic = relations::claim_reductions::advice::CyclePhase;
    type Inputs<C> = AdviceCyclePhaseInputClaims<C>;
    type Outputs<C> = AdviceCyclePhaseOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &AdviceCyclePhaseInputClaims<C>,
    ) -> Result<AdviceCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(advice_public_failed)?;
        Ok(match self.kind {
            JoltAdviceKind::Trusted => AdviceCyclePhaseOutputClaims {
                trusted: Some(opening_point),
                untrusted: None,
            },
            JoltAdviceKind::Untrusted => AdviceCyclePhaseOutputClaims {
                trusted: None,
                untrusted: Some(opening_point),
            },
        })
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &AdviceCyclePhaseInputClaims<C>,
        outputs: &AdviceCyclePhaseOutputClaims<OpeningClaim<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let opening = self.output(outputs)?;
        if self.layout.dimensions().has_address_phase() {
            Ok(opening.value)
        } else {
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(&self.reference_opening_point, &opening.point)
                .map_err(advice_public_failed)?;
            Ok(scale * opening.value)
        }
    }
}

/// Wire the consumed RAM value-check program-image contribution. (Verifier-side
/// constructor for the moved [`ProgramImageReductionCyclePhaseInputClaims`].)
pub fn program_image_reduction_cycle_phase_inputs_from_upstream<F: Field>(
    stage4: &Stage4ClearOutput<F>,
) -> Result<ProgramImageReductionCyclePhaseInputClaims<OpeningClaim<F>>, VerifierError> {
    let contribution = stage4
        .ram_val_check_init
        .program_image_contribution
        .as_ref()
        .ok_or_else(|| {
            program_image_public_failed("missing RAM value-check program-image contribution")
        })?;
    Ok(ProgramImageReductionCyclePhaseInputClaims {
        contribution: OpeningClaim {
            point: contribution.point.clone(),
            value: contribution.value,
        },
    })
}

pub struct ProgramImageReductionCyclePhase<F: Field> {
    symbolic: relations::claim_reductions::program_image::CyclePhase,
    layout: ProgramImageClaimReductionLayout,
    /// The RAM address component of the `RamVal` opening from RAM read-write
    /// checking; the `FinalScale` public compares the produced opening point
    /// against it.
    r_addr_rw: Vec<F>,
}

impl<F: Field> ProgramImageReductionCyclePhase<F> {
    pub fn new(layout: &ProgramImageClaimReductionLayout, r_addr_rw: Vec<F>) -> Self {
        Self {
            symbolic: relations::claim_reductions::program_image::CyclePhase::new(
                layout.dimensions(),
            ),
            layout: layout.clone(),
            r_addr_rw,
        }
    }
}

fn program_image_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::ProgramImageClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for ProgramImageReductionCyclePhase<F> {
    type Symbolic = relations::claim_reductions::program_image::CyclePhase;
    type Inputs<C> = ProgramImageReductionCyclePhaseInputClaims<C>;
    type Outputs<C> = ProgramImageReductionCyclePhaseOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &ProgramImageReductionCyclePhaseInputClaims<C>,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(program_image_public_failed)?;
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: opening_point,
        })
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &ProgramImageReductionCyclePhaseInputClaims<C>,
        outputs: &ProgramImageReductionCyclePhaseOutputClaims<OpeningClaim<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let opening = &outputs.program_image;
        if self.layout.dimensions().has_address_phase() {
            Ok(opening.value)
        } else {
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(&self.r_addr_rw, &opening.point)
                .map_err(program_image_public_failed)?;
            Ok(scale * opening.value)
        }
    }
}

/// The consumed staged `BytecodeValStage` openings from the bytecode read-RAF
/// address phase. (Verifier-side constructor for the moved
/// [`BytecodeReductionCyclePhaseInputClaims`].)
pub fn bytecode_reduction_cycle_phase_inputs_from_values<F: Field>(
    val_stages: Vec<OpeningClaim<F>>,
) -> BytecodeReductionCyclePhaseInputClaims<OpeningClaim<F>> {
    BytecodeReductionCyclePhaseInputClaims { val_stages }
}

pub struct BytecodeReductionCyclePhase<F: Field> {
    symbolic: relations::claim_reductions::bytecode::CyclePhase,
    layout: BytecodeClaimReductionLayout,
    weights: BytecodeReductionWeights<F>,
    chunk_count: usize,
}

impl<F: Field> BytecodeReductionCyclePhase<F> {
    pub fn new(
        layout: &BytecodeClaimReductionLayout,
        weights: BytecodeReductionWeights<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::bytecode::CyclePhase::new((
                layout.dimensions(),
                layout.chunk_count(),
            )),
            layout: layout.clone(),
            weights,
            chunk_count: layout.chunk_count(),
        }
    }

    fn output_weight_inputs(&self) -> BytecodeOutputWeightInputs<'_, F> {
        BytecodeOutputWeightInputs {
            r_bc: &self.weights.r_bc,
            chunk_rbc_weights: &self.weights.chunk_rbc_weights,
            lane_weights: &self.weights.lane_weights,
        }
    }
}

fn bytecode_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReductionCyclePhase<F> {
    type Symbolic = relations::claim_reductions::bytecode::CyclePhase;
    type Inputs<C> = BytecodeReductionCyclePhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionCyclePhaseOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReductionCyclePhaseInputClaims<C>,
    ) -> Result<BytecodeReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(bytecode_public_failed)?;
        Ok(if self.layout.dimensions().has_address_phase() {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(opening_point),
                chunks: Vec::new(),
            }
        } else {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: vec![opening_point; self.chunk_count],
            }
        })
    }

    fn expected_output<C: GetPoint<F>>(
        &self,
        _inputs: &BytecodeReductionCyclePhaseInputClaims<C>,
        outputs: &BytecodeReductionCyclePhaseOutputClaims<OpeningClaim<F>>,
        _challenges: &BytecodeReductionCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        if self.layout.dimensions().has_address_phase() {
            let intermediate = outputs.intermediate.as_ref().ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no intermediate")
            })?;
            return Ok(intermediate.value);
        }
        if outputs.chunks.len() != self.chunk_count {
            return Err(bytecode_public_failed(format!(
                "bytecode chunk claim count mismatch: expected {}, got {}",
                self.chunk_count,
                outputs.chunks.len()
            )));
        }
        let opening_point = outputs
            .chunks
            .first()
            .map(|chunk| chunk.point.as_slice())
            .ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no chunk openings")
            })?;
        let weights = self
            .layout
            .cycle_phase_final_output_weights_at_opening_point(
                self.output_weight_inputs(),
                opening_point,
            )
            .map_err(bytecode_public_failed)?;
        Ok(outputs
            .chunks
            .iter()
            .zip(weights)
            .map(|(chunk, weight)| chunk.value * weight)
            .sum())
    }
}
