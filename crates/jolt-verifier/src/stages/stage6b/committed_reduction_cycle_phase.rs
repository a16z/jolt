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
//! The output expression is bypassed (not the `derive_output_term` path) because the
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
use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage4::RamValCheckInitialEvaluation;
use crate::VerifierError;

/// Wire the consumed RAM value-check advice opening *value* for `kind`, in the
/// shared [`AdviceCyclePhaseInputClaims`] with only that kind's slot filled (the
/// relation's input `Expr` reads only its own kind's opening). Clear-only.
pub fn advice_cycle_phase_input_values_from_upstream<F: Field>(
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
    kind: JoltAdviceKind,
) -> AdviceCyclePhaseInputClaims<F> {
    let value = ram_val_check_init
        .advice_contribution(kind)
        .map(|contribution| contribution.opening_value);
    match kind {
        JoltAdviceKind::Trusted => AdviceCyclePhaseInputClaims {
            trusted: value,
            untrusted: None,
        },
        JoltAdviceKind::Untrusted => AdviceCyclePhaseInputClaims {
            trusted: None,
            untrusted: value,
        },
    }
}

/// The consumed advice opening *points* cell. The relation derives its produced
/// points from its own sumcheck point and reads no input point, so the cell is
/// empty; one constructor serves both proving modes.
pub fn advice_cycle_phase_input_points<F: Field>() -> AdviceCyclePhaseInputClaims<Vec<F>> {
    AdviceCyclePhaseInputClaims {
        trusted: None,
        untrusted: None,
    }
}

pub struct AdviceCyclePhase<F: Field> {
    symbolic: relations::claim_reductions::advice::CyclePhase,
    kind: JoltAdviceKind,
    layout: AdviceClaimReductionLayout,
    /// The RAM address point of the staged advice opening from RAM value-check;
    /// the `FinalScale` public compares the produced opening point against it.
    /// `None` in ZK, where `expected_output` (its only reader) never runs.
    reference_opening_point: Option<Vec<F>>,
}

impl<F: Field> AdviceCyclePhase<F> {
    pub fn new(
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Option<Vec<F>>,
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

    /// The produced advice opening for this kind: its value and point, or an error
    /// if the cycle phase produced no opening.
    fn output<'a>(
        &self,
        output_values: &AdviceCyclePhaseOutputClaims<F>,
        output_points: &'a AdviceCyclePhaseOutputClaims<Vec<F>>,
    ) -> Result<(F, &'a [F]), VerifierError> {
        let (value, point) = match self.kind {
            JoltAdviceKind::Trusted => (output_values.trusted, output_points.trusted()),
            JoltAdviceKind::Untrusted => (output_values.untrusted, output_points.untrusted()),
        };
        match (value, point) {
            (Some(value), Some(point)) => Ok((value, point)),
            _ => Err(advice_public_failed(
                "advice cycle phase produced no opening",
            )),
        }
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Precommitted cycle-phase reductions are bound on the offset-0 prefix of
    /// the batch challenge vector, not the front-loaded suffix.
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &AdviceCyclePhaseInputClaims<Vec<F>>,
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

    fn expected_output(
        &self,
        _input_points: &AdviceCyclePhaseInputClaims<Vec<F>>,
        output_values: &AdviceCyclePhaseOutputClaims<F>,
        output_points: &AdviceCyclePhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let (value, point) = self.output(output_values, output_points)?;
        if self.layout.dimensions().has_address_phase() {
            Ok(value)
        } else {
            let reference_opening_point =
                self.reference_opening_point.as_ref().ok_or_else(|| {
                    advice_public_failed("advice reference opening point unavailable")
                })?;
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(reference_opening_point, point)
                .map_err(advice_public_failed)?;
            Ok(scale * value)
        }
    }
}

/// Wire the consumed RAM value-check program-image contribution *value*.
/// Clear-only.
pub fn program_image_reduction_cycle_phase_input_values_from_upstream<F: Field>(
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
) -> Result<ProgramImageReductionCyclePhaseInputClaims<F>, VerifierError> {
    let value = ram_val_check_init
        .program_image_contribution_value
        .ok_or_else(|| {
            program_image_public_failed("missing RAM value-check program-image contribution")
        })?;
    Ok(ProgramImageReductionCyclePhaseInputClaims {
        contribution: value,
    })
}

/// The consumed program-image contribution *point* cell. The relation derives its
/// produced point from its own sumcheck point and reads no input point, so the
/// cell is empty; one constructor serves both proving modes.
pub fn program_image_reduction_cycle_phase_input_points<F: Field>(
) -> ProgramImageReductionCyclePhaseInputClaims<Vec<F>> {
    ProgramImageReductionCyclePhaseInputClaims {
        contribution: Vec::new(),
    }
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Precommitted cycle-phase reductions are bound on the offset-0 prefix of
    /// the batch challenge vector, not the front-loaded suffix.
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ProgramImageReductionCyclePhaseInputClaims<Vec<F>>,
    ) -> Result<ProgramImageReductionCyclePhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .cycle_phase_opening_point(sumcheck_point)
            .map_err(program_image_public_failed)?;
        Ok(ProgramImageReductionCyclePhaseOutputClaims {
            program_image: opening_point,
        })
    }

    fn expected_output(
        &self,
        _input_points: &ProgramImageReductionCyclePhaseInputClaims<Vec<F>>,
        output_values: &ProgramImageReductionCyclePhaseOutputClaims<F>,
        output_points: &ProgramImageReductionCyclePhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let value = output_values.program_image;
        if self.layout.dimensions().has_address_phase() {
            Ok(value)
        } else {
            let scale = self
                .layout
                .cycle_phase_scale_at_opening_point(&self.r_addr_rw, output_points.program_image())
                .map_err(program_image_public_failed)?;
            Ok(scale * value)
        }
    }
}

/// The consumed staged `BytecodeValStage` opening *values* from the bytecode
/// read-RAF address phase. (Verifier-side constructor for the moved
/// [`BytecodeReductionCyclePhaseInputClaims`].)
pub fn bytecode_reduction_cycle_phase_input_values_from_values<F: Field>(
    val_stages: Vec<F>,
) -> BytecodeReductionCyclePhaseInputClaims<F> {
    BytecodeReductionCyclePhaseInputClaims { val_stages }
}

/// The consumed staged `BytecodeValStage` opening *points* from the bytecode
/// read-RAF address phase.
pub fn bytecode_reduction_cycle_phase_input_points_from_points<F: Field>(
    val_stages: Vec<Vec<F>>,
) -> BytecodeReductionCyclePhaseInputClaims<Vec<F>> {
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Precommitted cycle-phase reductions are bound on the offset-0 prefix of
    /// the batch challenge vector, not the front-loaded suffix.
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReductionCyclePhaseInputClaims<Vec<F>>,
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

    fn expected_output(
        &self,
        _input_points: &BytecodeReductionCyclePhaseInputClaims<Vec<F>>,
        output_values: &BytecodeReductionCyclePhaseOutputClaims<F>,
        output_points: &BytecodeReductionCyclePhaseOutputClaims<Vec<F>>,
        _challenges: &BytecodeReductionCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        if self.layout.dimensions().has_address_phase() {
            let intermediate = output_values.intermediate.ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no intermediate")
            })?;
            return Ok(intermediate);
        }
        if output_values.chunks.len() != self.chunk_count {
            return Err(bytecode_public_failed(format!(
                "bytecode chunk claim count mismatch: expected {}, got {}",
                self.chunk_count,
                output_values.chunks.len()
            )));
        }
        let opening_point = output_points
            .chunks()
            .first()
            .map(Vec::as_slice)
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
        Ok(output_values
            .chunks
            .iter()
            .zip(weights)
            .map(|(chunk, weight)| *chunk * weight)
            .sum())
    }
}
