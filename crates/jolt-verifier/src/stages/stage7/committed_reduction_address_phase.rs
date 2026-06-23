//! The stage 7 committed-program claim-reduction address phases.
//!
//! In committed-program mode the bytecode value columns and the initial-RAM
//! program image are committed polynomials. Their two-phase reductions begin in
//! stage 6b (cycle phase) and, when active address-phase rounds remain, finish
//! here in stage 7. Each is a self-contained relation object: the bytecode
//! reduction opens the per-chunk `BytecodeChunk(i)` commitments under the
//! `ChunkOutputWeight(i)` publics, and the program-image reduction opens
//! `ProgramImageInit` under a single `FinalScale` public.
//!
//! Both publics are functions of the reduction's final opening point — the same
//! point `derive_opening_points` produces — so `resolve_public` recovers that
//! point from the output claims and asks the layout for the scale/weights at it,
//! exactly as stage 4's `RamValCheck` recovers the cycle from its output point.

use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::{
        bytecode::{self as bytecode_reduction, BytecodeOutputWeightInputs},
        program_image,
    },
    BytecodeClaimReductionLayout, BytecodeClaimReductionPublic, JoltPublicId, JoltRelationClaims,
    JoltRelationId, PrecommittedReductionLayout, ProgramImageClaimReductionLayout,
    ProgramImageClaimReductionPublic,
};
use jolt_field::Field;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::VerifierError;

/// Produced per-chunk `BytecodeChunk(i)` openings, all sharing the reduction's
/// final opening point. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReduction)]
pub struct BytecodeReductionAddressPhaseOutputClaims<C> {
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
}

/// Consumed intermediate opening from the stage-6b bytecode cycle phase.
#[derive(Clone, Debug, InputClaims)]
pub struct BytecodeReductionAddressPhaseInputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate, from = BytecodeClaimReductionCyclePhase)]
    pub cycle_phase_intermediate: C,
}

pub struct BytecodeReductionAddressPhase<F: Field> {
    claims: JoltRelationClaims<F>,
    layout: BytecodeClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    r_bc: Vec<F>,
    chunk_rbc_weights: Vec<F>,
    lane_weights: Vec<F>,
}

impl<F: Field> BytecodeReductionAddressPhase<F> {
    /// `weights` and `cycle_phase_variables` are the stage-6b bytecode cycle-phase
    /// outputs; they (and the layout) are all known before the stage-7 sumcheck,
    /// so a single construction serves both the input claim and the output check.
    pub fn new(
        layout: &BytecodeClaimReductionLayout,
        weights: BytecodeOutputWeightInputs<'_, F>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            claims: bytecode_reduction::address_phase(layout.dimensions(), layout.chunk_count()),
            layout: layout.clone(),
            cycle_phase_variables,
            r_bc: weights.r_bc.to_vec(),
            chunk_rbc_weights: weights.chunk_rbc_weights.to_vec(),
            lane_weights: weights.lane_weights.to_vec(),
        }
    }

    fn output_weight_inputs(&self) -> BytecodeOutputWeightInputs<'_, F> {
        BytecodeOutputWeightInputs {
            r_bc: &self.r_bc,
            chunk_rbc_weights: &self.chunk_rbc_weights,
            lane_weights: &self.lane_weights,
        }
    }
}

fn bytecode_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for BytecodeReductionAddressPhase<F> {
    type Inputs<C> = BytecodeReductionAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionAddressPhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReductionAddressPhaseInputClaims<C>,
    ) -> Result<BytecodeReductionAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(bytecode_public_failed)?;
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            chunks: vec![opening_point; self.layout.chunk_count()],
        })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &BytecodeReductionAddressPhaseInputClaims<C>,
        outputs: &BytecodeReductionAddressPhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::BytecodeClaimReduction(BytecodeClaimReductionPublic::ChunkOutputWeight(
            chunk_idx,
        )) = id
        else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        let opening_point = outputs.chunks.first().map(GetPoint::point).ok_or_else(|| {
            bytecode_public_failed("bytecode reduction produced no chunk openings")
        })?;
        let weights = self
            .layout
            .address_phase_final_output_weights_at_opening_point(
                self.output_weight_inputs(),
                opening_point,
            )
            .map_err(bytecode_public_failed)?;
        weights
            .get(*chunk_idx)
            .copied()
            .ok_or(VerifierError::MissingStageClaimPublic { id: *id })
    }
}

/// Produced `ProgramImageInit` opening at the reduction's final opening point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(ProgramImageClaimReduction)]
pub struct ProgramImageReductionAddressPhaseOutputClaims<C> {
    #[opening(committed = ProgramImageInit)]
    pub program_image: C,
}

/// Consumed intermediate opening from the stage-6b program-image cycle phase.
#[derive(Clone, Debug, InputClaims)]
pub struct ProgramImageReductionAddressPhaseInputClaims<C> {
    #[opening(committed = ProgramImageInit, from = ProgramImageClaimReductionCyclePhase)]
    pub cycle_phase: C,
}

pub struct ProgramImageReductionAddressPhase<F: Field> {
    claims: JoltRelationClaims<F>,
    layout: ProgramImageClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    reference_opening_point: Vec<F>,
}

impl<F: Field> ProgramImageReductionAddressPhase<F> {
    /// `reference_opening_point` is the RAM address point of the staged
    /// `ProgramImageInitContributionRw` opening (from stage 4). It and the
    /// cycle-phase variables are known before the stage-7 sumcheck.
    pub fn new(
        layout: &ProgramImageClaimReductionLayout,
        reference_opening_point: Vec<F>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            claims: program_image::address_phase(layout.dimensions()),
            layout: layout.clone(),
            cycle_phase_variables,
            reference_opening_point,
        }
    }
}

fn program_image_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::ProgramImageClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for ProgramImageReductionAddressPhase<F> {
    type Inputs<C> = ProgramImageReductionAddressPhaseInputClaims<C>;
    type Outputs<C> = ProgramImageReductionAddressPhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &ProgramImageReductionAddressPhaseInputClaims<C>,
    ) -> Result<ProgramImageReductionAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(program_image_public_failed)?;
        Ok(ProgramImageReductionAddressPhaseOutputClaims {
            program_image: opening_point,
        })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &ProgramImageReductionAddressPhaseInputClaims<C>,
        outputs: &ProgramImageReductionAddressPhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::ProgramImageClaimReduction(ProgramImageClaimReductionPublic::FinalScale) =
            id
        else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        self.layout
            .address_phase_scale_at_opening_point(
                &self.reference_opening_point,
                outputs.program_image.point(),
            )
            .map_err(program_image_public_failed)
    }
}
