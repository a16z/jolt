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
//! point `derive_opening_points` produces — so `derive_output_term` recovers that
//! point from the output claims and asks the layout for the scale/weights at it,
//! exactly as stage 4's `RamValCheck` recovers the cycle from its output point.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
    BytecodeReductionAddressPhaseInputClaims, BytecodeReductionAddressPhaseOutputClaims,
};
pub use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
    ProgramImageReductionAddressPhaseInputClaims, ProgramImageReductionAddressPhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::claim_reductions::bytecode::BytecodeOutputWeightInputs, BytecodeClaimReductionLayout,
    BytecodeClaimReductionPublic, JoltDerivedId, JoltRelationId, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout, ProgramImageClaimReductionPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::VerifierError;

pub struct BytecodeReductionAddressPhase<F: Field> {
    symbolic: relations::claim_reductions::bytecode::AddressPhase,
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
            symbolic: relations::claim_reductions::bytecode::AddressPhase::new((
                layout.dimensions(),
                layout.chunk_count(),
            )),
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

impl<F: Field> ConcreteSumcheck<F> for BytecodeReductionAddressPhase<F> {
    type Symbolic = relations::claim_reductions::bytecode::AddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
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

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &BytecodeReductionAddressPhaseInputClaims<C>,
        outputs: &BytecodeReductionAddressPhaseOutputClaims<OpeningClaim<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::BytecodeClaimReduction(BytecodeClaimReductionPublic::ChunkOutputWeight(
            chunk_idx,
        )) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
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
            .ok_or(VerifierError::MissingStageClaimDerived { id: *id })
    }
}

pub struct ProgramImageReductionAddressPhase<F: Field> {
    symbolic: relations::claim_reductions::program_image::AddressPhase,
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
            symbolic: relations::claim_reductions::program_image::AddressPhase::new(
                layout.dimensions(),
            ),
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

impl<F: Field> ConcreteSumcheck<F> for ProgramImageReductionAddressPhase<F> {
    type Symbolic = relations::claim_reductions::program_image::AddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
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

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &ProgramImageReductionAddressPhaseInputClaims<C>,
        outputs: &ProgramImageReductionAddressPhaseOutputClaims<OpeningClaim<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::ProgramImageClaimReduction(ProgramImageClaimReductionPublic::FinalScale) =
            id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        self.layout
            .address_phase_scale_at_opening_point(
                &self.reference_opening_point,
                outputs.program_image.point(),
            )
            .map_err(program_image_public_failed)
    }
}
