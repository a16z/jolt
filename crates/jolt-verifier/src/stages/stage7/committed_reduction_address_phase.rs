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

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

pub struct BytecodeReductionAddressPhase<F: Field> {
    symbolic: relations::claim_reductions::bytecode::AddressPhase,
    layout: BytecodeClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    /// The stage-6b bytecode cycle-phase output weights, consumed only by the
    /// clear-only `derive_output_term` (`ChunkOutputWeight`). `None` in ZK (BlindFold
    /// recomputes the weights), where this relation's `derive_output_term` never runs.
    weights: Option<OwnedBytecodeWeights<F>>,
}

/// Owned copy of the stage-6b bytecode cycle-phase output weights.
struct OwnedBytecodeWeights<F: Field> {
    r_bc: Vec<F>,
    chunk_rbc_weights: Vec<F>,
    lane_weights: Vec<F>,
}

impl<F: Field> BytecodeReductionAddressPhase<F> {
    /// `weights` are the stage-6b bytecode cycle-phase outputs (`None` in ZK,
    /// clear-only aux); `cycle_phase_variables` and the layout are known before the
    /// stage-7 sumcheck, so a single construction serves both the input claim and
    /// the output check.
    pub fn new(
        layout: &BytecodeClaimReductionLayout,
        weights: Option<BytecodeOutputWeightInputs<'_, F>>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::bytecode::AddressPhase::new((
                layout.dimensions(),
                layout.chunk_count(),
            )),
            layout: layout.clone(),
            cycle_phase_variables,
            weights: weights.map(|weights| OwnedBytecodeWeights {
                r_bc: weights.r_bc.to_vec(),
                chunk_rbc_weights: weights.chunk_rbc_weights.to_vec(),
                lane_weights: weights.lane_weights.to_vec(),
            }),
        }
    }

    fn output_weight_inputs(&self) -> Result<BytecodeOutputWeightInputs<'_, F>, VerifierError> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            bytecode_public_failed(
                "bytecode address phase has no output weights (ZK-only construction)",
            )
        })?;
        Ok(BytecodeOutputWeightInputs {
            r_bc: &weights.r_bc,
            chunk_rbc_weights: &weights.chunk_rbc_weights,
            lane_weights: &weights.lane_weights,
        })
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

    /// The bytecode address phase is bound on the offset-0 prefix of the batch
    /// challenge vector (two-phase reductions front-load the address rounds).
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReductionAddressPhaseInputClaims<Vec<F>>,
    ) -> Result<BytecodeReductionAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(bytecode_public_failed)?;
        Ok(BytecodeReductionAddressPhaseOutputClaims {
            chunks: vec![opening_point; self.layout.chunk_count()],
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &BytecodeReductionAddressPhaseInputClaims<Vec<F>>,
        output_points: &BytecodeReductionAddressPhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::BytecodeClaimReduction(BytecodeClaimReductionPublic::ChunkOutputWeight(
            chunk_idx,
        )) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = output_points
            .chunks()
            .first()
            .map(Vec::as_slice)
            .ok_or_else(|| {
                bytecode_public_failed("bytecode reduction produced no chunk openings")
            })?;
        let weights = self
            .layout
            .address_phase_final_output_weights_at_opening_point(
                self.output_weight_inputs()?,
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
    /// The RAM address point of the staged `ProgramImageInitContributionRw` opening
    /// (from stage 4). Consumed only by the clear-only `derive_output_term`
    /// (`FinalScale`), so it is `None` in ZK — where BlindFold recomputes the scale
    /// and this relation's `derive_output_term` never runs.
    reference_opening_point: Option<Vec<F>>,
}

impl<F: Field> ProgramImageReductionAddressPhase<F> {
    /// `reference_opening_point` is the RAM address point of the staged
    /// `ProgramImageInitContributionRw` opening (from stage 4), `None` in ZK
    /// (clear-only aux). It and the cycle-phase variables are known before the
    /// stage-7 sumcheck.
    pub fn new(
        layout: &ProgramImageClaimReductionLayout,
        reference_opening_point: Option<Vec<F>>,
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

    /// The program-image address phase is bound on the offset-0 prefix of the batch
    /// challenge vector (two-phase reductions front-load the address rounds).
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ProgramImageReductionAddressPhaseInputClaims<Vec<F>>,
    ) -> Result<ProgramImageReductionAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(program_image_public_failed)?;
        Ok(ProgramImageReductionAddressPhaseOutputClaims {
            program_image: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &ProgramImageReductionAddressPhaseInputClaims<Vec<F>>,
        output_points: &ProgramImageReductionAddressPhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::ProgramImageClaimReduction(ProgramImageClaimReductionPublic::FinalScale) =
            id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let reference_opening_point = self.reference_opening_point.as_ref().ok_or_else(|| {
            program_image_public_failed(
                "program-image address phase has no reference opening point (ZK-only construction)",
            )
        })?;
        self.layout
            .address_phase_scale_at_opening_point(
                reference_opening_point,
                output_points.program_image(),
            )
            .map_err(program_image_public_failed)
    }
}
