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
use jolt_field::JoltField;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage6b::outputs::BytecodeReductionWeights;
use crate::VerifierError;

#[derive(Clone)]
pub struct BytecodeReductionAddressPhase<F: JoltField> {
    symbolic: relations::claim_reductions::bytecode::AddressPhase,
    layout: BytecodeClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    weights: Option<BytecodeReductionWeights<F>>,
}

impl<F: JoltField> BytecodeReductionAddressPhase<F> {
    pub fn new(
        layout: &BytecodeClaimReductionLayout,
        weights: Option<BytecodeReductionWeights<F>>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::bytecode::AddressPhase::new((
                layout.dimensions(),
                layout.chunk_count(),
            )),
            layout: layout.clone(),
            cycle_phase_variables,
            weights,
        }
    }

    fn output_weight_inputs(&self) -> Result<BytecodeOutputWeightInputs<'_, F>, VerifierError> {
        Ok(self
            .weights
            .as_ref()
            .ok_or_else(|| {
                bytecode_public_failed(
                    "bytecode address phase has no output weights (ZK-only construction)",
                )
            })?
            .as_inputs())
    }
}

fn bytecode_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReductionAddressPhase<F> {
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
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
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
            .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() })
    }
}

#[derive(Clone)]
pub struct ProgramImageReductionAddressPhase<F: JoltField> {
    symbolic: relations::claim_reductions::program_image::AddressPhase,
    layout: ProgramImageClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    reference_opening_point: Option<Vec<F>>,
}

impl<F: JoltField> ProgramImageReductionAddressPhase<F> {
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

impl<F: JoltField> ConcreteSumcheck<F> for ProgramImageReductionAddressPhase<F> {
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
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
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
