//! Reduce staged bytecode-value claims to one whole-bytecode opening. The
//! summand is the committed row/lane grid times the lane-weight/address-equality
//! grid, bound over the canonical two-phase precommitted schedule.

use jolt_field::JoltField;
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::BytecodeReductionCyclePhase;
use jolt_witness::JoltWitnessPlane;

use crate::precommitted_reduction::bytecode_reduction_kernel;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, ReferenceBackend, SumcheckKernel,
};

impl<F: JoltField> PrepareKernel<F, BytecodeReductionCyclePhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReductionCyclePhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReductionCyclePhase<F>>>, KernelError<F>>
    {
        let layout = inputs.relation.layout();
        let program = witness.program_preprocessing();
        Ok(Box::new(bytecode_reduction_kernel(
            layout,
            inputs.relation.weights(),
            &program.bytecode.bytecode,
        )?))
    }
}
