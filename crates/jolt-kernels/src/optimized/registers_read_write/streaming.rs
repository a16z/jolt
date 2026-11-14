//! The register read/write check (stage 4) of the streaming register family:
//! the default kernel, which parks no `rd` indices because the family's
//! stage-5 value evaluation
//! ([`StreamingRegistersValEvaluation`](crate::optimized::registers_val_evaluation::StreamingRegistersValEvaluation))
//! reads the witness's rows.

use jolt_field::JoltField;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use super::OptimizedRegistersReadWrite;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

pub struct StreamingRegistersReadWrite;

impl<F: JoltField> PrepareKernel<F, RegistersReadWriteChecking<F>> for StreamingRegistersReadWrite {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        OptimizedRegistersReadWrite::prepare_kernel(None, witness, inputs)
    }
}
