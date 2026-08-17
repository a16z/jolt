//! The stage-4 `FieldRegistersReadWriteChecking` slot: fail-closed until the
//! FR witness wiring lands (milestone 11). The verifier-side relation and
//! batch composition exist so the FR-on prover COMPILES against the full
//! stage-4 shape, but no backend can honestly serve the FR openings yet — the
//! slot rejects at `prepare` rather than fabricating claims.

use jolt_field::Field;
use jolt_verifier::stages::stage4::field_registers_read_write_checking::FieldRegistersReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::KernelError;

impl<F: Field> PrepareKernel<F, FieldRegistersReadWriteChecking<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, FieldRegistersReadWriteChecking<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersReadWriteChecking<F>>>,
        KernelError<F>,
    > {
        Err(KernelError::Unsupported {
            reason: "field-inline proving pending witness wiring (milestone 11)",
        })
    }
}
