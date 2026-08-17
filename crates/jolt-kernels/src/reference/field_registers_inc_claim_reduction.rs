//! The stage-6b `FieldRegistersIncClaimReduction` slot: fail-closed until the
//! FR witness wiring lands (milestone 11). The verifier-side relation and
//! batch composition exist so the FR-on prover COMPILES against the full
//! stage-6b shape, but no backend can honestly serve the reduced `FieldRdInc`
//! opening yet — the slot rejects at `prepare` rather than fabricating claims.

use jolt_field::Field;
use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
use jolt_witness::JoltWitnessPlane;

use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::KernelError;

impl<F: Field> PrepareKernel<F, FieldRegistersIncClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, FieldRegistersIncClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersIncClaimReduction<F>>>,
        KernelError<F>,
    > {
        Err(KernelError::Unsupported {
            reason: "field-inline proving pending witness wiring (milestone 11)",
        })
    }
}
