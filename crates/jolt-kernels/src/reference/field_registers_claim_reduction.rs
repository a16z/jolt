//! The stage-2 `FieldRegistersClaimReduction` slot: fail-closed until the FR
//! witness wiring lands (milestone 11). The verifier-side relation and batch
//! composition exist so the FR-on prover COMPILES against the full stage-2
//! shape, but no backend can honestly serve the FR openings yet — the slot
//! rejects at `prepare` rather than fabricating claims.

use jolt_field::Field;
use jolt_verifier::stages::stage2::field_registers_claim_reduction::FieldRegistersClaimReduction;
use jolt_witness::JoltWitnessPlane;

use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::KernelError;

impl<F: Field> PrepareKernel<F, FieldRegistersClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, FieldRegistersClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersClaimReduction<F>>>,
        KernelError<F>,
    > {
        Err(KernelError::Unsupported {
            reason: "field-inline proving pending witness wiring (milestone 11)",
        })
    }
}
