//! The RAM RAF-evaluation slot: the stage-2 RAF sumcheck over the address
//! domain.

use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-2 RAM RAF-evaluation slot.
pub trait RamRafEvaluationProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        raf_dimensions: RamRafEvaluationDimensions,
        ram_log_k: usize,
        lowest_address: u64,
        tau_low: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRafEvaluation<F>>>, KernelError<F>>;
}
