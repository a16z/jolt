//! The RAM value-check slot: stage 4.

use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
use jolt_claims::protocols::jolt::relations::ram::RamValCheckChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-4 RAM value-check slot.
pub trait RamValCheckProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        init: RamValCheckInit<F>,
        r_address: &[F],
        r_cycle: &[F],
        challenges: &RamValCheckChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamValCheck<F>>>, KernelError<F>>;
}
