//! The RAM read/write-checking (stage 2) slot.

use jolt_claims::protocols::jolt::relations::ram::RamReadWriteChallenges;
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-2 RAM read/write-checking slot.
pub trait RamReadWriteProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        tau_low: &[F],
        challenges: &RamReadWriteChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>;
}
