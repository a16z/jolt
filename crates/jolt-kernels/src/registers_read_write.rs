//! The registers read/write-checking (stage 4) slot.

use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteChallenges;
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-4 registers read/write-checking slot.
pub trait RegistersReadWriteProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        r_cycle: &[F],
        challenges: &RegistersReadWriteChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>;
}
