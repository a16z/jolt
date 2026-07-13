//! The instruction read+RAF checking (stage 5) slot.

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::protocols::jolt_vm::Stage5InstructionReadRafRow;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-5 instruction read+RAF slot. The typed relation data is the
/// per-cycle lookup rows (index bits, table selection, operand interleaving)
/// and the upstream claim-reduction cycle point `r_reduction`; no witness
/// oracle is consumed — the rows are the complete witness for this relation.
pub trait InstructionReadRafProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Vec<Stage5InstructionReadRafRow>,
        challenges: &InstructionReadRafChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionReadRaf<F>>>, KernelError<F>>;
}
