//! The instruction read+RAF checking (stage 5) slot.

use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
use jolt_witness::WitnessBundle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The per-cycle witness of the instruction read+RAF relation: the lookup
/// index bits, the table selection, and the RAF flag (the negation of the
/// operand-interleaving marker).
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct InstructionReadRafWitness {
    pub lookup_index: LookupIndex,
    pub table_index: TableIndex,
    #[opening(InstructionRafFlag)]
    pub raf_flag: InstructionRafFlag,
}

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
        rows: Vec<InstructionReadRafWitness>,
        challenges: &InstructionReadRafChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionReadRaf<F>>>, KernelError<F>>;
}
