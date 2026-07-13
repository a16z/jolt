//! The bytecode read+RAF slots: the stage-6a address phase and the
//! stage-6b cycle phase.

use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafCyclePhaseCommittedChallenges,
};
use jolt_field::Field;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-6a bytecode read+RAF address-phase slot. The typed relation data
/// is the per-row stage-value table (the verifier's `read_raf_stage_values`
/// output over the padded bytecode), the five upstream stage cycle points,
/// the per-cycle bytecode indices (the PC pushforward source), and the
/// preprocessing entry index.
pub trait BytecodeReadRafAddressProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BytecodeReadRafDimensions,
        committed_program: bool,
        stage_values: Vec<[F; 5]>,
        stage_cycle_points: &[Vec<F>; 5],
        bytecode_indices: Vec<usize>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>;
}

/// The stage-6b bytecode read+RAF cycle-phase slot: a naive member driven
/// through the dispatch enum's *committed* anchor `Expr` — sound in full mode
/// because the committed `Expr` with constant `BytecodeValStage` tables (the
/// address fold values) and cycle-eq `StageCycleEq` publics computes the same
/// summand the full-mode `Expr` describes. The `BytecodeRa` opening tables are
/// address folds of the committed one-hot grids at the 6a address point's
/// committed-width chunks.
pub trait BytecodeReadRafCycleProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BytecodeReadRafDimensions,
        r_address: &[F],
        stage_cycle_points: &[Vec<F>; 5],
        entry_bytecode_index: usize,
        committed_chunk_bits: usize,
        stage_values_at_r_address: [F; 5],
        challenges: &BytecodeReadRafCyclePhaseCommittedChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>;
}
