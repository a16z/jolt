//! The bytecode read+RAF address-phase (stage 6a) slot. Bespoke: its
//! witness channels are non-oracle — the per-row stage-value table folds the
//! prover-retained program bytecode, and the PC pushforward source is the
//! per-cycle bytecode indices from the typed stage-6 rows — so the slot
//! cannot be served by the universal `PrepareKernel`. (The stage-6b cycle
//! phase CAN: its relation carries every input, including the address-folded
//! stage values, so it lives behind `JoltBackend::bytecode_read_raf_cycle`.)

use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;

use crate::{KernelError, ProofSession, SumcheckKernel};

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
        relation: &BytecodeReadRafAddressPhase<F>,
        dimensions: BytecodeReadRafDimensions,
        stage_values: Vec<[F; 5]>,
        stage_cycle_points: &[Vec<F>; 5],
        bytecode_indices: Vec<usize>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>;
}
