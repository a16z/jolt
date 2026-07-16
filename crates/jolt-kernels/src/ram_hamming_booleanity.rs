//! The RAM Hamming-weight booleanity (stage 6b) slot.

use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-6b RAM Hamming-weight booleanity slot.
pub trait RamHammingBooleanityProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        stage1_cycle_binding: &[F],
        challenges: &NoChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamHammingBooleanity<F>>>, KernelError<F>>;
}
