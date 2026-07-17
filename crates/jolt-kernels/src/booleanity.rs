//! The booleanity address-phase (stage 6a) slot. Bespoke: the reference
//! points and gamma are stage-level pre-batch draws carried in
//! `Stage6aCarriedChallenges` — neither the relation instance nor its
//! (empty) challenge struct carries them, so the slot cannot be served by
//! the universal `PrepareKernel`. (The stage-6b cycle phase CAN: its
//! relation carries every input, so it lives behind
//! `JoltBackend::booleanity_cycle`.)

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-6a booleanity address-phase slot. `reference_address` and
/// `reference_cycle` are the little-endian reference points carried in
/// `Stage6aCarriedChallenges`; `gamma` is the booleanity batching challenge.
pub trait BooleanityAddressProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BooleanityDimensions,
        reference_address: &[F],
        reference_cycle: &[F],
        gamma: F,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>;
}
