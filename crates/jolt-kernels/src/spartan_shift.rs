//! The Spartan shift slot: the stage-3 shift sumcheck.

use jolt_claims::protocols::jolt::relations::spartan::SpartanShiftChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::SpartanShift;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-3 Spartan-shift slot.
pub trait SpartanShiftProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_uniskip_tau_low: &[F],
        product_remainder_point: &[F],
        challenges: &SpartanShiftChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = SpartanShift<F>>>, KernelError<F>>;
}
