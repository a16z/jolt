//! The Spartan-outer (stage 1) slot: the uni-skip first-round polynomial
//! and the outer-remainder sumcheck member.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-1 slot: factory for a prepared Spartan-outer instance.
pub trait SpartanOuterProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SpartanOuterInstance<F>>, KernelError<F>>;
}

/// A prepared Spartan-outer instance: the uni-skip first-round polynomial,
/// then — once the uni-skip challenge is drawn — the remainder batch member.
pub trait SpartanOuterInstance<F: Field> {
    fn uniskip_first_round_poly(&self) -> Result<UnivariatePoly<F>, KernelError<F>>;

    fn into_remainder(
        self: Box<Self>,
        uniskip_challenge: F,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = OuterRemainder<F>>>, KernelError<F>>;
}
