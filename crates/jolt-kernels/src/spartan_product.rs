//! The Spartan product-virtualization (stage 2) uni-skip front slot and its
//! session carry.
//!
//! Same shape as [`spartan_outer`](crate::spartan_outer): the front mints the
//! instance, sends the uni-skip first-round polynomial (once `τ_high` is
//! drawn), then parks the instance as [`ParkedProductInstance`]; the
//! remainder batch member is served by `PrepareKernel<F, ProductRemainder<F>>`
//! on [`SessionCarriedKernels`].

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmWitnessPlane};
use jolt_witness::WitnessProvider;

use crate::{KernelError, PrepareKernel, ProofSession, SessionCarriedKernels, SumcheckKernel};

/// The stage-2 product slot: factory for a prepared product-virtualization
/// instance.
pub trait SpartanProductProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SpartanProductInstance<F>>, KernelError<F>>;
}

/// A prepared product-virtualization instance: the uni-skip first-round
/// polynomial (once `τ_high` is drawn), then the remainder batch member (once
/// the uni-skip challenge binds). `relation` is the stage's batch instance
/// (the source of `τ_high` and the uni-skip challenge; the kernel owns no
/// copy).
pub trait SpartanProductInstance<F: Field> {
    fn uniskip_first_round_poly(&self, tau_high: F) -> Result<UnivariatePoly<F>, KernelError<F>>;

    fn into_remainder(
        self: Box<Self>,
        relation: &ProductRemainder<F>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>>;
}

/// The stage-2 front's session carry: the product-virtualization instance,
/// parked after the uni-skip round for the remainder member's `prepare` to
/// reclaim.
pub struct ParkedProductInstance<F: Field>(pub Box<dyn SpartanProductInstance<F>>);

impl<F: Field> PrepareKernel<F, ProductRemainder<F>> for SessionCarriedKernels {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        let ParkedProductInstance(instance) =
            session
                .take::<ParkedProductInstance<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "the stage-2 front parked no product instance for the remainder member",
                })?;
        instance.into_remainder(inputs.relation)
    }
}
