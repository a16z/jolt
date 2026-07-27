//! The Spartan product-virtualization (stage 2) slot: the product uni-skip
//! first-round polynomial and the product-remainder batch member.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::JoltWitnessOracle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-2 product slot: factory for a prepared product-virtualization
/// instance.
pub trait SpartanProductProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[F],
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn SpartanProductInstance<F>>, KernelError<F>>;
}

/// A prepared product-virtualization instance: the uni-skip first-round
/// polynomial (once `τ_high` is drawn), then the remainder batch member (once
/// the uni-skip challenge binds).
pub trait SpartanProductInstance<F: Field> {
    fn uniskip_first_round_poly(&self, tau_high: F) -> Result<UnivariatePoly<F>, KernelError<F>>;

    fn into_remainder(
        self: Box<Self>,
        tau_high: F,
        uniskip_challenge: F,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = ProductRemainder<F>>>, KernelError<F>>;
}
