//! The Spartan-outer (stage 1) uni-skip front slot and its session carry.
//!
//! The uni-skip first round is a pre-phase, not a batch member: the stage
//! front mints the instance, sends its first-round polynomial, then parks the
//! instance in the [`ProofSession`] as [`ParkedOuterInstance`]. The remainder
//! IS a batch member, served by the universal `PrepareKernel<F,
//! OuterRemainder<F>>` on [`SessionCarriedKernels`], which reclaims the carry
//! and hands it the stage's relation (the source of `tau` and the uni-skip
//! challenge).

use crate::ProverInputs;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmWitnessPlane};
use jolt_witness::WitnessProvider;

use crate::{KernelError, PrepareKernel, ProofSession, SessionCarriedKernels, SumcheckKernel};

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
/// `inputs.relation` is the stage's batch instance (the source of the drawn
/// `tau` and the uni-skip challenge; the kernel owns no copy).
pub trait SpartanOuterInstance<F: Field> {
    fn uniskip_first_round_poly(&self) -> Result<UnivariatePoly<F>, KernelError<F>>;

    fn into_remainder(
        self: Box<Self>,
        inputs: &ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>>;
}

/// The stage-1 front's session carry: the Spartan-outer instance, parked
/// after the uni-skip round for the remainder member's `prepare` to reclaim.
pub struct ParkedOuterInstance<F: Field>(pub Box<dyn SpartanOuterInstance<F>>);

impl<F: Field> PrepareKernel<F, OuterRemainder<F>> for SessionCarriedKernels {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        inputs: ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        let ParkedOuterInstance(instance) =
            session
                .take::<ParkedOuterInstance<F>>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "the stage-1 front parked no outer instance for the remainder member",
                })?;
        instance.into_remainder(&inputs)
    }
}
