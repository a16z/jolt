//! The uni-skip front slot: the pre-phase round of a uni-skip stage (stage 1
//! Spartan outer, stage 2 product virtualization), generic over the remainder
//! relation the front feeds.
//!
//! The uni-skip first round is a pre-phase, not a batch member: the stage
//! front asks the slot to [`prepare`](UniskipKernel::prepare), sends the
//! [`first_round_poly`](UniskipKernel::first_round_poly), and moves on. No
//! instance object crosses the seam — the state the uni-skip round produces
//! lives in the [`ProofSession`] under a key private to the implementing
//! backend, and the same backend's `PrepareKernel<F, R>` remainder slot
//! reclaims it once the uni-skip challenge is drawn.

use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession};

/// The uni-skip front slot of a stage. `R` is never named by the methods —
/// it types the slot to the remainder relation the parked state becomes, so
/// the two uni-skip fronts are distinct [`JoltBackend`](crate::JoltBackend)
/// fields.
pub trait UniskipKernel<F, R>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    /// Compute the uni-skip first-round state and park it in the session
    /// under a backend-private key.
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau: &[F],
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<(), KernelError<F>>;

    /// The uni-skip first-round polynomial. `late_tau` carries the challenges
    /// drawn after [`prepare`](Self::prepare) (outer: empty; product:
    /// `&[tau_high]`).
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[F],
    ) -> Result<UnivariatePoly<F>, KernelError<F>>;
}
