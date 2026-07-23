//! The reference stage-7 precommitted slot server: reclaims (by move) the
//! plain carry its stage-6b cycle kernel parked via `park_residue` and mounts
//! a fresh address-phase kernel over it. The kernels, the carry, and the
//! round compute live in
//! [`precommitted_reduction`](crate::precommitted_reduction); the cycle-phase
//! table builders in the per-kind `reference::*_claim_reduction` modules.

use std::marker::PhantomData;

use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::precommitted_reduction::{AddressReductionKernel, PrecommittedReductionCarry};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// The stage-7 slot server for any precommitted address-phase relation `R`:
/// reclaims the carry stage 6b's cycle kernel parked under `R`'s key and
/// mounts the final-opening batch member. One instance per kind, wired in
/// [`reference()`](crate::JoltBackend::reference) with the kind's
/// missing-carry diagnostic.
pub struct ReferencePrecommittedAddress<R> {
    missing_carry: &'static str,
    _relation: PhantomData<fn() -> R>,
}

impl<R> ReferencePrecommittedAddress<R> {
    pub fn new(missing_carry: &'static str) -> Self {
        Self {
            missing_carry,
            _relation: PhantomData,
        }
    }
}

impl<F, R> PrepareKernel<F, R> for ReferencePrecommittedAddress<R>
where
    F: Field,
    R: ConcreteSumcheck<F> + 'static,
    AddressReductionKernel<F, R>: SumcheckKernel<F, Relation = R>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltVmWitnessPlane<F>,
        _inputs: ProverInputs<'_, F, R>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>> {
        let carry = session.take::<PrecommittedReductionCarry<F, R>>().ok_or(
            KernelError::InvariantViolation {
                reason: self.missing_carry,
            },
        )?;
        Ok(Box::new(AddressReductionKernel::new(carry)))
    }
}
