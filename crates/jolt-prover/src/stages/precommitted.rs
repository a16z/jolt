//! The typed adapter that lets a precommitted claim-reduction kernel join a
//! generated `prove_clear` batch as a `#[sumcheck(external)]` member.
//!
//! One [`PrecommittedReductionProver`] object spans two batches (the stage-6b
//! cycle phase and the stage-7 address phase) under two different relations,
//! so it cannot itself be a typed [`SumcheckKernel`]. Per batch, the recipe
//! wraps a borrow of it in this adapter together with the phase's wire-claim
//! curation (intermediate handoff claim vs. final opening, decided by the
//! schedule's `has_address_phase`), which is exactly the per-member claim
//! assembly the recipes previously hand-rolled after the round loop.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::JoltChallengeId;
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Field;
use jolt_kernels::precommitted_reduction::PrecommittedReductionProver;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckKernel,
    SumcheckKernelError, SumcheckOutputClaims,
};

pub(crate) struct PrecommittedKernelAdapter<'a, F, R, E>
where
    F: Field,
{
    member: &'a mut dyn PrecommittedReductionProver<F>,
    extract: E,
    _relation: PhantomData<fn() -> R>,
}

impl<'a, F, R, E> PrecommittedKernelAdapter<'a, F, R, E>
where
    F: Field,
{
    pub(crate) fn new(member: &'a mut dyn PrecommittedReductionProver<F>, extract: E) -> Self {
        Self {
            member,
            extract,
            _relation: PhantomData,
        }
    }
}

impl<F, R, E> ProveRounds<F> for PrecommittedKernelAdapter<'_, F, R, E>
where
    F: Field,
{
    fn num_rounds(&self) -> usize {
        self.member.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.member.prove_round(bind, round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.member.finish_rounds(bind)
    }
}

impl<F, R, E> SumcheckKernel<F> for PrecommittedKernelAdapter<'_, F, R, E>
where
    F: Field,
    R: ConcreteSumcheck<F>,
    SumcheckInputClaims<F, R>: InputClaims<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F>,
    ConcreteSumcheckChallenges<F, R>: SumcheckChallenges<F, JoltChallengeId>,
    E: FnMut(
        &dyn PrecommittedReductionProver<F>,
    ) -> Result<SumcheckOutputClaims<F, R>, SumcheckKernelError<F>>,
{
    type Relation = R;

    fn output_claims(&mut self) -> Result<SumcheckOutputClaims<F, R>, SumcheckKernelError<F>> {
        (self.extract)(&*self.member)
    }
}
