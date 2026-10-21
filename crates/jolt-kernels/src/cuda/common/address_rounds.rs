use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};

use crate::{KernelError, SumcheckKernel, SumcheckKernelError};

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, R: ConcreteSumcheck<F>")
)]
pub(crate) struct AddressRounds<F: JoltField, R: ConcreteSumcheck<F>> {
    inner: Box<dyn SumcheckKernel<F, Relation = R>>,
    active: Vec<usize>,
    rounds: usize,
    bound: usize,
    pending: Option<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    scale: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    inverse_scale: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    half: F,
}

impl<F: JoltField, R: ConcreteSumcheck<F> + 'static> AddressRounds<F, R> {
    pub(crate) fn wrap(
        inner: Box<dyn SumcheckKernel<F, Relation = R>>,
        dimensions: ReadWriteDimensions,
        rounds: usize,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>> {
        let mut active: Vec<_> = dimensions
            .address_opening_indices()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "the CUDA RAM address schedule is invalid",
            })?
            .collect();
        active.sort_unstable();
        if active.len() + dimensions.phase3_cycle_rounds() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "the CUDA RAM address schedule disagrees with the relation",
            });
        }
        if active.len() == rounds {
            return Ok(inner);
        }
        let scale = F::pow2(rounds - active.len());
        let half = F::from_u64(2)
            .inverse()
            .ok_or(KernelError::InvariantViolation {
                reason: "the CUDA RAM inactive-round scale is not invertible",
            })?;
        Ok(Box::new(Self {
            inner,
            active,
            rounds,
            bound: 0,
            pending: None,
            scale,
            inverse_scale: scale.inverse().ok_or(KernelError::InvariantViolation {
                reason: "the CUDA RAM inactive-round scale is not invertible",
            })?,
            half,
        }))
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        match self.active.binary_search(&self.bound) {
            Ok(index) if index + 1 == self.active.len() => {
                self.inner.finish_rounds(challenge)?;
            }
            Ok(_) => self.pending = Some(challenge),
            Err(_) => {
                self.scale *= self.half;
                self.inverse_scale += self.inverse_scale;
            }
        }
        self.bound += 1;
        Ok(())
    }
}

impl<F: JoltField, R: ConcreteSumcheck<F> + 'static> ProveRounds<F> for AddressRounds<F, R> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        match self.active.binary_search(&round) {
            Ok(index) => Ok(self.inner.prove_round(
                self.pending.take(),
                index,
                claim * self.inverse_scale,
            )? * self.scale),
            Err(_) => Ok(UnivariatePoly::new(vec![claim * self.half])),
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField, R: ConcreteSumcheck<F> + 'static> SumcheckKernel<F> for AddressRounds<F, R> {
    type Relation = R;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, R>,
    ) -> Result<SumcheckOutputClaims<F, R>, SumcheckKernelError<F>> {
        if self.bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds.saturating_sub(self.bound),
            });
        }
        self.inner.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &R,
        inputs: &SumcheckInputPoints<F, R>,
        outputs: &SumcheckOutputPoints<F, R>,
        challenges: &ConcreteSumcheckChallenges<F, R>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.inner
            .validate_derived_tables(relation, inputs, outputs, challenges)
    }
}
