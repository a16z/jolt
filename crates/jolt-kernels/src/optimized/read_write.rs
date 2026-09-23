//! Supported sparse read/write orders and inactive RAM address rounds.

use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};

use super::support::RoundProgress;
use crate::{KernelError, SumcheckKernel, SumcheckKernelError};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ReadWriteOrder {
    CycleFirst,
    AddressFirst,
}

impl ReadWriteOrder {
    pub(super) fn new<F: JoltField>(
        dimensions: ReadWriteDimensions,
    ) -> Result<Self, KernelError<F>> {
        dimensions
            .validate_phase_split()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "read/write phase split exceeds the cycle/address dimensions",
            })?;
        if dimensions.phase1_num_rounds() == dimensions.log_t()
            || dimensions.phase2_num_rounds() == 0
        {
            Ok(Self::CycleFirst)
        } else if dimensions.phase1_num_rounds() == 0
            && dimensions.phase2_num_rounds() == dimensions.log_k()
        {
            Ok(Self::AddressFirst)
        } else {
            Err(KernelError::Unsupported {
                reason:
                    "optimized read/write kernels support cycle-first or full address-first binding",
            })
        }
    }
}

/// Keeps address tables compact while the relation sums over unused cycles.
/// The canonical opening indices locate active rounds; each inactive bind
/// removes one factor of two without touching a table or an opening point.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, R: ConcreteSumcheck<F>")
)]
pub(super) struct RamAddressKernel<F: JoltField, R: ConcreteSumcheck<F>> {
    inner: Box<dyn SumcheckKernel<F, Relation = R>>,
    active_rounds: Vec<usize>,
    progress: RoundProgress,
    pending: Option<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    scale: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    unscale: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    half: F,
}

impl<F: JoltField, R: ConcreteSumcheck<F>> RamAddressKernel<F, R> {
    pub(super) fn wrap(
        inner: Box<dyn SumcheckKernel<F, Relation = R>>,
        dimensions: ReadWriteDimensions,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>>
    where
        R: 'static,
    {
        dimensions
            .validate_phase_split()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "invalid RAM address round geometry",
            })?;
        if inner.num_rounds() != dimensions.log_k() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address kernel has the wrong number of active rounds",
            });
        }
        if dimensions.phase3_cycle_rounds() == 0 {
            return Ok(inner);
        }
        let mut active_rounds: Vec<_> = dimensions
            .address_opening_indices()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "invalid RAM address round geometry",
            })?
            .collect();
        active_rounds.sort_unstable();
        let scale = F::pow2(dimensions.phase3_cycle_rounds());
        Ok(Box::new(Self {
            inner,
            active_rounds,
            progress: RoundProgress::new(dimensions.output_check_rounds()),
            pending: None,
            half: F::from_u64(2)
                .inverse()
                .ok_or(KernelError::InvariantViolation {
                    reason: "RAM inactive-round factor two is not invertible",
                })?,
            scale,
            unscale: scale.inverse().ok_or(KernelError::InvariantViolation {
                reason: "RAM inactive-round scale is not invertible",
            })?,
        }))
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        match self.active_rounds.binary_search(&self.progress.bound()) {
            Ok(index) if index + 1 == self.active_rounds.len() => {
                self.inner.finish_rounds(challenge)?;
            }
            Ok(_) => self.pending = Some(challenge),
            Err(_) => {
                self.scale *= self.half;
                self.unscale += self.unscale;
            }
        }
        self.progress.advance();
        Ok(())
    }
}

impl<F: JoltField, R: ConcreteSumcheck<F>> ProveRounds<F> for RamAddressKernel<F, R> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
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
        match self.active_rounds.binary_search(&round) {
            Ok(index) => {
                Ok(self
                    .inner
                    .prove_round(self.pending.take(), index, claim * self.unscale)?
                    * self.scale)
            }
            Err(_) => Ok(UnivariatePoly::new(vec![claim * self.half])),
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField, R: ConcreteSumcheck<F>> SumcheckKernel<F> for RamAddressKernel<F, R> {
    type Relation = R;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, R>,
    ) -> Result<SumcheckOutputClaims<F, R>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        self.inner.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &R,
        inputs: &SumcheckInputPoints<F, R>,
        outputs: &SumcheckOutputPoints<F, R>,
        challenges: &ConcreteSumcheckChallenges<F, R>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        self.inner
            .validate_derived_tables(relation, inputs, outputs, challenges)
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::Fr;

    use super::*;

    #[test]
    fn distinguishes_invalid_splits_from_unsupported_mixed_orders() {
        for dimensions in [
            ReadWriteDimensions::new(3, 2, 1, 1),
            ReadWriteDimensions::new(3, 2, 0, 1),
        ] {
            assert!(matches!(
                ReadWriteOrder::new::<Fr>(dimensions),
                Err(KernelError::Unsupported { .. })
            ));
        }
        for dimensions in [
            ReadWriteDimensions::new(3, 2, 4, 2),
            ReadWriteDimensions::new(3, 2, 0, 3),
        ] {
            assert!(matches!(
                ReadWriteOrder::new::<Fr>(dimensions),
                Err(KernelError::InvariantViolation { .. })
            ));
        }
    }
}
