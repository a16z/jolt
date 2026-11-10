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

/// Fixed batch coefficients, shared by every sparse-matrix phase.
#[derive(Clone, Copy)]
pub(super) struct ReadWriteBatch<F> {
    value: F,
    increment: [F; 2],
    code: F,
    count: F,
}

#[derive(Clone, Copy)]
pub(super) struct ReadWriteCoefficients<F> {
    value: F,
    increment: F,
    count: F,
}

impl<F: JoltField> ReadWriteBatch<F> {
    pub(super) fn new(gamma: F) -> Self {
        let code = gamma * gamma;
        Self {
            value: F::one() + gamma,
            increment: [gamma, gamma + code],
            code,
            count: code * gamma,
        }
    }

    pub(super) fn unbound(&self, protected: bool) -> ReadWriteCoefficients<F> {
        ReadWriteCoefficients {
            value: self.value,
            increment: self.increment[usize::from(protected)],
            count: self.count,
        }
    }

    pub(super) fn bound(&self, mask: F) -> ReadWriteCoefficients<F> {
        ReadWriteCoefficients {
            value: self.value,
            increment: self.increment[0] + self.code * mask,
            count: self.count,
        }
    }
}

impl<F: JoltField> ReadWriteCoefficients<F> {
    #[inline]
    pub(super) fn value(&self, val: F, inc: F) -> F {
        self.slope(val, inc) + self.count
    }

    #[inline]
    pub(super) fn slope(&self, val: F, inc: F) -> F {
        self.value * val + self.increment * inc
    }
}

/// A low-bit-bound interval has full interior blocks and at most two boundary
/// blocks. This stores O(s) endpoints/weights rather than a K-element table.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct IntervalMask<F: JoltField> {
    intervals: Vec<MaskInterval<F>>,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MaskInterval<F: JoltField> {
    start: usize,
    end: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    first: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    last: F,
}

impl<F: JoltField> MaskInterval<F> {
    fn value(&self, index: usize) -> F {
        if index < self.start || index >= self.end {
            F::zero()
        } else if index == self.start {
            self.first
        } else if index + 1 == self.end {
            self.last
        } else {
            F::one()
        }
    }
}

impl<F: JoltField> IntervalMask<F> {
    pub(super) fn new(intervals: &[(usize, usize)]) -> Self {
        Self {
            intervals: intervals
                .iter()
                .map(|&(start, end)| MaskInterval {
                    start,
                    end,
                    first: F::one(),
                    last: F::one(),
                })
                .collect(),
        }
    }

    /// Only used before any address bind. The sorted original intervals admit
    /// one binary search per surviving entry, with no additional trace storage.
    pub(super) fn contains_unbound(&self, index: usize) -> bool {
        let next = self
            .intervals
            .partition_point(|interval| interval.start <= index);
        next.checked_sub(1)
            .is_some_and(|previous| index < self.intervals[previous].end)
    }

    pub(super) fn value(&self, index: usize) -> F {
        self.intervals
            .iter()
            .map(|interval| interval.value(index))
            .sum()
    }

    pub(super) fn finish_address(&mut self) {
        let value = self.value(0);
        if let Some(interval) = self.intervals.first_mut() {
            interval.start = 0;
            interval.end = 1;
            interval.first = value;
            interval.last = value;
            self.intervals.truncate(1);
        }
    }

    pub(super) fn bind(&mut self, r: F) {
        for interval in &mut self.intervals {
            let start = interval.start / 2;
            let end = interval.end.div_ceil(2);
            let even = interval.value(2 * start);
            let first = even + r * (interval.value(2 * start + 1) - even);
            let even = interval.value(2 * (end - 1));
            let last = even + r * (interval.value(2 * (end - 1) + 1) - even);
            interval.start = start;
            interval.end = end;
            interval.first = first;
            interval.last = last;
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
        relation: &R,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = R>>, KernelError<F>>
    where
        R: 'static,
    {
        let mut active_rounds: Vec<_> = dimensions
            .address_opening_indices()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "read/write phase split exceeds the cycle/address dimensions",
            })?
            .collect();
        if inner.num_rounds() != dimensions.log_k() {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address kernel has the wrong number of active rounds",
            });
        }
        let gap = dimensions.phase3_cycle_rounds();
        let rounds = relation.rounds();
        if active_rounds.len() + gap != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address schedule disagrees with the relation's round count",
            });
        }
        if gap == 0 {
            return Ok(inner);
        }
        active_rounds.sort_unstable();
        let scale = F::pow2(gap);
        Ok(Box::new(Self {
            inner,
            active_rounds,
            progress: RoundProgress::new(rounds),
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
    use jolt_field::{Fr, One, Zero};

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

    #[test]
    fn interval_boundaries_match_boolean_domain_sum() {
        use jolt_field::Ring;

        for log_k in 0..=4 {
            let size = 1 << log_k;
            let bindings: Vec<_> = (0..log_k).map(|bit| Fr::from_u64(3 + bit as u64)).collect();
            for start in 0..size {
                for end in start + 1..=size {
                    let mut mask = IntervalMask::new(&[(start, end)]);
                    for bound in 0..=log_k {
                        for block in 0..size >> bound {
                            let expected: Fr = (start..end)
                                .filter(|index| index >> bound == block)
                                .map(|index| {
                                    bindings[..bound]
                                        .iter()
                                        .enumerate()
                                        .map(|(bit, &r)| {
                                            if (index >> bit) & 1 == 0 {
                                                Fr::one() - r
                                            } else {
                                                r
                                            }
                                        })
                                        .product::<Fr>()
                                })
                                .sum();
                            assert_eq!(
                                mask.value(block),
                                expected,
                                "[{start},{end}), bound={bound}, block={block}"
                            );
                        }
                        if let Some(&r) = bindings.get(bound) {
                            mask.bind(r);
                        }
                    }
                }
            }
        }
        let mut mask = IntervalMask::new(&[(1, 3), (5, 8)]);
        mask.bind(Fr::zero());
        mask.bind(Fr::one());
        mask.bind(Fr::zero());
        assert_eq!(mask.value(0), Fr::one());
    }
}
