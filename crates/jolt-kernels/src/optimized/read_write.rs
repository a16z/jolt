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
///
/// The RAM relation batches read, write, write-protected (`C`) and count
/// claims with powers `1, γ, γ², γ³`. The address mask `C` enters the
/// per-entry factor linearly, by scaling the increment.
#[derive(Clone, Copy)]
pub(super) struct ReadWriteBatch<F> {
    gamma: F,
    write_protected: F,
    count: F,
}

#[derive(Clone, Copy)]
pub(super) struct ReadWriteCoefficients<F> {
    gamma: F,
    write_protected_increment: F,
    count: F,
}

impl<F: JoltField> ReadWriteBatch<F> {
    pub(super) fn new(gamma: F) -> Self {
        let write_protected = gamma * gamma;
        Self {
            gamma,
            write_protected,
            count: write_protected * gamma,
        }
    }

    pub(super) fn unbound(&self, protected: bool) -> ReadWriteCoefficients<F> {
        ReadWriteCoefficients {
            gamma: self.gamma,
            write_protected_increment: if protected {
                self.write_protected
            } else {
                F::zero()
            },
            count: self.count,
        }
    }

    pub(super) fn bound(&self, mask: F) -> ReadWriteCoefficients<F> {
        ReadWriteCoefficients {
            gamma: self.gamma,
            write_protected_increment: self.write_protected * mask,
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
        // Unprotected entries need only the original read/write multiplication.
        let read_write = val + self.gamma * (val + inc);
        if self.write_protected_increment.is_zero() {
            read_write
        } else {
            read_write + self.write_protected_increment * inc
        }
    }
}

/// The RAM relation's write-protected address mask `C`, bound in lockstep
/// with the address variables. Its intervals stay sorted by start, so before
/// any address bind one binary search classifies an index. It holds O(s)
/// endpoints and weights, never a K-element table.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct IntervalMask<F: JoltField> {
    intervals: Vec<MaskInterval<F>>,
}

impl<F: JoltField> IntervalMask<F> {
    /// Rejects an empty interval, which binding cannot represent, and an
    /// unsorted or overlapping list, which the binary search would misclassify.
    pub(super) fn new(intervals: &[(usize, usize)]) -> Result<Self, KernelError<F>> {
        if intervals
            .windows(2)
            .any(|pair| pair[0].0 >= pair[0].1 || pair[1].0 < pair[0].1)
            || intervals.last().is_some_and(|&(start, end)| start >= end)
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM write-protected mask has an empty, unsorted or overlapping interval",
            });
        }
        Ok(Self {
            intervals: intervals
                .iter()
                .map(|&(start, end)| MaskInterval {
                    start,
                    end,
                    first: F::one(),
                    last: F::one(),
                })
                .collect(),
        })
    }

    /// Only used before any address bind. One binary search per surviving
    /// entry, with no additional trace storage.
    pub(super) fn contains_unbound(&self, index: usize) -> bool {
        let next = self
            .intervals
            .partition_point(|interval| interval.start <= index);
        next.checked_sub(1)
            .is_some_and(|previous| index < self.intervals[previous].end)
    }

    /// Binding keeps the list sorted by start, so the scan stops at the first
    /// interval past `index`.
    pub(super) fn value(&self, index: usize) -> F {
        let mut value = F::zero();
        for interval in &self.intervals {
            if interval.start > index {
                break;
            }
            value += interval.value(index);
        }
        value
    }

    pub(super) fn bind(&mut self, r: F) {
        for interval in &mut self.intervals {
            interval.bind(r);
        }
    }

    /// Collapses the fully bound mask to one point interval.
    pub(super) fn finish_address(&mut self) {
        let value = self.value(0);
        self.intervals = vec![MaskInterval {
            start: 0,
            end: 1,
            first: value,
            last: value,
        }];
    }
}

/// A low-bit-bound interval has full interior blocks and at most two boundary
/// blocks, so it stores two endpoints and two boundary weights.
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

    fn bind(&mut self, r: F) {
        let start = self.start / 2;
        let end = self.end.div_ceil(2);
        let even = self.value(2 * start);
        let first = even + r * (self.value(2 * start + 1) - even);
        let even = self.value(2 * (end - 1));
        let last = even + r * (self.value(2 * (end - 1) + 1) - even);
        self.start = start;
        self.end = end;
        self.first = first;
        self.last = last;
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
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_field::{Fr, One, Zero};

    use super::*;

    /// A mask's value on `block` after binding the low `bindings.len()` bits:
    /// the sum, over the indices of the block that lie in an interval, of the
    /// equality polynomial between their low bits and the bindings.
    fn block_sum(intervals: &[(usize, usize)], bindings: &[Fr], block: usize) -> Fr {
        intervals
            .iter()
            .flat_map(|&(start, end)| start..end)
            .filter(|index| index >> bindings.len() == block)
            .map(|index| {
                bindings
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
            .sum()
    }

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
                    let mut mask = IntervalMask::new(&[(start, end)]).unwrap();
                    for bound in 0..=log_k {
                        for block in 0..size >> bound {
                            assert_eq!(
                                mask.value(block),
                                block_sum(&[(start, end)], &bindings[..bound], block),
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
        let mut mask = IntervalMask::new(&[(1, 3), (5, 8)]).unwrap();
        mask.bind(Fr::zero());
        mask.bind(Fr::one());
        mask.bind(Fr::zero());
        assert_eq!(mask.value(0), Fr::one());
    }

    #[test]
    fn multi_interval_mask_matches_brute_force_sums() {
        use jolt_field::Ring;

        let log_k = 5;
        let intervals = [(0, 2), (3, 6), (7, 11), (12, 13), (20, 27), (28, 32)];
        let mut mask = IntervalMask::<Fr>::new(&intervals).unwrap();
        for index in 0..1 << log_k {
            let contains = intervals
                .iter()
                .any(|&(start, end)| (start..end).contains(&index));
            assert_eq!(mask.contains_unbound(index), contains, "{index}");
        }
        let bindings: Vec<Fr> = (0..log_k).map(|bit| Fr::from_u64(7 + bit as u64)).collect();
        for bound in 0..=log_k {
            for block in 0..1 << (log_k - bound) {
                assert_eq!(
                    mask.value(block),
                    block_sum(&intervals, &bindings[..bound], block),
                    "bound={bound}, block={block}"
                );
            }
            if let Some(&r) = bindings.get(bound) {
                mask.bind(r);
            }
        }
        mask.finish_address();
        assert_eq!(mask.value(0), block_sum(&intervals, &bindings, 0));

        for intervals in [&[(0, 4), (3, 5)][..], &[(2, 2)][..], &[(5, 7), (1, 3)][..]] {
            assert!(IntervalMask::<Fr>::new(intervals).is_err());
        }
    }
}
