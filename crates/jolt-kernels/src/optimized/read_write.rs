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
/// The RAM relation batches read, write, code (`C`), count and unmapped
/// (`N`) claims with powers `1, γ, γ², γ³, γ⁴`. Both address masks enter the
/// per-entry factor linearly: `C` scales the increment and `N` adds to the
/// count coefficient.
#[derive(Clone, Copy)]
pub(super) struct ReadWriteBatch<F> {
    gamma: F,
    code: F,
    count: F,
    unmapped: F,
}

#[derive(Clone, Copy)]
pub(super) struct ReadWriteCoefficients<F> {
    gamma: F,
    code_increment: F,
    count: F,
}

impl<F: JoltField> ReadWriteBatch<F> {
    pub(super) fn new(gamma: F) -> Self {
        let code = gamma * gamma;
        let count = code * gamma;
        Self {
            gamma,
            code,
            count,
            unmapped: count * gamma,
        }
    }

    pub(super) fn unbound(&self, masks: AddressMaskBits) -> ReadWriteCoefficients<F> {
        ReadWriteCoefficients {
            gamma: self.gamma,
            code_increment: if masks.protected {
                self.code
            } else {
                F::zero()
            },
            count: if masks.unmapped {
                self.count + self.unmapped
            } else {
                self.count
            },
        }
    }

    pub(super) fn bound(&self, masks: AddressMaskValues<F>) -> ReadWriteCoefficients<F> {
        ReadWriteCoefficients {
            gamma: self.gamma,
            code_increment: self.code * masks.protected,
            // Honest accesses touch no unmapped word, so `N` is zero on every
            // occupied column pair except at an interval's boundary block.
            count: if masks.unmapped.is_zero() {
                self.count
            } else {
                self.count + self.unmapped * masks.unmapped
            },
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
        if self.code_increment.is_zero() {
            read_write
        } else {
            read_write + self.code_increment * inc
        }
    }
}

/// The RAM relation's two public address masks: `C` over write-protected
/// words and `N` over unmapped words. They bind in lockstep with the address
/// variables.
///
/// Both masks share one interval list sorted by start, each interval tagged
/// with its mask. `C` and `N` are disjoint, so before any address bind one
/// binary search classifies an index for both masks, the same cost as `C`
/// alone. The list holds O(s) endpoints and weights, never a K-element table.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct AddressMasks<F: JoltField> {
    intervals: Vec<MaskInterval<F>>,
}

#[derive(Clone, Copy)]
pub(super) struct AddressMaskBits {
    protected: bool,
    unmapped: bool,
}

#[derive(Clone, Copy)]
pub(super) struct AddressMaskValues<F> {
    protected: F,
    unmapped: F,
}

impl<F: JoltField> AddressMaskValues<F> {
    pub(super) fn protected(&self) -> F {
        self.protected
    }

    pub(super) fn unmapped(&self) -> F {
        self.unmapped
    }

    /// The masks' values at `X = 2` on the line through `even` (`X = 0`) and
    /// `odd` (`X = 1`).
    pub(super) fn at_two(even: Self, odd: Self) -> Self {
        Self {
            protected: odd.protected + odd.protected - even.protected,
            unmapped: odd.unmapped + odd.unmapped - even.unmapped,
        }
    }
}

impl<F: JoltField> AddressMasks<F> {
    /// Merges the canonical (sorted, disjoint) `C` and `N` interval lists.
    /// Rejects lists that overlap each other, since one binary search would
    /// then misclassify the shared words.
    pub(super) fn new(
        protected: &[(usize, usize)],
        unmapped: &[(usize, usize)],
    ) -> Result<Self, KernelError<F>> {
        let tagged = |intervals: &[(usize, usize)], unmapped: bool| {
            intervals
                .iter()
                .map(move |&(start, end)| MaskInterval {
                    start,
                    end,
                    unmapped,
                    first: F::one(),
                    last: F::one(),
                })
                .collect::<Vec<_>>()
        };
        let mut intervals = tagged(protected, false);
        intervals.extend(tagged(unmapped, true));
        intervals.sort_unstable_by_key(|interval| interval.start);
        if intervals
            .windows(2)
            .any(|pair| pair[0].start >= pair[0].end || pair[1].start < pair[0].end)
            || intervals.last().is_some_and(|last| last.start >= last.end)
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM address masks are empty, unsorted or overlapping",
            });
        }
        Ok(Self { intervals })
    }

    /// Only used before any address bind. One binary search per surviving
    /// entry, with no additional trace storage.
    pub(super) fn contains_unbound(&self, index: usize) -> AddressMaskBits {
        let next = self
            .intervals
            .partition_point(|interval| interval.start <= index);
        match next
            .checked_sub(1)
            .map(|previous| &self.intervals[previous])
        {
            Some(interval) if index < interval.end => AddressMaskBits {
                protected: !interval.unmapped,
                unmapped: interval.unmapped,
            },
            _ => AddressMaskBits {
                protected: false,
                unmapped: false,
            },
        }
    }

    /// Binding keeps the list sorted by start, so the scan stops at the first
    /// interval past `index`.
    pub(super) fn value(&self, index: usize) -> AddressMaskValues<F> {
        let mut values = AddressMaskValues {
            protected: F::zero(),
            unmapped: F::zero(),
        };
        for interval in &self.intervals {
            if interval.start > index {
                break;
            }
            let value = interval.value(index);
            if interval.unmapped {
                values.unmapped += value;
            } else {
                values.protected += value;
            }
        }
        values
    }

    pub(super) fn bind(&mut self, r: F) {
        for interval in &mut self.intervals {
            interval.bind(r);
        }
    }

    /// Collapses the fully bound masks to one point interval per mask.
    pub(super) fn finish_address(&mut self) {
        let values = self.value(0);
        self.intervals = [(values.protected, false), (values.unmapped, true)]
            .into_iter()
            .map(|(value, unmapped)| MaskInterval {
                start: 0,
                end: 1,
                unmapped,
                first: value,
                last: value,
            })
            .collect();
    }
}

/// A low-bit-bound interval has full interior blocks and at most two boundary
/// blocks, so it stores two endpoints and two boundary weights.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MaskInterval<F: JoltField> {
    start: usize,
    end: usize,
    /// Whether the interval belongs to `N` rather than `C`.
    unmapped: bool,
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
                    // The same interval as `C` alone and as `N` alone.
                    let mut masks = [
                        AddressMasks::new(&[(start, end)], &[]).unwrap(),
                        AddressMasks::new(&[], &[(start, end)]).unwrap(),
                    ];
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
                            let [protected, unmapped] = &masks;
                            let label = format!("[{start},{end}), bound={bound}, block={block}");
                            assert_eq!(protected.value(block).protected(), expected, "{label}");
                            assert_eq!(protected.value(block).unmapped(), Fr::zero(), "{label}");
                            assert_eq!(unmapped.value(block).unmapped(), expected, "{label}");
                            assert_eq!(unmapped.value(block).protected(), Fr::zero(), "{label}");
                        }
                        if let Some(&r) = bindings.get(bound) {
                            for mask in &mut masks {
                                mask.bind(r);
                            }
                        }
                    }
                }
            }
        }
        let mut mask = AddressMasks::new(&[(1, 3)], &[(5, 8)]).unwrap();
        mask.bind(Fr::zero());
        mask.bind(Fr::one());
        mask.bind(Fr::zero());
        assert_eq!(mask.value(0).protected(), Fr::one());
        assert_eq!(mask.value(0).unmapped(), Fr::zero());
    }

    /// The merged list must give each mask the value it has on its own, both
    /// through the unbound binary search and after every bind.
    #[test]
    fn merged_masks_match_each_mask_alone() {
        use jolt_field::Ring;

        let log_k = 5;
        let protected = [(0, 2), (3, 6), (11, 12), (20, 27)];
        let unmapped = [(2, 3), (7, 11), (12, 13), (27, 32)];
        let mut merged = AddressMasks::<Fr>::new(&protected, &unmapped).unwrap();
        let mut alone = [
            AddressMasks::<Fr>::new(&protected, &[]).unwrap(),
            AddressMasks::<Fr>::new(&[], &unmapped).unwrap(),
        ];
        let contains = |intervals: &[(usize, usize)], index| {
            intervals
                .iter()
                .any(|&(start, end)| (start..end).contains(&index))
        };
        for index in 0..1 << log_k {
            let bits = merged.contains_unbound(index);
            assert_eq!(bits.protected, contains(&protected, index), "{index}");
            assert_eq!(bits.unmapped, contains(&unmapped, index), "{index}");
        }
        for bound in 0..=log_k {
            for block in 0..1 << (log_k - bound) {
                let values = merged.value(block);
                assert_eq!(values.protected(), alone[0].value(block).protected());
                assert_eq!(values.unmapped(), alone[1].value(block).unmapped());
            }
            if bound < log_k {
                let r = Fr::from_u64(7 + bound as u64);
                merged.bind(r);
                for mask in &mut alone {
                    mask.bind(r);
                }
            }
        }
        merged.finish_address();
        alone.iter_mut().for_each(AddressMasks::finish_address);
        assert_eq!(merged.value(0).protected(), alone[0].value(0).protected());
        assert_eq!(merged.value(0).unmapped(), alone[1].value(0).unmapped());

        for (protected, unmapped) in [
            (&[(0, 4)][..], &[(3, 5)][..]),
            (&[(2, 4)][..], &[(0, 3)][..]),
            (&[(2, 2)][..], &[][..]),
        ] {
            assert!(AddressMasks::<Fr>::new(protected, unmapped).is_err());
        }
    }
}
