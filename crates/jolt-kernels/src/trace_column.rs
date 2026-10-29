//! Lazy trace-column embeddings shared by commitment opening backends.

use std::ops::Range;

use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_field::JoltField;
#[cfg(not(feature = "akita"))]
use jolt_poly::EqPolynomial;
#[cfg(feature = "field-inline")]
use jolt_poly::{MultilinearPoly, TensorEqTable};
use jolt_utils::unsafe_allocate_zero_vec;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::commitment::CommitmentGrid;
#[cfg(feature = "field-inline")]
use crate::field_inline::FieldIncrementColumn;

/// Minimum per-range work before splitting scatter/sum drivers.
#[cfg(feature = "parallel")]
const MIN_RANGE: usize = 1 << 12;

/// A trace coefficient's grid index: `(k, t) ↦ t · t_stride + k · k_stride`,
/// dense columns at `k = 0`. Covers both proof orders with one formula:
/// cycle-major prefix-embeds the flat address-major `(K × T)` matrix
/// (`t_stride = 1`, `k_stride = 2^log_t`); address-major scatters
/// cycle-block-strided (`t_stride = cycle_stride`, `k_stride =
/// one_hot_stride`) — the reference embeddings' index maps verbatim.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TracePlacement {
    pub(crate) total_vars: usize,
    t_stride: usize,
    k_stride: usize,
}

impl TracePlacement {
    pub(crate) fn new(grid: CommitmentGrid) -> Self {
        match grid.order {
            TracePolynomialOrder::CycleMajor => Self {
                total_vars: grid.total_vars,
                t_stride: 1,
                k_stride: 1usize << grid.log_t,
            },
            TracePolynomialOrder::AddressMajor => Self {
                total_vars: grid.total_vars,
                t_stride: grid.cycle_stride(),
                k_stride: grid.one_hot_stride(),
            },
        }
    }

    #[inline(always)]
    pub(crate) const fn index(self, cycle: usize, address: usize) -> usize {
        cycle * self.t_stride + address * self.k_stride
    }

    #[cfg(not(feature = "akita"))]
    pub(crate) fn suffix_evaluation<F: JoltField>(
        self,
        point: &[F],
        prefix: usize,
        cycles: usize,
        address: usize,
    ) -> F {
        let low_bits = self.t_stride.trailing_zeros() as usize;
        let cycle_bits = cycles.trailing_zeros() as usize;
        let start = self.total_vars - low_bits - cycle_bits;
        let end = self.total_vars - low_bits;
        let index = self.index(0, address);
        let fixed = point
            .iter()
            .enumerate()
            .filter(|(i, _)| !(start..end).contains(i))
            .fold(F::one(), |product, (i, &coordinate)| {
                product
                    * if index & (1 << (self.total_vars - 1 - i)) == 0 {
                        F::one() - coordinate
                    } else {
                        coordinate
                    }
            });
        fixed * EqPolynomial::suffix_sum(&point[start..end], prefix)
    }

    /// An aligned dyadic cycle block maps to a Cartesian product of row and
    /// column indices. Sum its left weights once, then scatter that sum into
    /// its columns; no per-padding-cycle multiplication is needed.
    #[cfg(not(feature = "akita"))]
    pub(crate) fn fold_suffix<F: JoltField>(
        self,
        left: &[F],
        sigma: usize,
        cycles: Range<usize>,
        address: usize,
        value: F,
        result: &mut [F],
    ) {
        let mut prefix = cycles.start;
        let cycles = cycles.end;
        let stride_bits = self.t_stride.trailing_zeros() as usize;
        let row_stride = 1usize << stride_bits.saturating_sub(sigma);
        let mask = (1usize << sigma) - 1;
        while prefix < cycles {
            let bits = (prefix.trailing_zeros() as usize).min((cycles - prefix).ilog2() as usize);
            let column_bits = bits.min(sigma.saturating_sub(stride_bits));
            let row_bits = bits - column_bits;
            let index = self.index(prefix, address);
            let row = index >> sigma;
            let column = index & mask;
            let weight: F = (0..1usize << row_bits)
                .map(|offset| left[row + offset * row_stride])
                .sum();
            let contribution = value * weight;
            for offset in 0..1usize << column_bits {
                result[column + offset * self.t_stride] += contribution;
            }
            prefix += 1usize << bits;
        }
    }
}

/// Fold `total` source slots into a `num_cols`-sized accumulator through
/// `fill`, splitting into per-thread partial accumulators when parallel.
/// Field addition is exact, so the merge order cannot change the values.
pub(crate) fn scatter_fold<F: JoltField>(
    total: usize,
    num_cols: usize,
    fill: impl Fn(Range<usize>, &mut [F]) + Send + Sync,
) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if total > MIN_RANGE {
        let ranges = split_ranges(total);
        return ranges
            .into_par_iter()
            .map(|range| {
                let mut acc: Vec<F> = unsafe_allocate_zero_vec(num_cols);
                fill(range, &mut acc);
                acc
            })
            .reduce(
                || unsafe_allocate_zero_vec(num_cols),
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            );
    }
    let mut acc: Vec<F> = unsafe_allocate_zero_vec(num_cols);
    fill(0..total, &mut acc);
    acc
}

pub(crate) fn scatter_sum<F: JoltField>(
    total: usize,
    sum: impl Fn(Range<usize>) -> F + Send + Sync,
) -> F {
    #[cfg(feature = "parallel")]
    if total > MIN_RANGE {
        let ranges = split_ranges(total);
        return ranges
            .into_par_iter()
            .map(sum)
            .reduce(F::zero, |left, right| left + right);
    }
    sum(0..total)
}

#[cfg(feature = "parallel")]
fn split_ranges(total: usize) -> Vec<Range<usize>> {
    let max_ranges = rayon::current_num_threads() * 4;
    let ranges = (total / MIN_RANGE).clamp(1, max_ranges.max(1));
    let chunk = total.div_ceil(ranges);
    (0..total)
        .step_by(chunk)
        .map(|start| start..(start + chunk).min(total))
        .collect()
}

/// One dense trace-domain column (`T` values at address slot zero) as a lazy
/// view over the commitment grid: the placement of an increment column,
/// without materializing the `2^total_vars` grid it is embedded in. The
/// field-inline `FieldRdInc` openings use this view with either PCS.
#[cfg(feature = "field-inline")]
pub(crate) struct DenseTraceColumnPoly<F: JoltField> {
    values: FieldIncrementColumn<F>,
    placement: TracePlacement,
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> DenseTraceColumnPoly<F> {
    /// `None` when the column carries more cycles than the grid's trace
    /// dimension.
    pub fn new(values: FieldIncrementColumn<F>, grid: CommitmentGrid) -> Option<Self> {
        (values.len() <= 1usize << grid.log_t).then(|| Self {
            values,
            placement: TracePlacement::new(grid),
        })
    }

    #[inline]
    fn entries(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.values
            .nonzero_entries()
            .map(|(cycle, value)| (self.placement.index(cycle, 0), value))
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> MultilinearPoly<F> for DenseTraceColumnPoly<F> {
    fn num_vars(&self) -> usize {
        self.placement.total_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        debug_assert_eq!(point.len(), self.placement.total_vars);
        let eq = TensorEqTable::new(point);
        scatter_sum(self.values.prefix_len(), |range| {
            let mut acc = F::zero();
            for cycle in range {
                let value = self.values.value(cycle);
                if !value.is_zero() {
                    acc += value * eq.evaluate_index(self.placement.index(cycle, 0));
                }
            }
            acc
        })
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        emit_sorted_rows(self.entries().collect(), self.num_vars(), sigma, f);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        debug_assert_eq!(
            left.len(),
            1usize << self.num_vars().saturating_sub(sigma),
            "left vector length must equal number of rows"
        );
        let num_cols = 1usize << sigma;
        let mask = num_cols - 1;
        scatter_fold(self.values.prefix_len(), num_cols, |range, acc| {
            for cycle in range {
                let value = self.values.value(cycle);
                if !value.is_zero() {
                    let index = self.placement.index(cycle, 0);
                    acc[index & mask] += left[index >> sigma] * value;
                }
            }
        })
    }
}

/// Emit the `(2^{n-σ} × 2^σ)` matrix rows of a sparse entry set. Sorts the
/// entries and cursor-walks them into one reused row buffer — `O(N log N +
/// 2^n)` time, `O(N + 2^σ)` space. The batch opening never calls this
/// (it drives `fold_rows`); it serves the general [`MultilinearPoly`](jolt_poly::MultilinearPoly)
/// contract (`to_dense`, tests).
pub(crate) fn emit_sorted_rows<F: JoltField>(
    mut entries: Vec<(usize, F)>,
    num_vars: usize,
    sigma: usize,
    f: &mut dyn FnMut(usize, &[F]),
) {
    entries.sort_unstable_by_key(|&(index, _)| index);
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << num_vars.saturating_sub(sigma);
    let mut row_buffer: Vec<F> = unsafe_allocate_zero_vec(num_cols);
    let mut cursor = 0usize;
    for row in 0..num_rows {
        row_buffer.fill(F::zero());
        let row_base = row << sigma;
        while let Some(&(index, value)) = entries.get(cursor) {
            if index >= row_base + num_cols {
                break;
            }
            row_buffer[index - row_base] = value;
            cursor += 1;
        }
        f(row, &row_buffer);
    }
}
