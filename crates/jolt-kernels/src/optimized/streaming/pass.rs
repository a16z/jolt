//! Task-partitioned row passes: a pass splits the cycle domain into tasks of
//! whole split-eq groups, at most [`TASKS_PER_THREAD`] per pool thread, and
//! each task visits its rows once, serially, off the witness's resident rows.

use std::ops::Range;

use jolt_field::{Accumulator as _, JoltField};
use jolt_poly::GruenSplitEqPolynomial;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Rows per callback of a task's visit.
pub(crate) const VISIT_ROWS: usize = 1 << 12;

/// Byte cap on one streaming kernel's dense tables: its rounds recompute
/// their pairs from the rows until the tables fit.
/// 4 GiB, or every representable size where `usize` cannot hold 4 GiB.
pub(crate) const DENSE_TABLE_CAP: usize = 4usize.saturating_mul(1 << 30);

/// Tasks per pool thread in a [`task_ranges`] partition, so work stealing can
/// even out unequal row costs between tasks.
#[cfg(feature = "parallel")]
const TASKS_PER_THREAD: usize = 8;

/// The task partition of one pass over the cycles `0..cycles`: units of
/// `group` cycles and `min(TASKS_PER_THREAD · pool threads, cycles / group)`
/// tasks, task `i` holding units `⌊i·n/tasks⌋..⌊(i+1)·n/tasks⌋` of the
/// `n = cycles / group`. `cycles` is a multiple of `group`, so every task
/// covers whole `group`-cycle kernel groups. Per-task state allocated before
/// the pass is held `tasks` times at once.
pub(crate) fn task_ranges(cycles: usize, group: usize) -> Vec<Range<usize>> {
    let unit = group;
    let units = cycles / unit;
    #[cfg(feature = "parallel")]
    let tasks = (TASKS_PER_THREAD * rayon::current_num_threads()).min(units);
    #[cfg(not(feature = "parallel"))]
    let tasks = 1;
    (0..tasks)
        .map(|task| task * units / tasks * unit..(task + 1) * units / tasks * unit)
        .collect()
}

/// Splits every table at the task boundaries of `ranges`, where `entry(cycle)`
/// is the table index at which a boundary cycle's entries start, so each task
/// fills only its own slices of the final tables. Tables shorter than the
/// ranges' entries leave the last tasks' slices short or empty.
pub(crate) fn split_at_tasks<'a, T, const N: usize>(
    mut tables: [&'a mut [T]; N],
    ranges: &[Range<usize>],
    entry: impl Fn(usize) -> usize,
) -> Vec<[&'a mut [T]; N]> {
    ranges
        .iter()
        .map(|range| {
            let len = entry(range.end) - entry(range.start);
            tables.each_mut().map(|table| {
                let len = len.min(table.len());
                let (head, tail) = std::mem::take(table).split_at_mut(len);
                *table = tail;
                head
            })
        })
        .collect()
}

/// The cycles a pass with pairs of `2^log_pair_cycles` cycles visits:
/// `prefix` rounded up to whole split-eq groups, so the pairs past them,
/// `pass_cycles(..) >> log_pair_cycles` onward, lie in the constant suffix.
pub(crate) fn pass_cycles<F: JoltField>(
    split_eq: &GruenSplitEqPolynomial<F>,
    log_pair_cycles: usize,
    prefix: usize,
) -> usize {
    prefix.next_multiple_of(split_eq.e_in_current_len() << log_pair_cycles)
}

/// Contracts consecutive rows into the halves of `2^(log_half + 1)`-row
/// pairs: row `t` adds its `M` values to half `(t >> log_half) & 1` of pair
/// `t >> (log_half + 1)`. A walk adds every row of its range in order, so a
/// pair's last row completes it; halves persist across visit callbacks.
pub(crate) struct PairHalves<F, const M: usize> {
    log_half: usize,
    halves: [[F; 2]; M],
}

impl<F: JoltField, const M: usize> PairHalves<F, M> {
    pub(crate) fn new(log_half: usize) -> Self {
        Self {
            log_half,
            halves: [[F::zero(); 2]; M],
        }
    }

    /// Row `t`'s offset within its half: the index of its row weight.
    pub(crate) fn offset(&self, t: usize) -> usize {
        t & ((1 << self.log_half) - 1)
    }

    /// Adds row `t`'s values; the pair's last row returns the pair index and
    /// each value's `[low, high]` halves.
    pub(crate) fn add(&mut self, t: usize, values: [F; M]) -> Option<(usize, [[F; 2]; M])> {
        let half = (t >> self.log_half) & 1;
        for (halves, value) in self.halves.iter_mut().zip(values) {
            halves[half] += value;
        }
        let last = (2 << self.log_half) - 1;
        if t & last != last {
            return None;
        }
        Some((
            t >> (self.log_half + 1),
            std::mem::replace(&mut self.halves, [[F::zero(); 2]; M]),
        ))
    }
}

/// One task's Gruen-grouped sums
/// `Σ_x_out e_out[x_out] · Σ_x_in e_in[x_in] · terms(x_out·|e_in| + x_in)`,
/// each group's inner sum accumulated unreduced.
pub(crate) struct GruenSums<'a, F: JoltField, const N: usize> {
    e_out: &'a [F],
    e_in: &'a [F],
    inner: [F::Accumulator; N],
    pub(crate) total: [F; N],
}

impl<'a, F: JoltField, const N: usize> GruenSums<'a, F, N> {
    pub(crate) fn new(e_out: &'a [F], e_in: &'a [F]) -> Self {
        Self {
            e_out,
            e_in,
            inner: [F::Accumulator::default(); N],
            total: [F::zero(); N],
        }
    }

    /// Tasks hold whole groups ([`task_ranges`]) and add their pairs in
    /// order, so a group's last pair closes it.
    pub(crate) fn add(&mut self, pair: usize, terms: [F; N]) {
        let x_in = pair & (self.e_in.len() - 1);
        for (inner, term) in self.inner.iter_mut().zip(terms) {
            inner.fmadd(self.e_in[x_in], term);
        }
        if x_in + 1 == self.e_in.len() {
            let e_out = self.e_out[pair / self.e_in.len()];
            for (total, inner) in self.total.iter_mut().zip(&mut self.inner) {
                *total += e_out * std::mem::take(inner).reduce();
            }
        }
    }
}

/// One task's receiver of a [`gruen_pair_pass`]: writes each completed
/// pair's halves into the task's slices of the tables being filled, if any,
/// and folds the pair's summand terms into the task's Gruen sums.
pub(crate) struct PairSink<'a, F: JoltField, T, const M: usize, const N: usize> {
    slices: Option<[&'a mut [F]; M]>,
    first_pair: usize,
    terms: &'a T,
    sums: GruenSums<'a, F, N>,
}

impl<F, T, const M: usize, const N: usize> PairSink<'_, F, T, M, N>
where
    F: JoltField,
    T: Fn([[F; 2]; M]) -> [F; N],
{
    /// `halves[m]` is table `m`'s `[low, high]` pair; every pair of the
    /// task's range must arrive once, in order. Entries past the tables' end
    /// are dropped.
    pub(crate) fn emit(&mut self, pair: usize, halves: [[F; 2]; M]) {
        if let Some(slices) = &mut self.slices {
            let entry = 2 * (pair - self.first_pair);
            for (slice, pair_halves) in slices.iter_mut().zip(&halves) {
                let end = slice.len().min(entry + 2);
                if let Some(entries) = slice.get_mut(entry..end) {
                    entries.copy_from_slice(&pair_halves[..entries.len()]);
                }
            }
        }
        self.sums.add(pair, (self.terms)(halves));
    }
}

/// One pass of a streamed round of the split-eq summand
/// `Σ_pair E(pair) · terms(pair)` over the pairs before the constant suffix
/// starting at cycle `prefix`, its pairs spanning `2^log_pair_cycles` cycles
/// each: the [`pass_cycles`] split into [`task_ranges`] of whole Gruen
/// groups, and each task's `walk` visits its range once, serially, in
/// callbacks of at most `chunk` rows, passing every completed pair to its
/// [`PairSink`] in order. The caller adds the pairs past the pass. A filling
/// pass splits `tables` (pair `z` at entries `2z, 2z + 1`) at the task
/// boundaries, so each task writes only its own slices.
pub(crate) fn gruen_pair_pass<F, T, const M: usize, const N: usize>(
    split_eq: &GruenSplitEqPolynomial<F>,
    log_pair_cycles: usize,
    prefix: usize,
    tables: Option<[&mut [F]; M]>,
    terms: T,
    walk: impl Fn(Range<usize>, usize, &mut PairSink<'_, F, T, M, N>) -> Result<(), WitnessError> + Sync,
) -> Result<[F; N], WitnessError>
where
    F: JoltField,
    T: Fn([[F; 2]; M]) -> [F; N] + Sync,
{
    let e_out = split_eq.e_out_current();
    let e_in = split_eq.e_in_current();
    let group = e_in.len() << log_pair_cycles;
    let ranges = task_ranges(pass_cycles(split_eq, log_pair_cycles, prefix), group);
    let slices: Vec<Option<[&mut [F]; M]>> = match tables {
        Some(tables) => split_at_tasks(tables, &ranges, |cycle| 2 * (cycle >> log_pair_cycles))
            .into_iter()
            .map(Some)
            .collect(),
        None => ranges.iter().map(|_| None).collect(),
    };
    let task =
        |(range, slices): (&Range<usize>, Option<[&mut [F]; M]>)| -> Result<[F; N], WitnessError> {
            let mut sink = PairSink {
                slices,
                first_pair: range.start >> log_pair_cycles,
                terms: &terms,
                sums: GruenSums::new(e_out, e_in),
            };
            walk(range.clone(), VISIT_ROWS.min(group), &mut sink)?;
            Ok(sink.sums.total)
        };
    let add = |left: [F; N], right: [F; N]| std::array::from_fn(|i| left[i] + right[i]);
    #[cfg(feature = "parallel")]
    return ranges
        .par_iter()
        .zip(slices)
        .map(task)
        .try_reduce(|| [F::zero(); N], |left, right| Ok(add(left, right)));
    #[cfg(not(feature = "parallel"))]
    ranges
        .iter()
        .zip(slices)
        .try_fold([F::zero(); N], |sum, task_input| {
            Ok(add(sum, task(task_input)?))
        })
}

#[cfg(test)]
mod tests {
    use super::task_ranges;

    /// Tasks tile the domain in order, each a positive whole number of
    /// aligned groups.
    #[test]
    fn tasks_tile_the_domain_in_whole_aligned_units() {
        for (cycles, group) in [
            (1usize << 12, 1usize << 4),
            (1 << 12, 1 << 2),
            (1 << 12, 1 << 12),
            (1 << 20, 1),
            (1, 1),
        ] {
            let ranges = task_ranges(cycles, group);
            let mut next = 0;
            for range in &ranges {
                assert_eq!(range.start, next);
                assert_eq!(range.start % group, 0);
                assert_eq!(range.len() % group, 0);
                assert!(!range.is_empty());
                next = range.end;
            }
            assert_eq!(next, cycles);
        }
    }
}
