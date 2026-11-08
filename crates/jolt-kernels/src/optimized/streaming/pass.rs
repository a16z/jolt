//! Task-partitioned row passes: a pass splits the cycle domain into tasks of
//! whole split-eq groups, at most [`TASKS_PER_THREAD`] per pool thread, and
//! each task visits its rows once, serially
//! ([`BundleStore::visit`](crate::optimized::support::BundleStore::visit)).

use std::ops::Range;

use jolt_field::JoltField;
use jolt_poly::GruenSplitEqPolynomial;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Rows per callback of a task's visit.
pub(crate) const VISIT_ROWS: usize = 1 << 12;

/// Byte cap on one streaming kernel's dense tables: its rounds recompute
/// their pairs from the rows until the tables fit ([`dense_crossover`]).
pub(crate) const DENSE_TABLE_CAP: usize = 4 << 30;

/// Tasks per pool thread in a [`task_ranges`] partition, so work stealing can
/// even out unequal row costs between tasks.
#[cfg(feature = "parallel")]
const TASKS_PER_THREAD: usize = 8;

/// The task partition of one pass over the cycle domain `0..cycles`: units of
/// `U = min(cycles, group)` cycles and `min(TASKS_PER_THREAD · pool threads,
/// cycles / U)` tasks, task `i` holding units `⌊i·n/tasks⌋..⌊(i+1)·n/tasks⌋`
/// of the `n = cycles / U`. Both sizes are powers of two, so every task
/// covers whole `group`-cycle kernel groups. Per-task state allocated before
/// the pass is held `tasks` times at once.
pub(crate) fn task_ranges(cycles: usize, group: usize) -> Vec<Range<usize>> {
    let unit = cycles.min(group);
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
/// fills only its own slices of the final tables.
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
                let (head, tail) = std::mem::take(table).split_at_mut(len);
                *table = tail;
                head
            })
        })
        .collect()
}

/// The first bind depth `k` at which tables of `table_bytes` before any bind
/// fit `cap` bytes (`⌈table_bytes / 2^k⌉ ≤ cap`), or `rounds` when none of
/// the `rounds` depths does.
pub(crate) fn dense_crossover(table_bytes: usize, cap: usize, rounds: usize) -> usize {
    (0..rounds)
        .find(|&depth| table_bytes.div_ceil(1 << depth) <= cap)
        .unwrap_or(rounds)
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
/// `Σ_x_out e_out[x_out] · Σ_x_in e_in[x_in] · terms(x_out·|e_in| + x_in)`.
pub(crate) struct GruenSums<'a, F, const N: usize> {
    e_out: &'a [F],
    e_in: &'a [F],
    inner: [F; N],
    pub(crate) total: [F; N],
}

impl<'a, F: JoltField, const N: usize> GruenSums<'a, F, N> {
    pub(crate) fn new(e_out: &'a [F], e_in: &'a [F]) -> Self {
        Self {
            e_out,
            e_in,
            inner: [F::zero(); N],
            total: [F::zero(); N],
        }
    }

    /// Tasks hold whole groups ([`task_ranges`]) and add their pairs in
    /// order, so a group's last pair closes it.
    pub(crate) fn add(&mut self, pair: usize, terms: [F; N]) {
        let x_in = pair & (self.e_in.len() - 1);
        for (inner, term) in self.inner.iter_mut().zip(terms) {
            *inner += self.e_in[x_in] * term;
        }
        if x_in + 1 == self.e_in.len() {
            let e_out = self.e_out[pair / self.e_in.len()];
            for (total, inner) in self.total.iter_mut().zip(&mut self.inner) {
                *total += e_out * *inner;
                *inner = F::zero();
            }
        }
    }
}

/// One task's receiver of a [`gruen_pair_pass`]: writes each completed
/// pair's halves into the task's slices of the tables being filled, if any,
/// and folds the pair's summand terms into the task's Gruen sums.
pub(crate) struct PairSink<'a, F, T, const M: usize, const N: usize> {
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
    /// task's range must arrive once, in order.
    pub(crate) fn emit(&mut self, pair: usize, halves: [[F; 2]; M]) {
        if let Some(slices) = &mut self.slices {
            let entry = 2 * (pair - self.first_pair);
            for (slice, pair_halves) in slices.iter_mut().zip(&halves) {
                slice[entry..entry + 2].copy_from_slice(pair_halves);
            }
        }
        self.sums.add(pair, (self.terms)(halves));
    }
}

/// One pass of a streamed round of the split-eq summand
/// `Σ_pair E(pair) · terms(pair)`, its pairs spanning `2^log_pair_cycles`
/// cycles each: the cycle domain (`|e_out| · |e_in|` pairs) splits into
/// [`task_ranges`] of whole Gruen groups, and each task's `walk` visits its
/// range once, serially, in callbacks of at most `chunk` rows, passing every
/// completed pair to its [`PairSink`] in order. A filling pass splits
/// `tables` (pair `z` at entries `2z, 2z + 1`) at the task boundaries, so
/// each task writes only its own slices.
pub(crate) fn gruen_pair_pass<F, T, const M: usize, const N: usize>(
    split_eq: &GruenSplitEqPolynomial<F>,
    log_pair_cycles: usize,
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
    let ranges = task_ranges(e_out.len() * group, group);
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

/// `(q(0), q(∞))` terms of one pair of a two-factor product summand `a·b`,
/// from the factors' `[low, high]` halves.
pub(crate) fn product_endpoint_terms<F: JoltField>([a, b]: [[F; 2]; 2]) -> [F; 2] {
    [a[0] * b[0], (a[1] - a[0]) * (b[1] - b[0])]
}

#[cfg(test)]
mod tests {
    use super::task_ranges;
    #[cfg(feature = "parallel")]
    use super::TASKS_PER_THREAD;

    /// Tasks tile the domain in order, each a positive whole number of
    /// aligned units with the units spread evenly, and never outnumber
    /// `TASKS_PER_THREAD` per pool thread (one task without `parallel`).
    #[test]
    fn tasks_tile_the_domain_in_whole_aligned_units() {
        #[cfg(feature = "parallel")]
        let most_tasks = TASKS_PER_THREAD * rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let most_tasks = 1;
        for (cycles, group) in [
            (1usize << 12, 1usize << 4),
            (1 << 12, 1 << 2),
            (1 << 12, 1 << 12),
            (1 << 6, 1 << 8),
            (1 << 20, 1),
            (1, 1),
        ] {
            let unit = cycles.min(group);
            let units = cycles / unit;
            let ranges = task_ranges(cycles, group);
            assert_eq!(ranges.len(), most_tasks.min(units));
            let (fewest, most) = (units / ranges.len(), units.div_ceil(ranges.len()));
            let mut next = 0;
            for range in &ranges {
                assert_eq!(range.start, next);
                assert_eq!(range.start % unit, 0);
                assert_eq!(range.len() % unit, 0);
                assert!((fewest..=most).contains(&(range.len() / unit)) && !range.is_empty());
                next = range.end;
            }
            assert_eq!(next, cycles);
        }
    }
}
