//! Task-partitioned row passes: a pass splits the cycle domain into tasks of
//! whole split-eq groups, at most [`TASKS_PER_THREAD`] per pool thread, and
//! each task visits its rows once, serially
//! ([`BundleStore::visit`](crate::optimized::support::BundleStore::visit)).

use core::mem::MaybeUninit;
use core::slice::IterMut;
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
/// 4 GiB, or every representable size where `usize` cannot hold 4 GiB.
pub(crate) const DENSE_TABLE_CAP: usize = 4usize.saturating_mul(1 << 30);

/// Tasks per pool thread in a [`task_ranges`] partition, so work stealing can
/// even out unequal row costs between tasks.
#[cfg(feature = "parallel")]
const TASKS_PER_THREAD: usize = 8;

/// Tasks for a pass over `units` indivisible units: `min(TASKS_PER_THREAD ·
/// pool threads, units)`, at least one.
fn task_count(units: usize) -> usize {
    #[cfg(feature = "parallel")]
    return (TASKS_PER_THREAD * rayon::current_num_threads()).clamp(1, units.max(1));
    #[cfg(not(feature = "parallel"))]
    {
        let _ = units;
        1
    }
}

/// The task partition of one pass over the cycles `0..cycles`: units of
/// `group` cycles and [`task_count`] tasks, task `i` holding units
/// `⌊i·n/tasks⌋..⌊(i+1)·n/tasks⌋` of the
/// `n = cycles / group`. `cycles` is a multiple of `group`, so every task
/// covers whole `group`-cycle kernel groups; no cycles give one empty task.
/// Per-task state allocated before the pass is held `tasks` times at once.
pub(crate) fn task_ranges(cycles: usize, group: usize) -> Vec<Range<usize>> {
    let unit = group;
    let units = cycles / unit;
    let tasks = task_count(units);
    // `⌊task·units/tasks⌋` without the product `task·units`, which exceeds
    // `usize` for a pass over single cycles of a large enough domain (on a
    // 32-bit target, `2^32 / tasks` cycles).
    let start = |task: usize| (task * (units / tasks) + task * (units % tasks) / tasks) * unit;
    (0..tasks)
        .map(|task| start(task)..start(task + 1))
        .collect()
}

/// Splits `items`, sorted by `key`, into at most [`task_count`] contiguous
/// runs of about equal length, each holding every item of its keys.
pub(crate) fn key_runs<T>(items: &[T], key: impl Fn(&T) -> usize) -> Vec<&[T]> {
    let tasks = task_count(items.len());
    let mut runs = Vec::with_capacity(tasks);
    let mut rest = items;
    for task in 0..tasks {
        if rest.is_empty() {
            break;
        }
        let target = rest.len().div_ceil(tasks - task);
        let cut = match rest.get(target - 1) {
            Some(last) if target < rest.len() => {
                let boundary = key(last);
                target + rest[target..].partition_point(|item| key(item) == boundary)
            }
            _ => rest.len(),
        };
        let (run, tail) = rest.split_at(cut);
        runs.push(run);
        rest = tail;
    }
    runs
}

/// A task wrote a different number of items than its count pass reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FillMismatch;

/// One task's exactly sized slots of an [`exact_task_fill`] destination.
pub(crate) struct TaskSlots<'a, T> {
    slots: IterMut<'a, MaybeUninit<T>>,
    overflow: bool,
}

impl<'a, T> TaskSlots<'a, T> {
    /// A task's counted slots, written in order.
    pub(crate) fn new(slots: &'a mut [MaybeUninit<T>]) -> Self {
        Self {
            slots: slots.iter_mut(),
            overflow: false,
        }
    }

    /// Writes the task's next item; an item past its counted slots is dropped
    /// and fails the fill.
    pub(crate) fn push(&mut self, item: T) {
        match self.slots.next() {
            Some(slot) => {
                let _ = slot.write(item);
            }
            None => self.overflow = true,
        }
    }

    /// Whether the task wrote exactly its counted slots, every one of them
    /// initialized.
    pub(crate) fn finish(mut self) -> Result<(), FillMismatch> {
        if self.overflow || self.slots.next().is_some() {
            return Err(FillMismatch);
        }
        Ok(())
    }
}

/// Collects `tasks`' items in task order into one allocation of exactly their
/// total: a parallel pass counts each task's items, and a second writes each
/// task's items into its own slots, so no task holds a vector of its own and
/// the destination never grows. Either pass's error ends the fill; a task
/// that writes more or fewer items than it counted fails with
/// `mismatch(FillMismatch)`.
pub(crate) fn exact_task_fill<I, T, E>(
    tasks: &[I],
    count: impl Fn(&I) -> Result<usize, E> + Sync,
    fill: impl Fn(&I, &mut TaskSlots<'_, T>) -> Result<(), E> + Sync,
    mismatch: impl Fn(FillMismatch) -> E + Sync,
) -> Result<Vec<T>, E>
where
    I: Sync,
    T: Copy + Send,
    E: Send,
{
    #[cfg(feature = "parallel")]
    let counts: Vec<usize> = tasks.par_iter().map(&count).collect::<Result<_, E>>()?;
    #[cfg(not(feature = "parallel"))]
    let counts: Vec<usize> = tasks.iter().map(&count).collect::<Result<_, E>>()?;
    let total = counts.iter().sum();
    let mut items = Vec::with_capacity(total);
    let mut spare = &mut items.spare_capacity_mut()[..total];
    let task_slots: Vec<&mut [MaybeUninit<T>]> = counts
        .iter()
        .map(|&count| {
            let (head, tail) = core::mem::take(&mut spare).split_at_mut(count);
            spare = tail;
            head
        })
        .collect();
    let write = |(task, slots): (&I, &mut [MaybeUninit<T>])| -> Result<(), E> {
        let mut slots = TaskSlots::new(slots);
        fill(task, &mut slots)?;
        slots.finish().map_err(&mismatch)
    };
    #[cfg(feature = "parallel")]
    tasks.par_iter().zip(task_slots).try_for_each(write)?;
    #[cfg(not(feature = "parallel"))]
    tasks.iter().zip(task_slots).try_for_each(write)?;
    // SAFETY: the task slots tile the first `total` spare elements, and every
    // task wrote each of its slots before returning `Ok`.
    unsafe { items.set_len(total) };
    Ok(items)
}

/// Binds `items`, sorted by `key`, in place one key group at a time:
/// `bind(group, emit)` emits the group's bound items, never more than it
/// holds. Runs of whole groups ([`key_runs`]) bind in parallel, each writing
/// from its own start and reading the group being bound from a copy; the
/// runs' outputs then move down in order and `items` truncates to them,
/// keeping its capacity. A group that emits more items than it holds fails
/// with [`FillMismatch`] and leaves `items` partly bound.
pub(crate) fn bind_groups_in_place<T: Copy + Send + Sync>(
    items: &mut Vec<T>,
    key: impl Fn(&T) -> usize + Sync,
    bind: impl Fn(&[T], &mut dyn FnMut(T)) + Sync,
) -> Result<(), FillMismatch> {
    let lens: Vec<usize> = key_runs(items, &key).iter().map(|run| run.len()).collect();
    let mut runs: Vec<&mut [T]> = Vec::with_capacity(lens.len());
    let mut rest = items.as_mut_slice();
    for &len in &lens {
        let (run, tail) = core::mem::take(&mut rest).split_at_mut(len);
        runs.push(run);
        rest = tail;
    }
    let compact = |run: &mut [T]| -> Result<usize, FillMismatch> {
        let mut group = Vec::new();
        let (mut read, mut write, mut overflow) = (0, 0, false);
        while read < run.len() {
            let first = key(&run[read]);
            let end = read
                + run[read..]
                    .iter()
                    .take_while(|item| key(item) == first)
                    .count();
            group.clear();
            group.extend_from_slice(&run[read..end]);
            let start = write;
            bind(&group, &mut |item| {
                if write - start < group.len() {
                    run[write] = item;
                    write += 1;
                } else {
                    overflow = true;
                }
            });
            if overflow {
                return Err(FillMismatch);
            }
            read = end;
        }
        Ok(write)
    };
    #[cfg(feature = "parallel")]
    let written: Vec<usize> = runs
        .into_par_iter()
        .map(compact)
        .collect::<Result<_, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let written: Vec<usize> = runs.into_iter().map(compact).collect::<Result<_, _>>()?;
    let (mut start, mut end) = (0, 0);
    for (len, written) in lens.into_iter().zip(written) {
        items.copy_within(start..start + written, end);
        end += written;
        start += len;
    }
    items.truncate(end);
    Ok(())
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

/// The first bind depth `k` at which tables over `cycles` cycles before any
/// bind fit a cap of `cap_cycles` cycles (`⌈cycles / 2^k⌉ ≤ cap_cycles`), or
/// `rounds` when none of the `rounds` depths does. Both arguments count
/// cycles: a caller divides its byte cap by its tables' bytes per cycle, so
/// the byte size of the whole domain is never formed.
pub(crate) fn dense_crossover(cycles: usize, cap_cycles: usize, rounds: usize) -> usize {
    (0..rounds)
        .find(|&depth| cycles.div_ceil(1 << depth) <= cap_cycles)
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
#[cfg_attr(
    feature = "parallel",
    expect(clippy::unwrap_used, reason = "test module")
)]
mod tests {
    #[cfg(feature = "parallel")]
    use rayon::ThreadPoolBuilder;

    use super::{bind_groups_in_place, task_ranges, FillMismatch};

    #[test]
    fn group_overemit_fails_after_an_earlier_group_shrinks() {
        let check = || {
            let mut items = vec![0, 0, 1, 1];
            items.extend([2; 36]);
            let result = bind_groups_in_place(
                &mut items,
                |&item| item,
                |group, emit| match group[0] {
                    0 => emit(0),
                    1 => {
                        for _ in 0..3 {
                            emit(1);
                        }
                    }
                    _ => {
                        for &item in group {
                            emit(item);
                        }
                    }
                },
            );
            assert_eq!(result, Err(FillMismatch));
        };
        #[cfg(feature = "parallel")]
        for threads in [1, 3] {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(check);
        }
        #[cfg(not(feature = "parallel"))]
        check();
    }

    /// Tasks tile the cycles in order, each a positive whole number of
    /// aligned groups; no cycles give one empty task.
    #[test]
    fn tasks_tile_the_domain_in_whole_aligned_units() {
        assert!(matches!(task_ranges(0, 1 << 4).as_slice(), [task] if task.is_empty()));
        for (cycles, group) in [
            (1usize << 12, 1usize << 4),
            (1 << 12, 1 << 2),
            (1 << 12, 1 << 12),
            (364, 4),
            (3 << 12, 1 << 12),
            (1 << 20, 1),
            (1, 1),
            (1 << (usize::BITS - 1), 1),
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
