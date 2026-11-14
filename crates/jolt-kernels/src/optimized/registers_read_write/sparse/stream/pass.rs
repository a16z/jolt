//! The frontier's passes over the witness's rows: each regenerates its
//! round's level instead of reading stored entries. The Gruen groups that
//! hold a prefix row ([`StreamedRows`]) split into [`task_ranges`], each
//! visited once, serially; a
//! task merges its consecutive rows' cells ([`RegisterCycleRow::entries`])
//! level by level into the round's pairs, and every completed pair reaches the
//! task's [`GruenSums`]. A counting pass also counts each Gruen group's
//! next-level entries, and the round that fills that level writes its entries
//! into exactly those slots.

use std::cmp::Ordering;
use std::ops::Range;

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::EqPolynomial;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::layout::{
    merge_bind, Cell, MatrixEntry, SeedEntry, SparseEntry, MAX_ENTRIES_PER_CYCLE,
};
use super::super::ops::accumulate_pair_group;
use super::super::{CoeffLut, CycleState, LutIndex};
use super::LUT_LEVELS;
use crate::optimized::registers_read_write::rows::{raw_rd_inc, RegisterCycleRow, StreamedRows};
use crate::optimized::streaming::{
    split_at_tasks, task_ranges, FillMismatch, GruenSums, PairHalves, TaskSlots, VISIT_ROWS,
};

/// Why a streamed round's pass failed.
#[derive(Debug)]
pub(super) enum PassError {
    /// A row failed to extract.
    Rows,
    /// A filling round's tasks or rows disagree with the last pass's counts.
    Diverged,
}

impl From<WitnessError> for PassError {
    fn from(_: WitnessError) -> Self {
        Self::Rows
    }
}

/// One completed pair of the round's level: rows `2z` and `2z + 1`, in field
/// form, with the increment's halves `[inc(2z), inc(2z + 1)]` at that level.
struct Pair<'a, F> {
    z: usize,
    evens: &'a [SparseEntry<F, F>],
    odds: &'a [SparseEntry<F, F>],
    inc: [F; 2],
}

/// Rows `[even, odd]` of one level, each reserved for the entries a
/// level-`level` row holds at most: every register its `2^level` cycles
/// touch, [`MAX_ENTRIES_PER_CYCLE`] per cycle.
fn level_rows<E>(level: usize) -> [Vec<E>; 2] {
    let width =
        (MAX_ENTRIES_PER_CYCLE << level.min(REGISTER_ADDRESS_BITS)).min(1 << REGISTER_ADDRESS_BITS);
    [Vec::with_capacity(width), Vec::with_capacity(width)]
}

/// Merge-binds two sibling rows into `out`.
fn merge_into<E: Cell, B>(
    even: &[E],
    odd: &[E],
    bind: impl Fn(Option<&E>, Option<&E>) -> B,
    out: &mut Vec<B>,
) {
    out.clear();
    merge_bind(even, odd, &bind, |entry| out.push(entry));
}

/// One task's merge of consecutive rows into the round's level-`depth`
/// pairs: each level keeps its even row until the odd sibling arrives, then
/// merges the two into the next level. The state persists across visit
/// callbacks, so a pair may span several.
struct RowMerge<F: JoltField> {
    depth: usize,
    /// The even level-0 row a level-1 row waits on.
    leaf: ([SeedEntry; MAX_ENTRIES_PER_CYCLE], usize),
    /// Rows of levels `1..=min(depth, LUT_LEVELS)`.
    index: Vec<[Vec<SparseEntry<F, LutIndex>>; 2]>,
    /// Rows of levels `LUT_LEVELS + 1..=depth`; at `depth ≤ LUT_LEVELS`, the
    /// round's pair converted to field coefficients.
    field: Vec<[Vec<SparseEntry<F, F>>; 2]>,
    halves: PairHalves<F, 1>,
}

impl<F: JoltField> RowMerge<F> {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            leaf: RegisterCycleRow::default().entries(0),
            index: (1..=depth.min(LUT_LEVELS)).map(level_rows).collect(),
            field: if depth > LUT_LEVELS {
                (LUT_LEVELS + 1..=depth).map(level_rows).collect()
            } else {
                vec![level_rows(depth)]
            },
            halves: PairHalves::new(depth),
        }
    }

    /// Adds row `t`, its increment weighted by `weights` at its offset within
    /// its half; returns the pair whose last row this is. `luts[ℓ]` holds
    /// level `ℓ`'s `[ra, wa]` tables and `r[ℓ]` binds level `ℓ`.
    fn push(
        &mut self,
        t: usize,
        row: &RegisterCycleRow,
        weights: &[F],
        r: &[F],
        luts: &[[CoeffLut<F>; 2]],
    ) -> Option<Pair<'_, F>> {
        let weight = weights[self.halves.offset(t)];
        let inc = self.halves.add(t, [weight.mul_i128(raw_rd_inc(row))]);
        // `t < 2^32`: the kernel refuses a wider cycle domain.
        self.lift(t, row.entries(t as u32), r, luts);
        let (z, [inc]) = inc?;
        if (1..=LUT_LEVELS).contains(&self.depth) {
            let [ra, wa] = &luts[self.depth];
            for (out, index_row) in self.field[0].iter_mut().zip(&self.index[self.depth - 1]) {
                out.clear();
                out.extend(index_row.iter().map(|entry| entry.to_field(ra, wa)));
            }
        }
        let [evens, odds] = &self.field[self.field.len() - 1];
        Some(Pair {
            z,
            evens,
            odds,
            inc,
        })
    }

    /// Stores row `t`'s cells and merges each sibling pair it completes into
    /// the next level, up to the round's level: 0→1 by the seed bind, 1→2 and
    /// 2→3 by LUT-index composition, 3→4 dereferencing both coefficients
    /// first, deeper levels in field coefficients.
    fn lift(
        &mut self,
        t: usize,
        (cells, len): ([SeedEntry; MAX_ENTRIES_PER_CYCLE], usize),
        r: &[F],
        luts: &[[CoeffLut<F>; 2]],
    ) {
        let slot = |level: usize| (t >> level) & 1;
        if self.depth == 0 {
            let [ra, wa] = &luts[0];
            let out = &mut self.field[0][slot(0)];
            out.clear();
            out.extend(cells[..len].iter().map(|cell| cell.to_field(ra, wa)));
            return;
        }
        if slot(0) == 0 {
            self.leaf = (cells, len);
            return;
        }
        let [ra, wa] = &luts[0];
        merge_into(
            &self.leaf.0[..self.leaf.1],
            &cells[..len],
            |even, odd| SeedEntry::bind(even, odd, r[0], ra, wa),
            &mut self.index[0][slot(1)],
        );
        let unused = CycleState::<F>::unused_lut();
        for level in 1..self.depth {
            if slot(level) == 0 {
                return;
            }
            let [ra, wa] = &luts[level.min(LUT_LEVELS)];
            match level.cmp(&LUT_LEVELS) {
                Ordering::Less => {
                    let (lower, upper) = self.index.split_at_mut(level);
                    let [even, odd] = &lower[level - 1];
                    merge_into(
                        even,
                        odd,
                        |even, odd| SparseEntry::bind(even, odd, r[level], ra, wa),
                        &mut upper[0][slot(level + 1)],
                    );
                }
                Ordering::Equal => {
                    let [even, odd] = &self.index[level - 1];
                    let to_field = |entry: &SparseEntry<F, LutIndex>| entry.to_field(ra, wa);
                    merge_into(
                        even,
                        odd,
                        |even, odd| {
                            let (even, odd) = (even.map(to_field), odd.map(to_field));
                            SparseEntry::bind(
                                even.as_ref(),
                                odd.as_ref(),
                                r[level],
                                &unused,
                                &unused,
                            )
                        },
                        &mut self.field[0][slot(level + 1)],
                    );
                }
                Ordering::Greater => {
                    let (lower, upper) = self.field.split_at_mut(level - LUT_LEVELS);
                    let [even, odd] = &lower[level - LUT_LEVELS - 1];
                    merge_into(
                        even,
                        odd,
                        |even, odd| SparseEntry::bind(even, odd, r[level], &unused, &unused),
                        &mut upper[0][slot(level + 1)],
                    );
                }
            }
        }
    }
}

/// Round `r.len()`'s Gruen sums `Σ_z E(z) · terms(state, pair z)` over the
/// pairs of its level's Gruen groups that hold a prefix row, whose other
/// pairs add nothing; `state` is its task's entry of `task_states(ranges)`.
/// Every pair of a visited group reaches its task's sums, an empty one too,
/// so each group closes. Returns the sums and every task's final state, in
/// range order.
fn stream_pairs<F: JoltField, S: Send, const N: usize>(
    rows: &StreamedRows,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[CoeffLut<F>; 2]],
    task_states: impl FnOnce(&[Range<usize>]) -> Result<Vec<S>, PassError>,
    terms: impl Fn(&mut S, &Pair<'_, F>) -> [F; N] + Sync,
) -> Result<([F; N], Vec<S>), PassError> {
    let depth = r.len();
    let group = e_in.len() << (depth + 1);
    let ranges = task_ranges(rows.prefix_end(group, e_out.len() * group), group);
    let reversed: Vec<F> = r.iter().rev().copied().collect();
    let weights = EqPolynomial::<F>::evals(&reversed, None);
    let task = |(range, mut state): (&Range<usize>, S)| -> Result<([F; N], S), WitnessError> {
        let mut merge = RowMerge::new(depth);
        let mut sums = GruenSums::new(e_out, e_in);
        let chunk = VISIT_ROWS.min(group);
        rows.store.visit(range.clone(), chunk, |start, bundles| {
            for (t, row) in (start..).zip(bundles) {
                if let Some(pair) = merge.push(t, row, &weights, r, luts) {
                    sums.add(pair.z, terms(&mut state, &pair));
                }
            }
            Ok(())
        })?;
        Ok((sums.total, state))
    };
    let states = task_states(&ranges)?;
    #[cfg(feature = "parallel")]
    let tasks = ranges
        .par_iter()
        .zip(states)
        .map(task)
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let tasks = ranges
        .iter()
        .zip(states)
        .map(task)
        .collect::<Result<Vec<_>, _>>()?;
    let (sums, states): (Vec<[F; N]>, Vec<S>) = tasks.into_iter().unzip();
    let add = |left: [F; N], right: [F; N]| std::array::from_fn(|i| left[i] + right[i]);
    Ok((sums.into_iter().fold([F::zero(); N], add), states))
}

/// One pair's `[q(0), leading coefficient]` terms, the pair accumulation of
/// `sparse_quadratic` over regenerated entries.
fn quadratic_terms<F: JoltField>(pair: &Pair<'_, F>, unused: &CoeffLut<F>) -> [F; 2] {
    let mut inner = [F::Accumulator::default(); 2];
    accumulate_pair_group(
        pair.evens,
        pair.odds,
        [pair.inc[0], pair.inc[1] - pair.inc[0]],
        &mut inner,
        unused,
        unused,
    );
    inner.map(F::Accumulator::reduce)
}

/// The entries of a pair's bound row: one per column of either row, whatever
/// the coefficients.
fn column_union<F: JoltField>(pair: &Pair<'_, F>) -> usize {
    const _: () = assert!(REGISTER_ADDRESS_BITS <= 7);
    let mut columns = 0u128;
    for entry in pair.evens.iter().chain(pair.odds) {
        columns |= 1u128 << entry.col;
    }
    columns.count_ones() as usize
}

/// Round `r.len()`'s `[q(0), leading coefficient]` and each Gruen group's
/// entries at the next round's level.
pub(super) fn count<F: JoltField>(
    rows: &StreamedRows,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[CoeffLut<F>; 2]],
) -> Result<([F; 2], Vec<usize>), PassError> {
    let unused = CycleState::<F>::unused_lut();
    let group = e_in.len() << (r.len() + 1);
    let mut counts = vec![0; e_out.len()];
    let (quadratic, _) = stream_pairs(
        rows,
        e_in,
        e_out,
        r,
        luts,
        |tasks| {
            let bins = split_at_tasks([&mut counts[..]], tasks, |cycle| cycle / group);
            Ok(bins
                .into_iter()
                .zip(tasks)
                .map(|([bins], task)| (task.start / group, bins))
                .collect())
        },
        |(first, bins), pair| {
            bins[pair.z / e_in.len() - *first] += column_union(pair);
            quadratic_terms(pair, &unused)
        },
    )?;
    Ok((quadratic, counts))
}

/// A filling round's message, its level's entries and their increments.
type Filled<F> = ([F; 2], Vec<SparseEntry<F, F>>, Vec<F>);

/// Round `r.len()`'s `[q(0), leading coefficient]`, filling its level's
/// entries and the increments of its rows that hold a prefix cycle: the last
/// pass's Gruen group `g` owns `counts[g]` consecutive entry slots, each task
/// writes only its own groups' slots, and the entries are published only once
/// every task filled its slots exactly.
pub(super) fn fill<F: JoltField>(
    rows: &StreamedRows,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[CoeffLut<F>; 2]],
    counts: &[usize],
) -> Result<Filled<F>, PassError> {
    let unused = CycleState::<F>::unused_lut();
    let depth = r.len();
    let cycles = (e_out.len() * e_in.len()) << (depth + 1);
    let width = cycles / counts.len();
    let len: usize = counts.iter().sum();
    let mut entries: Vec<SparseEntry<F, F>> = Vec::with_capacity(len);
    let increments = rows.prefix_len.div_ceil(1 << depth);
    let mut inc: Vec<F> = unsafe_allocate_zero_vec(increments);
    let (quadratic, states) = stream_pairs(
        rows,
        e_in,
        e_out,
        r,
        luts,
        |tasks| {
            // Task ranges cover their cycles in order from 0, so whole-group
            // ends keep every task boundary on a group boundary. The groups
            // after the last task must own no slot.
            let first_slot = |cycle: usize| counts[..cycle / width].iter().sum::<usize>();
            let visited = tasks.last().map_or(0, |task| task.end);
            if width * counts.len() != cycles
                || !tasks.iter().all(|task| task.end.is_multiple_of(width))
                || first_slot(visited) != len
            {
                return Err(PassError::Diverged);
            }
            let slots = split_at_tasks(
                [&mut entries.spare_capacity_mut()[..len]],
                tasks,
                first_slot,
            );
            let halves = split_at_tasks([&mut inc[..]], tasks, |cycle| {
                (cycle >> depth).min(increments)
            });
            Ok(slots
                .into_iter()
                .zip(halves)
                .zip(tasks)
                .map(|(([slots], [halves]), task)| {
                    (task.start >> depth, TaskSlots::new(slots), halves)
                })
                .collect())
        },
        |(first, slots, halves), pair| {
            for entry in pair.evens.iter().chain(pair.odds) {
                slots.push(*entry);
            }
            let at = 2 * pair.z - *first;
            for (slot, half) in halves.iter_mut().skip(at).zip(pair.inc) {
                *slot = half;
            }
            quadratic_terms(pair, &unused)
        },
    )?;
    for (_, slots, _) in states {
        slots.finish().map_err(|FillMismatch| PassError::Diverged)?;
    }
    // SAFETY: the tasks' slots tile the first `len` spare slots of `entries`
    // (`split_at_tasks` by the counts, the visited groups owning all `len`
    // slots, checked above), and each task's slots finished with every slot
    // written.
    unsafe {
        entries.set_len(len);
    }
    Ok((quadratic, entries, inc))
}

/// Round `r.len()`'s `q(1)` for a Gruen round whose linear factor vanishes
/// at 1: each odd row's summand `ra·val + wa·(val + inc)`.
pub(super) fn at_one<F: JoltField>(
    rows: &StreamedRows,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[CoeffLut<F>; 2]],
) -> Result<F, PassError> {
    let unused = CycleState::<F>::unused_lut();
    let ([sum], _) = stream_pairs(
        rows,
        e_in,
        e_out,
        r,
        luts,
        |ranges| Ok(vec![(); ranges.len()]),
        |(), pair| {
            let mut lanes = [F::Accumulator::default(); 2];
            for odd in pair.odds {
                <SparseEntry<F, F> as MatrixEntry<F>>::accumulate_pair_evals(
                    Some(odd),
                    None,
                    [pair.inc[1], F::zero()],
                    &mut lanes,
                    &unused,
                    &unused,
                );
            }
            [lanes[0].reduce()]
        },
    )?;
    Ok(sum)
}
