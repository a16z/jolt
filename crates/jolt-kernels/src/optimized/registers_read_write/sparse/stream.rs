//! Cycle rounds 0 and 1 from the witness's row handle: each pass regenerates
//! the round's level instead of reading stored entries. The cycle domain
//! splits into [`task_ranges`] of whole Gruen groups, each visited once,
//! serially; a task merges its rows' cells ([`RegisterCycleRow::entries`]) in
//! row order with the existing seed bind, and every completed pair reaches the
//! task's [`GruenSums`].

use std::ops::Range;

use jolt_field::{Accumulator, JoltField};
use jolt_poly::EqPolynomial;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::layout::{merge_bind, MatrixEntry, SeedEntry, SparseEntry};
use super::ops::accumulate_pair_group;
use super::{CoeffLut, CycleState};
use crate::optimized::registers_read_write::rows::{raw_rd_inc, RegisterCycleRow};
use crate::optimized::support::{task_ranges, GruenSums, KernelRows, PairHalves, VISIT_ROWS};

/// One completed pair of the round's level: rows `2z` and `2z + 1`, in field
/// form, with the increment's halves `[inc(2z), inc(2z + 1)]` at that level.
struct Pair<'a, F> {
    z: usize,
    evens: &'a [SparseEntry<F, F>],
    odds: &'a [SparseEntry<F, F>],
    inc: [F; 2],
}

/// One task's merge of consecutive rows into level-`depth` pairs, `depth ≤ 1`.
struct RowMerge<F: JoltField> {
    depth: usize,
    /// The even level-0 row a level-1 row waits on.
    leaf: ([SeedEntry; 3], usize),
    evens: Vec<SparseEntry<F, F>>,
    odds: Vec<SparseEntry<F, F>>,
    halves: PairHalves<F, 1>,
}

impl<F: JoltField> RowMerge<F> {
    fn new(depth: usize) -> Self {
        let width = 3 << depth;
        Self {
            depth,
            leaf: RegisterCycleRow::default().entries(0),
            evens: Vec::with_capacity(width),
            odds: Vec::with_capacity(width),
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
        luts: &[[&CoeffLut<F>; 2]],
    ) -> Option<Pair<'_, F>> {
        let weight = weights[self.halves.offset(t)];
        let inc = self.halves.add(t, [weight.mul_i128(raw_rd_inc(row))]);
        let (cells, len) = row.entries(t as u32);
        if self.depth == 1 && t & 1 == 0 {
            self.leaf = (cells, len);
            return None;
        }
        let half = if (t >> self.depth) & 1 == 0 {
            &mut self.evens
        } else {
            &mut self.odds
        };
        half.clear();
        if self.depth == 0 {
            let [ra, wa] = luts[0];
            half.extend(cells[..len].iter().map(|cell| cell.to_field(ra, wa)));
        } else {
            let ([seed_ra, seed_wa], [ra, wa]) = (luts[0], luts[1]);
            merge_bind(
                &self.leaf.0[..self.leaf.1],
                &cells[..len],
                &|even, odd| SeedEntry::bind(even, odd, r[0], seed_ra, seed_wa),
                |entry| half.push(entry.to_field(ra, wa)),
            );
        }
        let (z, [inc]) = inc?;
        Some(Pair {
            z,
            evens: &self.evens,
            odds: &self.odds,
            inc,
        })
    }
}

/// Round `r.len()`'s Gruen sums `Σ_z E(z) · terms(pair z)` over its level's
/// pairs. Every pair reaches its task's sums, an empty one too, so each Gruen
/// group closes.
fn stream_pairs<F: JoltField, const N: usize>(
    rows: &KernelRows<RegisterCycleRow>,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[&CoeffLut<F>; 2]],
    terms: impl Fn(&Pair<'_, F>) -> [F; N] + Sync,
) -> Result<[F; N], WitnessError> {
    let depth = r.len();
    let group = e_in.len() << (depth + 1);
    let ranges = task_ranges(e_out.len() * group, group);
    let reversed: Vec<F> = r.iter().rev().copied().collect();
    let weights = EqPolynomial::<F>::evals(&reversed, None);
    let task = |range: &Range<usize>| -> Result<[F; N], WitnessError> {
        let mut merge = RowMerge::new(depth);
        let mut sums = GruenSums::new(e_out, e_in);
        rows.visit(range.clone(), VISIT_ROWS.min(group), |start, bundles| {
            for (t, row) in (start..).zip(bundles) {
                if let Some(pair) = merge.push(t, row, &weights, r, luts) {
                    sums.add(pair.z, terms(&pair));
                }
            }
            Ok(())
        })?;
        Ok(sums.total)
    };
    let add = |left: [F; N], right: [F; N]| std::array::from_fn(|i| left[i] + right[i]);
    #[cfg(feature = "parallel")]
    return ranges
        .par_iter()
        .map(task)
        .try_reduce(|| [F::zero(); N], |left, right| Ok(add(left, right)));
    #[cfg(not(feature = "parallel"))]
    ranges
        .iter()
        .try_fold([F::zero(); N], |sum, range| Ok(add(sum, task(range)?)))
}

/// Round `r.len()`'s `[q(0), leading coefficient]`, the pair accumulation of
/// `sparse_quadratic` over regenerated entries.
pub(super) fn quadratic<F: JoltField>(
    rows: &KernelRows<RegisterCycleRow>,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[&CoeffLut<F>; 2]],
) -> Result<[F; 2], WitnessError> {
    let unused = CycleState::<F>::unused_lut();
    stream_pairs(rows, e_in, e_out, r, luts, |pair| {
        let mut inner = [F::Accumulator::default(); 2];
        accumulate_pair_group(
            pair.evens,
            pair.odds,
            [pair.inc[0], pair.inc[1] - pair.inc[0]],
            &mut inner,
            &unused,
            &unused,
        );
        inner.map(F::Accumulator::reduce)
    })
}

/// Round `r.len()`'s `q(1)` for a Gruen round whose linear factor vanishes
/// at 1: each odd row's summand `ra·val + wa·(val + inc)`.
pub(super) fn at_one<F: JoltField>(
    rows: &KernelRows<RegisterCycleRow>,
    e_in: &[F],
    e_out: &[F],
    r: &[F],
    luts: &[[&CoeffLut<F>; 2]],
) -> Result<F, WitnessError> {
    let unused = CycleState::<F>::unused_lut();
    let [sum] = stream_pairs(rows, e_in, e_out, r, luts, |pair| {
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
    })?;
    Ok(sum)
}
