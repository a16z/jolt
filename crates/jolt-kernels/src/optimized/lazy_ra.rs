//! Lazily bound address-folded one-hot selectors for the stage-6b RA
//! virtualization kernels — the legacy `SharedRaPolynomials` /
//! `RaPolynomial` round state machine, generalized over the hot-index
//! source.
//!
//! The direct shape materializes every committed selector dense over the
//! cycle domain at prepare: `N × T` field elements, the stage-6b memory wall
//! at scale (the committed instruction RA family alone is `8 × T`). But an
//! unbound selector column is a point mass — `ra_i(·, j)` is
//! `eq(r_chunk_i, chunk_i(j))`, one scale-table lookup per cycle — and the
//! cycle binds preserve that structure until materialization: after `b`
//! binds the bound value at index `j` is the gather
//!
//! ```text
//! value(i, j) = Σ_{offset < 2^b} branch_tables[i][offset][index(i, j·2^b + offset)]
//! ```
//!
//! where branch table `offset` is the base scale table pre-scaled by that
//! offset's bound-bit eq weight (legacy `SharedRaRound1→2→3` pre-scaling).
//! Pre-scaling keeps the round-loop gathers multiplication-free — one table
//! lookup and one addition per branch — because the eq weights are folded
//! into the `N × 2^b × 2^w` tables at bind time (a few thousand entries)
//! instead of multiplied per cycle. Only the bind reaching
//! `materialize_branches` branches materializes dense vectors, at
//! `T / materialize_branches` length, and releases the branch tables and the
//! source. The stage-6b families take that count from a byte budget per dense
//! polynomial ([`materialize_branches`]); stage 5's read-RAF tables pass
//! theirs explicitly. Every read visits the source in consecutive blocks
//! ([`ChunkIndexSource::visit_blocks`]), never by cycle index.
//!
//! A polynomial whose base table is too large to tensor out (a virtual RA
//! over a wide lookup-index chunk) is served as the product of `factors`
//! point-mass columns: only its lead column's table carries the branch
//! weights, and every branch multiplies in the trailing columns' unscaled
//! entries — `factors − 1` multiplications per branch.
//!
//! Byte parity: every gathered value is the same polynomial of the same
//! table entries and challenges as the iterated `lo + r·(hi − lo)` dense
//! bind — identical monomials, exact field algebra (pre-scaling only
//! reassociates the weight product) — so round messages and output claims
//! are bit-identical. The consumers' in-module parity tests pin this
//! against the naive dense path.

use std::ops::Range;

use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial};
use jolt_sumcheck::SumcheckError;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::WitnessError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::task_ranges;
use crate::SumcheckKernelError;

/// Bytes of each dense polynomial a stage-6b family materializes: at 2^28
/// Fp128 cycles the 99 stage-6b polynomials then hold 6.19 GiB.
pub(crate) const MATERIALIZE_BYTES: usize = 1 << 26;

/// Unbound cycles per source block of a lazy pass; blocks hold whole pairs.
const BLOCK_CYCLES: usize = 1 << 14;

/// Per-cycle hot indices of `N` point-mass columns (one per committed
/// one-hot selector, or `factors` per factored polynomial) over one compact
/// backing row per cycle (typed witness rows, packed columns).
pub(crate) trait ChunkIndexSource: Send + Sync + 'static {
    type Row;

    fn num_columns(&self) -> usize;

    /// The unbound cycle-domain length.
    fn cycles(&self) -> usize;

    /// The scale-table index of column `i`'s hot address in a cycle's row;
    /// `None` when the cycle is cold for that column.
    fn index(&self, i: usize, row: &Self::Row) -> Option<usize>;

    /// Visits `range` in order on the calling thread: callback `k` gets the
    /// first cycle and the `min(block, cycles left)` rows from
    /// `range.start + k·block`; an empty range makes no callback. A source or
    /// visitor error ends the visit and is returned. `visit` must not enter
    /// Rayon.
    fn visit_blocks(
        &self,
        range: Range<usize>,
        block: usize,
        visit: impl FnMut(usize, &[Self::Row]) -> Result<(), WitnessError>,
    ) -> Result<(), WitnessError>;
}

/// `N` address-folded selector polynomials bound `LowToHigh`, lazily until
/// the bind reaching `materialize_branches` branches materializes dense.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: allocative::Allocative")
)]
pub(crate) enum LazyFoldedRa<F: JoltField, S> {
    /// Before materialization: per-column scale tables plus the compact
    /// index source. Polynomial `i` is the product of columns
    /// `i·factors … (i+1)·factors − 1`; its lead column's table holds the
    /// branch tables (the base table pre-scaled by each bound-bit pattern's
    /// eq weight), flattened offset-major — `tables[c][offset · stride_c + k]`
    /// with `stride_c = tables[c].len() / width` — and trailing columns keep
    /// their unscaled base tables.
    Lazy {
        tables: Vec<Vec<F>>,
        /// Bound-bit branch count (`2^binds`).
        width: usize,
        /// The branch count whose bind materializes dense polynomials of
        /// `cycles / materialize_branches` entries; a shorter domain ends
        /// lazy.
        materialize_branches: usize,
        /// Point-mass columns per polynomial.
        factors: usize,
        source: S,
    },
    /// Past materialization: plain dense multilinears
    /// (`T / materialize_branches` at entry).
    Dense(Vec<Polynomial<F>>),
}

impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    /// One scale table per selector polynomial, in polynomial order; dense
    /// polynomials materialize at `materialize_bytes` each at most
    /// ([`materialize_branches`]).
    pub(crate) fn new(tables: Vec<Vec<F>>, source: S, materialize_bytes: usize) -> Self {
        let branches = materialize_branches::<F>(source.cycles(), materialize_bytes);
        Self::factored(tables, 1, branches, source)
    }

    /// One base table per column, in column order; each run of `factors`
    /// consecutive columns is one polynomial. The bind reaching
    /// `materialize_branches` branches (a power of two above one)
    /// materializes dense.
    pub(crate) fn factored(
        tables: Vec<Vec<F>>,
        factors: usize,
        materialize_branches: usize,
        source: S,
    ) -> Self {
        debug_assert_eq!(tables.len(), source.num_columns());
        debug_assert!(factors > 0 && tables.len().is_multiple_of(factors));
        debug_assert!(materialize_branches.is_power_of_two() && materialize_branches > 1);
        Self::Lazy {
            tables,
            width: 1,
            materialize_branches,
            factors,
            source,
        }
    }

    pub(crate) fn num_polys(&self) -> usize {
        match self {
            Self::Lazy {
                tables, factors, ..
            } => tables.len() / factors,
            Self::Dense(polys) => polys.len(),
        }
    }

    /// Visits bound pair rows `rows` serially, a run of whole rows and its
    /// [`PairBlock`] per callback: lazy families read their source
    /// ([`visit_source_rows`]), dense ones make one callback.
    pub(crate) fn visit_rows(
        &self,
        rows: Range<usize>,
        mut visit: impl FnMut(Range<usize>, &PairBlock<'_, F, S>) -> Result<(), WitnessError>,
    ) -> Result<(), WitnessError> {
        match self {
            Self::Lazy {
                tables,
                width,
                factors,
                source,
                ..
            } => visit_source_rows(source, *width, rows, |rows, block| {
                visit(
                    rows,
                    &PairBlock::Lazy {
                        tables,
                        width: *width,
                        factors: *factors,
                        rows: block,
                    },
                )
            }),
            Self::Dense(polys) if !rows.is_empty() => visit(rows, &PairBlock::Dense(polys)),
            Self::Dense(_) => Ok(()),
        }
    }

    /// [`try_fold_groups`] over this family's pairs: `inner_step(inner,
    /// block, row, x_in, e_in)` reads row `row`'s pairs from `block`.
    pub(crate) fn try_fold_out_in<Inner, Outer: Send>(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        make_inner: impl Fn() -> Inner + Sync,
        inner_step: impl Fn(&mut Inner, &PairBlock<'_, F, S>, usize, usize, F) + Sync,
        outer_step: impl Fn(usize, F, Inner) -> Outer + Sync,
        merge: impl Fn(Outer, Outer) -> Outer + Sync,
    ) -> Result<Outer, SumcheckError<F>> {
        try_fold_groups(
            eq,
            make_inner,
            |rows, groups| {
                self.visit_rows(rows, |rows, block| {
                    for row in rows {
                        groups.row(row, |inner, x_in, e_in| {
                            inner_step(inner, block, row, x_in, e_in);
                        });
                    }
                    Ok(())
                })
            },
            outer_step,
            merge,
        )
    }

    /// The fully bound claims, in polynomial order (any state, so short
    /// cycle geometries extract correctly); a lazy family reads cycles
    /// `0..width` in one block.
    pub(crate) fn final_values(&self) -> Result<Vec<F>, SumcheckKernelError<F>> {
        match self {
            Self::Lazy {
                tables,
                width,
                factors,
                source,
                ..
            } => {
                let mut values = Vec::with_capacity(tables.len() / factors);
                source
                    .visit_blocks(0..*width, *width, |first, rows| {
                        let block = SourceBlock {
                            source,
                            first,
                            rows,
                        };
                        values.extend(
                            (0..tables.len() / factors)
                                .map(|i| gather_poly(tables, *factors, *width, &block, i, 0)),
                        );
                        Ok(())
                    })
                    .map_err(|_| SumcheckKernelError::InvariantViolation {
                        reason: "lazy RA index rows re-read failed after the rounds",
                    })?;
                Ok(values)
            }
            Self::Dense(polys) => Ok(polys.iter().map(|poly| poly.evals()[0]).collect()),
        }
    }

    /// Bind the next cycle variable `LowToHigh`: re-scale the branch tables
    /// until the bind reaching `materialize_branches` materializes dense,
    /// then use plain multilinear binds. A materialization failure is
    /// returned and leaves the family lazy at the doubled width.
    pub(crate) fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        match self {
            Self::Lazy {
                tables,
                width,
                materialize_branches,
                factors,
                source,
            } => {
                *tables = double_branches(std::mem::take(tables), *factors, challenge);
                *width *= 2;
                if *width == *materialize_branches {
                    let dense =
                        materialize(tables, *factors, source, *width).map_err(rows_unavailable)?;
                    let log_t = source.cycles().ilog2() as usize;
                    // Drops the branch tables and this source handle here;
                    // other `Arc` owners of the source's rows keep them.
                    *self = Self::Dense(dense);
                    crate::mem::purge_retained_memory(log_t);
                }
            }
            Self::Dense(polys) => {
                for poly in polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
            }
        }
        Ok(())
    }
}

/// The least power-of-two branch count, at least 16, whose dense polynomials
/// of `cycles / branches` field elements fit `bytes` each (`cycles` a power of
/// two); a shorter domain never materializes.
pub(crate) fn materialize_branches<F>(cycles: usize, bytes: usize) -> usize {
    let entries = (bytes / size_of::<F>()).max(1);
    cycles.div_ceil(entries).next_power_of_two().max(16)
}

/// [`ChunkIndexSource::visit_blocks`] over rows held in memory.
pub(crate) fn visit_resident<R>(
    rows: &[R],
    range: Range<usize>,
    block: usize,
    mut visit: impl FnMut(usize, &[R]) -> Result<(), WitnessError>,
) -> Result<(), WitnessError> {
    range
        .clone()
        .step_by(block)
        .try_for_each(|first| visit(first, &rows[first..range.end.min(first + block)]))
}

/// Unbound cycles `first .. first + rows.len()`, as one source callback
/// delivered them.
pub(crate) struct SourceBlock<'a, S: ChunkIndexSource> {
    source: &'a S,
    first: usize,
    rows: &'a [S::Row],
}

impl<S: ChunkIndexSource> SourceBlock<'_, S> {
    /// Column `i`'s hot index at unbound cycle `j` of this block.
    #[inline]
    pub(crate) fn index(&self, i: usize, j: usize) -> Option<usize> {
        self.source.index(i, &self.rows[j - self.first])
    }
}

/// Visits bound pair rows `rows` of a lazy family at width `width` serially,
/// a run of whole rows and the source block of their unbound cycles (at most
/// `max(BLOCK_CYCLES, 2·width)`) per callback.
pub(crate) fn visit_source_rows<S: ChunkIndexSource>(
    source: &S,
    width: usize,
    rows: Range<usize>,
    mut visit: impl FnMut(Range<usize>, &SourceBlock<'_, S>) -> Result<(), WitnessError>,
) -> Result<(), WitnessError> {
    let pair = 2 * width;
    source.visit_blocks(
        rows.start * pair..rows.end * pair,
        BLOCK_CYCLES.max(pair),
        |first, block| {
            let rows = first / pair..(first + block.len()) / pair;
            visit(
                rows,
                &SourceBlock {
                    source,
                    first,
                    rows: block,
                },
            )
        },
    )
}

/// The pairs of one run of bound rows in the family's current state.
pub(crate) enum PairBlock<'a, F: JoltField, S: ChunkIndexSource> {
    Lazy {
        tables: &'a [Vec<F>],
        width: usize,
        factors: usize,
        rows: &'a SourceBlock<'a, S>,
    },
    Dense(&'a [Polynomial<F>]),
}

impl<F: JoltField, S: ChunkIndexSource> PairBlock<'_, F, S> {
    /// Polynomial `i`'s `(lo, hi)` values at bound indices `2·row` and
    /// `2·row + 1`.
    #[inline]
    pub(crate) fn lo_hi(&self, i: usize, row: usize) -> (F, F) {
        match self {
            Self::Lazy {
                tables,
                width,
                factors,
                rows,
            } => (
                gather_poly(tables, *factors, *width, rows, i, 2 * row),
                gather_poly(tables, *factors, *width, rows, i, 2 * row + 1),
            ),
            Self::Dense(polys) => {
                let evals = polys[i].evals();
                (evals[2 * row], evals[2 * row + 1])
            }
        }
    }

    /// All polynomials' `(lo, hi)` pairs at `row`, into `out` (length
    /// `num_polys`). One state dispatch per row instead of `2N`, with
    /// per-polynomial table slices hoisted out of the gather loop — the
    /// round-message hot path.
    #[inline]
    pub(crate) fn lo_hi_all(&self, row: usize, out: &mut [(F, F)]) {
        match self {
            Self::Lazy {
                tables,
                width,
                factors,
                rows,
            } => {
                let (width, factors) = (*width, *factors);
                if factors == 1 {
                    for (i, (out, table)) in out.iter_mut().zip(tables.iter()).enumerate() {
                        *out = (
                            gather(table, width, rows, i, 2 * row),
                            gather(table, width, rows, i, 2 * row + 1),
                        );
                    }
                } else {
                    for (i, (out, tables)) in
                        out.iter_mut().zip(tables.chunks_exact(factors)).enumerate()
                    {
                        let column = i * factors;
                        *out = (
                            gather_product(tables, width, rows, column, 2 * row),
                            gather_product(tables, width, rows, column, 2 * row + 1),
                        );
                    }
                }
            }
            Self::Dense(polys) => {
                for (out, poly) in out.iter_mut().zip(polys.iter()) {
                    let evals = poly.evals();
                    *out = (evals[2 * row], evals[2 * row + 1]);
                }
            }
        }
    }
}

/// One task's walk over consecutive bound rows of whole `x_out` groups: each
/// row folds into its group's inner accumulator with its `e_in` weight, and a
/// group's last row closes it into the task total.
pub(crate) struct GruenGroups<'a, F, Inner, Outer> {
    e_out: &'a [F],
    e_in: &'a [F],
    make_inner: &'a dyn Fn() -> Inner,
    outer_step: &'a dyn Fn(usize, F, Inner) -> Outer,
    merge: &'a dyn Fn(Outer, Outer) -> Outer,
    inner: Inner,
    total: Option<Outer>,
}

impl<F: JoltField, Inner, Outer> GruenGroups<'_, F, Inner, Outer> {
    /// Folds bound row `row`; rows arrive in order (tasks hold whole groups).
    #[inline]
    pub(crate) fn row(&mut self, row: usize, step: impl FnOnce(&mut Inner, usize, F)) {
        let x_in = row & (self.e_in.len() - 1);
        step(&mut self.inner, x_in, self.e_in[x_in]);
        if x_in + 1 == self.e_in.len() {
            let x_out = row / self.e_in.len();
            let inner = std::mem::replace(&mut self.inner, (self.make_inner)());
            let value = (self.outer_step)(x_out, self.e_out[x_out], inner);
            self.total = Some(match self.total.take() {
                Some(total) => (self.merge)(total, value),
                None => value,
            });
        }
    }
}

/// `par_fold_out_in` over at most one task per pool thread, each holding whole
/// `x_out` groups ([`task_ranges`]): `task(rows, groups)` walks its rows once,
/// serially, through [`GruenGroups::row`]. Each group keeps its inner row
/// order; group values merge in another exact order. Source failures return.
pub(crate) fn try_fold_groups<F: JoltField, Inner, Outer: Send>(
    eq: &GruenSplitEqPolynomial<F>,
    make_inner: impl Fn() -> Inner + Sync,
    task: impl Fn(Range<usize>, &mut GruenGroups<'_, F, Inner, Outer>) -> Result<(), WitnessError>
        + Sync,
    outer_step: impl Fn(usize, F, Inner) -> Outer + Sync,
    merge: impl Fn(Outer, Outer) -> Outer + Sync,
) -> Result<Outer, SumcheckError<F>> {
    let (e_out, e_in) = (eq.e_out_current(), eq.e_in_current());
    let ranges = task_ranges(e_out.len() * e_in.len(), 1, e_in.len());
    let run = |rows: &Range<usize>| -> Result<Option<Outer>, WitnessError> {
        let mut groups = GruenGroups {
            e_out,
            e_in,
            make_inner: &make_inner,
            outer_step: &outer_step,
            merge: &merge,
            inner: make_inner(),
            total: None,
        };
        task(rows.clone(), &mut groups)?;
        Ok(groups.total)
    };
    let join = |left: Option<Outer>, right: Option<Outer>| match (left, right) {
        (Some(left), Some(right)) => Some(merge(left, right)),
        (left, None) => left,
        (None, right) => right,
    };
    #[cfg(feature = "parallel")]
    let total = ranges
        .par_iter()
        .map(run)
        .try_reduce(|| None, |left, right| Ok(join(left, right)));
    #[cfg(not(feature = "parallel"))]
    let total = ranges.iter().map(run).try_fold(None, |total, result| {
        Ok::<_, WitnessError>(join(total, result?))
    });
    Ok(total
        .map_err(rows_unavailable)?
        .unwrap_or_else(|| outer_step(0, F::zero(), make_inner())))
}

/// A lazy family's source failed inside a round, after `prepare` read it.
pub(crate) fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "lazy RA index rows",
    }
}

/// Polynomial `i`'s gather at unbound width `width`.
#[inline]
fn gather_poly<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    factors: usize,
    width: usize,
    rows: &SourceBlock<'_, S>,
    i: usize,
    j: usize,
) -> F {
    if factors == 1 {
        gather(&tables[i], width, rows, i, j)
    } else {
        let column = i * factors;
        gather_product(&tables[column..column + factors], width, rows, column, j)
    }
}

/// The eq-weighted branch gather at unbound width `width`: one lookup and
/// one add per hot branch, no multiplications (the weights are pre-scaled
/// into the branch tables).
#[inline]
fn gather<F: JoltField, S: ChunkIndexSource>(
    table: &[F],
    width: usize,
    rows: &SourceBlock<'_, S>,
    i: usize,
    j: usize,
) -> F {
    if width == 1 {
        return rows.index(i, j).map_or_else(F::zero, |k| table[k]);
    }
    let stride = table.len() / width;
    let mut sum = F::zero();
    let mut base = 0;
    for offset in 0..width {
        if let Some(k) = rows.index(i, j * width + offset) {
            sum += table[base + k];
        }
        base += stride;
    }
    sum
}

#[inline(never)]
fn gather_product<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    width: usize,
    rows: &SourceBlock<'_, S>,
    column: usize,
    j: usize,
) -> F {
    let (lead, middle, last) = (
        &tables[0],
        &tables[1..tables.len() - 1],
        &tables[tables.len() - 1],
    );
    let last_column = column + tables.len() - 1;
    let stride = lead.len() / width;
    let mut sum = F::Accumulator::default();
    'branches: for offset in 0..width {
        let cycle = j * width + offset;
        let Some(k) = rows.index(column, cycle) else {
            continue;
        };
        let mut term = lead[offset * stride + k];
        for (next, table) in middle.iter().enumerate() {
            let Some(k) = rows.index(column + 1 + next, cycle) else {
                continue 'branches;
            };
            term *= table[k];
        }
        if let Some(k) = rows.index(last_column, cycle) {
            sum.fmadd(term, last[k]);
        }
    }
    sum.reduce()
}

/// Doubles every polynomial's branch set for the next bound bit: the first
/// half keeps the existing branches scaled by `1 − challenge` (bit 0), the
/// second half by `challenge` (bit 1) — offset layout
/// `b0 + 2·b1 + 4·b2`, matching the low bits of the original cycle index.
/// Only lead columns carry branches; trailing columns pass through.
fn double_branches<F: JoltField>(tables: Vec<Vec<F>>, factors: usize, challenge: F) -> Vec<Vec<F>> {
    let one_minus = F::one() - challenge;
    let double = |(column, table): (usize, Vec<F>)| -> Vec<F> {
        if !column.is_multiple_of(factors) {
            return table;
        }
        let mut next = Vec::with_capacity(table.len() * 2);
        next.extend(table.iter().map(|value| one_minus * *value));
        next.extend(table.iter().map(|value| challenge * *value));
        next
    };
    #[cfg(feature = "parallel")]
    {
        tables.into_par_iter().enumerate().map(double).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        tables.into_iter().enumerate().map(double).collect()
    }
}

/// The switching bind's materialization: gather every polynomial dense at
/// `cycles / branches` length through the pre-scaled branch tables —
/// lookups and adds only. Each output is allocated once at its final length
/// and split at the task boundaries; every task visits its cycles once,
/// serially, writing all polynomials' entries in place.
fn materialize<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    factors: usize,
    source: &S,
    branches: usize,
) -> Result<Vec<Polynomial<F>>, WitnessError> {
    let len = source.cycles() / branches;
    let mut outputs: Vec<Vec<F>> = (0..tables.len() / factors)
        .map(|_| unsafe_allocate_zero_vec(len))
        .collect();
    let ranges = task_ranges(len, 1, 1);
    let mut slices: Vec<Vec<&mut [F]>> = ranges
        .iter()
        .map(|_| Vec::with_capacity(outputs.len()))
        .collect();
    for output in &mut outputs {
        let mut rest = output.as_mut_slice();
        for (range, task) in ranges.iter().zip(&mut slices) {
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(range.len());
            task.push(head);
            rest = tail;
        }
    }
    let fill = |(range, mut slices): (&Range<usize>, Vec<&mut [F]>)| {
        source.visit_blocks(
            range.start * branches..range.end * branches,
            BLOCK_CYCLES.max(branches),
            |first, rows| {
                let block = SourceBlock {
                    source,
                    first,
                    rows,
                };
                for j in first / branches..(first + rows.len()) / branches {
                    for (i, out) in slices.iter_mut().enumerate() {
                        out[j - range.start] = gather_poly(tables, factors, branches, &block, i, j);
                    }
                }
                Ok(())
            },
        )
    };
    #[cfg(feature = "parallel")]
    ranges.par_iter().zip(slices).try_for_each(fill)?;
    #[cfg(not(feature = "parallel"))]
    ranges.iter().zip(slices).try_for_each(fill)?;
    Ok(outputs.into_iter().map(Polynomial::new).collect())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::ops::Range;

    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField};
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial};
    use jolt_sumcheck::SumcheckError;
    use jolt_witness::WitnessError;

    use super::{
        materialize_branches, visit_resident, ChunkIndexSource, LazyFoldedRa, MATERIALIZE_BYTES,
    };
    use crate::SumcheckKernelError;

    const ADDRESSES: usize = 4;

    /// Each cycle's hot index per column; `None` is cold.
    struct Rows(Vec<Vec<Option<usize>>>);

    impl ChunkIndexSource for Rows {
        type Row = Vec<Option<usize>>;

        fn num_columns(&self) -> usize {
            self.0[0].len()
        }
        fn cycles(&self) -> usize {
            self.0.len()
        }
        fn index(&self, i: usize, row: &Self::Row) -> Option<usize> {
            row[i]
        }
        fn visit_blocks(
            &self,
            range: Range<usize>,
            block: usize,
            visit: impl FnMut(usize, &[Self::Row]) -> Result<(), WitnessError>,
        ) -> Result<(), WitnessError> {
            visit_resident(&self.0, range, block, visit)
        }
    }

    /// `family` over `2^log_t` cycles (`columns` columns, `factors` per
    /// polynomial) against directly bound dense products: every round's pairs,
    /// the materializing bind (`2^(log_t − bind)` entries, never earlier) and
    /// the final claims.
    fn check_family<F: JoltField>(
        log_t: usize,
        columns: usize,
        factors: usize,
        materialize_at: Option<usize>,
        family: impl FnOnce(Vec<Vec<F>>, Rows) -> LazyFoldedRa<F, Rows>,
    ) {
        let rows = Rows(
            (0..1usize << log_t)
                .map(|cycle| {
                    (0..columns)
                        .map(|column| {
                            (column == 0 || !(cycle + column).is_multiple_of(3))
                                .then_some((cycle * cycle + 3 * cycle + 5 * column) % ADDRESSES)
                        })
                        .collect()
                })
                .collect(),
        );
        let tables: Vec<Vec<F>> = (0..columns)
            .map(|column| {
                (0..ADDRESSES)
                    .map(|k| F::from_u64((11 * column + 3 * k + 2) as u64))
                    .collect()
            })
            .collect();
        let mut dense: Vec<Polynomial<F>> = (0..columns / factors)
            .map(|poly| {
                let value = |row: &Vec<Option<usize>>| {
                    (poly * factors..(poly + 1) * factors).fold(F::one(), |product, column| {
                        product * row[column].map_or(F::zero(), |k| tables[column][k])
                    })
                };
                Polynomial::new(rows.0.iter().map(value).collect())
            })
            .collect();
        let mut lazy = family(tables, rows);
        let mut pairs = vec![(F::zero(), F::zero()); dense.len()];
        for round in 0..=log_t {
            if round > 0 {
                let challenge = match round {
                    1 => F::zero(),
                    2 => F::one(),
                    _ => F::from_u64(7 + 31 * round as u64),
                };
                lazy.bind(challenge).unwrap();
                for poly in &mut dense {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
            }
            match &lazy {
                LazyFoldedRa::Dense(polys) => {
                    let bind = materialize_at.unwrap();
                    assert!(round >= bind, "materialized at round {round}");
                    if round == bind {
                        assert!(polys.iter().all(|poly| poly.len() == 1 << (log_t - bind)));
                    }
                }
                LazyFoldedRa::Lazy { .. } => {
                    assert!(
                        materialize_at.is_none_or(|bind| round < bind),
                        "round {round}"
                    );
                }
            }
            let half = dense[0].len() / 2;
            for start in (0..half).step_by(3) {
                lazy.visit_rows(start..half.min(start + 3), |rows, block| {
                    for row in rows {
                        block.lo_hi_all(row, &mut pairs);
                        for (pair, poly) in pairs.iter().zip(&dense) {
                            let expected = (poly.evals()[2 * row], poly.evals()[2 * row + 1]);
                            assert_eq!(*pair, expected, "round {round}, row {row}");
                        }
                    }
                    Ok(())
                })
                .unwrap();
            }
        }
        let finals: Vec<F> = dense.iter().map(|poly| poly.evals()[0]).collect();
        assert_eq!(lazy.final_values().unwrap(), finals);
    }

    fn check_families<F: JoltField>() {
        for log_t in [8, 10] {
            let bytes = (size_of::<F>() << log_t) / 64;
            check_family::<F>(log_t, 3, 1, Some(6), |tables, rows| {
                LazyFoldedRa::new(tables, rows, bytes)
            });
            check_family::<F>(log_t, 4, 2, Some(6), |tables, rows| {
                LazyFoldedRa::factored(tables, 2, 64, rows)
            });
        }
        check_family::<F>(6, 4, 2, Some(3), |tables, rows| {
            LazyFoldedRa::factored(tables, 2, 8, rows)
        });
        check_family::<F>(4, 3, 1, Some(4), |tables, rows| {
            LazyFoldedRa::new(tables, rows, MATERIALIZE_BYTES)
        });
        check_family::<F>(3, 3, 1, None, |tables, rows| {
            LazyFoldedRa::new(tables, rows, MATERIALIZE_BYTES)
        });
    }

    #[test]
    fn lazy_pairs_match_dense_binds() {
        check_families::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn lazy_pairs_match_dense_binds_fp128() {
        check_families::<Prime128OffsetA7F7>();
    }

    /// Fails each visit after its first callback.
    struct Failing(Rows);

    impl ChunkIndexSource for Failing {
        type Row = Vec<Option<usize>>;

        fn num_columns(&self) -> usize {
            self.0.num_columns()
        }
        fn cycles(&self) -> usize {
            self.0.cycles()
        }
        fn index(&self, i: usize, row: &Self::Row) -> Option<usize> {
            row[i]
        }
        fn visit_blocks(
            &self,
            range: Range<usize>,
            block: usize,
            mut visit: impl FnMut(usize, &[Self::Row]) -> Result<(), WitnessError>,
        ) -> Result<(), WitnessError> {
            visit_resident(&self.0 .0, range, block, |first, rows| {
                visit(first, rows)?;
                Err(WitnessError::Replay {
                    row: first + rows.len(),
                    reason: "injected".to_owned(),
                })
            })
        }
    }

    /// A failing source fails the round fold, the materializing bind (which
    /// leaves the family lazy) and the final values, each with a typed error.
    fn check_failures<F: JoltField>() {
        let rows = Rows((0..16).map(|cycle| vec![Some(cycle % ADDRESSES)]).collect());
        let tables = vec![(0..ADDRESSES).map(|k| F::from_u64(k as u64 + 1)).collect()];
        let mut lazy = LazyFoldedRa::factored(tables, 1, 4, Failing(rows));
        let eq = GruenSplitEqPolynomial::new(&[F::from_u64(3); 4], BindingOrder::LowToHigh);
        let fold = lazy.try_fold_out_in(
            &eq,
            F::zero,
            |_, _, _, _, _| {},
            |_, _, inner| inner,
            |a, b| a + b,
        );
        assert!(matches!(
            fold,
            Err(SumcheckError::MissingEvaluationSource { .. })
        ));
        lazy.bind(F::from_u64(2)).unwrap();
        let bind = lazy.bind(F::from_u64(5));
        assert!(matches!(
            bind,
            Err(SumcheckError::MissingEvaluationSource { .. })
        ));
        assert!(matches!(lazy, LazyFoldedRa::Lazy { width: 4, .. }));
        let finals = lazy.final_values();
        assert!(matches!(
            finals,
            Err(SumcheckKernelError::InvariantViolation { .. })
        ));
    }

    #[test]
    fn source_failures_return_typed_errors() {
        check_failures::<Fr>();
    }

    /// 64 MiB per dense polynomial: 16 branches up to 2^26 Fp128 cycles, 64
    /// at the 2^28 target (Fr's 32-byte elements double both).
    #[test]
    fn materialize_branches_fit_the_byte_budget() {
        assert_eq!(materialize_branches::<Fr>(1 << 28, MATERIALIZE_BYTES), 128);
        assert_eq!(materialize_branches::<Fr>(1 << 26, MATERIALIZE_BYTES), 32);
        assert_eq!(materialize_branches::<Fr>(1 << 25, MATERIALIZE_BYTES), 16);
        assert_eq!(materialize_branches::<Fr>(1 << 3, MATERIALIZE_BYTES), 16);
        assert_eq!(materialize_branches::<Fr>(1 << 8, 1), 256);
        #[cfg(feature = "akita")]
        {
            let branches = materialize_branches::<Prime128OffsetA7F7>;
            assert_eq!(branches(1 << 28, MATERIALIZE_BYTES), 64);
            assert_eq!(branches(1 << 27, MATERIALIZE_BYTES), 32);
            assert_eq!(branches(1 << 26, MATERIALIZE_BYTES), 16);
        }
    }
}
