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
//! first cycle binds preserve that structure: after `b < 4` binds the bound
//! value at index `j` is the gather
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
//! instead of multiplied per cycle. Only the fourth bind materializes dense
//! vectors, at `T/16` length, and drops the index source. Peak memory falls
//! from `N·T` field elements to the index source plus `N·T/16`. A consumer
//! whose base tables are too large to stay cache-resident as they double
//! materializes a bind earlier, at `T/8` (`max_width`).
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

use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The default widest lazy branch set: the fourth bind materializes `T/16`.
const DEFAULT_MAX_WIDTH: usize = 8;

/// Per-cycle hot indices of `N` point-mass columns (one per committed
/// one-hot selector, or `factors` per factored polynomial) over a shared
/// compact backing store (typed witness rows, packed columns).
pub(crate) trait ChunkIndexSource: Send + Sync {
    /// Number of point-mass columns served.
    fn num_columns(&self) -> usize;

    /// The unbound cycle-domain length.
    fn cycles(&self) -> usize;

    /// The scale-table index of column `i`'s hot address at unbound cycle
    /// `j`; `None` when the cycle is cold for that column.
    fn index(&self, i: usize, j: usize) -> Option<usize>;
}

/// `N` address-folded selector polynomials bound `LowToHigh`, lazily until
/// the bind past `max_width` branches (by default the fourth) materializes
/// dense.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: allocative::Allocative")
)]
pub(crate) enum LazyFoldedRa<F: JoltField, S> {
    /// Up to `log2(max_width)` binds: per-column scale tables plus the compact
    /// index source. Polynomial `i` is the product of columns
    /// `i·factors … (i+1)·factors − 1`; its lead column's table holds the
    /// branch tables (the base table pre-scaled by each bound-bit pattern's
    /// eq weight), flattened offset-major — `tables[c][offset · stride_c + k]`
    /// with `stride_c = tables[c].len() / width` — and trailing columns keep
    /// their unscaled base tables.
    Lazy {
        tables: Vec<Vec<F>>,
        /// Bound-bit branch count (`2^binds`: 1, 2, …, `max_width`).
        width: usize,
        /// The widest lazy branch set; the next bind materializes dense.
        max_width: usize,
        /// Point-mass columns per polynomial.
        factors: usize,
        source: S,
    },
    /// Past `max_width`: plain dense multilinears (`T/(2·max_width)` at
    /// entry).
    Dense(Vec<Polynomial<F>>),
}

impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    /// One scale table per selector polynomial, in polynomial order; the
    /// fourth bind materializes dense.
    pub(crate) fn new(tables: Vec<Vec<F>>, source: S) -> Self {
        Self::factored(tables, 1, DEFAULT_MAX_WIDTH, source)
    }

    /// One base table per column, in column order; each run of `factors`
    /// consecutive columns is one polynomial. The bind past branch width
    /// `max_width` (a power of two) materializes dense.
    pub(crate) fn factored(
        tables: Vec<Vec<F>>,
        factors: usize,
        max_width: usize,
        source: S,
    ) -> Self {
        debug_assert_eq!(tables.len(), source.num_columns());
        debug_assert!(factors > 0 && tables.len().is_multiple_of(factors));
        debug_assert!(max_width.is_power_of_two());
        Self::Lazy {
            tables,
            width: 1,
            max_width,
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

    /// The current (bound) evaluation of polynomial `i` at index `j` —
    /// exactly the value a dense representation would hold after the same
    /// binds.
    #[inline]
    pub(crate) fn value(&self, i: usize, j: usize) -> F {
        match self {
            Self::Lazy {
                tables,
                width,
                factors,
                source,
                ..
            } => gather_poly(tables, *factors, *width, source, i, j),
            Self::Dense(polys) => polys[i].evals()[j],
        }
    }

    /// The `(lo, hi) = (value(i, 2·row), value(i, 2·row + 1))` pair the
    /// round messages consume.
    #[inline]
    pub(crate) fn lo_hi(&self, i: usize, row: usize) -> (F, F) {
        (self.value(i, 2 * row), self.value(i, 2 * row + 1))
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
                source,
                ..
            } => {
                let (width, factors) = (*width, *factors);
                if factors == 1 {
                    for (i, (out, table)) in out.iter_mut().zip(tables).enumerate() {
                        *out = (
                            gather(table, width, source, i, 2 * row),
                            gather(table, width, source, i, 2 * row + 1),
                        );
                    }
                } else {
                    for (i, (out, tables)) in
                        out.iter_mut().zip(tables.chunks_exact(factors)).enumerate()
                    {
                        let column = i * factors;
                        *out = (
                            gather_product(tables, width, source, column, 2 * row),
                            gather_product(tables, width, source, column, 2 * row + 1),
                        );
                    }
                }
            }
            Self::Dense(polys) => {
                for (out, poly) in out.iter_mut().zip(polys) {
                    let evals = poly.evals();
                    *out = (evals[2 * row], evals[2 * row + 1]);
                }
            }
        }
    }

    /// The fully bound claims, in polynomial order (any state, so short
    /// cycle geometries extract correctly).
    pub(crate) fn final_values(&self) -> Vec<F> {
        (0..self.num_polys()).map(|i| self.value(i, 0)).collect()
    }

    /// Bind the next cycle variable `LowToHigh`: re-scale the branch tables
    /// until the bind past `max_width` materializes dense (and drops the
    /// source), then use plain multilinear binds.
    pub(crate) fn bind(&mut self, challenge: F) {
        *self = match std::mem::replace(self, Self::Dense(Vec::new())) {
            Self::Lazy {
                tables,
                width,
                max_width,
                factors,
                source,
            } => {
                let tables = double_branches(tables, factors, challenge);
                if width < max_width {
                    Self::Lazy {
                        tables,
                        width: width * 2,
                        max_width,
                        factors,
                        source,
                    }
                } else {
                    let log_t = source.cycles().ilog2() as usize;
                    let dense = Self::Dense(materialize(&tables, factors, &source, width * 2));
                    // Return branch tables and the final shared index handle.
                    drop(tables);
                    drop(source);
                    crate::mem::purge_retained_memory(log_t);
                    dense
                }
            }
            Self::Dense(mut polys) => {
                for poly in &mut polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                Self::Dense(polys)
            }
        };
    }
}

/// Polynomial `i`'s gather at unbound width `width`.
#[inline]
fn gather_poly<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    factors: usize,
    width: usize,
    source: &S,
    i: usize,
    j: usize,
) -> F {
    if factors == 1 {
        gather(&tables[i], width, source, i, j)
    } else {
        let column = i * factors;
        gather_product(&tables[column..column + factors], width, source, column, j)
    }
}

/// The eq-weighted branch gather at unbound width `width`: one lookup and
/// one add per hot branch, no multiplications (the weights are pre-scaled
/// into the branch tables).
#[inline]
fn gather<F: JoltField, S: ChunkIndexSource>(
    table: &[F],
    width: usize,
    source: &S,
    i: usize,
    j: usize,
) -> F {
    if width == 1 {
        return source.index(i, j).map_or_else(F::zero, |k| table[k]);
    }
    let stride = table.len() / width;
    let mut sum = F::zero();
    let mut base = 0;
    for offset in 0..width {
        if let Some(k) = source.index(i, j * width + offset) {
            sum += table[base + k];
        }
        base += stride;
    }
    sum
}

/// A factored polynomial's gather (its column `tables`, lead column
/// `column`): per hot branch, the pre-scaled lead entry times the trailing
/// columns' unscaled entries, the last product accumulated unreduced. Kept
/// out of line so single-factor gathers inline as small as before.
#[inline(never)]
fn gather_product<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    width: usize,
    source: &S,
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
        let Some(k) = source.index(column, cycle) else {
            continue;
        };
        let mut term = lead[offset * stride + k];
        for (next, table) in middle.iter().enumerate() {
            let Some(k) = source.index(column + 1 + next, cycle) else {
                continue 'branches;
            };
            term *= table[k];
        }
        if let Some(k) = source.index(last_column, cycle) {
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
/// lookups and adds only. The switch depth trades the dense tables'
/// footprint (`N · T / branches` field elements — the stage-6b peak at
/// large T) against one more gather round and double the branch tables;
/// measured on a 64-thread host, T/16 beats the original T/8 on both axes.
fn materialize<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    factors: usize,
    source: &S,
    branches: usize,
) -> Vec<Polynomial<F>> {
    debug_assert!(source.cycles() >= branches);
    let new_len = source.cycles() / branches;
    let materialize_poly = |i: usize| -> Polynomial<F> {
        let eval = |j: usize| gather_poly(tables, factors, branches, source, i, j);
        #[cfg(feature = "parallel")]
        let evals: Vec<F> = (0..new_len).into_par_iter().map(eval).collect();
        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = (0..new_len).map(eval).collect();
        Polynomial::new(evals)
    };
    let polys = tables.len() / factors;
    #[cfg(feature = "parallel")]
    {
        (0..polys).into_par_iter().map(materialize_poly).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..polys).map(materialize_poly).collect()
    }
}
