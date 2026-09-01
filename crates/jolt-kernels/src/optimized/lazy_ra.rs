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
//! from `N·T` field elements to the index source plus `N·T/16`.
//!
//! Byte parity: every gathered value is the same polynomial of the same
//! table entries and challenges as the iterated `lo + r·(hi − lo)` dense
//! bind — identical monomials, exact field algebra (pre-scaling only
//! reassociates the weight product) — so round messages and output claims
//! are bit-identical. The consumers' in-module parity tests pin this
//! against the naive dense path.

use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Per-cycle hot indices of `N` committed one-hot selector polynomials over
/// a shared compact backing store (typed witness rows, packed columns).
pub(crate) trait ChunkIndexSource: Send + Sync {
    /// Number of selector polynomials served.
    fn num_polys(&self) -> usize;

    /// The unbound cycle-domain length.
    fn cycles(&self) -> usize;

    /// The scale-table index of polynomial `i`'s hot address at unbound
    /// cycle `j`; `None` when the cycle is cold for that polynomial.
    fn index(&self, i: usize, j: usize) -> Option<usize>;
}

/// `N` address-folded selector columns bound `LowToHigh`, lazily until the
/// fourth bind materializes dense.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: allocative::Allocative")
)]
pub(crate) enum LazyFoldedRa<F: JoltField, S> {
    /// Fewer than four binds: per-polynomial branch scale tables (the base
    /// table pre-scaled by each bound-bit pattern's eq weight), flattened
    /// offset-major — `tables[i][offset · stride_i + k]` with
    /// `stride_i = tables[i].len() / width` — plus the compact index source.
    Lazy {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        tables: Vec<Vec<F>>,
        /// Bound-bit branch count (`2^binds`: 1, 2, 4, or 8).
        width: usize,
        source: S,
    },
    /// Four or more binds: plain dense multilinears (`T/16` at entry).
    Dense(Vec<Polynomial<F>>),
}

impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    /// One scale table per selector polynomial, in polynomial order.
    pub(crate) fn new(tables: Vec<Vec<F>>, source: S) -> Self {
        debug_assert_eq!(tables.len(), source.num_polys());
        Self::Lazy {
            tables,
            width: 1,
            source,
        }
    }

    pub(crate) fn num_polys(&self) -> usize {
        match self {
            Self::Lazy { tables, .. } => tables.len(),
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
                source,
            } => gather(&tables[i], *width, source, i, j),
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
                source,
            } => {
                let width = *width;
                for (i, (out, table)) in out.iter_mut().zip(tables).enumerate() {
                    *out = (
                        gather(table, width, source, i, 2 * row),
                        gather(table, width, source, i, 2 * row + 1),
                    );
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
    /// until the fourth bind materializes dense (and drops the source), then
    /// use plain multilinear binds.
    pub(crate) fn bind(&mut self, challenge: F) {
        *self = match std::mem::replace(self, Self::Dense(Vec::new())) {
            Self::Lazy {
                tables,
                width,
                source,
            } => {
                let tables = double_branches(tables, challenge);
                if width < 8 {
                    Self::Lazy {
                        tables,
                        width: width * 2,
                        source,
                    }
                } else {
                    let log_t = source.cycles().ilog2() as usize;
                    let dense = Self::Dense(materialize(&tables, &source, width * 2));
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

/// Doubles every polynomial's branch set for the next bound bit: the first
/// half keeps the existing branches scaled by `1 − challenge` (bit 0), the
/// second half by `challenge` (bit 1) — offset layout
/// `b0 + 2·b1 + 4·b2`, matching the low bits of the original cycle index.
fn double_branches<F: JoltField>(tables: Vec<Vec<F>>, challenge: F) -> Vec<Vec<F>> {
    let one_minus = F::one() - challenge;
    let double = |table: Vec<F>| -> Vec<F> {
        let mut next = Vec::with_capacity(table.len() * 2);
        next.extend(table.iter().map(|value| one_minus * *value));
        next.extend(table.iter().map(|value| challenge * *value));
        next
    };
    #[cfg(feature = "parallel")]
    {
        tables.into_par_iter().map(double).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        tables.into_iter().map(double).collect()
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
    source: &S,
    branches: usize,
) -> Vec<Polynomial<F>> {
    debug_assert!(source.cycles() >= branches);
    let new_len = source.cycles() / branches;
    let materialize_poly = |i: usize| -> Polynomial<F> {
        let table = tables[i].as_slice();
        let eval = |j: usize| gather(table, branches, source, i, j);
        #[cfg(feature = "parallel")]
        let evals: Vec<F> = (0..new_len).into_par_iter().map(eval).collect();
        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = (0..new_len).map(eval).collect();
        Polynomial::new(evals)
    };
    #[cfg(feature = "parallel")]
    {
        (0..tables.len())
            .into_par_iter()
            .map(materialize_poly)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..tables.len()).map(materialize_poly).collect()
    }
}
