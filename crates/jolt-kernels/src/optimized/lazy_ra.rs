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

pub(crate) trait ChunkIndexSource: Send + Sync {
    fn num_polys(&self) -> usize;

    fn cycles(&self) -> usize;

    fn index(&self, i: usize, j: usize) -> Option<usize>;
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: allocative::Allocative")
)]
pub(crate) enum LazyFoldedRa<F: JoltField, S> {
    Lazy {
        tables: Vec<Vec<F>>,
        width: usize,
        source: S,
    },
    Dense(Vec<Polynomial<F>>),
}

impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
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

    #[inline]
    pub(crate) fn lo_hi(&self, i: usize, row: usize) -> (F, F) {
        (self.value(i, 2 * row), self.value(i, 2 * row + 1))
    }

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

    pub(crate) fn final_values(&self) -> Vec<F> {
        (0..self.num_polys()).map(|i| self.value(i, 0)).collect()
    }

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
