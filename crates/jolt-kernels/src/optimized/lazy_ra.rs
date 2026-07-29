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
//! first cycle binds preserve that structure: after `b < 3` binds the bound
//! value at index `j` is the weighted gather
//!
//! ```text
//! value(i, j) = Σ_{offset < 2^b} weights[offset] · tables[i][index(i, j·2^b + offset)]
//! ```
//!
//! with `weights` the eq weights of the bound low bits. Only the third bind
//! materializes dense vectors, at `T/8` length, and drops the index source.
//! Peak memory falls from `N·T` field elements to the index source plus
//! `N·T/8`, with the `N × 2^w` scale tables unscaled and shared across all
//! states.
//!
//! Byte parity: every gathered value is the same polynomial of the same
//! table entries and challenges as the iterated `lo + r·(hi − lo)` dense
//! bind — identical monomials, exact field algebra — so round messages and
//! output claims are bit-identical. The consumers' in-module parity tests
//! pin this against the naive dense path.

use jolt_field::Field;
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

/// `N` address-folded selector columns bound `LowToHigh`, lazily for the
/// first three binds.
pub(crate) enum LazyFoldedRa<F: Field, S> {
    /// Fewer than three binds: unscaled per-polynomial scale tables, the
    /// compact index source, and the bound low bits' eq weights
    /// (`weights[b0 + 2·b1]`, length `2^binds`).
    Lazy {
        tables: Vec<Vec<F>>,
        source: S,
        weights: Vec<F>,
    },
    /// Three or more binds: plain dense multilinears (`T/8` at entry).
    Dense(Vec<Polynomial<F>>),
}

impl<F: Field, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    /// One scale table per selector polynomial, in polynomial order.
    pub(crate) fn new(tables: Vec<Vec<F>>, source: S) -> Self {
        debug_assert_eq!(tables.len(), source.num_polys());
        Self::Lazy {
            tables,
            source,
            weights: vec![F::one()],
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
                source,
                weights,
            } => {
                let table = &tables[i];
                let width = weights.len();
                if width == 1 {
                    return source.index(i, j).map_or_else(F::zero, |k| table[k]);
                }
                let mut sum = F::zero();
                for (offset, weight) in weights.iter().enumerate() {
                    if let Some(k) = source.index(i, j * width + offset) {
                        sum += *weight * table[k];
                    }
                }
                sum
            }
            Self::Dense(polys) => polys[i].evals()[j],
        }
    }

    /// The `(lo, hi) = (value(i, 2·row), value(i, 2·row + 1))` pair the
    /// round messages consume.
    #[inline]
    pub(crate) fn lo_hi(&self, i: usize, row: usize) -> (F, F) {
        (self.value(i, 2 * row), self.value(i, 2 * row + 1))
    }

    /// The fully bound claims, in polynomial order (any state, so short
    /// cycle geometries extract correctly).
    pub(crate) fn final_values(&self) -> Vec<F> {
        (0..self.num_polys()).map(|i| self.value(i, 0)).collect()
    }

    /// Bind the next cycle variable `LowToHigh`: double the branch weights
    /// for the first two binds, materialize dense (and drop the source) at
    /// the third, plain multilinear binds after.
    pub(crate) fn bind(&mut self, challenge: F) {
        let one_minus = F::one() - challenge;
        let doubled = |weights: &[F]| -> Vec<F> {
            let mut next = Vec::with_capacity(weights.len() * 2);
            next.extend(weights.iter().map(|weight| *weight * one_minus));
            next.extend(weights.iter().map(|weight| *weight * challenge));
            next
        };
        *self = match std::mem::replace(self, Self::Dense(Vec::new())) {
            Self::Lazy {
                tables,
                source,
                weights,
            } => {
                let weights = doubled(&weights);
                if weights.len() < 8 {
                    Self::Lazy {
                        tables,
                        source,
                        weights,
                    }
                } else {
                    Self::Dense(materialize(&tables, &source, &weights))
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

/// The third bind's materialization: gather every polynomial dense at
/// `cycles / 8` length through the eight bound-bit weights.
fn materialize<F: Field, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    source: &S,
    weights: &[F],
) -> Vec<Polynomial<F>> {
    debug_assert_eq!(weights.len(), 8);
    let new_len = source.cycles() / 8;
    let materialize_poly = |i: usize| -> Polynomial<F> {
        let table = &tables[i];
        let eval = |j: usize| -> F {
            let mut sum = F::zero();
            for (offset, weight) in weights.iter().enumerate() {
                if let Some(k) = source.index(i, 8 * j + offset) {
                    sum += *weight * table[k];
                }
            }
            sum
        };
        #[cfg(feature = "parallel")]
        let evals: Vec<F> = (0..new_len).into_par_iter().map(eval).collect();
        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = (0..new_len).map(eval).collect();
        Polynomial::new(evals)
    };
    (0..tables.len()).map(materialize_poly).collect()
}
