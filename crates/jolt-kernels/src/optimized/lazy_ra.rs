//! Lazily bound one-hot columns for sumcheck kernels over any field.
//!
//! [`ChunkIndexSource`] supplies the table indices of each column. Validated
//! construction uses [`LazyFoldedRa::try_new`]; binding proceeds least
//! significant bit first. The first three binds rescale branch tables, and
//! the fourth materializes dense columns of one sixteenth the original length.
//! The tables and source are then dropped in a background thread, with retained
//! memory purged through the memory helpers.

#[cfg(feature = "allocative")]
use allocative::Allocative;
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;

/// Stable table indices for one-hot columns.
///
/// The column count, cycle count and indices must remain unchanged while the
/// source is held by [`LazyFoldedRa`]. Neither construction nor reads check
/// this stability; state such as a call counter may change without changing returned values.
pub trait ChunkIndexSource: Send + Sync + 'static {
    /// Number of columns, unchanged throughout the source's lifetime.
    fn num_polys(&self) -> usize;

    /// Unbound cycle count, unchanged throughout the source's lifetime.
    fn cycles(&self) -> usize;

    /// Table index for column `i` at unbound cycle `j`, or `None` for zero.
    ///
    /// Calls made by a valid bound-column sequence have `i < num_polys()` and
    /// `j < cycles()`. The returned index must remain unchanged on repeated calls.
    fn index(&self, i: usize, j: usize) -> Option<usize>;

    /// Optional exclusive bound on every index of column `i`.
    ///
    /// `Some(bound)` promises every returned `Some(index)` has `index < bound`.
    /// Construction trusts this promise without scanning; a false promise can
    /// cause reads to panic or access an entry of another branch table.
    fn index_bound(&self, _i: usize) -> Option<usize> {
        None
    }
}

/// Construction failure for lazily bound one-hot columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LazyRaError {
    /// The table count differs from the source's column count.
    #[error("{tables} tables supplied for {polys} columns")]
    TableCount { tables: usize, polys: usize },
    /// A family must contain a column to enforce exhaustion during dense binds.
    #[error("at least one column is required")]
    NoColumns,
    /// The unbound cycle count is zero or is not a power of two.
    #[error("cycle count {cycles} is not a power of two")]
    CyclesNotPowerOfTwo { cycles: usize },
    /// A promised exclusive index bound exceeds its table's length.
    #[error("column {poly} index bound {bound} exceeds table length {len}")]
    IndexBoundExceedsTable {
        poly: usize,
        bound: usize,
        len: usize,
    },
    /// The least failing cycle of a scanned column names an invalid table index.
    #[error("column {poly}, cycle {cycle}: index {index} exceeds table length {len}")]
    IndexOutOfRange {
        poly: usize,
        cycle: usize,
        index: usize,
        len: usize,
    },
}

/// One-hot columns bound least significant bit first.
///
/// Column `i` initially has value `tables[i][source.index(i, j)]`, or zero for
/// `None`. Reads require a valid column and an index below the current length;
/// at most `log2(source.cycles())` binds are permitted. Source stability and
/// promised index bounds are the caller's responsibility.
#[cfg_attr(
    feature = "allocative",
    derive(Allocative),
    allocative(bound = "F: JoltField, S: Allocative")
)]
pub enum LazyFoldedRa<F: JoltField, S> {
    /// Branch tables with fewer than four bound variables.
    Lazy(LazyRaBranches<F, S>),
    /// Dense columns after the fourth bind.
    Dense(LazyRaDense<F>),
}

/// Opaque branch tables and stable source of a lazy one-hot column family.
///
/// Instances are produced by [`LazyFoldedRa::try_new`] and its binds.
#[cfg_attr(
    feature = "allocative",
    derive(Allocative),
    allocative(bound = "F: JoltField, S: Allocative")
)]
pub struct LazyRaBranches<F: JoltField, S> {
    pub(crate) tables: Vec<Vec<F>>,
    pub(crate) width: usize,
    pub(crate) source: S,
}

/// Opaque dense columns reached after four binds of a lazy family.
#[cfg_attr(
    feature = "allocative",
    derive(Allocative),
    allocative(bound = "F: JoltField")
)]
pub struct LazyRaDense<F: JoltField>(pub(crate) Vec<Polynomial<F>>);

impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    /// One scale table per selector polynomial, in polynomial order.
    pub(crate) fn new(tables: Vec<Vec<F>>, source: S) -> Self {
        debug_assert_eq!(tables.len(), source.num_polys());
        Self::Lazy(LazyRaBranches {
            tables,
            width: 1,
            source,
        })
    }

    /// Validate tables and their stable index source before constructing columns.
    ///
    /// Checks table count, a nonzero column count, a power-of-two cycle count,
    /// then each column's indices in order. A supplied `index_bound` is trusted
    /// and avoids a scan; otherwise the least out-of-range cycle is reported.
    /// Tables need no particular length, and an empty table is accepted when
    /// its column has no index. Stability of the source and the truth of its
    /// index-bound promises cannot be checked here.
    pub fn try_new(tables: Vec<Vec<F>>, source: S) -> Result<Self, LazyRaError> {
        let polys = source.num_polys();
        if tables.len() != polys {
            return Err(LazyRaError::TableCount {
                tables: tables.len(),
                polys,
            });
        }
        if polys == 0 {
            return Err(LazyRaError::NoColumns);
        }
        let cycles = source.cycles();
        if !cycles.is_power_of_two() {
            return Err(LazyRaError::CyclesNotPowerOfTwo { cycles });
        }
        for (poly, table) in tables.iter().enumerate() {
            let len = table.len();
            if let Some(bound) = source.index_bound(poly) {
                if bound > len {
                    return Err(LazyRaError::IndexBoundExceedsTable { poly, bound, len });
                }
            } else {
                let invalid = |cycle| {
                    source
                        .index(poly, cycle)
                        .filter(|&index| index >= len)
                        .map(|index| (cycle, index))
                };
                #[cfg(feature = "parallel")]
                let first = (0..cycles)
                    .into_par_iter()
                    .filter_map(invalid)
                    .find_first(|_| true);
                #[cfg(not(feature = "parallel"))]
                let first = (0..cycles).find_map(invalid);
                if let Some((cycle, index)) = first {
                    return Err(LazyRaError::IndexOutOfRange {
                        poly,
                        cycle,
                        index,
                        len,
                    });
                }
            }
        }
        Ok(Self::new(tables, source))
    }

    /// Number of columns in this family.
    pub fn num_polys(&self) -> usize {
        match self {
            Self::Lazy(LazyRaBranches { tables, .. }) => tables.len(),
            Self::Dense(LazyRaDense(polys)) => polys.len(),
        }
    }

    /// Current evaluation of column `i` at index `j` after low-to-high binds.
    ///
    /// Requires `i < num_polys()` and `j` below the current length. These
    /// conditions are not explicitly checked; invalid indices may panic.
    #[inline]
    pub fn value(&self, i: usize, j: usize) -> F {
        match self {
            Self::Lazy(LazyRaBranches {
                tables,
                width,
                source,
            }) => gather(&tables[i], *width, source, i, j),
            Self::Dense(LazyRaDense(polys)) => polys[i].evals()[j],
        }
    }

    /// Adjacent current entries `(value(i, 2·row), value(i, 2·row + 1))`.
    ///
    /// Requires `i < num_polys()` and `row` below half the current length,
    /// without an explicit precondition check.
    #[inline]
    pub fn lo_hi(&self, i: usize, row: usize) -> (F, F) {
        (self.value(i, 2 * row), self.value(i, 2 * row + 1))
    }

    /// Write adjacent pairs for the first `min(out.len(), num_polys())` columns.
    ///
    /// Remaining output entries are unchanged. Requires `row` below half the
    /// current length, without an explicit precondition check.
    #[inline]
    pub fn lo_hi_all(&self, row: usize, out: &mut [(F, F)]) {
        match self {
            Self::Lazy(LazyRaBranches {
                tables,
                width,
                source,
            }) => {
                let width = *width;
                for (i, (out, table)) in out.iter_mut().zip(tables).enumerate() {
                    *out = (
                        gather(table, width, source, i, 2 * row),
                        gather(table, width, source, i, 2 * row + 1),
                    );
                }
            }
            Self::Dense(LazyRaDense(polys)) => {
                for (out, poly) in out.iter_mut().zip(polys) {
                    let evals = poly.evals();
                    *out = (evals[2 * row], evals[2 * row + 1]);
                }
            }
        }
    }

    /// Entry zero of every column, in column order.
    ///
    /// These are the fully bound values after `log2(cycles())` binds; calling
    /// earlier returns the current entry zero without checking completion.
    pub fn final_values(&self) -> Vec<F> {
        (0..self.num_polys()).map(|i| self.value(i, 0)).collect()
    }

    /// Bind the next least significant variable by `lo + challenge·(hi − lo)`.
    ///
    /// The first three binds rescale branch tables; the fourth creates dense
    /// columns and drops the tables and source in a background thread.
    ///
    /// # Panics
    ///
    /// Panics in release builds if all `log2(cycles())` variables are bound.
    pub fn bind(&mut self, challenge: F) {
        *self = match std::mem::replace(self, Self::Dense(LazyRaDense(Vec::new()))) {
            Self::Lazy(LazyRaBranches {
                tables,
                width,
                source,
            }) => {
                assert!(width < source.cycles(), "no variables left to bind");
                let tables = double_branches(tables, challenge);
                if width < 8 {
                    Self::Lazy(LazyRaBranches {
                        tables,
                        width: width * 2,
                        source,
                    })
                } else {
                    let log_t = source.cycles().ilog2() as usize;
                    let dense = Self::Dense(LazyRaDense(materialize(&tables, &source, width * 2)));
                    crate::mem::drop_in_background_thread(tables);
                    crate::mem::drop_in_background_thread(source);
                    crate::mem::purge_retained_memory(log_t);
                    dense
                }
            }
            Self::Dense(LazyRaDense(mut polys)) => {
                for poly in &mut polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                Self::Dense(LazyRaDense(polys))
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
