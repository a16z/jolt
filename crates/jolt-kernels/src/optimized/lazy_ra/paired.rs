//! Sums of adjacent lazy branches keyed by both branch rows' index digits.
//!
//! Pair `p` covers branches `2p` and `2p + 1` of a lazy branch set over a
//! 16-address base table. Its entry `p * 289 + lo * 17 + hi` holds
//! `T[2p][lo] + T[2p + 1][hi]`, where digit 16 stands for a cold row whose
//! branch contributes zero. One lookup replaces two branch gathers; field
//! addition is exact, so every bound value is unchanged.

use std::ops::Range;

use jolt_field::{Accumulator, JoltField};

use super::ChunkIndexSource;

/// Base-table addresses a paired digit selects; digit `PAIR_ADDRESSES` is cold.
pub(super) const PAIR_ADDRESSES: usize = 16;
const RADIX: usize = PAIR_ADDRESSES + 1;
const STATES: usize = RADIX * RADIX;

pub(crate) struct PairedBranchTables<F> {
    tables: Vec<Vec<F>>,
}

impl<F: JoltField> PairedBranchTables<F> {
    /// Entries per column for a set of `branches` lazy branches.
    pub(crate) const fn entries(branches: usize) -> usize {
        branches / 2 * STATES
    }

    /// `tables[c]` holds an even number of 16-address branch blocks.
    pub(crate) fn new(tables: &[Vec<F>]) -> Self {
        let tables = tables
            .iter()
            .map(|table| {
                let mut paired = Vec::with_capacity(Self::entries(table.len() / PAIR_ADDRESSES));
                for branches in table.chunks_exact(2 * PAIR_ADDRESSES) {
                    let (low, high) = branches.split_at(PAIR_ADDRESSES);
                    for lo in low.iter().copied().chain([F::zero()]) {
                        for hi in high.iter().copied().chain([F::zero()]) {
                            paired.push(lo + hi);
                        }
                    }
                }
                paired
            })
            .collect();
        Self { tables }
    }

    pub(crate) fn columns(&self) -> usize {
        self.tables.len()
    }

    /// Each column in `columns`, summed over the `2 * pairs` branch rows from
    /// `start` with indices read through `source`.
    #[inline]
    pub(crate) fn visit_indexed<S: ChunkIndexSource>(
        &self,
        source: &S,
        columns: Range<usize>,
        start: usize,
        pairs: usize,
        mut visit: impl FnMut(usize, F),
    ) {
        for column in columns {
            let table = &self.tables[column];
            let mut sum = F::Accumulator::default();
            for pair in 0..pairs {
                let lo = source
                    .index(column, start + 2 * pair)
                    .unwrap_or(PAIR_ADDRESSES);
                let hi = source
                    .index(column, start + 2 * pair + 1)
                    .unwrap_or(PAIR_ADDRESSES);
                sum.add(table[pair * STATES + lo * RADIX + hi]);
            }
            visit(column, sum.reduce());
        }
    }

    /// Columns `0..columns` read consecutive 4-bit digits of each branch
    /// row's packed index, column `columns - 1` in the lowest digit.
    /// `packed(branch)` is that row's index with any lower digits shifted out;
    /// `columns` is at most 32, so every digit lies inside the `u128`.
    #[inline]
    pub(crate) fn visit_four_bit<const BRANCHES: usize>(
        &self,
        columns: usize,
        packed: impl Fn(usize) -> u128,
        mut visit: impl FnMut(usize, F),
    ) {
        const DIGITS_PER_LIMB: usize = u64::BITS as usize / 4;
        // Four-bit digits never cross a 64-bit limb boundary.
        for limb in 0..columns.div_ceil(DIGITS_PER_LIMB) {
            let end = columns - limb * DIGITS_PER_LIMB;
            let start = end.saturating_sub(DIGITS_PER_LIMB);
            let mut indices = [0u64; BRANCHES];
            for (branch, index) in indices.iter_mut().enumerate() {
                *index = (packed(branch) >> (limb * u64::BITS as usize)) as u64;
            }
            for column in (start..end).rev() {
                let table = &self.tables[column];
                let mut sum = F::Accumulator::default();
                for (pair, indices) in indices.chunks_exact_mut(2).enumerate() {
                    let lo = (indices[0] & 15) as usize;
                    let hi = (indices[1] & 15) as usize;
                    sum.add(table[pair * STATES + lo * RADIX + hi]);
                    indices[0] >>= 4;
                    indices[1] >>= 4;
                }
                visit(column, sum.reduce());
            }
        }
    }
}
