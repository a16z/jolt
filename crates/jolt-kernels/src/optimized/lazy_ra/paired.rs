//! Sums of adjacent lazy branches keyed by both branch rows' index digits.
//!
//! Pair `p` covers branches `2p` and `2p + 1` of a lazy branch set over a
//! 16-address base table. Its entry `p * 289 + lo * 17 + hi` holds
//! `T[2p][lo] + T[2p + 1][hi]`, where digit 16 stands for a cold row whose
//! branch contributes zero. One lookup replaces two branch gathers; field
//! addition is exact, so every bound value is unchanged.
//!
//! The visits sum each column's two to four lookups with plain field
//! additions: a deferred `F::Accumulator` reduction costs more than it saves
//! at that length, and the round messages call these visits every row.

use std::ops::Range;

use jolt_field::JoltField;

use super::ChunkIndexSource;

/// Base-table addresses a paired digit selects; digit `PAIR_ADDRESSES` is cold.
pub(super) const PAIR_ADDRESSES: usize = 16;
const RADIX: usize = PAIR_ADDRESSES + 1;
const STATES: usize = RADIX * RADIX;
/// Building the tables must cost at most one entry per this many live rows.
const MIN_ROWS_PER_CACHED_VALUE: usize = 16;

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
            let sum = (0..pairs)
                .map(|pair| {
                    let lo = source
                        .index(column, start + 2 * pair)
                        .unwrap_or(PAIR_ADDRESSES);
                    let hi = source
                        .index(column, start + 2 * pair + 1)
                        .unwrap_or(PAIR_ADDRESSES);
                    table[pair * STATES + lo * RADIX + hi]
                })
                .sum();
            visit(column, sum);
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
                let sum = indices
                    .chunks_exact_mut(2)
                    .enumerate()
                    .map(|(pair, indices)| {
                        let lo = (indices[0] & 15) as usize;
                        let hi = (indices[1] & 15) as usize;
                        let value = table[pair * STATES + lo * RADIX + hi];
                        indices[0] >>= 4;
                        indices[1] >>= 4;
                        value
                    })
                    .sum();
                visit(column, sum);
            }
        }
    }
}

/// `(lo, hi)` row pairs of a single-factor lazy set at branch width four or
/// eight, read through adjacent-branch sum tables.
pub(crate) struct PairedLoHi<'a, F: JoltField, S> {
    source: &'a S,
    width: PairWidth,
    tables: PairedBranchTables<F>,
}

#[derive(Clone, Copy)]
enum PairWidth {
    Four,
    Eight,
}

impl<'a, F: JoltField, S: ChunkIndexSource> PairedLoHi<'a, F, S> {
    /// `tables` are leading columns of a single-factor lazy set at `width`
    /// branches over `source`.
    pub(super) fn new(tables: &[Vec<F>], width: usize, source: &'a S) -> Option<Self> {
        let pair_width = match width {
            4 => PairWidth::Four,
            8 => PairWidth::Eight,
            _ => return None,
        };
        if tables
            .iter()
            .any(|table| table.len() != width * PAIR_ADDRESSES)
            || PairedBranchTables::<F>::entries(width)
                > source.constant_suffix_start() / MIN_ROWS_PER_CACHED_VALUE
        {
            return None;
        }
        Some(Self {
            source,
            width: pair_width,
            tables: PairedBranchTables::new(tables),
        })
    }

    #[inline]
    pub(crate) fn gather(&self, row: usize, out: &mut [(F, F)]) {
        match self.width {
            PairWidth::Four => self.gather_width::<4>(row, out),
            PairWidth::Eight => self.gather_width::<8>(row, out),
        }
    }

    #[inline(always)]
    fn gather_width<const WIDTH: usize>(&self, row: usize, out: &mut [(F, F)]) {
        for side in 0..2 {
            self.source.visit_paired_branches::<F, WIDTH>(
                &self.tables,
                (2 * row + side) * WIDTH,
                |column, value| {
                    if side == 0 {
                        out[column].0 = value;
                    } else {
                        out[column].1 = value;
                    }
                },
            );
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "checked selector fixtures")]
mod tests {
    use super::super::LazyFoldedRa;
    use crate::optimized::constant_suffix::ConstantSuffix;
    use crate::optimized::instruction_read_raf::{InstructionCycleRow, LookupIndexChunks};
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField};
    #[cfg(feature = "akita")]
    use jolt_witness::witnesses::FusedInc;
    use std::sync::Arc;

    fn check<F: JoltField>() {
        let cycles = 1 << 15;
        for prefix in [cycles - 5, cycles] {
            let indices: Vec<u128> = (0..prefix + usize::from(prefix < cycles))
                .map(|i| (i as u128 + 1).wrapping_mul(0xe387_9fb3_4ce9_ba74_1923_873a_e9b9_8379))
                .collect();
            let rows = Arc::new(ConstantSuffix::new(
                indices
                    .iter()
                    .map(|&index| {
                        InstructionCycleRow::new(
                            index,
                            None,
                            false,
                            0,
                            None,
                            #[cfg(feature = "akita")]
                            FusedInc::default(),
                        )
                    })
                    .collect(),
                cycles,
            ));
            let tables: Vec<Vec<F>> = (0..32)
                .map(|column| {
                    (0..16)
                        .map(|address| {
                            F::from_u128(
                                (column * 16 + address + 1) as u128 * 0xa6a7_c3d2_3fa2_41e1,
                            )
                        })
                        .collect()
                })
                .collect();
            for challenges in [
                [F::from_u64(17), F::from_u64(31), F::from_u64(43)],
                [F::zero(), F::one(), F::zero()],
            ] {
                let mut lazy = LazyFoldedRa::new(
                    tables.clone(),
                    LookupIndexChunks::new(Arc::clone(&rows), 32, 4),
                );
                for (round, &challenge) in challenges.iter().enumerate() {
                    lazy.bind(challenge);
                    let width = 1 << (round + 1);
                    if width == 2 {
                        assert!(lazy.paired_lo_hi(32).is_none());
                        continue;
                    }
                    let paired = lazy.paired_lo_hi(32).unwrap();
                    for row in [0, 17, (prefix - 1) / (2 * width), cycles / (2 * width) - 1] {
                        let mut actual = vec![(F::zero(), F::zero()); 32];
                        paired.gather(row, &mut actual);
                        for (column, pair) in actual.iter().enumerate() {
                            for (side, got) in [pair.0, pair.1].into_iter().enumerate() {
                                let mut expected = F::zero();
                                for branch in 0..width {
                                    let cycle = (2 * row + side) * width + branch;
                                    let key = indices[cycle.min(prefix)];
                                    let digit = ((key >> (4 * (31 - column))) & 15) as usize;
                                    let weight = challenges[..=round].iter().enumerate().fold(
                                        F::one(),
                                        |weight, (bit, &r)| {
                                            weight
                                                * if branch & (1 << bit) == 0 {
                                                    F::one() - r
                                                } else {
                                                    r
                                                }
                                        },
                                    );
                                    expected += weight * tables[column][digit];
                                }
                                assert_eq!(got, expected, "prefix {prefix}, width {width}, row {row}, column {column}, side {side}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn paired_instruction_gathers_match_multilinear_definition() {
        check::<Fr>();
        #[cfg(feature = "akita")]
        check::<Prime128OffsetA7F7>();
    }
}
