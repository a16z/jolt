use super::{BooleanityChunks, LazyFoldedRa};
use crate::optimized::lazy_ra::PairedBranchTables;
use jolt_field::JoltField;

/// Booleanity's `(lo, hi)` row pairs at branch width four or eight, read
/// through adjacent-branch sum tables.
pub(super) struct PairedLoHi<'a, F: JoltField> {
    source: &'a BooleanityChunks,
    width: PairWidth,
    tables: PairedBranchTables<F>,
}

#[derive(Clone, Copy)]
enum PairWidth {
    Four,
    Eight,
}

impl<'a, F: JoltField> PairedLoHi<'a, F> {
    const CHUNK_BITS: usize = 4;
    /// Building the tables must cost at most one entry per this many rows.
    const MIN_ROWS_PER_CACHED_VALUE: usize = 16;

    pub(super) fn new(
        tables: &'a LazyFoldedRa<F, BooleanityChunks>,
        columns: usize,
    ) -> Option<Self> {
        let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            source,
            ..
        } = tables
        else {
            return None;
        };
        let pair_width = match width {
            4 => PairWidth::Four,
            8 => PairWidth::Eight,
            _ => return None,
        };
        if source.chunk_bits != Self::CHUNK_BITS
            || PairedBranchTables::<F>::entries(*width)
                > source.prefix_len / Self::MIN_ROWS_PER_CACHED_VALUE
        {
            return None;
        }
        Some(Self {
            source,
            width: pair_width,
            tables: PairedBranchTables::new(&tables[..columns]),
        })
    }

    #[inline]
    pub(super) fn gather(&self, row: usize, out: &mut [(F, F)]) {
        match self.width {
            PairWidth::Four => self.gather_width::<4>(row, out),
            PairWidth::Eight => self.gather_width::<8>(row, out),
        }
    }

    #[inline(always)]
    fn gather_width<const WIDTH: usize>(&self, row: usize, out: &mut [(F, F)]) {
        let columns = self.tables.columns();
        let instruction = columns.min(self.source.instruction_chunks);
        // Instruction column `instruction - 1` lands in the lowest digit.
        let skip = (self.source.instruction_chunks - instruction) * Self::CHUNK_BITS;
        for side in 0..2 {
            let start = (2 * row + side) * WIDTH;
            let mut set = |column: usize, value: F| {
                if side == 0 {
                    out[column].0 = value;
                } else {
                    out[column].1 = value;
                }
            };
            self.tables.visit_four_bit::<WIDTH>(
                instruction,
                |branch| self.source.rows[start + branch].lookup_index() >> skip,
                &mut set,
            );
            self.tables
                .visit_indexed(self.source, instruction..columns, start, WIDTH / 2, set);
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "checked selector fixtures")]
mod tests {
    use super::*;
    use crate::optimized::booleanity::{ColumnSelector, ConstantSuffix, InstructionCycleRow};
    use crate::optimized::lazy_ra::ChunkIndexSource;
    use jolt_field::Fr;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    #[cfg(feature = "akita")]
    use jolt_witness::witnesses::FusedInc;
    use jolt_witness::witnesses::RaChunkSelector;
    use std::sync::Arc;

    fn check<F: JoltField>() {
        let cycles = 1 << 15;
        for prefix in [cycles - 11, cycles] {
            for width in [4usize, 8] {
                let rows = (0..prefix + usize::from(prefix < cycles))
                    .map(|row| {
                        let value = (row as u128 + 1)
                            .wrapping_mul(0xe387_9fb3_4ce9_ba74_1923_873a_e9b9_8379);
                        InstructionCycleRow::new(
                            value,
                            None,
                            false,
                            row % 65536,
                            if row % 3 == 0 {
                                None
                            } else {
                                Some((row % 4096) as u64)
                            },
                            #[cfg(feature = "akita")]
                            FusedInc::default(),
                        )
                    })
                    .collect();
                let selectors = (0..32)
                    .map(|i| ColumnSelector::Instruction(RaChunkSelector::new(i, 32, 4).unwrap()))
                    .chain(
                        (0..4).map(|i| {
                            ColumnSelector::Bytecode(RaChunkSelector::new(i, 4, 4).unwrap())
                        }),
                    )
                    .chain(
                        (0..3).map(|i| ColumnSelector::Ram(RaChunkSelector::new(i, 3, 4).unwrap())),
                    )
                    .collect();
                let source = BooleanityChunks {
                    rows: Arc::new(ConstantSuffix::new(rows, cycles)),
                    prefix_len: prefix,
                    selectors,
                    instruction_chunks: 32,
                    chunk_bits: 4,
                };
                let tables = (0..39)
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
                let mut lazy = LazyFoldedRa::new(tables, source);
                for round in 0..width.ilog2() {
                    lazy.bind(F::from_u128(u128::MAX - u128::from(round)));
                }
                let LazyFoldedRa::Lazy { tables, source, .. } = &lazy else {
                    unreachable!()
                };
                for columns in [1, 31, 32, 33, 39] {
                    let paired = PairedLoHi::new(&lazy, columns).unwrap();
                    for row in [
                        0,
                        1,
                        13,
                        31,
                        255,
                        (prefix - 1) / (2 * width),
                        cycles / (2 * width) - 1,
                    ] {
                        let mut actual = vec![(F::zero(), F::zero()); columns];
                        paired.gather(row, &mut actual);
                        for column in 0..columns {
                            let expected: [F; 2] = std::array::from_fn(|side| {
                                (0..width)
                                    .filter_map(|branch| {
                                        source
                                            .index(column, (2 * row + side) * width + branch)
                                            .map(|index| tables[column][branch * 16 + index])
                                    })
                                    .sum()
                            });
                            assert_eq!(actual[column], (expected[0], expected[1]));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn paired_branches_match_defining_sum_fr() {
        check::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn paired_branches_match_defining_sum_fp128() {
        check::<Prime128OffsetA7F7>();
    }
}
