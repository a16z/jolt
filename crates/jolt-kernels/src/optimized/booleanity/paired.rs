use super::{BooleanityChunks, ChunkIndexSource, LazyFoldedRa};
use jolt_field::JoltField;

/// Cache sums of adjacent lazy branches over their small categorical alphabet.
/// Cold RAM has its own digit, whose branch value is zero.
pub(super) struct PairedBranchTables<'a, F: JoltField> {
    source: &'a BooleanityChunks,
    width: usize,
    values: Vec<Vec<F>>,
}

impl<'a, F: JoltField> PairedBranchTables<'a, F> {
    const CHUNK_BITS: usize = 4;
    const ADDRESSES: usize = 1 << Self::CHUNK_BITS;
    const RADIX: usize = Self::ADDRESSES + 1;
    const STATES: usize = Self::RADIX * Self::RADIX;
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
        if source.chunk_bits != Self::CHUNK_BITS
            || !matches!(width, 4 | 8)
            || width / 2 * Self::STATES > source.prefix_len / Self::MIN_ROWS_PER_CACHED_VALUE
        {
            return None;
        }
        let values = tables[..columns]
            .iter()
            .map(|table| {
                let mut paired = Vec::with_capacity(width / 2 * Self::STATES);
                for branches in table.chunks_exact(2 * Self::ADDRESSES) {
                    for high in branches[Self::ADDRESSES..]
                        .iter()
                        .copied()
                        .chain(std::iter::once(F::zero()))
                    {
                        for low in branches[..Self::ADDRESSES]
                            .iter()
                            .copied()
                            .chain(std::iter::once(F::zero()))
                        {
                            paired.push(low + high);
                        }
                    }
                }
                paired
            })
            .collect();
        Some(Self {
            source,
            width: *width,
            values,
        })
    }

    #[inline]
    pub(super) fn gather(&self, row: usize, out: &mut [(F, F)]) {
        match self.width {
            4 => self.gather_width::<4>(row, out),
            8 => self.gather_width::<8>(row, out),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn gather_width<const WIDTH: usize>(&self, row: usize, out: &mut [(F, F)]) {
        let columns = self.values.len();
        let instruction = columns.min(self.source.instruction_chunks);
        for branch in (0..WIDTH).step_by(2) {
            let base = branch / 2 * Self::STATES;
            let low_row = 2 * row * WIDTH + branch;
            let high_row = (2 * row + 1) * WIDTH + branch;
            if instruction != 0 {
                let skip = (self.source.instruction_chunks - instruction) * Self::CHUNK_BITS;
                let mut lo0 = self.source.rows[low_row].lookup_index() >> skip;
                let mut lo1 = self.source.rows[low_row + 1].lookup_index() >> skip;
                let mut hi0 = self.source.rows[high_row].lookup_index() >> skip;
                let mut hi1 = self.source.rows[high_row + 1].lookup_index() >> skip;
                for column in (0..instruction).rev() {
                    let lo = (lo0 & (Self::ADDRESSES - 1) as u128) as usize
                        + Self::RADIX * (lo1 & (Self::ADDRESSES - 1) as u128) as usize;
                    let hi = (hi0 & (Self::ADDRESSES - 1) as u128) as usize
                        + Self::RADIX * (hi1 & (Self::ADDRESSES - 1) as u128) as usize;
                    let values = (
                        self.values[column][base + lo],
                        self.values[column][base + hi],
                    );
                    if branch == 0 {
                        out[column] = values;
                    } else {
                        out[column].0 += values.0;
                        out[column].1 += values.1;
                    }
                    lo0 >>= Self::CHUNK_BITS;
                    lo1 >>= Self::CHUNK_BITS;
                    hi0 >>= Self::CHUNK_BITS;
                    hi1 >>= Self::CHUNK_BITS;
                }
            }
            for (column, pair) in out.iter_mut().enumerate().take(columns).skip(instruction) {
                let state = |start| {
                    self.source.index(column, start).unwrap_or(Self::ADDRESSES)
                        + Self::RADIX
                            * self
                                .source
                                .index(column, start + 1)
                                .unwrap_or(Self::ADDRESSES)
                };
                let values = (
                    self.values[column][base + state(low_row)],
                    self.values[column][base + state(high_row)],
                );
                if branch == 0 {
                    *pair = values;
                } else {
                    pair.0 += values.0;
                    pair.1 += values.1;
                }
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "checked selector fixtures")]
mod tests {
    use super::*;
    use crate::optimized::booleanity::{ColumnSelector, ConstantSuffix, InstructionCycleRow};
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
                    let paired = PairedBranchTables::new(&lazy, columns).unwrap();
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
