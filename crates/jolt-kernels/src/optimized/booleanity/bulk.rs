//! Decode the instruction chunks together, loading each packed row once.
//! Bytecode and RAM retain their checked selectors, including cold RAM rows.

use super::{BooleanityChunks, ChunkIndexSource};
use jolt_field::JoltField;

impl ChunkIndexSource for BooleanityChunks {
    fn constant_suffix_start(&self) -> usize {
        self.prefix_len
    }
    fn num_columns(&self) -> usize {
        self.selectors.len()
    }
    fn cycles(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        self.selectors[i].index(&self.rows[j])
    }

    #[inline(always)]
    fn for_each_category_pair(
        &self,
        width: usize,
        row: usize,
        radix: usize,
        columns: usize,
        mut visit: impl FnMut(usize, usize, usize),
    ) {
        debug_assert!(width == 1 || width == 2);
        let instruction = columns.min(self.instruction_chunks);
        if instruction != 0 {
            let skip = (self.instruction_chunks - instruction) * self.chunk_bits;
            let mask = (1u128 << self.chunk_bits) - 1;
            let mut lo0 = self.rows[2 * row * width].lookup_index() >> skip;
            let mut hi0 = self.rows[(2 * row + 1) * width].lookup_index() >> skip;
            let mut lo1 = if width == 2 {
                self.rows[2 * row * width + 1].lookup_index() >> skip
            } else {
                0
            };
            let mut hi1 = if width == 2 {
                self.rows[(2 * row + 1) * width + 1].lookup_index() >> skip
            } else {
                0
            };
            for column in (0..instruction).rev() {
                let lo = (lo0 & mask) as usize + radix * (lo1 & mask) as usize;
                let hi = (hi0 & mask) as usize + radix * (hi1 & mask) as usize;
                visit(column, lo, hi);
                lo0 >>= self.chunk_bits;
                hi0 >>= self.chunk_bits;
                lo1 >>= self.chunk_bits;
                hi1 >>= self.chunk_bits;
            }
        }
        for column in instruction..columns {
            let mut states = [0; 2];
            let mut weight = 1;
            for branch in 0..width {
                for (side, state) in states.iter_mut().enumerate() {
                    *state += weight
                        * self
                            .index(column, (2 * row + side) * width + branch)
                            .unwrap_or(radix - 1);
                }
                weight *= radix;
            }
            visit(column, states[0], states[1]);
        }
    }

    #[inline]
    fn gather_pairs<F: JoltField>(
        &self,
        tables: &[Vec<F>],
        width: usize,
        row: usize,
        out: &mut [(F, F)],
    ) {
        let instruction = out.len().min(self.instruction_chunks);
        let stride = 1usize << self.chunk_bits;
        let mask = (stride - 1) as u128;
        if instruction != 0 {
            let skip = (self.instruction_chunks - instruction) * self.chunk_bits;
            for branch in 0..width {
                let mut lo = self.rows[2 * row * width + branch].lookup_index() >> skip;
                let mut hi = self.rows[(2 * row + 1) * width + branch].lookup_index() >> skip;
                let base = branch * stride;
                for (pair, table) in out[..instruction]
                    .iter_mut()
                    .zip(&tables[..instruction])
                    .rev()
                {
                    let values = (
                        table[base + (lo & mask) as usize],
                        table[base + (hi & mask) as usize],
                    );
                    if branch == 0 {
                        *pair = values;
                    } else {
                        pair.0 += values.0;
                        pair.1 += values.1;
                    }
                    lo >>= self.chunk_bits;
                    hi >>= self.chunk_bits;
                }
            }
        }
        for (column, (pair, table)) in out.iter_mut().zip(tables).enumerate().skip(instruction) {
            if width == 1 {
                *pair = (
                    self.index(column, 2 * row).map_or(F::zero(), |i| table[i]),
                    self.index(column, 2 * row + 1)
                        .map_or(F::zero(), |i| table[i]),
                );
                continue;
            }
            *pair = (F::zero(), F::zero());
            for branch in 0..width {
                if let Some(index) = self.index(column, 2 * row * width + branch) {
                    pair.0 += table[branch * stride + index];
                }
                if let Some(index) = self.index(column, (2 * row + 1) * width + branch) {
                    pair.1 += table[branch * stride + index];
                }
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "checked selector fixtures")]
mod tests {
    use super::super::{ColumnSelector, ConstantSuffix, InstructionCycleRow};
    use super::*;
    use jolt_field::Fr;
    use jolt_witness::witnesses::RaChunkSelector;
    use std::sync::Arc;

    fn check<F: JoltField>() {
        for bits in [1, 2, 4, 8] {
            let instruction = 128 / bits;
            let addresses = 1usize << bits;
            for prefix in [0usize, 13, 17, 63] {
                let rows: Vec<_> = (0..=prefix)
                    .map(|i| {
                        let x =
                            (i as u128 + 1).wrapping_mul(0xe387_9fb3_4ce9_ba74_1923_873a_e9b9_8379);
                        InstructionCycleRow::new(
                            x,
                            Some(1),
                            false,
                            i * 17,
                            if i % 3 == 0 {
                                None
                            } else if i % 5 == 0 {
                                Some(0)
                            } else {
                                Some((i * 29) as u64)
                            },
                            #[cfg(feature = "akita")]
                            jolt_witness::witnesses::FusedInc::default(),
                        )
                    })
                    .collect();
                let rows = Arc::new(ConstantSuffix::new(rows, 64));
                let selectors =
                    (0..instruction)
                        .map(|i| {
                            ColumnSelector::Instruction(
                                RaChunkSelector::new(i, instruction, bits).unwrap(),
                            )
                        })
                        .chain((0..2).map(|i| {
                            ColumnSelector::Bytecode(RaChunkSelector::new(i, 2, bits).unwrap())
                        }))
                        .chain((0..3).map(|i| {
                            ColumnSelector::Ram(RaChunkSelector::new(i, 3, bits).unwrap())
                        }))
                        .collect();
                let source = BooleanityChunks {
                    prefix_len: rows.prefix_len(),
                    rows,
                    selectors,
                    instruction_chunks: instruction,
                    chunk_bits: bits,
                };
                for width in [1, 2, 4, 8] {
                    let tables: Vec<Vec<F>> = (0..source.num_columns())
                        .map(|column| {
                            (0..width * addresses)
                                .map(|i| {
                                    let value = F::from_u64((column * 101 + i * 37 + 13) as u64);
                                    if (column + i) % 2 == 0 {
                                        -value
                                    } else {
                                        value
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    for columns in [
                        1,
                        instruction - 1,
                        instruction,
                        instruction + 1,
                        source.num_columns(),
                    ] {
                        for row in 0..64 / (2 * width) {
                            let mut pairs = vec![(F::zero(), F::zero()); columns];
                            source.gather_pairs(&tables[..columns], width, row, &mut pairs);
                            for column in 0..columns {
                                let expected: [F; 2] = std::array::from_fn(|side| {
                                    (0..width)
                                        .filter_map(|branch| {
                                            source
                                                .index(column, (2 * row + side) * width + branch)
                                                .map(|i| tables[column][branch * addresses + i])
                                        })
                                        .sum()
                                });
                                assert_eq!(pairs[column], (expected[0], expected[1]));
                            }
                            if width <= 2 {
                                let radix = addresses + 1;
                                let mut seen = vec![false; columns];
                                source.for_each_category_pair(
                                    width,
                                    row,
                                    radix,
                                    columns,
                                    |column, lo, hi| {
                                        assert!(!seen[column]);
                                        seen[column] = true;
                                        let expected: [usize; 2] = std::array::from_fn(|side| {
                                            (0..width)
                                                .map(|branch| {
                                                    radix.pow(branch as u32)
                                                        * source
                                                            .index(
                                                                column,
                                                                (2 * row + side) * width + branch,
                                                            )
                                                            .unwrap_or(addresses)
                                                })
                                                .sum()
                                        });
                                        assert_eq!((lo, hi), (expected[0], expected[1]));
                                    },
                                );
                                assert!(seen.into_iter().all(|visited| visited));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bulk_instruction_decoding_matches_selectors_fr() {
        check::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn bulk_instruction_decoding_matches_selectors_fp128() {
        check::<jolt_field::Prime128OffsetA7F7>();
    }
}
