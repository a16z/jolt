use super::{LazyFoldedRa, LookupIndexChunks};
use crate::optimized::lazy_ra::{ChunkIndexSource, PairedBranchTables};
use jolt_field::JoltField;

pub(super) struct PairedInstructionGather<'a, F: JoltField> {
    source: &'a LookupIndexChunks,
    width: PairWidth,
    tables: PairedBranchTables<F>,
}

enum PairWidth {
    Four,
    Eight,
}

impl<'a, F: JoltField> PairedInstructionGather<'a, F> {
    pub(super) fn new(folded: &'a LazyFoldedRa<F, LookupIndexChunks>) -> Option<Self> {
        let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            source,
            ..
        } = folded
        else {
            return None;
        };
        let pair_width = match width {
            4 => PairWidth::Four,
            8 => PairWidth::Eight,
            _ => return None,
        };
        // One table entry per sixteen live rows amortizes construction; the
        // branch-table length pins the source's four-bit address geometry.
        if tables.iter().any(|table| table.len() != width * 16)
            || PairedBranchTables::<F>::entries(*width) > source.constant_suffix_start() / 16
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
    pub(super) fn gather(&self, row: usize, out: &mut [(F, F)]) {
        match self.width {
            PairWidth::Four => self.gather_width::<4>(row, out),
            PairWidth::Eight => self.gather_width::<8>(row, out),
        }
    }

    #[inline(always)]
    fn gather_width<const WIDTH: usize>(&self, row: usize, out: &mut [(F, F)]) {
        for side in 0..2 {
            let start = (2 * row + side) * WIDTH;
            self.tables.visit_four_bit::<WIDTH>(
                self.tables.columns(),
                |branch| self.source.packed_index(start + branch),
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
    use super::*;
    use crate::optimized::constant_suffix::ConstantSuffix;
    use crate::optimized::instruction_read_raf::InstructionCycleRow;
    use jolt_field::Fr;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
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
                        assert!(PairedInstructionGather::new(&lazy).is_none());
                        continue;
                    }
                    let paired = PairedInstructionGather::new(&lazy).unwrap();
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
