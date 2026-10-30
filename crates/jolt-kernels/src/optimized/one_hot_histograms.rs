use jolt_field::{Accumulator, JoltField};
use jolt_poly::{EqPolynomial, TensorEqTable};
use jolt_witness::witnesses::{BytecodePc, LookupIndex, RaChunkSelector, RemappedRamAddress};
use jolt_witness::{WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::constant_suffix::ConstantSuffix;
use super::instruction_read_raf::InstructionCycleRow;
use crate::KernelError;

/// Per-family chunk selectors in canonical layout order.
pub(crate) struct FamilySelectors {
    instruction: Vec<RaChunkSelector>,
    bytecode: Vec<RaChunkSelector>,
    ram: Vec<RaChunkSelector>,
}

impl FamilySelectors {
    pub(crate) fn new<F: JoltField>(
        counts: (usize, usize, usize),
        chunk_bits: usize,
    ) -> Result<Self, KernelError<F>> {
        let family = |count: usize| -> Result<Vec<RaChunkSelector>, KernelError<F>> {
            (0..count)
                .map(|index| {
                    RaChunkSelector::new(index, count, chunk_bits).map_err(KernelError::from)
                })
                .collect()
        };
        Ok(Self {
            instruction: family(counts.0)?,
            bytecode: family(counts.1)?,
            ram: family(counts.2)?,
        })
    }
}

/// Fold equality weights into small selector histograms without a trace-sized
/// equality table. The outer tensor weight is applied once per histogram
/// bucket, after summing its inner weights with deferred reduction.
pub(crate) fn pushforwards<F: JoltField>(
    rows: &ConstantSuffix<InstructionCycleRow>,
    point: &[F],
    selectors: &FamilySelectors,
    k_chunk: usize,
    omit_digit_zero: bool,
) -> Result<Vec<Vec<F>>, WitnessError> {
    debug_assert_eq!(rows.len(), 1usize << point.len());
    pushforward_access(
        |index| {
            let row = &rows[index];
            Ok(HistogramRow {
                lookup_index: LookupIndex(row.lookup_index()),
                bytecode_pc: BytecodePc(row.bytecode_pc()),
                remapped_ram_address: RemappedRamAddress(row.remapped_ram_address()),
            })
        },
        rows.prefix_len(),
        point,
        selectors,
        k_chunk,
        omit_digit_zero,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub(crate) struct HistogramRow {
    lookup_index: LookupIndex,
    bytecode_pc: BytecodePc,
    remapped_ram_address: RemappedRamAddress,
}

pub(crate) fn pushforward_access<F: JoltField>(
    row_at: impl Fn(usize) -> Result<HistogramRow, WitnessError> + Sync,
    prefix: usize,
    point: &[F],
    selectors: &FamilySelectors,
    k_chunk: usize,
    omit_digit_zero: bool,
) -> Result<Vec<Vec<F>>, WitnessError> {
    let instruction = selectors.instruction.len();
    let bytecode = selectors.bytecode.len();
    let ram = selectors.ram.len();
    let total = instruction + bytecode + ram;
    let eq = TensorEqTable::new(point);
    let in_len = eq.e_in().len();
    let bits = k_chunk.trailing_zeros();
    let mask = (k_chunk - 1) as u128;
    let native_decode = bits.is_power_of_two() && bits <= 32 && instruction <= 128 / bits as usize;
    struct State<F: JoltField> {
        block: Vec<Vec<F::Accumulator>>,
        sums: Vec<Vec<F::Accumulator>>,
    }
    let zero = || State::<F> {
        block: vec![vec![F::Accumulator::default(); k_chunk]; total],
        sums: vec![vec![F::Accumulator::default(); k_chunk]; total],
    };
    let add_row = |buckets: &mut [Vec<F::Accumulator>], row: &HistogramRow, weight: F| {
        let (instruction_buckets, other) = buckets.split_at_mut(instruction);
        let (bytecode_buckets, ram_buckets) = other.split_at_mut(bytecode);
        let index = row.lookup_index.0;
        if native_decode {
            for (buckets, mut limb) in instruction_buckets
                .rchunks_mut(64 / bits as usize)
                .zip([index as u64, (index >> 64) as u64])
            {
                if omit_digit_zero && limb == 0 {
                    continue;
                }
                for bucket in buckets.iter_mut().rev() {
                    let digit = limb as usize & (k_chunk - 1);
                    if !omit_digit_zero || digit != 0 {
                        bucket[digit].add(weight);
                    }
                    limb >>= bits;
                }
            }
        } else {
            let mut index = index;
            for bucket in instruction_buckets.iter_mut().rev() {
                let digit = (index & mask) as usize;
                if !omit_digit_zero || digit != 0 {
                    bucket[digit].add(weight);
                }
                index >>= bits;
            }
        }
        let mut pc = row.bytecode_pc.0;
        for bucket in bytecode_buckets.iter_mut().rev() {
            let digit = pc & (k_chunk - 1);
            if !omit_digit_zero || digit != 0 {
                bucket[digit].add(weight);
            }
            pc >>= bits;
        }
        if let Some(mut address) = row.remapped_ram_address.0 {
            for bucket in ram_buckets.iter_mut().rev() {
                bucket[address as usize & (k_chunk - 1)].add(weight);
                address >>= bits;
            }
        }
    };
    let scatter = |mut state: State<F>, outer: usize| -> Result<State<F>, WitnessError> {
        let base = outer * in_len;
        for (inner, &weight) in eq.e_in().iter().take(prefix - base).enumerate() {
            add_row(&mut state.block, &row_at(base + inner)?, weight);
        }
        let outer_weight = eq.e_out()[outer];
        for (sums, block) in state.sums.iter_mut().zip(&mut state.block) {
            for (sum, bucket) in sums.iter_mut().zip(block) {
                let value = std::mem::take(bucket).reduce();
                sum.fmadd(outer_weight, value);
            }
        }
        Ok(state)
    };
    let merge = |mut left: State<F>, right: State<F>| {
        for (left, right) in left.sums.iter_mut().zip(right.sums) {
            for (left, right) in left.iter_mut().zip(right) {
                left.merge(right);
            }
        }
        left
    };
    let blocks = prefix.div_ceil(in_len);
    #[cfg(feature = "parallel")]
    let mut state = (0..blocks)
        .into_par_iter()
        .try_fold(zero, scatter)
        .try_reduce(zero, |left, right| Ok(merge(left, right)))?;
    #[cfg(not(feature = "parallel"))]
    let mut state = {
        let _ = merge;
        (0..blocks).try_fold(zero(), scatter)?
    };
    add_row(
        &mut state.sums,
        &row_at(prefix)?,
        EqPolynomial::suffix_sum(point, prefix),
    );
    Ok(state
        .sums
        .into_iter()
        .map(|buckets| buckets.into_iter().map(|bucket| bucket.reduce()).collect())
        .collect())
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod pushforward_tests {
    use super::{pushforwards, FamilySelectors, InstructionCycleRow};
    use crate::optimized::constant_suffix::ConstantSuffix;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField};
    use jolt_poly::EqPolynomial;
    #[cfg(feature = "akita")]
    use jolt_witness::witnesses::FusedInc;

    #[cfg(feature = "akita")]
    fn inc(index: u128) -> i128 {
        let magnitude = (index % (1u128 << 64)) as i128;
        if index.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        }
    }

    fn row(index: u128, pc: usize, address: Option<u64>) -> InstructionCycleRow {
        InstructionCycleRow::new(
            index,
            None,
            false,
            pc,
            address,
            #[cfg(feature = "akita")]
            FusedInc(inc(index)),
        )
    }

    fn check<F: JoltField>() {
        for bits in [1, 2, 3, 4, 8] {
            for counts in [(128 / bits, 3, 4), (2, 2, 2), (0, 2, 1), (2, 0, 0)] {
                let selectors = FamilySelectors::new::<F>(counts, bits).unwrap();
                for prefix in [0, 1, 15, 16, 17, 33, 63] {
                    for hot_tail in [false, true] {
                        let tail = if hot_tail {
                            row(u128::MAX, 37, Some(0))
                        } else {
                            row(0, 0, None)
                        };
                        let mut rows = vec![tail; 64];
                        for (j, item) in rows[..prefix].iter_mut().enumerate() {
                            let value = (j as u128 + 1) * 0xdead_beef_1234_5678;
                            let value = (value << 64) | value;
                            *item = row(
                                value,
                                j % 23,
                                if j % 3 == 0 {
                                    None
                                } else {
                                    Some((j % 19) as u64)
                                },
                            );
                        }
                        let compact: ConstantSuffix<_> = rows.clone().into();
                        for boolean_point in [false, true] {
                            let point: Vec<_> = (0..6)
                                .map(|i| {
                                    F::from_u64(if boolean_point { i % 2 } else { 17 + i * 3 })
                                })
                                .collect();
                            let weights = EqPolynomial::<F>::evals(&point, None);
                            for omit_zero in [false, true] {
                                let mut expected = vec![
                                    vec![F::zero(); 1 << bits];
                                    counts.0 + counts.1 + counts.2
                                ];
                                for (row, &weight) in rows.iter().zip(&weights) {
                                    let mut column = 0;
                                    for selector in &selectors.instruction {
                                        expected[column]
                                            [selector.chunk_u128(row.lookup_index())] += weight;
                                        column += 1;
                                    }
                                    for selector in &selectors.bytecode {
                                        expected[column]
                                            [selector.chunk_usize(row.bytecode_pc())] += weight;
                                        column += 1;
                                    }
                                    for selector in &selectors.ram {
                                        if let Some(address) = row.remapped_ram_address() {
                                            expected[column]
                                                [selector.chunk_usize(address as usize)] += weight;
                                        }
                                        column += 1;
                                    }
                                }
                                if omit_zero {
                                    let ram = counts.0 + counts.1..counts.0 + counts.1 + counts.2;
                                    for (index, bucket) in expected.iter_mut().enumerate() {
                                        if !ram.contains(&index) {
                                            bucket[0] = F::zero();
                                        }
                                    }
                                }
                                assert_eq!(
                                    pushforwards(
                                        &compact,
                                        &point,
                                        &selectors,
                                        1 << bits,
                                        omit_zero
                                    )
                                    .unwrap(),
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_histograms_match_dense_equality() {
        check::<Fr>();
    }
    #[cfg(feature = "akita")]
    #[test]
    fn tensor_histograms_match_dense_equality_akita() {
        check::<Prime128OffsetA7F7>();
    }
}
