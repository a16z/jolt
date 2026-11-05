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
    struct State<F: JoltField> {
        instruction: InstructionHistogram<F>,
        other: Vec<Vec<F::Accumulator>>,
        sums: Vec<Vec<F::Accumulator>>,
    }
    let zero_sums = || vec![vec![F::Accumulator::default(); k_chunk]; total];
    let zero = || State::<F> {
        instruction: InstructionHistogram::new(instruction, k_chunk, in_len, omit_digit_zero),
        other: vec![vec![F::Accumulator::default(); k_chunk]; bytecode + ram],
        sums: zero_sums(),
    };
    let bits = k_chunk.trailing_zeros();
    let add_other = move |buckets: &mut [Vec<F::Accumulator>], row: &HistogramRow, weight: F| {
        let (bytecode_buckets, ram_buckets) = buckets.split_at_mut(bytecode);
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
            let row = row_at(base + inner)?;
            state.instruction.add(row.lookup_index.0, weight);
            add_other(&mut state.other, &row, weight);
        }
        let outer_weight = eq.e_out()[outer];
        let (instruction_sums, other_sums) = state.sums.split_at_mut(instruction);
        state.instruction.fold_into(instruction_sums, outer_weight);
        for (sums, block) in other_sums.iter_mut().zip(&mut state.other) {
            for (sum, bucket) in sums.iter_mut().zip(block) {
                let value = std::mem::take(bucket).reduce();
                sum.fmadd(outer_weight, value);
            }
        }
        Ok(state)
    };
    let merge = |mut left: Vec<Vec<F::Accumulator>>, right: Vec<Vec<F::Accumulator>>| {
        for (left, right) in left.iter_mut().zip(right) {
            for (left, right) in left.iter_mut().zip(right) {
                left.merge(right);
            }
        }
        left
    };
    let blocks = prefix.div_ceil(in_len);
    #[cfg(feature = "parallel")]
    let mut sums = (0..blocks)
        .into_par_iter()
        .try_fold(zero, scatter)
        .map(|state| state.map(|state| state.sums))
        .try_reduce_with(|left, right| Ok(merge(left, right)))
        .transpose()?
        .unwrap_or_else(zero_sums);
    #[cfg(not(feature = "parallel"))]
    let mut sums = {
        let _ = merge;
        (0..blocks).try_fold(zero(), scatter)?.sums
    };
    let tail = row_at(prefix)?;
    let weight = EqPolynomial::suffix_sum(point, prefix);
    let (instruction_sums, other_sums) = sums.split_at_mut(instruction);
    add_instruction_digits(
        instruction_sums,
        tail.lookup_index.0,
        weight,
        k_chunk,
        omit_digit_zero,
    );
    add_other(other_sums, &tail, weight);
    Ok(sums
        .into_iter()
        .map(|buckets| buckets.into_iter().map(|bucket| bucket.reduce()).collect())
        .collect())
}

const PAIRED_K_CHUNK: usize = 16;
const PAIR_BUCKETS: usize = PAIRED_K_CHUNK * PAIRED_K_CHUNK;

// Marginalization and clearing grow with accumulator width. Pair selectors
// only on large enough blocks, with zero buckets omitted: dense zero updates
// can dominate the larger tables even when the instruction count falls.
// Pairs always omit zero digits, so only `Digits` carries the zero policy.
enum InstructionHistogram<F: JoltField> {
    Digits {
        buckets: Vec<Vec<F::Accumulator>>,
        k_chunk: usize,
        omit_zero: bool,
    },
    Pairs(Vec<[F::SumAccumulator; PAIR_BUCKETS]>),
}

impl<F: JoltField> InstructionHistogram<F> {
    fn new(columns: usize, k_chunk: usize, block_len: usize, omit_zero: bool) -> Self {
        // Blocks need 32 rows per accumulator byte to amortize clearing and
        // marginalizing the tables: 1,024 rows for Fp128 and 4,096 for Fr.
        // Smaller Fr blocks measured slower than direct digits.
        if omit_zero
            && k_chunk == PAIRED_K_CHUNK
            && columns > 0
            && columns <= 32
            && columns.is_multiple_of(2)
            && block_len >= 1024
            && std::mem::size_of::<F::SumAccumulator>() <= block_len / 32
        {
            Self::Pairs(vec![
                [F::SumAccumulator::default(); PAIR_BUCKETS];
                columns / 2
            ])
        } else {
            Self::Digits {
                buckets: vec![vec![F::Accumulator::default(); k_chunk]; columns],
                k_chunk,
                omit_zero,
            }
        }
    }

    fn add(&mut self, index: u128, weight: F) {
        match self {
            Self::Digits {
                buckets,
                k_chunk,
                omit_zero,
            } => add_instruction_digits(buckets, index, weight, *k_chunk, *omit_zero),
            Self::Pairs(tables) => {
                add_instruction_digits(tables, index, weight, PAIR_BUCKETS, true);
            }
        }
    }

    fn fold_into(&mut self, sums: &mut [Vec<F::Accumulator>], weight: F) {
        match self {
            Self::Digits { buckets, .. } => {
                for (sums, buckets) in sums.iter_mut().zip(buckets) {
                    for (sum, bucket) in sums.iter_mut().zip(buckets) {
                        sum.fmadd(weight, std::mem::take(bucket).reduce());
                    }
                }
            }
            Self::Pairs(tables) => {
                for (table, sums) in tables.iter_mut().zip(sums.chunks_exact_mut(2)) {
                    let (high, low) = sums.split_at_mut(1);
                    for digit in 1..PAIRED_K_CHUNK {
                        let mut high_sum = table[PAIRED_K_CHUNK * digit];
                        let mut low_sum = table[digit];
                        for other in 1..PAIRED_K_CHUNK {
                            high_sum.merge(table[PAIRED_K_CHUNK * digit + other]);
                            low_sum.merge(table[PAIRED_K_CHUNK * other + digit]);
                        }
                        high[0][digit].fmadd(weight, high_sum.reduce());
                        low[0][digit].fmadd(weight, low_sum.reduce());
                    }
                    table.fill(F::SumAccumulator::default());
                }
            }
        }
    }
}

fn add_instruction_digits<A: Accumulator, B: AsMut<[A]>>(
    buckets: &mut [B],
    index: u128,
    weight: A::Element,
    k_chunk: usize,
    omit_zero: bool,
) {
    let bits = k_chunk.trailing_zeros();
    if bits.is_power_of_two() && bits <= 32 && buckets.len() <= 128 / bits as usize {
        for (buckets, mut limb) in buckets
            .rchunks_mut(64 / bits as usize)
            .zip([index as u64, (index >> 64) as u64])
        {
            if omit_zero && limb == 0 {
                continue;
            }
            for bucket in buckets.iter_mut().rev() {
                let bucket = bucket.as_mut();
                let digit = limb as usize & (k_chunk - 1);
                if !omit_zero || digit != 0 {
                    bucket[digit].add(weight);
                }
                limb >>= bits;
            }
        }
    } else {
        let mut index = index;
        for bucket in buckets.iter_mut().rev() {
            let bucket = bucket.as_mut();
            let digit = (index & (k_chunk - 1) as u128) as usize;
            if !omit_zero || digit != 0 {
                bucket[digit].add(weight);
            }
            index >>= bits;
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod pushforward_tests {
    use super::{pushforwards, FamilySelectors, InstructionCycleRow, InstructionHistogram};
    use crate::optimized::constant_suffix::ConstantSuffix;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Accumulator, Fr, JoltField};
    use jolt_poly::EqPolynomial;
    #[cfg(feature = "akita")]
    use jolt_poly::TensorEqTable;
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

    fn trace(prefix: usize, len: usize, tail: InstructionCycleRow) -> Vec<InstructionCycleRow> {
        let mut rows = vec![tail; len];
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
        rows
    }

    fn dense_pushforwards<F: JoltField>(
        rows: &[InstructionCycleRow],
        point: &[F],
        selectors: &FamilySelectors,
        k_chunk: usize,
        omit_zero: bool,
    ) -> Vec<Vec<F>> {
        let (instruction, bytecode, ram) = (
            selectors.instruction.len(),
            selectors.bytecode.len(),
            selectors.ram.len(),
        );
        let mut expected = vec![vec![F::zero(); k_chunk]; instruction + bytecode + ram];
        for (row, &weight) in rows.iter().zip(&EqPolynomial::<F>::evals(point, None)) {
            let mut column = 0;
            for selector in &selectors.instruction {
                expected[column][selector.chunk_u128(row.lookup_index())] += weight;
                column += 1;
            }
            for selector in &selectors.bytecode {
                expected[column][selector.chunk_usize(row.bytecode_pc())] += weight;
                column += 1;
            }
            for selector in &selectors.ram {
                if let Some(address) = row.remapped_ram_address() {
                    expected[column][selector.chunk_usize(address as usize)] += weight;
                }
                column += 1;
            }
        }
        if omit_zero {
            let ram = instruction + bytecode..instruction + bytecode + ram;
            for (index, bucket) in expected.iter_mut().enumerate() {
                if !ram.contains(&index) {
                    bucket[0] = F::zero();
                }
            }
        }
        expected
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
                        let rows = trace(prefix, 64, tail);
                        let compact: ConstantSuffix<_> = rows.clone().into();
                        for boolean_point in [false, true] {
                            let point: Vec<_> = (0..6)
                                .map(|i| {
                                    F::from_u64(if boolean_point { i % 2 } else { 17 + i * 3 })
                                })
                                .collect();
                            for omit_zero in [false, true] {
                                assert_eq!(
                                    pushforwards(
                                        &compact,
                                        &point,
                                        &selectors,
                                        1 << bits,
                                        omit_zero
                                    )
                                    .unwrap(),
                                    dense_pushforwards(
                                        &rows,
                                        &point,
                                        &selectors,
                                        1 << bits,
                                        omit_zero
                                    )
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_marginals<F: JoltField>() {
        for columns in [2, 16, 18, 32] {
            for omit_zero in [false, true] {
                let mut histogram = InstructionHistogram::<F>::new(columns, 16, 4096, omit_zero);
                let mut sums = vec![vec![F::Accumulator::default(); 16]; columns];
                let mut expected = vec![vec![F::zero(); 16]; columns];
                for outer in [17, 0, 23] {
                    let outer = F::from_u64(outer);
                    for i in 0..300u128 {
                        let index = if i < 256 {
                            i | (i << 120)
                        } else {
                            i.wrapping_mul(0xdead_beef_0123_4567_89ab_cdef_fedc_ba98)
                        };
                        let weight = F::from_u128(((i + 1) << 96) | (i * i + 31));
                        histogram.add(index, weight);
                        for (column, buckets) in expected.iter_mut().enumerate() {
                            let digit = ((index >> (4 * (columns - column - 1))) & 15) as usize;
                            if !omit_zero || digit != 0 {
                                buckets[digit] += weight * outer;
                            }
                        }
                    }
                    histogram.fold_into(&mut sums, outer);
                    let actual: Vec<Vec<F>> = sums
                        .iter()
                        .map(|buckets| buckets.iter().map(|bucket| bucket.reduce()).collect())
                        .collect();
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn paired_instruction_marginals_match_direct_buckets() {
        check_marginals::<Fr>();
    }

    #[cfg(feature = "akita")]
    #[test]
    fn paired_instruction_marginals_match_direct_buckets_akita() {
        check_marginals::<Prime128OffsetA7F7>();
    }

    #[cfg(feature = "akita")]
    fn check_paired<F: JoltField>() {
        let point: Vec<_> = (0..20).map(|i| F::from_u64(17 + 3 * i)).collect();
        let block_len = TensorEqTable::new(&point).e_in().len();
        let counts = (32, 3, 4);
        assert!(matches!(
            InstructionHistogram::<F>::new(counts.0, 16, block_len, true),
            InstructionHistogram::Pairs(_)
        ));
        let selectors = FamilySelectors::new::<F>(counts, 4).unwrap();
        for prefix in [0, block_len, 2 * block_len + 37] {
            let rows = trace(prefix, 1 << point.len(), row(u128::MAX, 37, Some(0)));
            let compact: ConstantSuffix<_> = rows.clone().into();
            assert_eq!(
                pushforwards(&compact, &point, &selectors, 16, true).unwrap(),
                dense_pushforwards(&rows, &point, &selectors, 16, true)
            );
        }
    }

    #[cfg(feature = "akita")]
    #[test]
    fn paired_histograms_match_dense_equality_through_pushforwards() {
        check_paired::<Prime128OffsetA7F7>();
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
