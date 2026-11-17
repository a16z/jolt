use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

/// The unsigned byte of `x` addressed by the low three bits of `y`.
/// XLEN=8 uses eight one-bit lanes for full-hypercube tests.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ExtractByteUnsignedTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for ExtractByteUnsignedTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let width = XLEN / 8;
        (x >> (width * (y as usize & 7))) & ((1u64 << width) - 1)
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let width = XLEN / 8;
        (0..8)
            .map(|lane| {
                let mut indicator = F::one();
                for bit in 0..3 {
                    let y: F = r[2 * (XLEN - 1 - bit) + 1].into();
                    indicator *= if lane & (1 << bit) == 0 {
                        F::one() - y
                    } else {
                        y
                    };
                }
                let mut value = F::zero();
                for bit in 0..width {
                    let x: F = r[2 * (XLEN - 1 - lane * width - bit)].into();
                    value += x * F::from_u64(1 << bit);
                }
                indicator * value
            })
            .sum()
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for ExtractByteUnsignedTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[
            Prefixes::ExtractByte0,
            Prefixes::ExtractByte1,
            Prefixes::ExtractByte2,
            Prefixes::ExtractByte3,
            Prefixes::ExtractByte4,
            Prefixes::ExtractByte5,
            Prefixes::ExtractByte6,
            Prefixes::ExtractByte7,
            Prefixes::ByteOffset0,
            Prefixes::ByteOffset1,
            Prefixes::ByteOffset2,
            Prefixes::ByteOffset3,
            Prefixes::ByteOffset4,
            Prefixes::ByteOffset5,
            Prefixes::ByteOffset6,
            Prefixes::ByteOffset7,
        ]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        debug_assert_eq!(XLEN, 64);
        &[
            Suffixes::ByteOffset0,
            Suffixes::ByteOffset1,
            Suffixes::ByteOffset2,
            Suffixes::ByteOffset3,
            Suffixes::ByteOffset4,
            Suffixes::ByteOffset5,
            Suffixes::ByteOffset6,
            Suffixes::ByteOffset7,
            Suffixes::ExtractByteUnsigned,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(suffixes.len(), 9);
        // Once an offset bit enters the MSB-first prefix, all remaining x bits
        // belong to lane zero. Before then every offset prefix equals one.
        // Thus the suffix lane sum needs only the lane-zero offset prefix.
        let mut value = prefixes[Prefixes::ByteOffset0] * suffixes[8];
        for (prefix, suffix) in self.prefixes()[..8].iter().zip(suffixes) {
            value += prefixes[*prefix] * suffix;
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleave_bits;
    use crate::tables::test_utils::{
        mle_full_hypercube_test, mle_random_test, prefix_suffix_materialization_test,
        prefix_suffix_test,
    };
    use jolt_field::{Fr, Ring};

    #[test]
    fn byte_array_oracle() {
        for word in [0u64, u64::MAX, 0x80ff_017f_55aa_00fe, 1 << 63] {
            for (offset, byte) in word.to_le_bytes().iter().enumerate() {
                for address in [offset as u64, 0xffff_ffff_ffff_fff8 | offset as u64] {
                    assert_eq!(
                        ExtractByteUnsignedTable::<64>
                            .materialize_entry(interleave_bits(word, address)),
                        u64::from(*byte)
                    );
                }
            }
        }
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, ExtractByteUnsignedTable<8>>();
    }

    #[test]
    fn non_boolean_mle_matches_dense_table() {
        let r: Vec<Fr> = (2..18).map(Fr::from_u64).collect();
        let mut values: Vec<Fr> = (0..1u128 << 16)
            .map(|index| {
                // Independent one-bit-lane oracle for the reduced hypercube.
                let mut address = 0usize;
                for bit in 0..3 {
                    address |= (((index >> (2 * bit)) & 1) as usize) << bit;
                }
                Fr::from_u64(((index >> (2 * address + 1)) & 1) as u64)
            })
            .collect();
        for challenge in &r {
            let half = values.len() / 2;
            for i in 0..half {
                values[i] = values[i] + *challenge * (values[i + half] - values[i]);
            }
            values.truncate(half);
        }
        assert_eq!(
            ExtractByteUnsignedTable::<8>.evaluate_mle::<Fr, Fr>(&r),
            values[0]
        );
    }

    #[test]
    fn mle_random() {
        mle_random_test::<64, Fr, ExtractByteUnsignedTable<64>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<64, Fr, ExtractByteUnsignedTable<64>>();
        for phase_bits in [1, 2, 4] {
            prefix_suffix_materialization_test::<64, Fr, ExtractByteUnsignedTable<64>>(
                phase_bits, 3,
            );
        }
    }
}
