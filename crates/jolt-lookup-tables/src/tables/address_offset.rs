use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::address_offset::AddressOffsetPrefix;
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Intra-doubleword offset of a floor-addressed memory row with a
/// `WIDTH`-byte access: `index & (8 − WIDTH)`, i.e. `ea & 7`, `ea & 6` and
/// `ea & 4` for byte, halfword and word accesses.
///
/// The (non-interleaved) index is the effective address `rs1 + imm`. R1CS
/// constraint 0 subtracts the output from `ea` to form `RamAddress`, which
/// therefore keeps exactly the offset bits natural alignment clears; see that
/// constraint for why RAM RAF evaluation then rejects a misaligned access.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AddressOffsetTable<const XLEN: usize, const WIDTH: usize>;

impl<const XLEN: usize, const WIDTH: usize> AddressOffsetTable<XLEN, WIDTH> {
    // `8 − WIDTH` clears the low log2(WIDTH) offset bits only because
    // WIDTH ∈ {1, 2, 4} is a power of two.
    pub(crate) const OFFSET_MASK: u128 = (8 - WIDTH) as u128;

    const PREFIXES: &'static [Prefixes] = &[AddressOffsetPrefix::<WIDTH>::VARIANT];

    const SUFFIXES: &'static [Suffixes] = match WIDTH {
        1 => &[Suffixes::One, Suffixes::AddressOffsetB],
        2 => &[Suffixes::One, Suffixes::AddressOffsetH],
        4 => &[Suffixes::One, Suffixes::AddressOffsetW],
        _ => panic!("unsupported WIDTH"),
    };
}

impl<const XLEN: usize, const WIDTH: usize> LookupTable for AddressOffsetTable<XLEN, WIDTH> {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index & Self::OFFSET_MASK) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for bit in 0..3 {
            if (Self::OFFSET_MASK >> bit) & 1 == 1 {
                let r_bit: F = r[r.len() - 1 - bit].into();
                result += F::from_u64(1 << bit) * r_bit;
            }
        }
        result
    }
}

impl<const XLEN: usize, const WIDTH: usize> PrefixSuffixDecomposition<XLEN>
    for AddressOffsetTable<XLEN, WIDTH>
{
    fn prefixes(&self) -> &'static [Prefixes] {
        Self::PREFIXES
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        Self::SUFFIXES
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, offset] = suffixes.try_into().unwrap();
        prefixes[AddressOffsetPrefix::<WIDTH>::VARIANT] * one + offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{
        mle_full_hypercube_test, mle_random_test, prefix_suffix_materialization_test,
        prefix_suffix_test,
    };
    use crate::XLEN;
    use jolt_field::Fr;

    macro_rules! address_offset_tests {
        ($module:ident, $width:literal) => {
            mod $module {
                use super::*;

                #[test]
                fn mle_random() {
                    mle_random_test::<XLEN, Fr, AddressOffsetTable<XLEN, $width>>();
                }

                #[test]
                fn prefix_suffix() {
                    prefix_suffix_test::<XLEN, Fr, AddressOffsetTable<XLEN, $width>>();
                }

                /// Two-round phases put a phase boundary inside the low three
                /// index bits, so the offset is split between the prefix
                /// checkpoint, the phase bits and the suffix.
                #[test]
                fn prefix_suffix_small_phases() {
                    prefix_suffix_materialization_test::<XLEN, Fr, AddressOffsetTable<XLEN, $width>>(
                        2, 3,
                    );
                }

                #[test]
                fn mle_full_hypercube() {
                    mle_full_hypercube_test::<8, Fr, AddressOffsetTable<8, $width>>();
                }
            }
        };
    }

    address_offset_tests!(byte, 1);
    address_offset_tests!(halfword, 2);
    address_offset_tests!(word, 4);
}
