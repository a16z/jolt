use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualChangeDivisorTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for VirtualChangeDivisorTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (dividend, divisor) = uninterleave_bits(index);

        match XLEN {
            #[cfg(test)]
            8 => {
                let dividend = dividend as i8;
                let divisor = divisor as i8;
                if dividend == i8::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u8 as u64
                }
            }
            32 => {
                let dividend = dividend as i32;
                let divisor = divisor as i32;
                if dividend == i32::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u32 as u64
                }
            }
            64 => {
                let dividend = dividend as i64;
                let divisor = divisor as i64;
                if dividend == i64::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u64
                }
            }
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut divisor_value = F::zero();
        for i in 0..XLEN {
            let bit_value = r[2 * i + 1];
            let shift = XLEN - 1 - i;
            divisor_value += F::from_u128(1u128 << shift) * bit_value;
        }

        let mut x_product = r[0].into();
        for i in 1..XLEN {
            x_product *= F::one() - r[2 * i];
        }

        let mut y_product = F::one();
        for i in 0..XLEN {
            y_product = y_product * r[2 * i + 1];
        }

        let adjustment = F::from_u64(2) - F::from_u128(1u128 << XLEN);

        divisor_value + x_product * y_product * adjustment
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualChangeDivisorTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightOperand,
            Suffixes::ChangeDivisor,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_operand, change_divisor] = suffixes.try_into().unwrap();

        prefixes[Prefixes::RightOperand] * one
            + right_operand
            + prefixes[Prefixes::ChangeDivisor] * change_divisor
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::VirtualChangeDivisorTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualChangeDivisorTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualChangeDivisorTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualChangeDivisorTable<XLEN>>();
    }
}
