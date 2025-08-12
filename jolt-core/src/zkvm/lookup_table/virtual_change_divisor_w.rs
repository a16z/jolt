use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use crate::zkvm::lookup_table::prefixes::Prefixes;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualChangeDivisorWTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for VirtualChangeDivisorWTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (remainder, divisor) = uninterleave_bits(index);
        match WORD_SIZE {
            8 => {
                let remainder = ((remainder & 0xF) as i8) << 4 >> 4;
                let divisor = ((divisor & 0xF) as i8) << 4 >> 4;

                if remainder == -8 && divisor == -1 {
                    1
                } else {
                    divisor as u8 as u64
                }
            }
            64 => {
                let remainder = remainder as u32 as i32;
                let divisor = divisor as u32 as i32;

                if remainder == i32::MIN && divisor == -1 {
                    1
                } else {
                    divisor as i64 as u64
                }
            }
            _ => panic!("Unsupported {WORD_SIZE} word size"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let sign_bit = r[WORD_SIZE + 1];

        let mut divisor_value = F::zero();
        for i in WORD_SIZE / 2..WORD_SIZE {
            let bit_value = r[2 * i + 1];
            let shift = WORD_SIZE - 1 - i;
            if shift >= 64 {
                divisor_value += F::from_u128(1u128 << shift) * bit_value;
            } else {
                divisor_value += F::from_u64(1u64 << shift) * bit_value;
            }
        }

        let mut x_product = r[WORD_SIZE];
        for i in WORD_SIZE / 2 + 1..WORD_SIZE {
            x_product *= F::one() - r[2 * i];
        }

        let mut y_product = F::one();
        for i in WORD_SIZE / 2..WORD_SIZE {
            y_product *= r[2 * i + 1];
        }

        let sign_extension =
            F::from_u128((1u128 << WORD_SIZE) - (1u128 << (WORD_SIZE / 2))) * sign_bit;

        let adjustment = F::from_u64(2) - F::from_u128(1u128 << WORD_SIZE);

        divisor_value + adjustment * x_product * y_product + sign_extension
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for VirtualChangeDivisorWTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightOperandW,
            Suffixes::ChangeDivisorW,
            Suffixes::SignExtensionRightOperand,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_operand_w, change_divisor_w, sign_extension] = suffixes.try_into().unwrap();

        prefixes[Prefixes::RightOperandW] * one
            + right_operand_w
            + prefixes[Prefixes::ChangeDivisorW] * change_divisor_w
            + prefixes[Prefixes::SignExtensionRightOperand] * sign_extension
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::VirtualChangeDivisorWTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualChangeDivisorWTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualChangeDivisorWTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualChangeDivisorWTable<XLEN>>();
    }
}
