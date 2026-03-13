use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualChangeDivisorWTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable<XLEN> for VirtualChangeDivisorWTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (dividend, divisor) = uninterleave_bits(index);
        match XLEN {
            #[cfg(test)]
            8 => {
                let dividend = ((dividend & 0xF) as i8) << 4 >> 4;
                let divisor = ((divisor & 0xF) as i8) << 4 >> 4;
                if dividend == -8 && divisor == -1 {
                    1
                } else {
                    divisor as u8 as u64
                }
            }
            64 => {
                let dividend = dividend as u32 as i32;
                let divisor = divisor as u32 as i32;
                if dividend == i32::MIN && divisor == -1 {
                    1
                } else {
                    divisor as i64 as u64
                }
            }
            _ => panic!("Unsupported {XLEN} word size"),
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let sign_bit = r[XLEN + 1];

        let mut divisor_value = F::zero();
        for i in XLEN / 2..XLEN {
            let bit_value = r[2 * i + 1];
            let shift = XLEN - 1 - i;
            divisor_value += F::from_u64(1u64 << shift) * bit_value;
        }

        let mut x_product = r[XLEN].into();
        for i in XLEN / 2 + 1..XLEN {
            x_product *= F::one() - r[2 * i];
        }

        let mut y_product = F::one();
        for i in XLEN / 2..XLEN {
            y_product = y_product * r[2 * i + 1];
        }

        let sign_extension = F::from_u128((1u128 << XLEN) - (1u128 << (XLEN / 2))) * sign_bit;
        let adjustment = F::from_u64(2) - F::from_u128(1u128 << XLEN);

        divisor_value + adjustment * x_product * y_product + sign_extension
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for VirtualChangeDivisorWTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::RightOperandW,
            Suffixes::ChangeDivisorW,
            Suffixes::SignExtensionRightOperand,
        ]
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, right_operand_w, change_divisor_w, sign_extension] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightOperandW] * one
            + right_operand_w
            + prefixes[Prefixes::ChangeDivisorW] * change_divisor_w
            + prefixes[Prefixes::SignExtensionRightOperand] * sign_extension
    }
}
