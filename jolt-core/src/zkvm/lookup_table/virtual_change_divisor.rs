use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualChangeDivisorTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for VirtualChangeDivisorTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        // VirtualChangeDivisor takes remainder as first operand and divisor as second
        // If remainder == MIN && divisor == -1, return 1, otherwise return divisor
        let (remainder, divisor) = uninterleave_bits(index);

        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                let remainder = remainder as i8;
                let divisor = divisor as i8;
                if remainder == i8::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u8 as u64
                }
            }
            32 => {
                let remainder = remainder as i32;
                let divisor = divisor as i32;
                if remainder == i32::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u32 as u64
                }
            }
            64 => {
                let remainder = remainder as i64;
                let divisor = divisor as i64;
                if remainder == i64::MIN && divisor == -1 {
                    1
                } else {
                    divisor as u64
                }
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        // MLE: f(x, y) = Σᵢ yᵢ * 2^(n-1-i) + x₀ * ∏ᵢ₌₁ⁿ⁻¹(1 - xᵢ) * ∏ᵢ₌₀ⁿ⁻¹ yᵢ * (2 - 2^n)

        let mut divisor_value = F::zero();
        for i in 0..WORD_SIZE {
            let bit_value = r[2 * i + 1];
            let shift = WORD_SIZE - 1 - i;
            if shift >= 64 {
                divisor_value += F::from_u128(1u128 << shift) * bit_value;
            } else {
                divisor_value += F::from_u64(1u64 << shift) * bit_value;
            }
        }

        let mut x_product = r[0];
        for i in 1..WORD_SIZE {
            x_product *= F::one() - r[2 * i];
        }

        let mut y_product = F::one();
        for i in 0..WORD_SIZE {
            y_product *= r[2 * i + 1];
        }

        let adjustment = if WORD_SIZE == 64 {
            F::from_u64(2) - F::from_u128(1u128 << 64)
        } else {
            F::from_u64(2) - F::from_u64(1u64 << WORD_SIZE)
        };

        divisor_value + x_product * y_product * adjustment
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for VirtualChangeDivisorTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![]
    }

    fn combine<F: JoltField>(&self, _prefixes: &[PrefixEval<F>], _suffixes: &[SuffixEval<F>]) -> F {
        todo!("combine for VirtualChangeDivisorTable")
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
    #[ignore] // Remove when prefix-suffix decomposition is implemented
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
