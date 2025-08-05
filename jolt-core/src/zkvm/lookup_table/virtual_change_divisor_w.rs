use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualChangeDivisorWTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for VirtualChangeDivisorWTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        // VirtualChangeDivisorW handles 32-bit division in 64-bit mode
        // If lower 32 bits represent (INT32_MIN, -1), return 1, otherwise return divisor
        let (remainder, divisor) = uninterleave_bits(index);

        // Check if lower 32 bits match the special case
        let remainder_lower32 = remainder as u32 as i32;
        let divisor_lower32 = divisor as u32 as i32;

        if remainder_lower32 == i32::MIN && divisor_lower32 == -1 {
            1
        } else {
            divisor
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        // MLE: f(x, y) = Σᵢ yᵢ * 2^(n-1-i) + x₃₁ * ∏ᵢ₌₀³⁰(1 - xᵢ) * ∏ᵢ₌₀³¹ yᵢ * (2 - 2^32)
        // For W variant, we check lower 32 bits for INT32_MIN and -1
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

        // Check lower 32 bits for x == INT32_MIN (bit 31 = 1, bits 0-30 = 0)
        let mut x_product = if 31 < WORD_SIZE {
            r[2 * 31] // x₃₁ should be 1
        } else {
            F::zero()
        };

        for i in 0..31.min(WORD_SIZE) {
            x_product *= F::one() - r[2 * i]; // (1 - xᵢ) for i < 31
        }

        // Check lower 32 bits for y == -1 (all 32 bits = 1)
        let mut y_product = F::one();
        for i in 0..32.min(WORD_SIZE) {
            y_product *= r[2 * i + 1];
        }

        let adjustment = F::from_u64(2) - F::from_u64(1u64 << 32);

        divisor_value + x_product * y_product * adjustment
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for VirtualChangeDivisorWTable<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![]
    }

    fn combine<F: JoltField>(&self, _prefixes: &[PrefixEval<F>], _suffixes: &[SuffixEval<F>]) -> F {
        todo!("combine for VirtualChangeDivisorWTable")
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::instruction_lookups::WORD_SIZE;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::VirtualChangeDivisorWTable;

    #[test]
    #[ignore] // Remove when prefix-suffix decomposition is implemented
    fn prefix_suffix() {
        prefix_suffix_test::<WORD_SIZE, Fr, VirtualChangeDivisorWTable<WORD_SIZE>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualChangeDivisorWTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualChangeDivisorWTable<WORD_SIZE>>();
    }
}
