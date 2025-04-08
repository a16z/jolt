use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct AssertValidDiv0Instruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for AssertValidDiv0Instruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn lookup_entry(&self) -> u64 {
        let divisor = self.0;
        let quotient = self.1;
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
            }
        } else {
            1
        }
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (divisor, quotient) = uninterleave_bits(index);
        if divisor == 0 {
            match WORD_SIZE {
                8 => (quotient == u8::MAX as u32).into(),
                32 => (quotient == u32::MAX).into(),
                _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
            }
        } else {
            1
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64, rng.next_u32() as u64),
            64 => Self(rng.next_u64(), rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        let mut divisor_is_zero = F::one();
        let mut is_valid_div_by_zero = F::one();

        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            divisor_is_zero *= F::one() - x_i;
            is_valid_div_by_zero *= (F::one() - x_i) * y_i;
        }

        F::one() - divisor_is_zero + is_valid_div_by_zero
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for AssertValidDiv0Instruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::LeftOperandIsZero,
            Suffixes::DivByZero,
        ]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, left_operand_is_zero, div_by_zero] = suffixes.try_into().unwrap();
        // If the divisor is *not* zero, both:
        // `prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero` and
        // `prefixes[Prefixes::DivByZero] * div_by_zero`
        // will be zero.
        //
        // If the divisor *is* zero, returns 1 (on the Boolean hypercube)
        // iff the quotient is valid (i.e. 2^WORD_SIZE - 1).
        one - prefixes[Prefixes::LeftOperandIsZero] * left_operand_is_zero
            + prefixes[Prefixes::DivByZero] * div_by_zero
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::AssertValidDiv0Instruction;

    #[test]
    fn assert_valid_div0_materialize_entry() {
        materialize_entry_test::<Fr, AssertValidDiv0Instruction<32>>();
    }

    #[test]
    fn assert_valid_div0_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, AssertValidDiv0Instruction<8>>();
    }

    #[test]
    fn assert_valid_div0_mle_random() {
        instruction_mle_random_test::<Fr, AssertValidDiv0Instruction<32>>();
    }

    #[test]
    fn assert_valid_div0_prefix_suffix() {
        prefix_suffix_test::<Fr, AssertValidDiv0Instruction<32>>();
    }
}
