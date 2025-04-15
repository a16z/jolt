use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;
use crate::field::JoltField;
use crate::jolt::instruction::prefixes::Prefixes;
use crate::subprotocols::sparse_dense_shout::{LookupBits, PrefixSuffixDecomposition};
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualSRLInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for VirtualSRLInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn lookup_entry(&self) -> u64 {
        let mut x = LookupBits::new(self.0 as u64, WORD_SIZE);
        let mut y = LookupBits::new(self.1 as u64, WORD_SIZE);

        let mut entry = 0;
        for _ in 0..WORD_SIZE {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            entry *= 1 + y_i as u64;
            entry += (x_i * y_i) as u64;
        }
        entry
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mut x = LookupBits::new(x as u64, WORD_SIZE);
        let mut y = LookupBits::new(y as u64, WORD_SIZE);

        let mut entry = 0;
        for _ in 0..WORD_SIZE {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            entry *= 1 + y_i as u64;
            entry += (x_i * y_i) as u64;
        }
        entry
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        let shift = rng.next_u64() % WORD_SIZE as u64;
        let y = (1 << WORD_SIZE) - (1 << shift);
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), y),
            32 => Self(rng.next_u32() as u64, y),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result *= F::one() + y_i;
            result += x_i * y_i;
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for VirtualSRLInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::RightShift, Suffixes::RightShiftHelper]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [right_shift, right_shift_helper] = suffixes.try_into().unwrap();
        prefixes[Prefixes::RightShift] * right_shift_helper + right_shift
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::VirtualSRLInstruction;
    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    #[test]
    fn virtual_srl_materialize_entry() {
        materialize_entry_test::<Fr, VirtualSRLInstruction<32>>();
    }

    #[test]
    fn virtual_srl_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, VirtualSRLInstruction<8>>();
    }

    #[test]
    fn virtual_srl_mle_random() {
        instruction_mle_random_test::<Fr, VirtualSRLInstruction<32>>();
    }

    #[test]
    fn virtual_srl_prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualSRLInstruction<32>>();
    }
}
