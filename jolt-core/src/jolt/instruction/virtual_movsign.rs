use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{field::JoltField, utils::interleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MOVSIGNInstruction<const WORD_SIZE: usize>(pub u64);

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

impl<const WORD_SIZE: usize> JoltInstruction for MOVSIGNInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, 0)
    }

    fn lookup_entry(&self) -> u64 {
        match WORD_SIZE {
            32 => {
                if self.0 & SIGN_BIT_32 != 0 {
                    ALL_ONES_32
                } else {
                    0
                }
            }
            64 => {
                if self.0 & SIGN_BIT_64 != 0 {
                    ALL_ONES_64
                } else {
                    0
                }
            }
            _ => panic!("only implemented for u32 / u64"),
        }
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit = 1 << (2 * WORD_SIZE - 1);
        if index & sign_bit != 0 {
            (1 << WORD_SIZE) - 1
        } else {
            0
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64),
            64 => Self(rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        // 2 ^ {WORD_SIZE - 1} * x_0
        debug_assert!(r.len() == 2 * WORD_SIZE);

        let sign_bit = r[0];
        let ones: u64 = (1 << WORD_SIZE) - 1;
        sign_bit * F::from_u64(ones)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for MOVSIGNInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one] = suffixes.try_into().unwrap();
        let ones: u64 = (1 << WORD_SIZE) - 1;
        F::from_u64(ones) * prefixes[Prefixes::LeftOperandMsb] * one
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::MOVSIGNInstruction;

    #[test]
    fn virtual_movsign_materialize_entry() {
        materialize_entry_test::<Fr, MOVSIGNInstruction<32>>();
    }

    #[test]
    fn virtual_movsign_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, MOVSIGNInstruction<8>>();
    }

    #[test]
    fn virtual_movsign_mle_random() {
        instruction_mle_random_test::<Fr, MOVSIGNInstruction<32>>();
    }

    #[test]
    fn virtual_movsign_prefix_suffix() {
        prefix_suffix_test::<Fr, MOVSIGNInstruction<32>>();
    }
}
