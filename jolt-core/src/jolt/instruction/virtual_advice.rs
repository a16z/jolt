use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::instruction_utils::{chunk_operand_usize, concatenate_lookups};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ADVICEInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ADVICEInstruction<WORD_SIZE> {
    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(msb_chunk_index + 1..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Same for both 32-bit and 64-bit word sizes
        self.0
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        index % (1 << WORD_SIZE)
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
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            result += F::from_u64(1 << (WORD_SIZE - 1 - i)) * r[WORD_SIZE + i];
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for ADVICEInstruction<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWord]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerWord] * one + lower_word
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::ADVICEInstruction;
    use crate::{
        jolt::instruction::{
            test::{
                instruction_mle_full_hypercube_test, instruction_mle_random_test,
                materialize_entry_test, prefix_suffix_test,
            },
            JoltInstruction,
        },
        jolt_instruction_test,
    };

    #[test]
    fn advice_materialize_entry() {
        materialize_entry_test::<Fr, ADVICEInstruction<32>>();
    }

    #[test]
    fn advice_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, ADVICEInstruction<8>>();
    }

    #[test]
    fn advice_mle_random() {
        instruction_mle_random_test::<Fr, ADVICEInstruction<32>>();
    }

    #[test]
    fn advice_prefix_suffix() {
        prefix_suffix_test::<Fr, ADVICEInstruction<32>>();
    }

    #[test]
    fn advice_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = ADVICEInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ADVICEInstruction::<WORD_SIZE>(0),
            ADVICEInstruction::<WORD_SIZE>(1),
            ADVICEInstruction::<WORD_SIZE>(100),
            ADVICEInstruction::<WORD_SIZE>(1 << 8),
            ADVICEInstruction::<WORD_SIZE>(1 << 16),
            ADVICEInstruction::<WORD_SIZE>(u32_max),
            ADVICEInstruction::<WORD_SIZE>(u32_max - 101),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn advice_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = ADVICEInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ADVICEInstruction::<WORD_SIZE>(0),
            ADVICEInstruction::<WORD_SIZE>(1),
            ADVICEInstruction::<WORD_SIZE>(100),
            ADVICEInstruction::<WORD_SIZE>(1 << 8),
            ADVICEInstruction::<WORD_SIZE>(1 << 16),
            ADVICEInstruction::<WORD_SIZE>(1 << 32),
            ADVICEInstruction::<WORD_SIZE>(1 << (48 + 2)),
            ADVICEInstruction::<WORD_SIZE>(u64_max),
            ADVICEInstruction::<WORD_SIZE>(u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
