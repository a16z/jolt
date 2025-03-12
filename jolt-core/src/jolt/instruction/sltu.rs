use crate::{
    field::JoltField,
    jolt::instruction::suffixes::{lt::LessThanSuffix, one::OneSuffix},
    subprotocols::sparse_dense_shout::{LookupBits, SparseDenseSumcheckAlt},
    utils::{interleave_bits, uninterleave_bits},
};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{suffixes::Suffixes, JoltInstruction};
use crate::{
    jolt::{
        instruction::SubtableIndices,
        subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SLTUInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SLTUInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];

        let mut sum = F::zero();
        let mut eq_prod = F::one();

        for i in 0..C - 1 {
            sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        // Do not need to update `eq_prod` for the last iteration
        sum + ltu[C - 1] * eq_prod
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C - 1)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn eta(&self) -> usize {
        1
    }

    fn subtable_entry(&self, _: usize, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x < y).into()
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, self.1 as u32)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for 32-bit and 64-bit word sizes
        (self.0 < self.1).into()
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

    fn subtable_mle<F: JoltField>(&self, _: usize, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        // \sum_i (1 - x_i) * y_i * \prod_{j < i} ((1 - x_j) * (1 - y_j) + x_j * y_j)
        let mut result = F::zero();
        let mut eq_term = F::one();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += (F::one() - x_i) * y_i * eq_term;
            eq_term *= F::one() - x_i - y_i + x_i * y_i + x_i * y_i;
        }
        result
    }
}

impl<const WORD_SIZE: usize, F: JoltField> SparseDenseSumcheckAlt<WORD_SIZE, F>
    for SLTUInstruction<WORD_SIZE>
{
    const NUM_PREFIXES: usize = 2;

    fn combine(prefixes: &[F], suffixes: &[F]) -> F {
        prefixes[0] * suffixes[0] + prefixes[1] * suffixes[1]
    }

    fn suffixes() -> Vec<Suffixes<WORD_SIZE>> {
        vec![Suffixes::One(OneSuffix), Suffixes::LessThan(LessThanSuffix)]
    }

    fn update_prefix_checkpoints(checkpoints: &mut [Option<F>], r_x: F, r_y: F, j: usize) {
        let lt_checkpoint = checkpoints[0].unwrap_or(F::zero());
        let eq_checkpoint = checkpoints[1].unwrap_or(F::one());
        let lt_updated = lt_checkpoint + eq_checkpoint * (F::one() - r_x) * r_y;
        let eq_updated = eq_checkpoint * (r_x * r_y + (F::one() - r_x) * (F::one() - r_y));
        checkpoints[0] = Some(lt_updated);
        checkpoints[1] = Some(eq_updated);
    }

    fn prefix_mle(
        l: usize,
        checkpoints: &[Option<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        match l {
            0 => {
                let mut lt = checkpoints[0].unwrap_or(F::zero());
                let mut eq = checkpoints[1].unwrap_or(F::one());

                if let Some(r_x) = r_x {
                    let c = F::from_u32(c);
                    lt += (F::one() - r_x) * c * eq;
                    let (x, y) = b.uninterleave();
                    if u64::from(x) < u64::from(y) {
                        eq *= r_x * c + (F::one() - r_x) * (F::one() - c);
                        lt += eq;
                    }
                } else {
                    let c = F::from_u32(c);
                    let y_msb = b.pop_msb();
                    if y_msb == 1 {
                        // lt += eq * (1 - c) * y_msb
                        lt += eq * (F::one() - c);
                    }
                    let (x, y) = b.uninterleave();
                    if u64::from(x) < u64::from(y) {
                        if y_msb == 1 {
                            lt += eq * c;
                        } else {
                            lt += eq * (F::one() - c);
                        }
                    }
                }

                lt
            }
            1 => {
                let eq = checkpoints[1].unwrap_or(F::one());

                if let Some(r_x) = r_x {
                    let (x, y) = b.uninterleave();
                    if x == y {
                        let y = F::from_u32(c);
                        eq * (r_x * y + (F::one() - r_x) * (F::one() - y))
                    } else {
                        F::zero()
                    }
                } else {
                    let y_msb = b.pop_msb();
                    let (x, y) = b.uninterleave();
                    if x == y {
                        let c = F::from_u32(c);
                        if y_msb == 1 {
                            eq * c
                        } else {
                            eq * (F::one() - c)
                        }
                    } else {
                        F::zero()
                    }
                }
            }
            _ => unimplemented!("Unexpected value l={l}"),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{
        instruction_mle_test_large, instruction_mle_test_small,
        jolt::instruction::{test::prefix_suffix_test, JoltInstruction},
        jolt_instruction_test,
    };

    use super::SLTUInstruction;

    #[test]
    fn sltu_prefix_suffix() {
        prefix_suffix_test::<Fr, SLTUInstruction<32>>();
    }

    instruction_mle_test_small!(sltu_mle_small, SLTUInstruction<8>);
    instruction_mle_test_large!(sltu_mle_large, SLTUInstruction<32>);

    #[test]
    fn sltu_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = SLTUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            jolt_instruction_test!(SLTUInstruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SLTUInstruction::<WORD_SIZE>(100, 0),
            SLTUInstruction::<WORD_SIZE>(0, 100),
            SLTUInstruction::<WORD_SIZE>(1, 0),
            SLTUInstruction::<WORD_SIZE>(0, u32_max),
            SLTUInstruction::<WORD_SIZE>(u32_max, 0),
            SLTUInstruction::<WORD_SIZE>(u32_max, u32_max),
            SLTUInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            SLTUInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn sltu_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = SLTUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u64();
            jolt_instruction_test!(SLTUInstruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            SLTUInstruction::<WORD_SIZE>(100, 0),
            SLTUInstruction::<WORD_SIZE>(0, 100),
            SLTUInstruction::<WORD_SIZE>(1, 0),
            SLTUInstruction::<WORD_SIZE>(0, u64_max),
            SLTUInstruction::<WORD_SIZE>(u64_max, 0),
            SLTUInstruction::<WORD_SIZE>(u64_max, u64_max),
            SLTUInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            SLTUInstruction::<WORD_SIZE>(1 << 32, u64_max),
            SLTUInstruction::<WORD_SIZE>(1 << 63, 1 << 63 - 1),
            SLTUInstruction::<WORD_SIZE>(1 << 63 - 1, 1 << 63),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
