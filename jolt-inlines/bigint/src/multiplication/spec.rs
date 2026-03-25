use super::sequence_builder::BigintMul256;
use super::{INPUT_LIMBS, OUTPUT_LIMBS};
use jolt_inlines_sdk::host::Xlen;
use jolt_inlines_sdk::spec::rand::rngs::StdRng;
use jolt_inlines_sdk::spec::rand::Rng;
use jolt_inlines_sdk::spec::{InlineMemoryLayout, InlineSpec, InlineTestHarness, INLINE};

impl InlineSpec for BigintMul256 {
    type Input = ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS]);
    type Output = [u64; OUTPUT_LIMBS];

    fn random_input(rng: &mut StdRng) -> Self::Input {
        (
            core::array::from_fn(|_| rng.gen()),
            core::array::from_fn(|_| rng.gen()),
        )
    }

    fn reference(input: &Self::Input) -> Self::Output {
        let (lhs, rhs) = input;
        let mut result = [0u64; OUTPUT_LIMBS];

        for (i, &lhs_limb) in lhs.iter().enumerate() {
            for (j, &rhs_limb) in rhs.iter().enumerate() {
                let product = (lhs_limb as u128) * (rhs_limb as u128);
                let low = product as u64;
                let high = (product >> 64) as u64;

                let result_position = i + j;

                let (sum, carry1) = result[result_position].overflowing_add(low);
                result[result_position] = sum;

                let mut carry = carry1 as u64;
                if high != 0 || carry != 0 {
                    let (sum_with_hi, carry_hi) = result[result_position + 1].overflowing_add(high);
                    let (sum_with_carry, carry_carry) = sum_with_hi.overflowing_add(carry);
                    result[result_position + 1] = sum_with_carry;
                    carry = (carry_hi as u64) + (carry_carry as u64);

                    let mut carry_position = result_position + 2;
                    while carry != 0 && carry_position < OUTPUT_LIMBS {
                        let (sum, c) = result[carry_position].overflowing_add(carry);
                        result[carry_position] = sum;
                        carry = c as u64;
                        carry_position += 1;
                    }
                }
            }
        }
        result
    }

    fn create_harness() -> InlineTestHarness {
        let layout = InlineMemoryLayout::two_inputs(32, 32, 64);
        InlineTestHarness::new(layout, Xlen::Bit64)
    }

    fn instruction() -> INLINE {
        InlineTestHarness::create_default_instruction(
            crate::multiplication::INLINE_OPCODE,
            crate::multiplication::BIGINT256_MUL_FUNCT3,
            crate::multiplication::BIGINT256_MUL_FUNCT7,
        )
    }

    fn load(harness: &mut InlineTestHarness, input: &Self::Input) {
        harness.setup_registers();
        harness.load_input64(&input.0);
        harness.load_input2_64(&input.1);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let vec = harness.read_output64(OUTPUT_LIMBS);
        vec.try_into().unwrap()
    }
}
