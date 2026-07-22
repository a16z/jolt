use jolt_inlines_sdk::{InlineReference, InlineSpec};
use rand::RngCore;
use jolt_tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

use super::{exec, INPUT_LIMBS, OUTPUT_LIMBS};
use crate::multiplication::sequence_builder::BigintMul256;

pub type BigIntInput = ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS]);
pub type BigIntOutput = [u64; OUTPUT_LIMBS];

impl InlineReference for BigintMul256 {
    type Input = BigIntInput;
    type Output = BigIntOutput;

    fn reference((lhs, rhs): &Self::Input) -> Self::Output {
        exec::bigint_mul(*lhs, *rhs)
    }
}

impl InlineSpec for BigintMul256 {
    fn edge_cases() -> impl IntoIterator<Item = Self::Input> {
        let zero = [0u64; INPUT_LIMBS];
        let one = [1u64, 0, 0, 0];
        let max = [u64::MAX; INPUT_LIMBS];
        let two = [2u64, 0, 0, 0];
        let pow2_64 = [0, 1, 0, 0];
        let pow2_128 = [0, 0, 1, 0];
        let single_limb_max = [u64::MAX, 0, 0, 0];
        let alternating = [
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
        ];
        let msb_only = [0x8000000000000000; INPUT_LIMBS];
        let lsb_only = [1; INPUT_LIMBS];
        let sequential = [1, 2, 3, 4];
        let sequential2 = [5, 6, 7, 8];
        let high_limb_only = [0, 0, 0, u64::MAX];
        let low_limb_only = [u64::MAX, 0, 0, 0];
        let middle_limbs = [0, u64::MAX, u64::MAX, 0];
        let mersenne_like = [u64::MAX, u64::MAX, u64::MAX, 0x7FFFFFFFFFFFFFFF];

        [
            (zero, zero),
            (zero, one),
            (one, zero),
            (one, one),
            (max, zero),
            (zero, max),
            (max, one),
            (one, max),
            (max, max),
            (pow2_64, pow2_128),
            (max, two),
            (single_limb_max, single_limb_max),
            (alternating, alternating),
            (msb_only, msb_only),
            (lsb_only, lsb_only),
            (sequential, sequential2),
            (high_limb_only, high_limb_only),
            (low_limb_only, low_limb_only),
            (middle_limbs, middle_limbs),
            (mersenne_like, two),
        ]
    }

    fn random(rng: &mut impl RngCore) -> Self::Input {
        (
            core::array::from_fn(|_| rng.next_u64()),
            core::array::from_fn(|_| rng.next_u64()),
        )
    }

    fn harness() -> InlineTestHarness {
        InlineTestHarness::new(InlineMemoryLayout::two_inputs(32, 32, 64))
    }

    fn load(harness: &mut InlineTestHarness, (lhs, rhs): &Self::Input) {
        harness.load_input64(lhs);
        harness.load_input2_64(rhs);
    }

    fn read(harness: &mut InlineTestHarness) -> Self::Output {
        let result = harness.read_output64(OUTPUT_LIMBS);
        result.try_into().unwrap()
    }
}
