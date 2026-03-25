use super::{INPUT_LIMBS, OUTPUT_LIMBS};
use crate::multiplication::spec::BigintMul256Spec;
use jolt_inlines_sdk::spec::InlineSpec;

pub mod bigint_verify {
    use super::*;

    pub fn assert_exec_trace_equiv(
        lhs: &[u64; INPUT_LIMBS],
        rhs: &[u64; INPUT_LIMBS],
        expected: &[u64; OUTPUT_LIMBS],
    ) {
        let input = (*lhs, *rhs);
        let mut harness = BigintMul256Spec::create_harness();
        BigintMul256Spec::load(&mut harness, &input);
        harness.execute_inline(BigintMul256Spec::instruction());
        let result = BigintMul256Spec::read(&mut harness);

        assert_eq!(&result, expected, "BigInt multiplication result mismatch");
    }
}
