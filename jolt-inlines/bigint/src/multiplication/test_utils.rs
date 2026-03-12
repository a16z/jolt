use super::{BIGINT256_MUL_FUNCT3, BIGINT256_MUL_FUNCT7, INLINE_OPCODE, INPUT_LIMBS, OUTPUT_LIMBS};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

pub type BigIntInput = ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS]);
pub type BigIntOutput = [u64; OUTPUT_LIMBS];

pub fn create_bigint_harness() -> InlineTestHarness {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| crate::init_inlines().unwrap());

    let layout = InlineMemoryLayout::two_inputs(32, 32, 64);
    InlineTestHarness::new(layout, Xlen::Bit64)
}

pub fn instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(
        INLINE_OPCODE,
        BIGINT256_MUL_FUNCT3,
        BIGINT256_MUL_FUNCT7,
    )
}

pub mod bigint_verify {
    use super::*;

    pub fn assert_exec_trace_equiv(
        lhs: &[u64; INPUT_LIMBS],
        rhs: &[u64; INPUT_LIMBS],
        expected: &[u64; OUTPUT_LIMBS],
    ) {
        let mut harness = create_bigint_harness();
        harness.setup_registers();
        harness.load_input64(lhs);
        harness.load_input2_64(rhs);
        harness.execute_inline(instruction());

        let result_vec = harness.read_output64(OUTPUT_LIMBS);
        let mut result = [0u64; OUTPUT_LIMBS];
        result.copy_from_slice(&result_vec);

        assert_eq!(&result, expected, "BigInt multiplication result mismatch");
    }
}
