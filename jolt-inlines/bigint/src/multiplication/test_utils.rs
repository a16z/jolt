use super::{BIGINT256_MUL_FUNCT3, BIGINT256_MUL_FUNCT7, INLINE_OPCODE, INPUT_LIMBS, OUTPUT_LIMBS};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{bigint_helpers, InlineTestHarness};

pub type BigIntInput = ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS]);
pub type BigIntOutput = [u64; OUTPUT_LIMBS];

pub const RS1: u8 = 10;
pub const RS2: u8 = 11;
pub const RS3: u8 = 12;

pub fn create_bigint_harness() -> InlineTestHarness {
    bigint_helpers::bigint256_mul_harness(Xlen::Bit64)
}

pub fn instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_instruction(
        INLINE_OPCODE,
        BIGINT256_MUL_FUNCT3,
        BIGINT256_MUL_FUNCT7,
        RS1,
        RS2,
        RS3,
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
        harness.setup_registers(RS1, RS2, Some(RS3));
        harness.load_input64(lhs);
        harness.load_input2_64(rhs);
        harness.execute_inline(instruction());

        let result_vec = harness.read_output64(OUTPUT_LIMBS);
        let mut result = [0u64; OUTPUT_LIMBS];
        result.copy_from_slice(&result_vec);

        assert_eq!(&result, expected, "BigInt multiplication result mismatch");
    }
}
